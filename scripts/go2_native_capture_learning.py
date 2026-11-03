"""133 native-action learning, exact cached states and sensor-only deployment."""
import copy, inspect
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from go2_late_phase_capture import candidate_hold
from go2_bilateral_pilot import CONTROL
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_complete_recovery_env import complete_recovery_terms, REWARD_PART_NAMES
from go2_recovery_state_env import recovery_reward
from validate_go2_context_states import local_force, command_at


def attach_learner(env, stack, model=None):
    native=env.modify_joint_action; reset=env.reset; hold=dict(active=False)
    env.learner_action=None;env.learner_calls=[]
    def modify(action):
        requested=native(action);log=env.recovery_selection_log[-1]
        active=candidate_hold(log,hold);teacher=np.asarray(log['teacher_observation'],np.float32)
        raw=Go2TransitionEnvV3.observe(env).copy()
        # Straight command means the teacher's correction channel is zero.
        assert abs(env.command[2])<1e-10 and env.last_correction==0
        np.testing.assert_array_equal(raw,teacher)
        clipped=None
        if active and (model is not None or env.learner_action is not None):
            if env.learner_action is not None:clipped=np.clip(env.learner_action,-1,1).astype(np.float32)
            else:clipped=np.asarray(model.predict(stack.formal[1].normalize_obs(teacher.copy()),deterministic=True)[0],np.float32)
            applied=clipped.astype(np.float64)
        else:applied=requested
        used=clipped is not None
        env.learner_calls.append(dict(observation=teacher,action=np.zeros(12,np.float32) if clipped is None else clipped.copy(),applied=used,eligible=active,requested=np.asarray(requested).copy(),actual=np.asarray(applied).copy()))
        log.update(learner_applied=used,learner_eligible=active,learner_hold=hold['active'],learner_requested_action=log['applied_action'],learner_action=None if clipped is None else clipped.tolist())
        log['applied_action']=np.asarray(applied).tolist()
        return applied
    def wrapped_reset(*args,**kwargs):
        hold.update(active=False);env.learner_action=None;env.learner_calls=[]
        return reset(*args,**kwargs)
    env.modify_joint_action=modify;env.reset=wrapped_reset
    return env


def state_dicts(env):
    result=[];seen=set();functions=set()
    def walk(function):
        if not inspect.isfunction(function) or id(function) in functions:return
        functions.add(id(function))
        for name,value in inspect.getclosurevars(function).nonlocals.items():
            if isinstance(value,dict) and name in ['state','hold','scope','fast_state','initial'] and id(value) not in seen:
                result.append(value);seen.add(id(value))
            if inspect.isfunction(value):walk(value)
    for function in [env.modify_joint_action,env.step,env.reset]:walk(function)
    return result


def cache(env):
    dictionaries=state_dicts(env)
    return dict(data=copy.copy(env.data),friction=env.model.geom_friction.copy(),
        control={key:copy.deepcopy(getattr(env,key)) for key in CONTROL},
        dictionaries=dictionaries,values=copy.deepcopy(dictionaries))


def restore(env,saved):
    env.data=copy.copy(saved['data']);env.model.geom_friction[:]=saved['friction']
    for key,value in saved['control'].items():setattr(env,key,copy.deepcopy(value))
    for target,value in zip(saved['dictionaries'],saved['values']):target.clear();target.update(copy.deepcopy(value))
    env.recovery_selection_log=[];env.scoped_calls=[];env.observed_burst_calls=[]
    env.learner_calls=[];env.probe_telemetry=[];env.traction_trace=[];env.traction_changes=[]
    env.scoped_query_snapshots={};env.scoped_extra_before={};env.observed_burst_forks={};env.validation_full_queries={}
    env.learner_action=None


def prepare_inputs(env,record,zero=False):
    protocol=record['protocol'];i=env.step_count
    env.command[:]=command_at(protocol,i)[0]
    point=next(p for p in reversed(protocol['friction_schedule']) if p['step']<=i)
    env.model.geom_friction[[env.floor]+sorted(env.feet),0]=point['mu']
    env.data.xfrc_applied[:]=0
    if not zero:
        force=local_force(protocol['events'],i);rotation=env.data.xmat[env.base].reshape(3,3)
        env.data.xfrc_applied[env.base]=np.r_[rotation@force[:3],rotation@force[3:]]


def drive(env,record,stack,action=None):
    prepare_inputs(env,record)
    observation=env.observe().copy()
    adapter=stack.adapter[0].predict(stack.adapter[1].normalize_obs(observation.copy()),deterministic=True)[0]
    env.learner_action=action
    result=env.step(adapter);env.learner_action=None
    return observation,adapter,result


class NativeCaptureEnv(gym.Env):
    def __init__(self,envs,records,caches,stack,worker,out):
        self.envs=envs;self.records=records;self.caches=caches;self.stack=stack;self.worker=worker
        self.index=-1;self.out=out;out.mkdir(parents=True,exist_ok=False);self.trace=[]
        self.observation_space=spaces.Box(-np.inf,np.inf,(50,),np.float32)
        self.action_space=spaces.Box(-1,1,(12,),np.float32)
    def flush(self):
        if self.trace:
            path=self.out/f'episode-{self.index:05d}.npz'
            assert not path.exists();np.savez_compressed(path,**{k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]})
        self.trace=[]
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.flush();self.index+=1;bid=(self.index+self.worker)%len(self.envs)
        self.env=self.envs[bid];self.record=self.records[bid];restore(self.env,self.caches[bid]);self.stable=0
        prepare_inputs(self.env,self.record)
        return Go2TransitionEnvV3.observe(self.env).copy(),{}
    def step(self,action):
        observation=Go2TransitionEnvV3.observe(self.env).copy();previous=self.env.last_action.copy()
        clipped=np.clip(action,-1,1).astype(np.float32)
        _,_,(_,_,terminated,truncated,info)=drive(self.env,self.record,self.stack,clipped)
        call=self.env.learner_calls[-1]
        np.testing.assert_array_equal(observation,call['observation'])
        if call['applied']:
            np.testing.assert_array_equal(call['actual'],clipped.astype(np.float64))
            np.testing.assert_array_equal(self.env.data.ctrl,np.clip(self.env.nominal+.35*clipped.astype(np.float64),self.env.model.actuator_ctrlrange[:,0],self.env.model.actuator_ctrlrange[:,1]))
        _,parts=recovery_reward(self.env,previous,self.env.last_action.copy(),terminated)
        lin,ang=self.env.velocity();additions,self.stable,metrics=complete_recovery_terms(lin,ang,self.env.command,self.stable);parts.update(additions)
        reward=float(sum(parts.values())*.1)
        self.trace.append(dict(observation=observation,action=clipped,actor_applied=call['applied'],eligible=call['eligible'],
            reward=reward,reward_parts=np.asarray([parts[k] for k in REWARD_PART_NAMES]),qpos=self.env.data.qpos.copy(),qvel=self.env.data.qvel.copy(),
            targets=self.env.data.ctrl.copy(),world_wrench=self.env.data.xfrc_applied[self.env.base].copy(),time=float(self.env.data.time),
            terminated=terminated,truncated=truncated))
        info.update(actor_applied=call['applied'],record=self.record['case'])
        if not(terminated or truncated):prepare_inputs(self.env,self.record)
        obs=Go2TransitionEnvV3.observe(self.env).copy()
        if terminated or truncated:self.flush()
        return obs,reward,terminated,truncated,info
    def close(self):self.flush()
