"""111 current-stack recovery interaction pilot; no controller access to force labels."""
import copy,inspect,pickle
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import torch
from archive_project import ROOT
from diagnose_go2_update import read
import evaluate_go2_robust as ev
from screen_go2_mass_course import factory,adapter_policy
from screen_go2_recovery_specialist import load_expert
from go2_unified_recovery_selection import attach_unified_recovery_selection
from go2_transition_env_v3 import Go2TransitionEnvV3
from validate_go2_context_states import attach_initial_state,command_at,local_force
from go2_recovery_state_env import recovery_reward

CONTROL=['command','last_action','step_count','gait_phase','filtered_command','filtered_frequency','in_step',
 'is_reversal','turn_sign','turn_age','heading_error','target_heading','measured_heading','last_heading',
 'last_correction','residual_stepping','_velocity']


class Stack:
    def __init__(self,cfg):
        self.cfg=cfg;ev.CONFIG=cfg;spec=cfg['support_context'];setting=cfg['unified_recovery_selection']
        self.D=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint'])
        self.formal=ev.load_policy(ROOT/cfg['motion_composition']['formal_joint'])
        self.default=load_expert(ROOT/spec['default_train'],spec['default_steps'])
        self.low=load_expert(ROOT/spec['low_train'],spec['low_steps']);self.adapter=adapter_policy(cfg,self.D)
        with np.load(ROOT/setting['bank']/'bank.npz') as z:self.bank=z['prototypes'].copy()
        self.replacement=None;self.action=None;self.expert_input=None;self.used=False

    def make(self,record):
        stack=self;cfg=self.cfg
        class Default:
            def predict(self,observation,deterministic=True):
                stack.expert_input=np.asarray(observation).copy();stack.used=True
                if stack.action is not None:return stack.action.copy(),None
                return (stack.replacement or stack.default[0]).predict(observation,deterministic=deterministic)
        p=record['protocol'];plain=factory(self.D,dict(id='bilateral',factor='friction',level=p['friction'],scope='111-current-stack-pilot'))
        env=attach_unified_recovery_selection(plain('external',episode_seconds=p['seconds']),*self.formal,Default(),self.low[0],self.default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],self.bank,cfg['unified_recovery_selection'])
        return attach_initial_state(env,record['profile'])

    def drive(self,env,record,action=None):
        p=record['protocol'];i=env.step_count;cmd,phase=command_at(p,i);env.command[:]=cmd
        point=next(x for x in reversed(p['friction_schedule']) if x['step']<=i)
        env.model.geom_friction[[env.floor]+sorted(env.feet),0]=point['mu']
        local=local_force(p['events'],i);R=env.data.xmat[env.base].reshape(3,3)
        env.data.xfrc_applied[:]=0;env.data.xfrc_applied[env.base]=np.r_[R@local[:3],R@local[3:]]
        self.action=None if action is None else np.asarray(action).copy();self.used=False;self.expert_input=None
        obs=env.observe().copy();a,_=self.adapter[0].predict(self.adapter[1].normalize_obs(obs.copy()),deterministic=True)
        transition=env.step(a);self.action=None
        return obs,a,transition

    def close(self):
        for m,n in [self.D,self.formal,self.default,self.low,self.adapter]:n.close()


def capture(env):
    closure=inspect.getclosurevars(env.modify_joint_action).nonlocals
    return dict(data=copy.copy(env.data),friction=env.model.geom_friction.copy(),
        control={k:copy.deepcopy(getattr(env,k)) for k in CONTROL},
        selection=copy.deepcopy(closure['state']),fast=copy.deepcopy(closure['fast_state']))


def restore(env,s):
    # MjData copy retains both complete integration inputs and derived contacts;
    # there is no extra mj_forward which could change the first control action.
    env.data=copy.copy(s['data']);env.model.geom_friction[:]=s['friction']
    for k,v in s['control'].items():setattr(env,k,copy.deepcopy(v))
    closure=inspect.getclosurevars(env.modify_joint_action).nonlocals
    for key,target in [('selection','state'),('fast','fast_state')]:
        closure[target].clear();closure[target].update(copy.deepcopy(s[key]))
    env.recovery_selection_log=[];env.probe_telemetry=[]


class InteractionEnv(gym.Env):
    """Reset from exact observed entries, act through unchanged D/yaw/router/PD.

    A pilot action replaces default only. Last/low/zero-weight transitions are
    excluded from actor loss by AppliedOnlyBuffer, rather than pretending that
    every sampled action influenced the robot. GAE retains the resulting rewards.
    """
    def __init__(self,cfg,records,caches,worker,archive):
        self.cfg=cfg;self.records=records;self.caches=caches;self.worker=worker;self.index=-1
        self.stack=Stack(cfg);self.envs=[self.stack.make(r) for r in records]
        self.action_space=spaces.Box(-1.,1.,(12,),np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(50,),np.float32)
        self.archive=archive;self.archive.mkdir(parents=True,exist_ok=False);self.trace=[]

    def flush(self,reason):
        if self.trace:
            path=self.archive/f'episode-{self.index:05d}'
            np.savez_compressed(path.with_suffix('.npz'),**{k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]})
            from diagnose_go2_update import dump
            dump(path.with_suffix('.json'),dict(record=self.record['case'],worker=self.worker,episode=self.index,reason=reason,steps=len(self.trace)))
        self.trace=[]

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.flush('reset');self.index+=1
        bid=(self.worker+self.index)%len(self.records);self.record=self.records[bid];self.env=self.envs[bid]
        restore(self.env,self.caches[bid]);self.elapsed=0
        return Go2TransitionEnvV3.observe(self.env).copy(),{}

    def step(self,action):
        obs=Go2TransitionEnvV3.observe(self.env).copy();previous=self.env.last_action.copy()
        clipped=np.clip(action,-1,1).astype(np.float32)
        _,_,(_,base,term,trunc,info)=self.stack.drive(self.env,self.record,clipped);self.elapsed+=1
        weight=self.env.recovery_selection_log[-1]['weight'];applied=bool(self.stack.used and weight>0)
        if applied:
            expected=self.stack.default[1].normalize_obs(obs.copy())
            np.testing.assert_array_equal(self.stack.expert_input,expected)
        reward,parts=recovery_reward(self.env,previous,self.env.last_action.copy(),term);reward*=self.cfg['bilateral_pilot']['reward_scale']
        # End after native recovery hands back, or at the declared two-second
        # suffix horizon; normal locomotion remains supplied by frozen actors.
        trunc=bool(trunc or weight==0 or self.elapsed>=self.cfg['bilateral_pilot']['episode_steps'])
        self.trace.append(dict(observation=obs,action=clipped,actor_applied=applied,weight=weight,reward=reward,
            reward_parts=np.asarray(list(parts.values())),qpos=self.env.data.qpos.copy(),qvel=self.env.data.qvel.copy(),
            targets=self.env.data.ctrl.copy(),force=self.env.data.xfrc_applied[self.env.base].copy(),time=float(self.env.data.time),
            terminated=term,truncated=trunc))
        info.update(actor_applied=applied,record=self.record['case'])
        result=Go2TransitionEnvV3.observe(self.env).copy()
        if term or trunc:self.flush('terminated' if term else 'native_exit_or_horizon')
        return result,reward,term,trunc,info

    def close(self):
        self.flush('training_end_partial')
        for e in self.envs:e.close()
        self.stack.close()
