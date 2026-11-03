"""155 contact-augmented residual actions over immutable145; native sensor inputs only."""
import copy
import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from diagnose_go2_update import dump
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_native_capture_learning import restore, prepare_inputs
from go2_recovery_state_env import recovery_reward
from go2_complete_recovery_env import complete_recovery_terms, REWARD_PART_NAMES


def contact_observation(env, norm):
    raw=Go2TransitionEnvV3.observe(env).copy()
    raw[11]+=np.float32(.25*env.last_correction)
    feet=env.meter.sample(env.data)
    # Flat-floor clearance and world velocity rotated into body axes. No material/force/time labels.
    velocity=feet['velocity']@env.data.xmat[env.base].reshape(3,3)
    extra=np.r_[feet['contact'].astype(np.float32),np.clip(feet['height']/.1,-10,10),np.clip(velocity.ravel()/.5,-10,10)]
    obs=np.r_[norm.normalize_obs(raw.copy()),extra].astype(np.float32)
    assert obs.shape==(70,) and np.isfinite(obs).all()
    return obs


def combine(parent,residual,scale):
    # Preserve original float32/float64 output exactly at the identity endpoint.
    if not np.any(residual):return np.asarray(parent).copy()
    return np.clip(np.asarray(parent,np.float64)+scale*np.asarray(residual,np.float64),-1,1)


def attach_contact_residual(env,norm,scale,model=None):
    native=env.modify_joint_action;reset=env.reset
    env.contact_calls=[];env.residual_override=None
    def modify(action):
        obs=contact_observation(env,norm);parent=native(action);log=env.recovery_selection_log[-1]
        eligible=bool(log['learner_eligible']);residual=np.zeros(12,np.float32);called=False
        if eligible and (env.residual_override is not None or model is not None):
            value=env.residual_override if env.residual_override is not None else model.predict(obs.copy(),deterministic=True)[0]
            residual=np.clip(value,-1,1).astype(np.float32);called=True
        actual=combine(parent,residual,scale) if called else np.asarray(parent).copy()
        changed=bool(not np.array_equal(actual,parent))
        env.contact_calls.append(dict(step=int(env.step_count),observation=obs,parent=np.asarray(parent).copy(),residual=residual,
                                      actual=actual.copy(),eligible=eligible,actor_applied=called,changed=changed,dtype=str(actual.dtype)))
        log.update(contact_residual_applied=called,contact_residual_changed=changed,parent145_shadow_action=np.asarray(parent).tolist(),
                   contact_residual=residual.tolist(),actual_output_dtype=str(actual.dtype))
        log['applied_action']=actual.tolist()
        return actual
    def wrapped_reset(*a,**kw):
        env.contact_calls=[];env.residual_override=None;return reset(*a,**kw)
    env.modify_joint_action=modify;env.reset=wrapped_reset
    return env


def drive(env,record,stack,residual=None):
    prepare_inputs(env,record);obs=env.observe().copy()
    action=stack.adapter[0].predict(stack.adapter[1].normalize_obs(obs.copy()),deterministic=True)[0]
    env.residual_override=residual
    result=env.step(action);env.residual_override=None
    return obs,action,result


def packed(env,obs,action):
    return dict(observations=obs,actions=action,qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy(),
                world_wrench=env.data.xfrc_applied[env.base].copy(),feet_world=env.data.geom_xpos[env.meter.feet].copy(),heading=env.data.xmat[env.base].reshape(3,3)[:,0].copy())


class ContactResidualEnv(gym.Env):
    def __init__(self,envs,records,caches,stack,norm,out):
        self.envs=envs;self.records=records;self.caches=caches;self.stack=stack;self.norm=norm;self.out=out
        out.mkdir(parents=True,exist_ok=False);self.index=-1;self.trace=[];self.counts=np.zeros(len(envs),np.int64);self.applied=np.zeros(len(envs),np.int64)
        self.observation_space=spaces.Box(-np.inf,np.inf,(70,),np.float32);self.action_space=spaces.Box(-1,1,(12,),np.float32)
    def flush(self):
        if self.trace:
            path=self.out/f'episode-{self.index:05d}';assert not path.with_suffix('.npz').exists()
            np.savez_compressed(path.with_suffix('.npz'),**{k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]})
            dump(path.with_suffix('.json'),dict(episode=self.index,source_index=int(self.bid),case=self.record['case'],seed=self.record['seed'],
                source_folder=self.record['folder'],start_step=int(self.caches[self.bid]['control']['step_count']),steps=len(self.trace),
                terminated=bool(self.trace[-1]['terminated']),truncated=bool(self.trace[-1]['truncated'])))
        self.trace=[]
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.flush();self.index+=1
        self.bid=int(np.argmax((self.counts.sum()+1)/len(self.counts)-self.counts));self.env=self.envs[self.bid];self.record=self.records[self.bid]
        restore(self.env,self.caches[self.bid]);self.env.contact_calls=[];self.env.residual_override=None;self.stable=0
        prepare_inputs(self.env,self.record)
        return contact_observation(self.env,self.norm),{}
    def step(self,action):
        obs=contact_observation(self.env,self.norm);previous=self.env.last_action.copy();clipped=np.clip(action,-1,1).astype(np.float32)
        _,_,(_,_,term,trunc,info)=drive(self.env,self.record,self.stack,clipped);call=self.env.contact_calls[-1]
        np.testing.assert_array_equal(obs,call['observation'])
        _,parts=recovery_reward(self.env,previous,self.env.last_action.copy(),term)
        lin,ang=self.env.velocity();extra,self.stable,_=complete_recovery_terms(lin,ang,self.env.command,self.stable);parts.update(extra)
        reward=float(sum(parts.values())*.1);self.counts[self.bid]+=1;self.applied[self.bid]+=int(call['actor_applied'])
        self.trace.append(dict(observation=obs,residual=clipped,parent_action=call['parent'],actual_action=call['actual'],
            actor_applied=call['actor_applied'],eligible=call['eligible'],reward=reward,reward_parts=np.asarray([parts[k] for k in REWARD_PART_NAMES]),
            qpos=self.env.data.qpos.copy(),qvel=self.env.data.qvel.copy(),targets=self.env.data.ctrl.copy(),
            world_wrench=self.env.data.xfrc_applied[self.env.base].copy(),time=float(self.env.data.time),fresh_velocity=lin.copy(),fresh_angular=ang.copy(),
            legacy_velocity=np.asarray([info['vx'],info['vy'],info['yaw_rate']]),terminated=term,truncated=trunc))
        info.update(actor_applied=call['actor_applied'],record=self.record['case'])
        if not(term or trunc):prepare_inputs(self.env,self.record)
        nextobs=contact_observation(self.env,self.norm)
        if term or trunc:self.flush()
        return nextobs,reward,term,trunc,info
    def close(self):self.flush()


def save_calls(env,folder):
    (folder/'contact-calls.pkl').write_bytes(pickle.dumps(env.contact_calls,protocol=5))
    for name,calls in [('learner-shadow-actions',env.learner_calls)]:
        np.savez_compressed(folder/(name+'.npz'),**{k:np.asarray([c[k] for c in calls]) for k in calls[0]})
