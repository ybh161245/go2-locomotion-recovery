"""Compact integration snapshots and independent recovery episodes over unchanged native physics."""
import copy,pickle
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from archive_project import ROOT
from go2_mass_course_env import Go2MassCourseEnv
from go2_motion_metrics import sample_row

CONTROL_KEYS=['command','last_action','step_count','gait_phase','filtered_command','filtered_frequency','in_step',
    'events','mass_scale','previous_mass_scale','total_steps','episode','episode_meta','schedule','use_sequence',
    '_velocity','episode_seconds','training_command_mode','command_mode']
SIGNATURE=mujoco.mjtState.mjSTATE_INTEGRATION


def max_tilt(env):
    w,x,y,z=env.data.qpos[3:7]
    return float(max(abs(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))),abs(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))))


def capture_state(env):
    state=np.empty(mujoco.mj_stateSize(env.model,SIGNATURE));mujoco.mj_getState(env.model,env.data,state,SIGNATURE)
    return dict(integration=state,control={k:copy.deepcopy(getattr(env,k)) for k in CONTROL_KEYS},
        rng=copy.deepcopy(env.np_random.bit_generator.state),mass=float(env.mass_scale),
        body_mass=env.model.body_mass.copy(),body_inertia=env.model.body_inertia.copy())


def restore_state(env,snapshot):
    env.reset(seed=106101,options={'command':[.2,0,0],'events':snapshot['control']['events'],'mass_scale':snapshot['mass']})
    np.testing.assert_array_equal(env.model.body_mass,snapshot['body_mass']);np.testing.assert_array_equal(env.model.body_inertia,snapshot['body_inertia'])
    mujoco.mj_setState(env.model,env.data,snapshot['integration'],SIGNATURE)
    # Rebuild kinematics and contacts, then restore the exact integration/warmstart inputs.
    mujoco.mj_forward(env.model,env.data)
    mujoco.mj_setState(env.model,env.data,snapshot['integration'],SIGNATURE)
    for k,v in snapshot['control'].items():setattr(env,k,copy.deepcopy(v))
    env.np_random.bit_generator.state=copy.deepcopy(snapshot['rng'])
    actual=np.empty_like(snapshot['integration']);mujoco.mj_getState(env.model,env.data,actual,SIGNATURE)
    np.testing.assert_array_equal(actual,snapshot['integration'])
    return env.observe()


def recovery_reward(env,previous,action,terminated):
    lin,ang=env.velocity();tilt=max_tilt(env);err=float(np.square(lin[:2]-env.command[:2]).sum())
    parts=dict(posture=4*np.exp(-tilt**2/np.radians(4)**2),tracking=2*np.exp(-err/.04),
        angular=.5*np.exp(-np.square(ang[:2]).sum()/.25),height=.5*np.exp(-(env.data.qpos[2]-.29)**2/.01),
        action_change=-.01*np.square(action-previous).sum(),joint_velocity=-.00005*np.square(env.data.qvel[6:]).sum(),
        effort=-.0002*np.square(env.data.actuator_force).sum(),excess=-5*(max(0,tilt-np.radians(8))/.1)**2,
        termination=-40. if terminated else 0.)
    return float(sum(parts.values())),{k:float(v) for k,v in parts.items()}


def row_and_state(env,obs,action,info,term):
    row=sample_row(env,env.meter,info,env.data.time,'straight')
    row.update(vy=float(info['vy']),terminated=bool(term),command_vx=float(env.command[0]),command_yaw=float(env.command[2]))
    values=dict(observations=obs.copy(),actions=np.asarray(action).copy(),qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),
        targets=env.data.ctrl.copy(),world_wrench=env.data.xfrc_applied[env.base].copy())
    return row,values


def suffix_metrics(rows,start_step,prefix_peak,complete):
    after=[r for r in rows if r['time']>6.8+1e-8]
    band=[np.hypot(r['vx']-.2,r['vy'])<=.08 and abs(r['yaw_rate'])<=.15 for r in after]
    recovery=next((max(0.,after[i]['time']-.02-6.8) for i in range(len(band)-24) if all(band[i:i+25])),None)
    peak=max([prefix_peak]+[max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows])
    term=bool(rows[-1]['terminated'])
    return dict(steps=len(rows),start_step=start_step,terminated=term,peak_tilt=float(peak),recovery_seconds=recovery,
        passed=bool(complete and not term and peak<=8 and recovery is not None and recovery<=2))


class RecoveryStateEnv(gym.Env):
    """Only the independent expert trains here; the original D actor is never mutated."""
    def __init__(self,cfg,worker=0,archive=None):
        self.cfg=cfg;self.worker=worker;self.index=-1;self.env=Go2MassCourseEnv('external',config=cfg)
        with (ROOT/cfg['recovery_specialist']['bank']/'bank.pkl').open('rb') as f:self.bank=pickle.load(f)
        self.action_space=self.env.action_space;self.observation_space=self.env.observation_space
        self.archive=Path(archive) if archive else None
        if self.archive:self.archive.mkdir(parents=True,exist_ok=False)
        self.trace=[];self.elapsed=0

    def flush(self,reason):
        if self.archive and self.trace:
            path=self.archive/f'episode-{self.index:05d}'
            np.savez_compressed(path.with_suffix('.npz'),**{k:np.asarray([r[k] for r in self.trace]) for k in self.trace[0]})
            import json
            path.with_suffix('.json').write_text(json.dumps(dict(worker=self.worker,episode=self.index,bank_id=self.entry['id'],
                mass=self.entry['snapshot']['mass'],start_step=self.entry['snapshot']['control']['step_count'],steps=len(self.trace),reason=reason),indent=2),encoding='utf-8')
        self.trace=[]

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.flush('reset_partial');self.index+=1;self.elapsed=0
        bid=int(options['bank_index']) if options and 'bank_index' in options else (self.worker+self.index)%len(self.bank)
        self.entry=self.bank[bid];obs=restore_state(self.env,self.entry['snapshot'])
        return obs,dict(bank_id=self.entry['id'])

    def step(self,action):
        obs=self.env.observe().copy();previous=self.env.last_action.copy();clipped=np.clip(action,-1,1).astype(np.float64)
        _,base_reward,term,native_trunc,info=self.env.step(clipped);self.elapsed+=1
        reward,parts=recovery_reward(self.env,previous,clipped,term)
        trunc=bool(native_trunc or self.elapsed>=self.cfg['recovery_specialist']['episode_steps'])
        item=dict(observation=obs,action=clipped,reward=reward,base_reward=base_reward,qpos=self.env.data.qpos.copy(),
            qvel=self.env.data.qvel.copy(),target=self.env.data.ctrl.copy(),world_wrench=self.env.data.xfrc_applied[self.env.base].copy(),
            previous_action=previous,time=float(self.env.data.time),terminated=term,truncated=trunc,
            reward_parts=np.asarray(list(parts.values())))
        self.trace.append(item)
        if term or trunc:self.flush('terminated' if term else 'time_limit')
        info.update(recovery_reward_parts=parts,bank_id=self.entry['id'])
        return self.env.observe(),reward,term,trunc,info

    def close(self):self.flush('training_end_partial');self.env.close()
