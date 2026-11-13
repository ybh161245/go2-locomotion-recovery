"""Fixed proprioceptive prototype selector; no access to force schedule/mass/time."""
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_recovery_state_env import max_tilt


def attach_selector(env,expert,norm,spec):
    state=dict(active=False,risk=0,safe=0,age=0,cooldown=0,weight=0.)
    env.recovery_selection_log=[]
    def modify(base_action):
        lin,ang=env.velocity();tilt=np.degrees(max_tilt(env));cmd=env.command.copy()
        permitted=abs(cmd[0]-spec['only_vx'])<1e-8 and abs(cmd[1])<1e-8 and abs(cmd[2])<=spec['max_requested_yaw']
        risk=permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on'])
        state['cooldown']=max(0,state['cooldown']-1)
        state['risk']=state['risk']+1 if risk else 0
        if not state['active'] and not state['cooldown'] and state['risk']>=spec['risk_steps']:
            state.update(active=True,age=0,safe=0)
        if state['active']:
            state['age']+=1
            safe=tilt<=spec['safe_tilt_deg'] and np.linalg.norm(lin[:2]-cmd[:2])<=spec['safe_planar_error'] and abs(ang[2]-cmd[2])<=spec['safe_yaw_error'] and np.linalg.norm(ang[:2])<=spec['safe_angular_xy']
            state['safe']=state['safe']+1 if safe else 0
            if not permitted or state['age']>=spec['max_active_steps'] or (state['age']>=spec['min_active_steps'] and state['safe']>=spec['safe_steps']):
                state.update(active=False,risk=0,cooldown=spec['cooldown_steps'])
        target=float(state['active']);state['weight']+=float(np.clip(target-state['weight'],-1/spec['blend_steps'],1/spec['blend_steps']))
        base=np.clip(np.asarray(base_action,dtype=np.float64),-1,1)
        if state['weight']:
            obs=Go2TransitionEnvV3.observe(env);action,_=expert.predict(norm.normalize_obs(obs.copy()),deterministic=True)
            applied=(1-state['weight'])*base+state['weight']*np.clip(action,-1,1)
        else:action=base.copy();applied=base.copy()
        assert np.max(np.abs(applied))<=1+1e-12
        env.recovery_selection_log.append(dict(active=state['active'],weight=state['weight'],risk=bool(risk),tilt=tilt,
            vy=float(lin[1]),angular_xy=float(np.linalg.norm(ang[:2])),permitted=bool(permitted),
            base_action=base.tolist(),expert_action=np.asarray(action).tolist(),applied_action=applied.tolist()))
        return applied
    env.modify_joint_action=modify
    return env
