"""078 sensor-only negative lateral acceleration capture, with a latched braking domain."""
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_recovery_state_env import max_tilt

def attach_fast_capture_router(env,formal,formal_norm,expert,expert_norm,spec,fast,blend_steps=4):
    state=dict(active=False,risk=0,safe=0,age=0,cooldown=0,weight=0.,formal_weight=0.,previous_vy=None,fast_latched=False)
    env.recovery_selection_log=[]
    def modify(D_action):
        cmd=env.command.copy();turn=abs(cmd[2])>spec['max_requested_yaw']
        state['formal_weight']+=float(np.clip(float(turn)-state['formal_weight'],-1/blend_steps,1/blend_steps));fw=state['formal_weight']
        base_obs=Go2TransitionEnvV3.observe(env).copy();teacher_obs=base_obs.copy();teacher_obs[11]+=np.float32(.25*env.last_correction)
        if fw:
            fa,_=formal.predict(formal_norm.normalize_obs(teacher_obs.copy()),deterministic=True)
            routed=(1-fw)*np.clip(np.asarray(D_action,dtype=np.float64),-1,1)+fw*np.clip(fa,-1,1)
        else:fa=np.asarray(D_action).copy();routed=D_action
        lin,ang=env.velocity();tilt=np.degrees(max_tilt(env));filtered=env.filtered_command.copy()
        previous=state['previous_vy'];acceleration=0. if previous is None else (float(lin[1])-previous)/fast['control_dt_s'];state['previous_vy']=float(lin[1])
        old_permitted=abs(cmd[0]-spec['only_vx'])<1e-8 and abs(cmd[1])<1e-8 and abs(cmd[2])<=spec['max_requested_yaw']
        domain=fw==0 and abs(filtered[2])<=spec['max_requested_yaw'];stop=bool(np.max(np.abs(cmd))<1e-8)
        braking=stop and filtered[0]>=fast['brake_filtered_vx_min']
        eligible=domain and (old_permitted or braking)
        fast_signal=bool(eligible and lin[1]<=fast['vy_max'] and acceleration<=fast['lateral_acceleration_max'])
        state['cooldown']=max(0,state['cooldown']-1)
        fast_entry=bool(fast_signal and not state['active'] and not state['cooldown'])
        if fast_entry:state.update(active=True,age=0,safe=0,fast_latched=True,weight=1.)
        permitted=bool(domain and (old_permitted or (state['fast_latched'] and stop)))
        risk=(permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on'])) and abs(lin[1])>=spec['vy_confirmation']
        state['risk']=state['risk']+1 if risk else 0
        if not state['active'] and not state['cooldown'] and state['risk']>=spec['risk_steps']:state.update(active=True,age=0,safe=0)
        if state['active']:
            state['age']+=1
            safe=tilt<=spec['safe_tilt_deg'] and np.linalg.norm(lin[:2]-cmd[:2])<=spec['safe_planar_error'] and abs(ang[2]-cmd[2])<=spec['safe_yaw_error'] and np.linalg.norm(ang[:2])<=spec['safe_angular_xy']
            state['safe']=state['safe']+1 if safe else 0
            if not permitted or state['age']>=spec['max_active_steps'] or (state['age']>=spec['min_active_steps'] and state['safe']>=spec['safe_steps']):state.update(active=False,risk=0,cooldown=spec['cooldown_steps'])
        state['weight']+=float(np.clip(float(state['active'])-state['weight'],-1/spec['blend_steps'],1/spec['blend_steps']));base=np.clip(np.asarray(routed,dtype=np.float64),-1,1)
        if state['weight']:
            obs=Go2TransitionEnvV3.observe(env);action,_=expert.predict(expert_norm.normalize_obs(obs.copy()),deterministic=True)
            applied=(1-state['weight'])*base+state['weight']*np.clip(action,-1,1)
        else:action=base.copy();applied=base.copy()
        assert np.max(np.abs(applied))<=1+1e-12
        env.recovery_selection_log.append(dict(active=state['active'],weight=state['weight'],risk=bool(risk),tilt=tilt,vy=float(lin[1]),angular_xy=float(np.linalg.norm(ang[:2])),
            permitted=permitted,base_action=base.tolist(),expert_action=np.asarray(action).tolist(),applied_action=applied.tolist(),formal_weight=fw,D_action=np.asarray(D_action).tolist(),
            formal_action=np.asarray(fa).tolist(),filtered_command=filtered.tolist(),requested_command=cmd.tolist(),last_correction=float(env.last_correction),teacher_observation=teacher_obs.tolist(),
            base_observation=base_obs.tolist(),old_permitted=bool(old_permitted),domain=bool(domain),previous_vy=previous,lateral_acceleration=acceleration,
            fast_eligible=bool(eligible),fast_signal=fast_signal,fast_entry=fast_entry,braking=bool(braking),fast_latched=state['fast_latched']))
        if not state['active'] and state['weight']==0:state['fast_latched']=False
        return applied
    env.modify_joint_action=modify;return env
