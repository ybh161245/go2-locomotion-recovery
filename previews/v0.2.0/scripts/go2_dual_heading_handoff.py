"""249 continue a fast-entry heading reference through low speed and stand."""
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3

def attach(env):
    native=env.modify_joint_action;reset=env.reset
    anchor=None
    env.handoff_calls=[]
    def modify(action):
        nonlocal anchor
        actual=np.asarray(native(action)).copy();call=env.dual_calls[-1]
        yaw=env.yaw()
        if abs(env.command[2])>1e-8:anchor=None
        elif anchor is None and call['weight']>0:anchor=yaw
        log=env.recovery_selection_log[-1]
        allowed=(anchor is not None and call['weight']==0 and abs(env.command[2])<1e-8
            and log['formal_weight']==0 and log['weight']==0 and not log['learner_eligible'])
        error=0. if anchor is None else float((anchor-yaw+np.pi)%(2*np.pi)-np.pi)
        extra=float(np.clip(.4*error,-.06,.06)) if allowed else 0.
        raw=Go2TransitionEnvV3.observe(env).copy();obs=raw.copy()
        if extra:
            plain=env.teacher.predict(env.teacher_norm.normalize_obs(raw.copy()),deterministic=True)[0]
            np.testing.assert_array_equal(np.clip(plain,-1,1),actual)
            obs[11]+=np.float32(.25*extra)
            actual=np.clip(env.teacher.predict(env.teacher_norm.normalize_obs(obs.copy()),deterministic=True)[0],-1,1).astype(actual.dtype)
            call['actual']=actual.copy();call['policy_input']=obs.copy()
        env.handoff_calls.append(dict(anchor=anchor,yaw=yaw,error=error,extra=extra,allowed=allowed,
            teacher_input=obs,actual=actual.copy(),shadow=np.asarray(env.dual_calls[-1]['low']).copy()))
        return actual
    def clear(*a,**kw):
        nonlocal anchor
        anchor=None;env.handoff_calls.clear()
        return reset(*a,**kw)
    env.modify_joint_action=modify;env.reset=clear
    return env
