"""248 retain212 low routing, introduce244 on straight high requests."""
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3
from run_go2_acceleration_bias_reset import LineController

def attach(env,actor,norm):
    native=env.modify_joint_action
    env.dual_calls=[]
    state=dict(weight=0.,latched=False,controller=LineController())
    def modify(action):
        low=np.asarray(native(action)).copy()
        command=env.command
        if abs(command[2])>1e-8:state['latched']=False
        elif command[0]>.4:state['latched']=True
        elif env.filtered_command[0]<=.4:state['latched']=False
        log=env.recovery_selection_log[-1]
        allowed=not (log['weight']>0 or log['learner_eligible'] or log['formal_weight']>0)
        target=float(state['latched'] and allowed)
        state['weight']+=float(np.clip(target-state['weight'],-1/15,1/15))
        if abs(state['weight'])<1e-12:state['weight']=0.
        weight=state['weight'];raw=Go2TransitionEnvV3.observe(env).copy()
        high=np.zeros(12,np.float32);policy_input=raw.copy()
        if weight:
            policy_input,_=state['controller'].observe(env,raw)
            high=np.clip(actor.predict(norm.normalize_obs(policy_input.copy()),deterministic=True)[0],-1,1)
            actual=((1-weight)*low.astype(np.float64)+weight*high.astype(np.float64))
        else:
            actual=low.copy();state['controller']=LineController()
        env.dual_calls.append(dict(weight=weight,latched=state['latched'],allowed=allowed,raw=raw,
            policy_input=policy_input,low=low,high=high.copy(),actual=actual.copy()))
        return actual
    env.modify_joint_action=modify
    return env
