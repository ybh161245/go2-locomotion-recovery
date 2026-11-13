"""141 fixed random-feature residual actor: pure normalized observation -> actual action."""
import numpy as np
from stable_baselines3 import PPO
from go2_late_phase_capture import candidate_hold
from go2_transition_env_v3 import Go2TransitionEnvV3

def features(x,frequency,phase):
    a=np.asarray(x,np.float64)
    return np.sqrt(2/frequency.shape[1])*np.cos(a@frequency+phase)

class RandomFeatureActor:
    def __init__(self,parent_path,weights):
        self.parent=PPO.load(parent_path,device='cpu')
        with np.load(weights) as z:
            self.frequency=z['frequency'].copy();self.phase=z['phase'].copy();self.coefficient=z['coefficient'].copy()
    def predict(self,observation,deterministic=True):
        assert deterministic
        x=np.asarray(observation,np.float32);assert x.shape==(50,)
        base=np.asarray(self.parent.predict(x.copy(),deterministic=True)[0],np.float32).astype(np.float64)
        # No exemplar lookup, time index, case, material or applied force enters this function.
        actual=np.clip(base+features(x,self.frequency,self.phase)@self.coefficient,-1,1)
        return actual,None

def attach_rf_learner(env,stack,model):
    native=env.modify_joint_action;reset=env.reset;hold=dict(active=False);env.learner_calls=[]
    def modify(action):
        requested=native(action);log=env.recovery_selection_log[-1];active=candidate_hold(log,hold)
        teacher=np.asarray(log['teacher_observation'],np.float32);raw=Go2TransitionEnvV3.observe(env).copy();raw[11]+=np.float32(.25*env.last_correction)
        np.testing.assert_array_equal(raw,teacher)
        clipped=np.asarray(model.predict(stack.formal[1].normalize_obs(teacher.copy()),deterministic=True)[0],np.float64) if active else None
        applied=clipped if clipped is not None else requested
        env.learner_calls.append(dict(observation=teacher,action=np.zeros(12,np.float64) if clipped is None else clipped.copy(),
            applied=active,eligible=active,requested=np.asarray(requested).copy(),actual=np.asarray(applied).copy()))
        log.update(learner_applied=active,learner_eligible=active,learner_hold=hold['active'],
            learner_requested_action=log['applied_action'],learner_action=None if clipped is None else clipped.tolist(),
            learner_output_dtype=None if clipped is None else 'float64',learned_representation='fixed-random-feature-residual')
        log['applied_action']=np.asarray(applied).tolist();return applied
    def wrapped_reset(*args,**kwargs):hold.update(active=False);env.learner_calls=[];return reset(*args,**kwargs)
    env.modify_joint_action=modify;env.reset=wrapped_reset;return env
