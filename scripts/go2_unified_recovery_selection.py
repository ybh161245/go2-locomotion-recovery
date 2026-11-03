"""098 query a fixed positive-entry memory at each original fast recovery, without startup gating."""
import copy,inspect
import numpy as np
from go2_fast_capture_router import attach_fast_capture_router

def attach_unified_recovery_selection(env,formal,formal_norm,default,low,shared_norm,selector,fast,bank,spec):
    state=dict(choice=None,distance=None,entry=False);features=np.asarray(spec['features']);prototypes=np.asarray(bank,dtype=np.float32)
    class Expert:
        def predict(self,observation,deterministic=True):
            if state['choice'] is None:
                query=np.asarray(observation,dtype=np.float32)[features];distance=float(np.sqrt(np.mean((prototypes-query)**2,axis=1)).min());state.update(choice='low' if distance<=spec['radius'] else 'default',distance=distance,entry=True)
            return (low if state['choice']=='low' else default).predict(observation,deterministic=deterministic)
    attach_fast_capture_router(env,formal,formal_norm,Expert(),shared_norm,selector,fast,4);native=env.modify_joint_action;native_reset=env.reset;fast_state=inspect.getclosurevars(native).nonlocals['state'];initial=copy.deepcopy(fast_state)
    def modify(action):
        state['entry']=False
        if fast_state['weight']==0:state.update(choice=None,distance=None)
        result=native(action);log=env.recovery_selection_log[-1];log.update(actual_expert=state['choice'] if log['weight']>0 else 'none',selection_choice=state['choice'],selection_distance=state['distance'],selection_entry=state['entry']);return result
    def reset(*args,**kwargs):
        state.update(choice=None,distance=None,entry=False);fast_state.clear();fast_state.update(copy.deepcopy(initial));env.recovery_selection_log=[];return native_reset(*args,**kwargs)
    env.modify_joint_action=modify;env.reset=reset;return env
