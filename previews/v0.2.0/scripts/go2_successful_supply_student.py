"""123 direct student supply in the frozen118 self-observed positive scope."""
import copy
import numpy as np
from go2_unified_recovery_selection import attach_unified_recovery_selection
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_neighbor_entry_causal import snapshot


def scope_condition(raw,spec):
    return bool(raw[1]>=spec['scaled_vy_on'] and abs(float(raw[9])-spec['scaled_filtered_vx'])<=spec['command_atol']
        and abs(float(raw[10]))<=spec['command_atol'] and abs(float(raw[11]))<=spec['scaled_max_yaw'])


def attach_successful_supply_student(env,formal,formal_norm,parent,learned,low,norm,selector,fast,bank,selection,spec):
    state=dict(enabled=None,entry=False);env.scoped_calls=[];env.scoped_query_snapshots={};env.scoped_extra_before={}
    class Expert:
        def predict(self,observation,deterministic=True):
            assert deterministic;raw=Go2TransitionEnvV3.observe(env).copy();state['entry']=False
            if state['enabled'] is None:state.update(enabled=scope_condition(raw,spec),entry=True)
            a,_=parent.predict(observation,deterministic=True)
            if state['enabled']:
                b,_=learned.predict(observation,deterministic=True);result=np.asarray(b,dtype=np.float32).copy()
            else:b=np.zeros_like(a);result=np.asarray(a).copy()
            assert result.dtype==np.dtype('float32')
            env.scoped_calls.append(dict(normalized_observation=np.asarray(observation).copy(),raw_observation=raw,parent_action=np.asarray(a).copy(),learned_action=np.asarray(b).copy(),learned_evaluated=bool(state['enabled']),applied=result.copy(),output_is_float64=result.dtype==np.dtype('float64'),scope_entry=state['entry']))
            return result,None
    attach_unified_recovery_selection(env,formal,formal_norm,Expert(),low,norm,selector,fast,bank,selection);native=env.modify_joint_action;native_step=env.step;native_reset=env.reset
    def modify(action):
        prior=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0.
        if prior==0:state.update(enabled=None,entry=False)
        state['entry']=False;result=native(action);log=env.recovery_selection_log[-1];called=log['actual_expert']=='default';log.update(scope_enabled=state['enabled'],scope_entry=bool(state['entry'] and called),scope_learned=bool(called and state['enabled']),expert_output_dtype='float32' if log['actual_expert'] in ['default','low'] else 'base-float64');return result
    def step(action):
        i=env.step_count;prior=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0.;candidate=snapshot(env,native) if prior==0 else None;extra=copy.deepcopy(state);result=native_step(action)
        if env.recovery_selection_log[-1]['selection_entry']:
            assert candidate is not None;env.scoped_query_snapshots[i]=candidate;env.scoped_extra_before[i]=extra
        return result
    def reset(*args,**kwargs):
        state.update(enabled=None,entry=False);env.scoped_calls=[];env.scoped_query_snapshots={};env.scoped_extra_before={};return native_reset(*args,**kwargs)
    env.modify_joint_action=modify;env.step=step;env.reset=reset;return env
