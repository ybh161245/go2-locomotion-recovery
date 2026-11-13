"""130 self-observed negative recovery-entry three-frame formal pulse."""
import copy,inspect
import numpy as np
from go2_successful_supply_student import attach_successful_supply_student
from go2_neighbor_entry_causal import snapshot
from go2_bilateral_pilot import CONTROL


def trigger_condition(log,spec):
    raw=log['base_observation']
    return bool(log['selection_entry'] and log['actual_expert']=='default'
        and log['weight']==spec['entry_weight']
        and raw[1]<=spec['scaled_vy_max']
        and abs(float(raw[9])-spec['scaled_filtered_vx'])<=spec['command_atol']
        and abs(float(raw[10]))<=spec['command_atol']
        and abs(float(raw[11]))<=spec['scaled_max_yaw'])


def attach_observed_entry_burst(env,stack,student,cfg):
    attach_successful_supply_student(env,*stack.formal,stack.default[0],student,stack.low[0],stack.default[1],
        cfg['confirmed_selector']['selector'],cfg['fast_capture'],stack.bank,cfg['unified_recovery_selection'],cfg['scoped_expert_blend']['gate'])
    native_modify=env.modify_joint_action;native_step=env.step;native_reset=env.reset
    closure=inspect.getclosurevars(native_modify).nonlocals;unified=closure['native'];scope=closure['state']
    spec=cfg['observed_entry_burst']['gate'];state=dict(remaining=0)
    env.observed_burst_calls=[];env.observed_burst_forks={}
    def modify(action):
        requested=native_modify(action);log=env.recovery_selection_log[-1]
        before=state['remaining'];entry=bool(before==0 and trigger_condition(log,spec))
        if entry:state['remaining']=spec['frames']
        direct=state['remaining']>0
        teacher=np.asarray(log['teacher_observation'],np.float32)
        formal=np.zeros(12,np.float32)
        if direct:
            formal,_=stack.formal[0].predict(stack.formal[1].normalize_obs(teacher.copy()),deterministic=True)
            applied=np.asarray(formal,dtype=np.float64).copy();state['remaining']-=1
        else:applied=requested
        log.update(shadow_applied_action=log['applied_action'],direct_override=direct,direct_formal_action=np.asarray(formal).tolist(),
            burst_entry=entry,burst_remaining_before=before,burst_remaining_after=state['remaining'])
        log['applied_action']=np.asarray(applied).tolist()
        env.observed_burst_calls.append(dict(teacher_observation=teacher.copy(),formal_action=np.asarray(formal).copy(),requested=np.asarray(requested).copy(),applied=np.asarray(applied).copy(),direct_override=direct,burst_entry=entry,remaining_before=before,remaining_after=state['remaining']))
        return applied
    def step(action):
        i=env.step_count;prior=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0.
        end=bool(env.recovery_selection_log and env.recovery_selection_log[-1].get('direct_override') and state['remaining']==0)
        before=dict(native=snapshot(env,unified),scope=copy.deepcopy(scope),full_control={k:copy.deepcopy(getattr(env,k)) for k in CONTROL},burst=copy.deepcopy(state)) if prior==0 or end else None
        if end:env.observed_burst_forks[i]=dict(reason='end',**before)
        result=native_step(action)
        if env.recovery_selection_log[-1]['burst_entry']:
            assert before is not None
            env.observed_burst_forks[i]=dict(reason='entry',**before)
        return result
    def reset(*args,**kwargs):
        state.update(remaining=0);env.observed_burst_calls=[];env.observed_burst_forks={}
        return native_reset(*args,**kwargs)
    env.modify_joint_action=modify;env.step=step;env.reset=reset;return env
