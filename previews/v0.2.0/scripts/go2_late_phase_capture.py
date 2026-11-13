"""132 offline cuts and one own-observed continuation; frozen130 remains intact."""
import copy,inspect
import numpy as np
from go2_neighbor_entry_causal import snapshot
from go2_bilateral_pilot import CONTROL


def candidate_hold(log,state):
    """Live candidate decision: only original entry/counter and actual fast weight."""
    if log['burst_entry']:state['active']=True
    if state['active'] and log['weight']==0 and not log['direct_override']:state['active']=False
    return bool(log['direct_override'] or (state['active'] and log['weight']>0))


def attach_late_phase_capture(env,stack,variant,cuts):
    native_modify=env.modify_joint_action;native_step=env.step;native_reset=env.reset
    outer=inspect.getclosurevars(native_modify).nonlocals
    if variant['controller']=='observed130':student_modify=outer['native_modify'];burst=outer['state']
    else:student_modify=outer['modify'];burst=dict(remaining=0)
    inner=inspect.getclosurevars(student_modify).nonlocals;unified=inner['native'];scope=inner['state'];hold=dict(active=False)
    env.late_capture_calls=[];env.late_capture_snapshots={}
    def modify(action):
        requested=native_modify(action);log=env.recovery_selection_log[-1];before=copy.deepcopy(hold)
        if variant['live_hold']:direct=candidate_hold(log,hold)
        else:direct=bool(variant['cut'] is not None and env.step_count>=variant['cut'])
        teacher=np.asarray(log['teacher_observation'],np.float32);formal=np.zeros(12,np.float32)
        if direct:
            formal,_=stack.formal[0].predict(stack.formal[1].normalize_obs(teacher.copy()),deterministic=True)
            applied=np.asarray(formal,np.float64).copy()
        else:applied=requested
        log.update(late_shadow_applied_action=log['applied_action'],late_direct_override=direct,
            late_formal_action=np.asarray(formal).tolist(),late_hold_before=before['active'],late_hold_after=hold['active'])
        log['applied_action']=np.asarray(applied).tolist()
        env.late_capture_calls.append(dict(teacher_observation=teacher.copy(),formal_action=np.asarray(formal).copy(),
            requested=np.asarray(requested).copy(),applied=np.asarray(applied).copy(),direct_override=direct,
            hold_before=before['active'],hold_after=hold['active']))
        return applied
    def step(action):
        i=env.step_count
        if i in cuts:env.late_capture_snapshots[i]=dict(native=snapshot(env,unified),scope=copy.deepcopy(scope),
            full_control={k:copy.deepcopy(getattr(env,k)) for k in CONTROL},burst=copy.deepcopy(burst),hold=copy.deepcopy(hold))
        return native_step(action)
    def reset(*args,**kwargs):
        hold.update(active=False);env.late_capture_calls=[];env.late_capture_snapshots={}
        return native_reset(*args,**kwargs)
    env.modify_joint_action=modify;env.step=step;env.reset=reset;return env
