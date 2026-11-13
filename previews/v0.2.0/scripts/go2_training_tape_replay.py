"""134 offline training-action tape, explicitly indexed by known step, never deployable."""
import copy
import mujoco
import numpy as np
from go2_native_capture_learning import cache
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_bilateral_pilot import CONTROL

def pack_native_cache(env):
    saved=cache(env);data=saved['data'];values=np.empty(mujoco.mj_stateSize(env.model,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model,data,values,mujoco.mjtState.mjSTATE_INTEGRATION)
    return dict(integration=values,control=saved['control'],friction=saved['friction'],state_dicts=saved['values'])

def attach_tape(env,tape,entry,enabled):
    modify=env.modify_joint_action;step=env.step;reset=env.reset
    env.tape_snapshots={};env.tape_calls=[]
    def wrapped_modify(action):
        i=env.step_count;j=i-entry;expected=None
        if enabled and j>=0:
            assert j<len(tape['action']);expected=Go2TransitionEnvV3.observe(env).copy()
            np.testing.assert_array_equal(expected,tape['observation'][j])
            if tape['actor_applied'][j]:env.learner_action=tape['action'][j].copy()
        applied=modify(action);env.learner_action=None;log=env.recovery_selection_log[-1]
        if enabled and j>=0:
            assert log['learner_applied']==bool(tape['actor_applied'][j]) and log['learner_eligible']==bool(tape['eligible'][j])
            if tape['actor_applied'][j]:np.testing.assert_array_equal(applied,tape['action'][j].astype(np.float64))
        log.update(tape_mode=enabled,tape_index=j if enabled and j>=0 else None,tape_applied=bool(enabled and j>=0 and tape['actor_applied'][j]),offline_time_index_oracle=True if enabled else False)
        env.tape_calls.append(dict(step=i,tape_index=j if enabled else -1,applied=log['tape_applied'],actual=np.asarray(applied).copy()))
        return applied
    def wrapped_step(action):
        i=env.step_count
        # Inputs have already been prepared by the full-protocol driver.
        if i in [entry,entry+1,364,416,470,473]:env.tape_snapshots[i]=pack_native_cache(env)
        result=step(action)
        if enabled and i>=entry:
            j=i-entry
            for key,value in [('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl),('world_wrench',env.data.xfrc_applied[env.base])]:np.testing.assert_array_equal(value,tape[key][j],err_msg=key+' '+str(i))
            assert bool(result[2])==bool(tape['terminated'][j]) and bool(result[3])==bool(tape['truncated'][j])
        return result
    def wrapped_reset(*args,**kwargs):
        env.tape_snapshots={};env.tape_calls=[];return reset(*args,**kwargs)
    env.modify_joint_action=wrapped_modify;env.step=wrapped_step;env.reset=wrapped_reset
    return env
