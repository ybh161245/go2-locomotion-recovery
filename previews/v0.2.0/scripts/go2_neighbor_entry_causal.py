"""100 offline intervention only: preserve unified routing and replace requested default actions."""
import copy,inspect
import mujoco
import numpy as np
from go2_unified_recovery_selection import attach_unified_recovery_selection


def snapshot(env,native):
    signature=mujoco.mjtState.mjSTATE_INTEGRATION;integration=np.empty(mujoco.mj_stateSize(env.model,signature));mujoco.mj_getState(env.model,env.data,integration,signature);closure=inspect.getclosurevars(native).nonlocals
    return dict(integration=integration,selection=copy.deepcopy(closure['state']),fast=copy.deepcopy(closure['fast_state']),control={k:copy.deepcopy(getattr(env,k)) for k in ['command','filtered_command','last_action','step_count','gait_phase','last_correction']},friction=env.model.geom_friction.copy())


def attach_neighbor_entry_causal(env,formal,formal_norm,default,low,norm,selector,fast,bank,selection,experiment):
    driver=dict(step=0,D_action=None);env.neighbor_entry_snapshots={}
    class Default:
        def predict(self,observation,deterministic=True):
            if experiment['override_step'] is not None and driver['step']>=experiment['override_step']:
                if experiment['replacement']=='D':return np.asarray(driver['D_action'],dtype=np.float32).copy(),None
                return low.predict(observation,deterministic=deterministic)
            return default.predict(observation,deterministic=deterministic)
    attach_unified_recovery_selection(env,formal,formal_norm,Default(),low,norm,selector,fast,bank,selection);native_modify=env.modify_joint_action;native_step=env.step
    def modify(action):driver['D_action']=np.asarray(action).copy();return native_modify(action)
    def step(action):
        i=env.step_count;driver['step']=i
        if i==experiment['snapshot_step']:env.neighbor_entry_snapshots[i]=snapshot(env,native_modify)
        transition=native_step(action);log=env.recovery_selection_log[-1];requested=log['actual_expert'];override=requested=='default' and experiment['override_step'] is not None and i>=experiment['override_step'];log.update(requested_expert=requested,diagnostic_override=bool(override),actual_expert=experiment['replacement'] if override else requested);return transition
    env.modify_joint_action=modify;env.step=step;return env
