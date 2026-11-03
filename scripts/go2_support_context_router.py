"""087 retain the original fast selector; choose frozen expert from observed support-slip history."""
import copy,inspect
import numpy as np
from go2_fast_capture_router import attach_fast_capture_router

def context_transition(state,cmd,filtered,vy,support_speed,spec):
    eligible=abs(cmd[1])<=1e-8 and abs(cmd[2])<=spec['max_yaw'] and abs(filtered[2])<=spec['max_yaw'] and cmd[0]>=spec['minimum_requested_vx'] and filtered[0]>0 and cmd[0]-filtered[0]>=spec['minimum_filter_gap']
    evidence=bool(eligible and abs(vy)>=spec['minimum_abs_vy'] and support_speed>=spec['minimum_support_speed'])
    state['count']=state['count']+1 if evidence else 0;changed=bool(not state['low'] and state['count']>=spec['confirmation_steps'])
    if changed:state['low']=True
    return bool(eligible),evidence,changed

def attach_support_context(env,formal,formal_norm,default_expert,low_expert,shared_expert_norm,selector,fast,context_spec):
    context=dict(low=False,count=0)
    class Experts:
        def predict(self,observation,deterministic=True):return (low_expert if context['low'] else default_expert).predict(observation,deterministic=deterministic)
    attach_fast_capture_router(env,formal,formal_norm,Experts(),shared_expert_norm,selector,fast,4)
    native_modify=env.modify_joint_action;native_reset=env.reset;fast_state=inspect.getclosurevars(native_modify).nonlocals['state'];initial_fast=copy.deepcopy(fast_state)
    def reset(*args,**kwargs):
        context.update(low=False,count=0);fast_state.clear();fast_state.update(copy.deepcopy(initial_fast));env.recovery_selection_log=[];return native_reset(*args,**kwargs)
    def modify(D_action):
        cmd=env.command.copy();filtered=env.filtered_command.copy();lin,_=env.velocity();feet=env.meter.sample(env.data);contacts=feet['contact'];speed=float(np.mean(feet['speed_xy'][contacts])) if contacts.any() else 0.
        eligible,evidence,changed=context_transition(context,cmd,filtered,float(lin[1]),speed,context_spec);result=native_modify(D_action)
        env.recovery_selection_log[-1].update(support_context='low' if context['low'] else 'default',context_count=context['count'],context_eligible=eligible,context_evidence=evidence,context_changed=changed,support_contact_count=int(contacts.sum()),support_speed_mean=speed)
        return result
    env.reset=reset;env.modify_joint_action=modify;return env
