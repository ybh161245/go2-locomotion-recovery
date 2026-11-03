"""Long normal command sequences with native V3 rewards and zero training force."""
import numpy as np
from go2_robust_env import Go2RobustEnv

def build_schedule(config,worker,episode):
    c=config['sequence_course'];slot=(7*episode+worker)%c['total_slots']
    if slot>=c['normal_slots']:
        return [dict(start_step=0,end_step=c['straight_slot_s']*50,command=[.2,0.,0.],phase='straight-slot')],slot
    rng=np.random.default_rng(np.random.SeedSequence([c['seed'],worker,episode]));sign=1 if (worker+episode)%2==0 else -1
    fast=float(rng.choice(c['fast_vx']));tv=float(rng.choice(c['turn_vx']));yaw=c['turn_yaw_abs']
    specs=[('stand','stand_s',[0,0,0]),('start','low_straight_s',[.2,0,0]),('accelerate','fast_straight_s',[fast,0,0]),('brake','brake_s',[0,0,0]),('turn','turn_s',[tv,0,sign*yaw]),('reverse-1','turn_s',[tv,0,-sign*yaw]),('reverse-2','turn_s',[tv,0,sign*yaw]),('straighten','straighten_s',[.2,0,0]),('final-stop','final_stop_s',[0,0,0])]
    schedule=[];start=0
    for phase,key,command in specs:
        seconds=int(rng.integers(c[key][0],c[key][1]+1));end=start+seconds*50
        schedule.append(dict(start_step=start,end_step=end,command=command,phase=phase));start=end
    return schedule,slot

class Go2SequenceEnv(Go2RobustEnv):
    def __init__(self,command_mode='mixed',**kwargs):
        self.use_sequence=command_mode!='external';self.schedule=[]
        # Disable V2's three-second command sampling. External preflight stays12s.
        super().__init__('external',**kwargs)

    def reset(self,*,seed=None,options=None):
        episode=self.episode+1
        if seed is None:seed=self.config['seed']+self.worker*100000+episode
        if self.use_sequence:
            self.schedule,slot=build_schedule(self.config,self.worker,episode)
            self.episode_seconds=self.schedule[-1]['end_step']*.02
            assert int(self.episode_seconds/.02)==self.schedule[-1]['end_step']
            opts=dict(options or {},command=self.schedule[0]['command'],events=[])
        else:
            self.schedule=[];slot=None;opts=options
        _,info=super().reset(seed=seed,options=opts)
        self.episode_meta.update(profile=('continuous-normal' if len(self.schedule)>1 else 'straight-slot') if self.use_sequence else 'explicit-preflight',slot=slot,command_schedule=self.schedule,episode_limit_steps=int(self.episode_seconds/.02),initial_command=self.command.tolist())
        if self.trace is not None:self.trace.update(base_reward=[],normal_anchor_cost=[],recovery_cost=[])
        return self.observe(),info

    def step(self,action):
        _,reward,term,trunc,info=super().step(action)
        if self.use_sequence and not (term or trunc):
            segment=next(s for s in self.schedule if s['start_step']<=self.step_count<s['end_step'])
            self.command[:]=segment['command']
        return self.observe(),reward,term,trunc,info

    def adjust_reward(self,reward,info):
        if self.trace is not None:
            self.trace['base_reward'].append(reward);self.trace['normal_anchor_cost'].append(0.);self.trace['recovery_cost'].append(0.)
        return reward
