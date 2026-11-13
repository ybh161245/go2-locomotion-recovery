"""Command slew limits and turn-focused reward; still learned joint actions."""
import numpy as np
from go2_comfort_env import Go2ComfortEnv,ENV_CONFIG as PARENT_CONFIG

ENV_CONFIG=dict(PARENT_CONFIG,experiment='turn-transition-v1',linear_command_acceleration=.8,
                yaw_command_acceleration=2.,frequency_slew_hz_s=2.,
                extra_yaw_tracking_weight=2.,extra_yaw_error_scale=.08,
                command_distribution='20% stop,30% straight,50% signed turns; every 3 seconds')


class Go2TransitionEnv(Go2ComfortEnv):
    def __init__(self,command_mode='mixed',**kwargs):
        self.training_command_mode=command_mode
        self.filtered_command=np.zeros(3)
        self.filtered_frequency=1.45
        self.in_step=False
        super().__init__('external',**kwargs)

    def _sample_command(self):
        if self.training_command_mode=='external':return
        p=self.np_random.uniform()
        if p<.2:self.command[:]=0
        elif p<.5:self.command[:]=[self.np_random.choice([.2,.4,.6]),0,0]
        else:self.command[:]=[self.np_random.choice([.2,.3]),0,self.np_random.choice([-.5,.5])]

    def reset(self,**kwargs):
        self.filtered_command[:]=0;self.filtered_frequency=1.45;self.in_step=False
        return super().reset(**kwargs)

    def observe(self):
        if self.in_step:return super().observe()
        requested=self.command.copy()
        self.command[:]=self.filtered_command
        obs=super().observe();self.command[:]=requested
        return obs

    def frequency(self):
        return self.filtered_frequency

    def step(self,action):
        requested=self.command.copy()
        limit=np.array([.8,.8,2.])*.02
        self.filtered_command+=np.clip(requested-self.filtered_command,-limit,limit)
        target_hz=1.75 if abs(self.filtered_command[2])>.1 else float(np.clip(1.35+.5*abs(self.filtered_command[0]),1.35,1.75))
        self.filtered_frequency+=float(np.clip(target_hz-self.filtered_frequency,-.04,.04))
        self.command[:]=self.filtered_command;self.in_step=True
        try:obs,reward,term,trunc,info=super().step(action)
        finally:self.command[:]=requested;self.in_step=False
        lin,ang=self.velocity()
        reward+=float(2.*np.exp(-(ang[2]-self.filtered_command[2])**2/.08))
        info.update(requested_vx=float(requested[0]),requested_yaw=float(requested[2]),
                    filtered_vx=float(self.filtered_command[0]),filtered_yaw=float(self.filtered_command[2]),
                    velocity_error=float(np.linalg.norm(lin[:2]-requested[:2])),
                    yaw_error=float(abs(ang[2]-requested[2])))
        if self.training_command_mode!='external' and self.step_count%150==0:self._sample_command()
        return self.observe(),reward,term,trunc,info
