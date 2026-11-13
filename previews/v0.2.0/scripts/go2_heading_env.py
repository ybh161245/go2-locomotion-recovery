"""Inference-only heading feedback experiment; PPO weights stay unchanged."""
import numpy as np
from go2_transition_env_v3 import Go2TransitionEnvV3,ENV_CONFIG as PARENT_CONFIG

ENV_CONFIG=dict(PARENT_CONFIG,experiment='heading-feedback-v1',heading_gain=.4,max_heading_correction=.06,
                note='World yaw reference integrates requested yaw; extra policy command only while turning. No new policy training.')


class Go2HeadingEnv(Go2TransitionEnvV3):
    def __init__(self,command_mode='mixed',gain=.4,max_correction=.06,**kwargs):
        self.heading_gain=gain;self.max_correction=max_correction
        self.heading_error=0.;self.target_heading=0.;self.measured_heading=0.
        self.last_heading=0.;self.turn_active=False
        super().__init__(command_mode,**kwargs)

    def yaw(self):
        w,x,y,z=self.data.qpos[3:7]
        return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

    def reset(self,**kwargs):
        self.turn_active=False;self.heading_error=0.;self.target_heading=0.;self.measured_heading=0.
        result=super().reset(**kwargs)
        self.last_heading=self.yaw()
        return result

    def sync(self):
        active=abs(self.command[2])>.025
        if active and not self.turn_active:
            self.target_heading=0.;self.measured_heading=0.;self.heading_error=0.;self.last_heading=self.yaw()
        if not active:self.heading_error=0.
        self.turn_active=active

    def observe(self):
        if not self.in_step:self.sync()
        obs=super().observe()
        if self.turn_active:
            correction=float(np.clip(self.heading_gain*self.heading_error,-self.max_correction,self.max_correction))
            obs[11]+=np.float32(.25*correction)
        return obs

    def step(self,action):
        self.sync();requested=self.command.copy()
        if self.turn_active:self.target_heading+=requested[2]*.02
        obs,reward,term,trunc,info=super().step(action)
        angle=self.yaw();delta=(angle-self.last_heading+np.pi)%(2*np.pi)-np.pi
        self.last_heading=angle
        if self.turn_active:
            self.measured_heading+=delta;self.heading_error=self.target_heading-self.measured_heading
        else:self.heading_error=0.
        info.update(heading_error_rad=self.heading_error,heading_correction_rad_s=float(np.clip(self.heading_gain*self.heading_error,-self.max_correction,self.max_correction)))
        return self.observe(),reward,term,trunc,info
