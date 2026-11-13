"""Train a bounded yaw-command adapter over the immutable v3 joint policy."""
from functools import partial
import numpy as np
from gymnasium import spaces
from go2_transition_env_v3 import Go2TransitionEnvV3,ENV_CONFIG as BASE_CONFIG
from go2_balance_env import frozen_teacher,TEACHER

ENV_CONFIG=dict(BASE_CONFIG,experiment='turn-residual-v1',teacher_run=TEACHER.name,
    controller='Frozen 12-joint PPO plus trained scalar yaw-command residual',
    observation_dimension=53,action_dimension=1,heading_reset_on_reversal=True,
    correction_ramp_seconds=.3,heading_reward_weight=1.,heading_error_scale=.01,
    raw_yaw_error_cost=8.,residual_action_cost=.02,extra_roll_pitch_rate_cost=.5,
    extra_turn_slip_cost=2.,teacher_weights_and_statistics_frozen=True)


class Go2TurnResidualEnv(Go2TransitionEnvV3):
    def __init__(self,command_mode='mixed',correction_limit=.06,reversal_hold=0.,recovery_gain=0.,recovery_limit=.06,**kwargs):
        self.recovery_gain=recovery_gain;self.recovery_limit=recovery_limit
        self.reversal_hold=reversal_hold;self.is_reversal=False
        self.correction_limit=correction_limit
        self.turn_sign=0;self.turn_age=0.;self.heading_error=0.
        self.target_heading=0.;self.measured_heading=0.;self.last_heading=0.
        self.last_correction=0.
        self.residual_stepping=False
        super().__init__(command_mode,**kwargs)
        self.observation_space=spaces.Box(-np.inf,np.inf,(53,),np.float32)
        self.action_space=spaces.Box(-1.,1.,(1,),np.float32)
        self.teacher,self.teacher_norm=frozen_teacher()

    def yaw(self):
        w,x,y,z=self.data.qpos[3:7]
        return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

    def sync_turn(self):
        sign=int(np.sign(self.command[2])) if abs(self.command[2])>.025 else 0
        if sign!=self.turn_sign:
            self.is_reversal=bool(sign and self.turn_sign)
            self.turn_sign=sign;self.turn_age=0.;self.heading_error=0.
            self.target_heading=0.;self.measured_heading=0.;self.last_heading=self.yaw()
            self.last_correction=0.

    def reset(self,**kwargs):
        self.is_reversal=False
        self.turn_sign=0;self.turn_age=0.;self.heading_error=0.
        self.target_heading=0.;self.measured_heading=0.;self.last_heading=0.;self.last_correction=0.
        result=super().reset(**kwargs)
        self.last_heading=self.yaw()
        return self.observe(),result[1]

    def observe(self):
        if not self.in_step and not self.residual_stepping:self.sync_turn()
        base=Go2TransitionEnvV3.observe(self)
        extra=[np.clip(self.heading_error,-.5,.5),self.command[2],min(self.turn_age/2.,1.)]
        return np.concatenate([base,extra]).astype(np.float32)

    def step(self,action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(1,) or not np.isfinite(action).all():raise ValueError('Expected one finite residual action')
        self.sync_turn();requested=self.command.copy();turning=self.turn_sign!=0
        scalar=float(np.clip(action[0],-1,1))
        age=max(0.,self.turn_age-self.reversal_hold) if self.is_reversal else self.turn_age
        self.last_correction=scalar*self.correction_limit*min(age/.3,1.) if turning else 0.
        if self.is_reversal and self.recovery_gain:
            ramp=float(np.clip((self.turn_age-1.)/.5,0,1))
            self.last_correction+=ramp*float(np.clip(self.recovery_gain*self.heading_error,-self.recovery_limit,self.recovery_limit))
        teacher_obs=Go2TransitionEnvV3.observe(self).copy()
        teacher_obs[11]+=np.float32(.25*self.last_correction)
        joint_action,_=self.teacher.predict(self.teacher_norm.normalize_obs(teacher_obs),deterministic=True)
        joint_action=self.modify_joint_action(joint_action)
        if turning:self.target_heading+=requested[2]*.02
        self.residual_stepping=True
        try:obs,reward,term,trunc,info=super().step(joint_action)
        finally:self.residual_stepping=False
        angle=self.yaw();delta=(angle-self.last_heading+np.pi)%(2*np.pi)-np.pi;self.last_heading=angle
        if turning:
            self.measured_heading+=delta;self.heading_error=self.target_heading-self.measured_heading;self.turn_age+=.02
            lin,ang=self.velocity();feet=self.meter.sample(self.data)
            reward+=float(np.exp(-self.heading_error**2/.01)-8.*(ang[2]-requested[2])**2
                -.02*scalar**2-.5*np.square(ang[:2]).sum()
                -2.*np.mean(feet['contact']*np.minimum(feet['speed_xy']**2,1.)))
        info.update(heading_error_rad=self.heading_error,yaw_command_correction=self.last_correction,
            joint_action=joint_action.tolist(),frozen_joint_policy=True)
        return self.observe(),reward,term,trunc,info

    def modify_joint_action(self,joint_action):
        """Identity extension point; existing controllers retain exactly their old actions."""
        return joint_action


def configured(config):
    return partial(Go2TurnResidualEnv,correction_limit=config['environment']['correction_limit'],reversal_hold=config['environment'].get('reversal_hold',0.),recovery_gain=config['environment'].get('recovery_gain',0.),recovery_limit=config['environment'].get('recovery_limit',.06))
