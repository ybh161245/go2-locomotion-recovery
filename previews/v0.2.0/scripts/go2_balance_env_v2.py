"""Joint heading/velocity/posture fine-tuning with a frozen baseline teacher."""
from functools import lru_cache
from pathlib import Path
import multiprocessing
import numpy as np
from go2_heading_env import Go2HeadingEnv,ENV_CONFIG as PARENT_CONFIG
from go2_transition_env_v3 import Go2TransitionEnvV3

ROOT=Path(__file__).resolve().parents[1]
TEACHER=ROOT/'artifacts/runs/20261023-140238-938755-go2-transition-v3-mixed-ppo'
ENV_CONFIG=dict(PARENT_CONFIG,experiment='heading-balance-v2',heading_gain=.2,max_heading_correction=.04,
    teacher_run=TEACHER.name,teacher_regularization_straight=12.,teacher_regularization_turn=2.,
    heading_reward_weight=.25,heading_reward_error_scale=.01,
    requested_yaw_error_cost=80.,extra_turn_slip_cost=4.,extra_roll_pitch_rate_cost=1.,
    note='PPO fine-tuning jointly penalizes heading error, requested yaw error and motion deviations. Frozen teacher statistics never update.')


@lru_cache(maxsize=1)
def frozen_teacher():
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    if multiprocessing.current_process().name!='MainProcess':torch.set_num_threads(1)
    model=PPO.load(TEACHER/'final-model.zip',device='cpu')
    norm=VecNormalize.load(TEACHER/'final-vecnormalize.pkl',DummyVecEnv([lambda:Go2TransitionEnvV3('external')]))
    norm.training=False;norm.norm_reward=False
    return model,norm


class Go2BalanceEnvV2(Go2HeadingEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,gain=.2,max_correction=.04,**kwargs)
        self.teacher,self.teacher_norm=frozen_teacher()

    def step(self,action):
        requested=self.command.copy()
        teacher_obs=Go2TransitionEnvV3.observe(self)
        teacher_action,_=self.teacher.predict(self.teacher_norm.normalize_obs(teacher_obs.copy()),deterministic=True)
        obs,reward,term,trunc,info=super().step(action)
        feet=self.meter.sample(self.data);lin,ang=self.velocity()
        turning=abs(requested[2])>.025
        imitation=float(np.mean((np.asarray(action)-teacher_action)**2))
        addition=-(2. if turning else 12.)*imitation
        if turning:
            addition+=float(.25*np.exp(-self.heading_error**2/.01))
            addition-=80.*float((ang[2]-requested[2])**2)
            addition-=4.*float(np.mean(feet['contact']*np.minimum(feet['speed_xy']**2,1.)))
            addition-=1.*float(np.square(ang[:2]).sum())
        info.update(balance_reward=addition,teacher_action_deviation=imitation)
        return obs,reward+addition,term,trunc,info
