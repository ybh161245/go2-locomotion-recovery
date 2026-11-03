"""Fine-tune vertical motion while retaining v2 command and yaw control."""
from go2_transition_env_v2 import Go2TransitionEnvV2,ENV_CONFIG as PARENT_CONFIG

ENV_CONFIG=dict(PARENT_CONFIG,experiment='turn-transition-v3',extra_vertical_velocity_cost=20.)


class Go2TransitionEnvV3(Go2TransitionEnvV2):
    def step(self,action):
        moving=abs(self.command[0])+abs(self.command[2])>.025
        obs,reward,term,trunc,info=super().step(action)
        lin,_=self.velocity()
        extra=20.*float(lin[2]**2) if moving else 0.
        info['vertical_finetune_cost']=extra
        return obs,reward-extra,term,trunc,info
