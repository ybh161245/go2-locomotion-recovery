"""Second reward experiment: stronger, per-foot swing/stance objectives."""
import numpy as np
from go2_motion_env import Go2MotionEnv,ENV_CONFIG as V1_CONFIG

ENV_CONFIG=dict(V1_CONFIG,experiment='motion-quality-v2',
                extra_phase_clearance_weight=4.,extra_contact_weight=3.,
                per_foot_height_error_scale=.00015,
                note_v2='Fine-tune v1; stronger per-foot phase clearance. No actuator or joint trajectory change.')


class Go2MotionEnvV2(Go2MotionEnv):
    def step(self,action):
        moving=np.linalg.norm(self.command)>.025
        obs,reward,term,trunc,info=super().step(action)
        if moving:
            feet=self.meter.sample(self.data)
            phase=(self.step_count*.02*2.+np.array(ENV_CONFIG['phase_offsets']))%1.
            swing=phase>=.6
            progress=np.clip((phase-.6)/.4,0,1)
            desired=.05*np.sin(np.pi*progress)**2
            per_foot=np.exp(-(np.maximum(feet['height'],0)-desired)**2/.00015)
            addition=4.*float(per_foot.mean())+3.*float(np.mean(feet['contact']==~swing))
            reward+=addition;info['v2_phase_reward']=addition
        return obs,float(reward),term,trunc,info
