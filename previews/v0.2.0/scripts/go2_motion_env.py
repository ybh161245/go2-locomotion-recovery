"""Motion-quality reward experiment; learned joint targets, no IK trajectory."""
import mujoco
import numpy as np
from go2_env import Go2VelocityEnv, ENV_CONFIG as BASE_CONFIG
from go2_motion_metrics import MotionMeter

ENV_CONFIG = dict(BASE_CONFIG, experiment='motion-quality-v1',
                  swing_clearance=.05, stance_fraction=.60,
                  phase_offsets=[0.,.5,.5,0.], phase_mask_at_stop=True,
                  reward_note='Added contact rhythm, swing clearance, slip and quiet-standing objectives; original servo/model unchanged.')


class Go2MotionEnv(Go2VelocityEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.meter=MotionMeter(self.model)

    def observe(self):
        obs=super().observe()
        if np.linalg.norm(self.command)<.025:
            obs[-2:]=0
        return obs

    def step(self,action):
        # Parent may sample a new command at the end of the step. Reward must
        # use the command that actually governed this transition.
        cmd=self.command.copy()
        obs,reward,terminated,truncated,info=super().step(action)
        mujoco.mj_forward(self.model,self.data)
        feet=self.meter.sample(self.data)
        lin,ang=self.velocity()
        grav=self.gravity()
        moving=np.linalg.norm(cmd)>.025
        slip=float(np.mean(feet['contact']*np.minimum(feet['speed_xy']**2,1.)))
        pose_cost=2.*np.square(grav[:2]).sum()+.2*np.square(ang[:2]).sum()+2.*lin[2]**2
        if moving:
            phase=(self.step_count*.02*2.+np.array(ENV_CONFIG['phase_offsets']))%1.
            swing=phase>=ENV_CONFIG['stance_fraction']
            progress=np.clip((phase-.6)/.4,0,1)
            desired_height=.05*np.sin(np.pi*progress)**2
            height_reward=float(np.exp(-np.mean((np.maximum(feet['height'],0)-desired_height)**2)/.0005))
            contact_reward=float(np.mean(feet['contact']==~swing))
            addition=1.5*height_reward+1.2*contact_reward-3.*slip-pose_cost
        else:
            # Static support and small joint motion explicitly matter at rest.
            quiet=float(np.exp(-np.mean(self.data.qvel[6:]**2)/.15))
            contact_reward=float(np.mean(feet['contact']))
            height_reward=0.
            addition=1.5*quiet+1.5*contact_reward-8.*slip-pose_cost-4.*np.square(lin).sum()
            addition-=.5*float(np.mean((self.data.qpos[7:]-self.nominal)**2))
        reward+=float(addition)
        info.update(quality_reward=float(addition),foot_slip_cost=slip,
                    contact_reward=contact_reward,clearance_reward=height_reward)
        return self.observe(),float(reward),terminated,truncated,info
