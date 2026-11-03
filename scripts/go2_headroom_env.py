"""Frozen-policy experiment: release only the clipped RL calf mean, in radians."""
import copy
import mujoco
import numpy as np
import torch
from go2_env import ENV_CONFIG
from go2_motion_metrics import MotionMeter
from go2_turn_residual_env import Go2TurnResidualEnv


class MeanRecorder:
    def __init__(self, model, env):
        self.model, self.env = model, env

    def predict(self, observation, deterministic=True):
        assert deterministic
        action, state = self.model.predict(observation, deterministic=True)
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(observation)
            mean = self.model.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
        np.testing.assert_allclose(np.clip(mean, -1, 1), action, atol=2e-6, rtol=0)
        self.env.raw_joint_mean = mean.copy()
        return action, state


class HeadroomEnv(Go2TurnResidualEnv):
    def __init__(self, *args, headroom_rad=0., **kwargs):
        assert 0 <= headroom_rad <= .09
        self.headroom_rad = float(headroom_rad)
        self.raw_joint_mean = None
        self.mapping_trace = []
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        self.mapping_trace = []
        self.raw_joint_mean = None
        return super().reset(**kwargs)

    def action_targets(self, action):
        original = super().action_targets(action)
        desired = original.copy()
        assert self.raw_joint_mean is not None
        delta = float(np.clip(ENV_CONFIG['action_scale'] * (float(self.raw_joint_mean[8]) + 1), -self.headroom_rad, 0))
        desired[8] += delta
        self.mapping_trace.append(dict(raw_mean=self.raw_joint_mean.copy(), clipped_action=action.copy(),
            original_targets=original.copy(), mapped_targets=desired.copy(), residual_rad=delta))
        return desired

    def step(self, action):
        result = super().step(action)
        record = self.mapping_trace[-1]
        record.update(last_action=self.last_action.copy(), actual_targets=self.data.ctrl.copy(),
            joint_angles=self.data.qpos[7:].copy(), sampled_motor_force=self.data.actuator_force.copy())
        np.testing.assert_array_equal(record['last_action'], record['clipped_action'])
        return result


def factory(stack_config, joint, headroom_rad):
    spec = stack_config['environment']
    instances = []
    def make(*args, **kwargs):
        env = HeadroomEnv(*args, headroom_rad=headroom_rad,
            correction_limit=spec['correction_limit'], reversal_hold=spec.get('reversal_hold', 0),
            recovery_gain=spec.get('recovery_gain', 0), recovery_limit=spec.get('recovery_limit', .06), **kwargs)
        env.model = copy.copy(env.model)
        env.data = mujoco.MjData(env.model)
        env.meter = MotionMeter(env.model)
        env.teacher = MeanRecorder(joint[0], env)
        env.teacher_norm = joint[1]
        instances.append(env)
        return env
    make.instances = instances
    return make


def save_trace(make, folder):
    records = make.instances[-1].mapping_trace
    np.savez_compressed(folder/'mapping.npz', **{k: np.asarray([r[k] for r in records]) for k in records[0]})
    env = make.instances[-1]
    fields = ['actuator_ctrlrange', 'actuator_forcerange', 'actuator_gainprm', 'actuator_biasprm', 'jnt_range']
    np.savez_compressed(folder/'physics.npz', **{k: getattr(env.model, k).copy() for k in fields})
