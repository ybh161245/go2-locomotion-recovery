"""112 complete-recovery objective and horizon package over the frozen 111 stack.

This changes both reward coverage and episode duration as one declared course
package; it is not a single-factor reward or horizon ablation. The learner still
sees the original 50 observations and supplies the original 12 joint actions only
when the native router asks for default. Force, material, protocol time and the
reward's stability counter are not added to the actor or chooser observation.
"""
import numpy as np

from go2_bilateral_pilot import InteractionEnv
from go2_recovery_state_env import recovery_reward
from go2_transition_env_v3 import Go2TransitionEnvV3


REWARD_PART_NAMES = (
    'posture', 'tracking', 'angular', 'height', 'action_change',
    'joint_velocity', 'effort', 'excess', 'termination',
    'yaw', 'speed_band', 'continuous_stability', 'time_cost',
)
REWARD_SCALE = .1


def complete_recovery_terms(lin, ang, command, previous_stable_steps):
    """Unscaled additions computed only from current body velocity and command.

    Returns (parts, stable_steps, metrics). The streak is reward bookkeeping,
    starts at zero on every reset, and never changes controller state or inputs.
    No withdrawal-time label is used: these terms apply at every suffix step.
    """
    planar_error = float(np.linalg.norm(np.asarray(lin)[:2] - np.asarray(command)[:2]))
    yaw_error = float(ang[2] - command[2])
    inside_band = bool(planar_error <= .08 and abs(yaw_error) <= .15)
    stable_steps = int(previous_stable_steps) + 1 if inside_band else 0
    parts = dict(
        yaw=float(2. * np.exp(-(yaw_error / .15) ** 2)),
        speed_band=1. if inside_band else 0.,
        continuous_stability=.5 * min(stable_steps, 25) / 25.,
        time_cost=0. if inside_band else -.5,
    )
    metrics = dict(
        passed_band=inside_band,
        stable_steps=stable_steps,
        planar_error=planar_error,
        yaw_error=yaw_error,
        yaw_error_abs=abs(yaw_error),
    )
    return parts, stable_steps, metrics


class CompleteRecoveryEnv(InteractionEnv):
    """Same constructor as InteractionEnv: (cfg, records, caches, worker, archive).

    Native recovery handback does not end the episode. Frozen base/low policies
    keep running, and later native default calls may use the learner again. The
    actor_applied flag preserves AppliedOnlyBuffer's exclusion of inactive
    actions while GAE retains all resulting rewards through the full protocol.
    The finite rollout ends only at the declared protocol end or native physical
    termination. For 111's 319/320 entries this yields at most 281/280 steps.

    reward_parts follows REWARD_PART_NAMES and stores unscaled components;
    the scalar reward is their sum times REWARD_SCALE. Stability metrics are
    saved per frame and exposed in info for auditing, not in the observation.
    """

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.stable_steps = 0
        self.protocol_end_step = round(self.record['protocol']['seconds'] / .02)
        self.suffix_start_step = int(self.env.step_count)
        assert self.suffix_start_step < self.protocol_end_step
        assert observation.shape == (50,)
        return observation, info

    def step(self, action):
        assert self.env.step_count < self.protocol_end_step
        observation = Go2TransitionEnvV3.observe(self.env).copy()
        assert observation.shape == (50,)
        previous = self.env.last_action.copy()
        clipped = np.clip(action, -1, 1).astype(np.float32)
        assert clipped.shape == (12,) and np.isfinite(clipped).all()

        _, _, (_, _, terminated, native_truncated, info) = self.stack.drive(
            self.env, self.record, clipped)
        self.elapsed += 1
        weight = self.env.recovery_selection_log[-1]['weight']
        applied = bool(self.stack.used and weight > 0)
        if applied:
            expected = self.stack.default[1].normalize_obs(observation.copy())
            np.testing.assert_array_equal(self.stack.expert_input, expected)

        _, parts = recovery_reward(
            self.env, previous, self.env.last_action.copy(), terminated)
        lin, ang = self.env.velocity()
        additions, self.stable_steps, metrics = complete_recovery_terms(
            lin, ang, self.env.command, self.stable_steps)
        parts.update(additions)
        assert tuple(parts) == REWARD_PART_NAMES
        reward = float(sum(parts.values()) * REWARD_SCALE)

        # Protocol time is an environmental episode boundary, never a policy or
        # reward input. A zero native recovery weight is intentionally absent.
        reached_end = self.env.step_count >= self.protocol_end_step
        assert not native_truncated or reached_end
        truncated = bool(native_truncated or reached_end)
        assert self.elapsed == self.env.step_count - self.suffix_start_step
        self.trace.append(dict(
            observation=observation,
            action=clipped,
            actor_applied=applied,
            weight=weight,
            reward=reward,
            reward_parts=np.asarray([parts[k] for k in REWARD_PART_NAMES]),
            qpos=self.env.data.qpos.copy(),
            qvel=self.env.data.qvel.copy(),
            targets=self.env.data.ctrl.copy(),
            force=self.env.data.xfrc_applied[self.env.base].copy(),
            time=float(self.env.data.time),
            terminated=terminated,
            truncated=truncated,
            **metrics,
        ))
        info.update(actor_applied=applied, record=self.record['case'],
                    recovery_reward_parts=parts, **metrics)
        result = Go2TransitionEnvV3.observe(self.env).copy()
        assert result.shape == (50,)
        if terminated or truncated:
            self.flush('terminated' if terminated else 'full_protocol_end')
        return result, reward, terminated, truncated, info
