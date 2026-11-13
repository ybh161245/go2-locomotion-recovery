"""Release the two existing phase rewards for high-speed requests."""
import numpy as np
from run_go2_speed_course_pilot_validated import Course


class RunningRewardEnv(Course):
    def __init__(self, mode="mixed", *, release_phase=True, **kwargs):
        self.release_phase = release_phase
        super().__init__(mode, **kwargs)

    def _sample_command(self):
        if self.training_command_mode == "external":
            return
        self.command[:] = [self.np_random.choice([0., .4, .8, 1.], p=[.2, .2, .3, .3]), 0., 0.]

    def step(self, action):
        requested = self.command.copy()
        observation, reward, term, trunc, info = super().step(action)
        active = requested[0] > .6 and np.linalg.norm(self.filtered_command) > .025
        original_phase_reward = float(1.5 * info["clearance_reward"] + 1.2 * info["contact_reward"]) if active else 0.
        removed = original_phase_reward if self.release_phase else 0.
        reward -= removed
        self.trace[-1].update(reward=reward, phase_reward=original_phase_reward, removed_phase_reward=removed)
        info.update(high_speed_phase_reward=original_phase_reward, removed_phase_reward=removed)
        return observation, reward, term, trunc, info
