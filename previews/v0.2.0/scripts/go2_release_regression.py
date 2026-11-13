"""107 passive instrumentation only; candidate remains original098 observation-driven selection."""
import numpy as np


def attach_release_capture(env):
    native_reset=env.reset;native_step=env.step;env.release_io={'observations':[],'actions':[]}
    def reset(*args,**kwargs):
        result=native_reset(*args,**kwargs);feet=env.meter.sample(env.data);env.validation_initial=dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),foot_contact=feet['contact'].copy(),foot_speed=feet['speed_xy'].copy(),command=env.command.copy(),filtered_command=env.filtered_command.copy());return result
    def step(action):
        env.release_io['observations'].append(env.observe().copy());env.release_io['actions'].append(np.asarray(action).copy());return native_step(action)
    env.reset=reset;env.step=step;return env
