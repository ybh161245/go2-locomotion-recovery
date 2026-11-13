"""212: preserve an existing exit heading while switching between .2 and .4 m/s."""

import numpy as np


def attach(env, spec):
    native, reset = env.modify_joint_action, env.reset
    anchor = previous_yaw = None
    env.exit_hold_calls = []

    def modify(action):
        nonlocal anchor, previous_yaw
        shadow = np.asarray(native(action))
        log = env.recovery_selection_log[-1]
        command = env.command
        yaw = env.yaw()
        straight = tuple(command) in ((.2, 0., 0.), (.4, 0., 0.))
        if not straight:
            anchor = None
        elif command[0] == .4 and previous_yaw is not None and abs(previous_yaw) > .025:
            anchor = yaw
        previous_yaw = float(command[2])
        error = (anchor-yaw+np.pi) % (2*np.pi)-np.pi if anchor is not None else 0.
        allowed = anchor is not None and log["formal_weight"] == 0 and log["weight"] == 0 and not log["learner_eligible"]
        extra = float(np.clip(spec["gain"]*error, -spec["limit"], spec["limit"])) if allowed else 0.
        actual = shadow
        observation = None
        if extra:
            np.testing.assert_array_equal(shadow, np.clip(action, -1, 1))
            observation = np.asarray(log["teacher_observation"], np.float32).copy()
            observation[11] += np.float32(.25*extra)
            prediction, _ = env.teacher.predict(env.teacher_norm.normalize_obs(observation.copy()), deterministic=True)
            actual = np.clip(prediction, -1, 1).astype(shadow.dtype)
        env.exit_hold_calls.append(dict(step=env.step_count, anchor=anchor, yaw=yaw, error=error,
            allowed=allowed, extra=extra, shadow=shadow.copy(), actual=actual.copy(), observation=observation))
        return actual

    def clear(*args, **kwargs):
        nonlocal anchor, previous_yaw
        anchor = previous_yaw = None
        env.exit_hold_calls.clear()
        return reset(*args, **kwargs)

    env.modify_joint_action, env.reset = modify, clear
    return env
