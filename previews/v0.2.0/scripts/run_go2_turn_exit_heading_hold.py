"""208: fixed heading hold on the D policy after a turn exits into a trot."""

import argparse
import pickle
import shutil
import traceback
from unittest.mock import patch

import numpy as np

import go2_compare
import go2_task
import run_go2_turn_exit_trot as parent
from go2_157 import ROOT, read, sha, write_new
from run_go2_full_heading_feedback import attach as attach197
from run_go2_reversal_push_task import equal_npz, measure, verify_inputs


CONFIG = ROOT/"configs/go2-turn-exit-heading-hold.json"
PLAN = ROOT/"docs/experiments/2026-11-06-turn-exit-heading-hold.md"
RUN = ROOT/"artifacts/runs/20261106-turn-exit-heading-hold"
PUBLIC = ROOT/"docs/data/go2-turn-exit-heading-hold"
SOURCE = ROOT/"scripts/run_go2_turn_exit_heading_hold.py"


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
        trot = tuple(command) == (.4, 0., 0.)
        if not trot:
            anchor = None
        elif previous_yaw is not None and abs(previous_yaw) > .025:
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


def prepare():
    assert not any(p.exists() for p in (CONFIG, RUN, PUBLIC))
    previous = read(parent.CONFIG)
    verify_inputs(previous)
    summary = parent.PUBLIC/"summary.json"
    inputs = go2_compare.candidate_inputs()
    for path in (SOURCE, PLAN, parent.CONFIG, summary, ROOT/"scripts/run_go2_turn_exit_trot.py",
                 ROOT/"scripts/go2_task.py", ROOT/"docs/data/go2-current/selection.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    records = read(summary)["records"]
    for record in records:
        for name in ("states.npz", "initial-state.npz", "physics.npz", "steps.csv"):
            path = ROOT/record["result"]["folder"]/name
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(CONFIG, dict(experiment=208, inputs=inputs, cases=previous["definitions"], reference=records,
        mechanism=dict(gain=.4, limit=.06), exit_gate=previous["exit_gate"], min_worst_push_gain_deg=.5,
        max_physical=4, training=0, fits=0))


def first_second(folder):
    with np.load(folder/"states.npz") as states:
        heading = states["heading"][[999, 1049]]
    yaw = np.arctan2(heading[:, 1], heading[:, 0])
    return float(np.degrees((yaw[1]-yaw[0]+np.pi) % (2*np.pi)-np.pi))


def run():
    config = read(CONFIG)
    verify_inputs(config)
    RUN.mkdir(exist_ok=False)
    for path in (CONFIG, PLAN, SOURCE):
        shutil.copy2(path, RUN/path.name)
    original_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, evaluator = original_load(spec)

        def factory208(reference):
            plain, make = factory(reference)

            def make208(*args, **kwargs):
                env = attach(attach197(make(*args, **kwargs), stack), config["mechanism"])
                instances.append(env)
                return env

            return plain, make208

        return cfg, stack, factory208, evaluator

    records = []
    try:
        for case in config["cases"]:
            side, task = case["side"], case["task"]
            zero = None
            for kind in ("zero", "pressure"):
                with patch.object(go2_task, "load_stack", load):
                    result = go2_task.run_task(task, RUN/f"{side}-{kind}", kind)
                env = instances.pop()
                folder = ROOT/result["folder"]
                calls = env.exit_hold_calls
                with (folder/"exit-hold-calls.pkl").open("xb") as stream:
                    pickle.dump(calls, stream, protocol=5)
                old = next(r for r in config["reference"] if r["side"] == side and r["kind"] == kind)
                reference = ROOT/old["result"]["folder"]
                for filename in ("initial-state.npz", "physics.npz"):
                    equal_npz(folder, reference, filename)
                first = next((i for i, c in enumerate(calls) if not np.array_equal(c["actual"], c["shadow"])), len(calls))
                equal_npz(folder, reference, "states.npz", first)
                with np.load(folder/"states.npz") as states:
                    target = np.stack([c["actual"] for c in calls]).astype(np.float64)
                    np.testing.assert_array_equal(np.clip(env.action_targets(target), *env.model.actuator_ctrlrange.T), states["targets"])
                if kind == "zero":
                    zero = folder
                else:
                    equal_npz(folder, zero, "states.npz", round(task["push"]["start_s"]*50))
                transition = parent.exit_metrics(result, config["exit_gate"])
                metrics = measure(result, task)
                record = dict(side=side, kind=kind, result=result, exit=transition, metrics=metrics,
                    first_changed_step=first, active_frames=sum(c["extra"] != 0 for c in calls),
                    first_second_old_deg=first_second(reference), first_second_new_deg=first_second(folder),
                    end_old_deg=old["exit"]["heading_change_deg"], end_new_deg=transition["heading_change_deg"],
                    original_and_exit_pass=result["passed"] and transition["passed"])
                records.append(record)
                print(side, kind, record["first_second_old_deg"], record["first_second_new_deg"],
                      record["end_old_deg"], record["end_new_deg"], record["original_and_exit_pass"], flush=True)
        verify_inputs(config)
        pushes = [r for r in records if r["kind"] == "pressure"]
        gain = max(abs(r["first_second_old_deg"]) for r in pushes)-max(abs(r["first_second_new_deg"]) for r in pushes)
        both_improve = all(abs(r["first_second_new_deg"]) < abs(r["first_second_old_deg"]) for r in pushes)
        end_held = max(abs(r["end_new_deg"]) for r in records) <= max(abs(r["end_old_deg"]) for r in records)
        protected = all(r["original_and_exit_pass"] for r in records)
        summary = dict(experiment=208, records=records, worst_push_first_second_gain_deg=gain,
            both_pressure_first_second_improved=both_improve, worst_end_protected=end_held, original_protected=protected,
            qualified=protected and both_improve and end_held and gain >= config["min_worst_push_gain_deg"],
            new_physical=len(records), training=0, fits=0, visual="pending")
        write_new(RUN/"summary.json", summary)
        PUBLIC.mkdir(exist_ok=False)
        write_new(PUBLIC/"summary.json", summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    prepare() if parser.parse_args().prepare else run()
