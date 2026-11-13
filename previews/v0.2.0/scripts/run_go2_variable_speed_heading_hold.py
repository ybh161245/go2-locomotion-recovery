"""212: one bounded development comparison on slowing and accelerating after a turn."""

import argparse
import csv
import pickle
import shutil
import traceback
from math import hypot
from unittest.mock import patch

import numpy as np

import go2_exit_hold as entry
import go2_task
from go2_157 import ROOT, read, sha, write_new
from go2_variable_speed_heading_hold import attach
from run_go2_full_heading_feedback import attach as attach197
from run_go2_reversal_push_task import equal_npz, measure, verify_inputs


CONFIG = ROOT/"configs/go2-variable-speed-heading-hold.json"
PLAN = ROOT/"docs/experiments/2026-11-06-variable-speed-heading-hold.md"
RUN = ROOT/"artifacts/runs/20261106-variable-speed-heading-hold"
PUBLIC = ROOT/"docs/data/go2-variable-speed-heading-hold"
SOURCE = ROOT/"scripts/run_go2_variable_speed_heading_hold.py"


def prepare():
    assert not any(path.exists() for path in (CONFIG, RUN, PUBLIC))
    inputs, mechanism = entry.candidate_inputs()
    task = dict(sequence=dict(seed=512101, segments=[
        dict(name=n, motion=m, seconds=s) for n, m, s in (
            ("initial stand", "stand", 2), ("walk", "walk", 2), ("right", "right", 6),
            ("exit fast", "trot", 2), ("slow straight", "walk", 3),
            ("accelerate again", "trot", 3), ("final stand", "stand", 4))]),
        push=dict(start_s=15.06, duration_s=.2, body_y_n=30))
    assert go2_task.validate_task(task)[-1][1] == 1100
    for path in (SOURCE, PLAN, ROOT/"scripts/go2_variable_speed_heading_hold.py",
                 ROOT/"scripts/go2_task.py", ROOT/"scripts/run_go2_reversal_push_task.py",
                 ROOT/"scripts/run_go2_full_heading_new_conditions.py", ROOT/"docs/data/go2-current/selection.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(CONFIG, dict(experiment=212, task=task, inputs=inputs, mechanism=mechanism,
        max_physical=4, training=0, fits=0, min_worst_heading_gain_deg=.5))


def metrics(result, task):
    folder = ROOT/result["folder"]
    with (folder/"steps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    stable = []
    for start, end, command, name in go2_task.validate_task(task):
        if start not in (500, 600, 750):
            continue
        band = [hypot(float(r["vx"])-command[0], float(r["vy"])) <= .08 and abs(float(r["yaw_rate"])) <= .15
                for r in rows[start:end]]
        seconds = next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])), None)
        stable.append(dict(name=name, seconds=seconds, passed=seconds is not None and seconds <= 2))
    with np.load(folder/"states.npz") as state:
        w, x, y, z = state["qpos"][[499, 599, 749, 899], 3:7].T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        heading = np.degrees((yaw-yaw[0]+np.pi) % (2*np.pi)-np.pi).tolist()
    return dict(speed_transitions=stable, heading_at_10_12_15_18_s_deg=heading,
                heading_end_deg=heading[-1], passed=result["passed"] and all(s["passed"] for s in stable))


def run():
    config = read(CONFIG)
    verify_inputs(config)
    RUN.mkdir(exist_ok=False)
    for path in (CONFIG, PLAN, SOURCE, ROOT/"scripts/go2_variable_speed_heading_hold.py"):
        shutil.copy2(path, RUN/path.name)
    original_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, evaluator = original_load(spec)

        def wrapped(reference):
            plain, make = factory(reference)

            def make212(*args, **kwargs):
                env = attach(attach197(make(*args, **kwargs), stack), config["mechanism"])
                instances.append(env)
                return env

            return plain, make212

        return cfg, stack, wrapped, evaluator

    records = []
    try:
        for controller in ("208", "212"):
            zero = None
            for kind in ("zero", "pressure"):
                task = config["task"]
                output = RUN/f"{controller}-{kind}"
                if controller == "208":
                    result = entry.run("208", task, kind, output)
                else:
                    with patch.object(go2_task, "load_stack", load):
                        result = go2_task.run_task(task, output, kind)
                    env = instances.pop()
                    folder = ROOT/result["folder"]
                    calls = env.exit_hold_calls
                    with (folder/"exit-hold-calls.pkl").open("xb") as stream:
                        pickle.dump(calls, stream, protocol=5)
                    baseline = next(r for r in records if r["controller"] == "208" and r["kind"] == kind)
                    before = ROOT/baseline["result"]["folder"]
                    equal_npz(folder, before, "states.npz", 600)
                    for name in ("initial-state.npz", "physics.npz"):
                        equal_npz(folder, before, name)
                    assert all(c["anchor"] == calls[500]["anchor"] for c in calls[500:900])
                    assert all(c["anchor"] is None and c["extra"] == 0 for c in calls[:500]+calls[900:])
                    with np.load(folder/"states.npz") as state:
                        actual = np.stack([c["actual"] for c in calls]).astype(np.float64)
                        np.testing.assert_array_equal(np.clip(env.action_targets(actual), *env.model.actuator_ctrlrange.T), state["targets"])
                    result["controller"] = 212
                    with np.load(folder/"states.npz") as state:
                        result["heading_hold_segments"] = entry.heading_segments(calls, state)
                    write_new(output/"result.json", result)
                folder = ROOT/result["folder"]
                if kind == "zero":
                    zero = folder
                else:
                    equal_npz(folder, zero, "states.npz", 753)
                values = metrics(result, task)
                records.append(dict(controller=controller, kind=kind, result=result,
                    motion=values, metrics=measure(result, task), passed=values["passed"]))
                print("212", controller, kind, values, flush=True)
        pairs = []
        for kind in ("zero", "pressure"):
            old, new = [next(r for r in records if r["controller"] == c and r["kind"] == kind) for c in ("208", "212")]
            pairs.append(dict(kind=kind, old_pass=old["passed"], new_pass=new["passed"],
                old_heading_deg=old["motion"]["heading_end_deg"], new_heading_deg=new["motion"]["heading_end_deg"],
                new_termination=not old["result"]["terminated"] and new["result"]["terminated"],
                deltas={k: new["metrics"][k]-v for k, v in old["metrics"].items()}))
        gain = max(abs(p["old_heading_deg"]) for p in pairs)-max(abs(p["new_heading_deg"]) for p in pairs)
        verify_inputs(config)
        protected = all(p["new_pass"] and not p["new_termination"] for p in pairs)
        improved = all(abs(p["new_heading_deg"]) < abs(p["old_heading_deg"]) for p in pairs)
        summary = dict(experiment=212, records=records, pairs=pairs, worst_gain_deg=gain,
            qualified=protected and improved and gain >= config["min_worst_heading_gain_deg"],
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
