"""209: protect inactive old tasks, then compare frozen208 on two new exit histories."""

import argparse
import csv
import pickle
import shutil
import traceback
from math import hypot
from unittest.mock import patch

import numpy as np

import go2_compare
import go2_task
import run_go2_turn_exit_heading_hold as hold
from go2_157 import ROOT, read, sha, write_new
from run_go2_reversal_push_task import equal_npz, measure, verify_inputs


CONFIG = ROOT/"configs/go2-exit-hold-new-conditions.json"
PLAN = ROOT/"docs/experiments/2026-11-06-exit-hold-retention-new-conditions.md"
RUN = ROOT/"artifacts/runs/20261106-exit-hold-new-conditions"
PUBLIC = ROOT/"docs/data/go2-exit-hold-new-conditions"
SOURCE = ROOT/"scripts/run_go2_exit_hold_new_conditions.py"


def inactive_record(result):
    path = ROOT/result["folder"]/"steps.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    anchor = previous = None
    for row in rows:
        vx, yaw = (float(row[key]) for key in ("command_vx", "command_yaw"))
        if (vx, yaw) != (.4, 0.):
            anchor = None
        elif previous is not None and abs(previous) > .025:
            anchor = True
        previous = yaw
        assert anchor is None, path
    assert result["passed"]
    return dict(folder=result["folder"], steps=len(rows), active_frames=0,
                passed=result["passed"], original_checks=len(result["checks"]),
                source_sha256=sha(path))


def prepare():
    assert not any(p.exists() for p in (CONFIG, RUN, PUBLIC))
    old = read(hold.CONFIG)
    verify_inputs(old)
    assert read(hold.PUBLIC/"summary.json")["qualified"]
    inputs = dict(old["inputs"])
    retention = []
    for name in ("go2-full-heading-retention", "go2-reversal-push-task", "go2-opposite-reversal-push-task"):
        path = ROOT/"docs/data"/name/"summary.json"
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
        for record in read(path)["records"]:
            if name == "go2-full-heading-retention" and not record["label"].startswith("normal"):
                continue
            if name != "go2-full-heading-retention" and record["controller"] != "197":
                continue
            result = record["result"]
            retention.append(inactive_record(result))
    assert len(retention) == 7 and sum(r["original_checks"] for r in retention) == 492
    for path in (SOURCE, PLAN, hold.SOURCE, hold.CONFIG, hold.PUBLIC/"summary.json",
                 ROOT/"scripts/run_go2_full_heading_feedback.py", ROOT/"scripts/run_go2_reversal_push_task.py",
                 ROOT/"scripts/run_go2_full_heading_new_conditions.py"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for record in retention:
        inputs[record["folder"]+"/steps.csv"] = record["source_sha256"]
    cases = []
    for side, fast, onset, force, seed in (("right", 4.4, 20.3, 30, 509101),
                                           ("left", 3.6, 19.68, -30, 509102)):
        base = next(c for c in old["cases"] if c["side"] == side)["task"]
        task = {
            "sequence": {"seed": seed, "segments": [dict(s) for s in base["sequence"]["segments"]]},
            "push": dict(start_s=onset, duration_s=.2, body_y_n=force)}
        task["sequence"]["segments"][2]["seconds"] = fast
        go2_task.validate_task(task)
        cases.append(dict(side=side, task=task))
    write_new(CONFIG, dict(experiment=209, cases=cases, inputs=inputs, retention=retention,
        max_physical=8, training=0, fits=0, mechanism=old["mechanism"],
        min_worst_push_gain_deg=old["min_worst_push_gain_deg"]))


def angles_and_exit(result, task):
    schedule = go2_task.validate_task(task)
    start, end, command, _ = next(s for s in schedule if s[3] == "exit trot")
    folder = ROOT/result["folder"]
    with (folder/"steps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[start:end]
    band = [hypot(float(r["vx"])-command[0], float(r["vy"])) <= .08 and abs(float(r["yaw_rate"])) <= .15 for r in rows]
    stable = next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])), None)
    with np.load(folder/"states.npz") as state:
        heading = state["heading"][[start-1, start+49, end-1]]
        yaw = np.arctan2(heading[:, 1], heading[:, 0])
        delta = np.degrees((yaw-yaw[0]+np.pi) % (2*np.pi)-np.pi)
    return dict(first_second_deg=float(delta[1]), end_deg=float(delta[2]), stable_s=stable,
                passed=stable is not None and stable <= 2 and len(rows) == end-start)


def run():
    config = read(CONFIG)
    verify_inputs(config)
    RUN.mkdir(exist_ok=False)
    PUBLIC.mkdir(exist_ok=False)
    for path in (SOURCE, PLAN, CONFIG):
        shutil.copy2(path, RUN/path.name)
    write_new(PUBLIC/"frozen208.json", dict(candidate=208, sha256=config["inputs"],
               mechanism=config["mechanism"], retained=config["retention"], new_physical_at_freeze=0))
    native_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, evaluator = native_load(spec)

        def wrapped(reference):
            plain, make = factory(reference)

            def make208(*args, **kwargs):
                env = hold.attach(hold.attach197(make(*args, **kwargs), stack), config["mechanism"])
                instances.append(env)
                return env

            return plain, make208

        return cfg, stack, wrapped, evaluator

    records = []
    try:
        for case in config["cases"]:
            for arm in ("197", "208"):
                zero = None
                for kind in ("zero", "pressure"):
                    task, side = case["task"], case["side"]
                    output = RUN/f"{side}-{arm}-{kind}"
                    if arm == "197":
                        result = go2_compare.run("197", task, kind, output)
                    else:
                        with patch.object(go2_task, "load_stack", load):
                            result = go2_task.run_task(task, output, kind)
                    folder = ROOT/result["folder"]
                    if kind == "zero":
                        zero = folder
                    else:
                        equal_npz(folder, zero, "states.npz", round(task["push"]["start_s"]*50))
                        for name in ("initial-state.npz", "physics.npz"):
                            equal_npz(folder, zero, name)
                    if arm == "208":
                        env = instances.pop()
                        calls = env.exit_hold_calls
                        with (folder/"exit-hold-calls.pkl").open("xb") as stream:
                            pickle.dump(calls, stream, protocol=5)
                        old = next(r for r in records if r["side"] == side and r["arm"] == "197" and r["kind"] == kind)
                        before = ROOT/old["result"]["folder"]
                        first = next((i for i, c in enumerate(calls) if not np.array_equal(c["actual"], c["shadow"])), len(calls))
                        equal_npz(folder, before, "states.npz", first)
                        for name in ("initial-state.npz", "physics.npz"):
                            equal_npz(folder, before, name)
                        with np.load(folder/"states.npz") as state:
                            actions = np.stack([c["actual"] for c in calls]).astype(np.float64)
                            np.testing.assert_array_equal(np.clip(env.action_targets(actions), *env.model.actuator_ctrlrange.T), state["targets"])
                    transition = angles_and_exit(result, task)
                    records.append(dict(side=side, arm=arm, kind=kind, result=result,
                        exit=transition, metrics=measure(result, task), passed=result["passed"] and transition["passed"]))
                    print("209", side, arm, kind, records[-1]["passed"], transition, flush=True)
        pairs = []
        for case in config["cases"]:
            for kind in ("zero", "pressure"):
                old, new = [next(r for r in records if r["side"] == case["side"] and r["arm"] == arm and r["kind"] == kind)
                            for arm in ("197", "208")]
                pairs.append(dict(side=case["side"], kind=kind, old=old["exit"], new=new["exit"],
                    old_pass=old["passed"], new_pass=new["passed"],
                    new_termination=not old["result"]["terminated"] and new["result"]["terminated"],
                    deltas={k: new["metrics"][k]-v for k, v in old["metrics"].items()}))
        pushes = [p for p in pairs if p["kind"] == "pressure"]
        gain = max(abs(p["old"]["first_second_deg"]) for p in pushes)-max(abs(p["new"]["first_second_deg"]) for p in pushes)
        angles_pass = all(abs(p["new"]["first_second_deg"]) < abs(p["old"]["first_second_deg"]) for p in pushes) and gain >= config["min_worst_push_gain_deg"]
        end_pass = max(abs(p["new"]["end_deg"]) for p in pairs) <= max(abs(p["old"]["end_deg"]) for p in pairs)
        protected = all(p["new_pass"] and not p["new_termination"] for p in pairs)
        verify_inputs(config)
        summary = dict(experiment=209, records=records, pairs=pairs, worst_push_gain_deg=gain,
            qualified=protected and angles_pass and end_pass, original_task_qualified=protected,
            heading_qualified=angles_pass and end_pass, old_reuse=7, new_physical=len(records), training=0, fits=0, visual="pending")
        write_new(RUN/"summary.json", summary)
        write_new(PUBLIC/"summary.json", summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    prepare() if parser.parse_args().prepare else run()
