"""213: old-scope identity plus one opposite-direction prospective speed-change task."""

import argparse
import csv
import pickle
import shutil
import traceback
from unittest.mock import patch

import numpy as np

import go2_exit_hold as entry
import go2_task
import run_go2_variable_speed_heading_hold as development
from go2_157 import ROOT, read, sha, write_new
from run_go2_reversal_push_task import equal_npz, measure, verify_inputs


CONFIG = ROOT/"configs/go2-variable-speed-new-conditions.json"
PLAN = ROOT/"docs/experiments/2026-11-06-variable-speed-new-conditions.md"
RUN = ROOT/"artifacts/runs/20261106-variable-speed-new-conditions"
PUBLIC = ROOT/"docs/data/go2-variable-speed-new-conditions"
SOURCE = ROOT/"scripts/run_go2_variable_speed_new_conditions.py"


def protection():
    records = []
    for name in ("go2-full-heading-retention", "go2-exit-hold-new-conditions", "go2-two-exit-mission"):
        for record in read(ROOT/"docs/data"/name/"summary.json")["records"]:
            if name == "go2-full-heading-retention" and not record["label"].startswith("normal"):
                continue
            if name == "go2-exit-hold-new-conditions" and record["arm"] != "208":
                continue
            if name == "go2-two-exit-mission" and record["controller"] != "208":
                continue
            result = record["result"]
            path = ROOT/result["folder"]/"steps.csv"
            old = new = previous_yaw = None
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for i, row in enumerate(rows):
                vx, yaw = float(row["command_vx"]), float(row["command_yaw"])
                command = (vx, float(row.get("command_vy", 0)), yaw)
                if command != (.4, 0., 0.):
                    old = None
                elif previous_yaw is not None and abs(previous_yaw) > .025:
                    old = i
                if command not in ((.2, 0., 0.), (.4, 0., 0.)):
                    new = None
                elif vx == .4 and previous_yaw is not None and abs(previous_yaw) > .025:
                    new = i
                assert old == new, (path, i)
                previous_yaw = yaw
            assert result["passed"]
            records.append(dict(folder=result["folder"], steps=len(rows), original_checks=len(result["checks"]),
                all_anchor_indices_equal=True, command_sha256=sha(path)))
    assert len(records) == 9
    return records


def prepare():
    assert not any(p.exists() for p in (CONFIG, RUN, PUBLIC))
    prior = read(development.CONFIG)
    verify_inputs(prior)
    assert read(development.PUBLIC/"summary.json")["qualified"]
    retained = protection()
    inputs = dict(prior["inputs"])
    for path in (SOURCE, PLAN, development.CONFIG, development.PUBLIC/"summary.json",
                 ROOT/"scripts/complete_go2_variable_speed_heading_hold.py"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for record in retained:
        inputs[record["folder"]+"/steps.csv"] = record["command_sha256"]
    task = dict(sequence=dict(seed=513101, segments=[
        dict(name=n, motion=m, seconds=s) for n,m,s in (
            ("initial stand", "stand", 2), ("walk", "walk", 2), ("left", "left", 6),
            ("exit fast", "trot", 2.4), ("slow straight", "walk", 2.6),
            ("accelerate again", "trot", 3), ("final stand", "stand", 4))]),
        push=dict(start_s=15.12, duration_s=.2, body_y_n=-30))
    assert go2_task.validate_task(task)[-1][1] == 1100
    write_new(CONFIG, dict(experiment=213, task=task, inputs=inputs, retained=retained,
        mechanism=prior["mechanism"], min_worst_heading_gain_deg=.5, max_physical=4, training=0, fits=0))


def metrics(result, task):
    folder = ROOT/result["folder"]
    with (folder/"steps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    stable = []
    for start, end, command, name in go2_task.validate_task(task):
        if name not in ("exit fast", "slow straight", "accelerate again"):
            continue
        band = [np.hypot(float(r["vx"])-command[0], float(r["vy"])) <= .08
                and abs(float(r["yaw_rate"])) <= .15 for r in rows[start:end]]
        seconds = next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])), None)
        stable.append(dict(name=name, seconds=seconds, passed=seconds is not None and seconds <= 2))
    with np.load(folder/"states.npz") as state:
        w,x,y,z = state["qpos"][[499,899],3:7].T
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        delta = float(np.degrees((yaw[1]-yaw[0]+np.pi) % (2*np.pi)-np.pi))
    return dict(speed_transitions=stable, heading_end_deg=delta,
                passed=result["passed"] and all(s["passed"] for s in stable))


def run():
    config = read(CONFIG)
    verify_inputs(config)
    assert protection() == config["retained"]
    RUN.mkdir(exist_ok=False)
    PUBLIC.mkdir(exist_ok=False)
    for p in (SOURCE, CONFIG, PLAN):
        shutil.copy2(p, RUN/p.name)
    write_new(PUBLIC/"frozen212.json", dict(candidate=212, sha256=config["inputs"],
        mechanism=config["mechanism"], retained=config["retained"], new_physical_at_freeze=0))
    native_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, evaluator = native_load(spec)

        def wrapped(reference):
            plain, make = factory(reference)

            def make212(*args, **kwargs):
                env = development.attach(development.attach197(make(*args, **kwargs), stack), config["mechanism"])
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
                out = RUN/f"{controller}-{kind}"
                if controller == "208":
                    result = entry.run(controller, task, kind, out)
                else:
                    with patch.object(go2_task, "load_stack", load):
                        result = go2_task.run_task(task, out, kind)
                    env = instances.pop()
                    folder = ROOT/result["folder"]
                    calls = env.exit_hold_calls
                    with (folder/"exit-hold-calls.pkl").open("xb") as stream:
                        pickle.dump(calls, stream, protocol=5)
                    old = next(r for r in records if r["controller"] == "208" and r["kind"] == kind)
                    before = ROOT/old["result"]["folder"]
                    equal_npz(folder, before, "states.npz", 620)
                    for name in ("initial-state.npz", "physics.npz"):
                        equal_npz(folder, before, name)
                    with np.load(folder/"states.npz") as state:
                        actual = np.stack([c["actual"] for c in calls]).astype(np.float64)
                        np.testing.assert_array_equal(np.clip(env.action_targets(actual), *env.model.actuator_ctrlrange.T), state["targets"])
                folder = ROOT/result["folder"]
                if kind == "zero":
                    zero = folder
                else:
                    equal_npz(folder, zero, "states.npz", 756)
                values = metrics(result, task)
                records.append(dict(controller=controller, kind=kind, result=result, motion=values,
                    metrics=measure(result, task), passed=values["passed"]))
                print("213", controller, kind, values, flush=True)
        pairs = []
        for kind in ("zero", "pressure"):
            old,new = [next(r for r in records if r["controller"] == c and r["kind"] == kind) for c in ("208","212")]
            pairs.append(dict(kind=kind, old_pass=old["passed"], new_pass=new["passed"],
                old_heading_deg=old["motion"]["heading_end_deg"], new_heading_deg=new["motion"]["heading_end_deg"],
                new_termination=not old["result"]["terminated"] and new["result"]["terminated"],
                deltas={k:new["metrics"][k]-v for k,v in old["metrics"].items()}))
        gain = max(abs(p["old_heading_deg"]) for p in pairs)-max(abs(p["new_heading_deg"]) for p in pairs)
        protected = all(p["new_pass"] and not p["new_termination"] for p in pairs)
        improved = all(abs(p["new_heading_deg"]) < abs(p["old_heading_deg"]) for p in pairs)
        verify_inputs(config)
        summary = dict(experiment=213, records=records, pairs=pairs, worst_gain_deg=gain,
            qualified=protected and improved and gain >= .5, retained_records=9,
            new_physical=len(records), training=0, fits=0, visual="pending")
        write_new(RUN/"summary.json",summary)
        write_new(PUBLIC/"summary.json",summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    prepare() if parser.parse_args().prepare else run()
