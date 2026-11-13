"""207: verify both turn exits into a trot with frozen197 and matched pushes."""

import argparse
import csv
from math import hypot
import shutil
import traceback

import numpy as np

import go2_compare
import go2_task
import run_go2_opposite_reversal_push_task as opposite
import run_go2_reversal_push_task as previous
from go2_157 import ROOT, read, sha, write_new


CONFIG = ROOT / "configs/go2-turn-exit-trot.json"
PLAN = ROOT / "docs/experiments/2026-11-06-turn-exit-trot.md"
RUN = ROOT / "artifacts/runs/20261106-turn-exit-trot"
PUBLIC = ROOT / "docs/data/go2-turn-exit-trot"
RUNTIME = ROOT / "scripts/run_go2_turn_exit_trot.py"
EXIT_STEP, STOP_STEP = 1000, 1200


def prepare():
    assert not any(path.exists() for path in (CONFIG, RUN, PUBLIC))
    inputs = go2_compare.candidate_inputs()
    definitions = []
    for side, parent in (("right", previous), ("left", opposite)):
        old = read(parent.CONFIG)
        previous.verify_inputs(old)
        task = old["task"]
        task["sequence"]["segments"].insert(-1, dict(name="exit trot", motion="trot", seconds=4))
        task["push"]["start_s"] = 20.06
        schedule = go2_task.validate_task(task)
        assert schedule[-2][:2] == (EXIT_STEP, STOP_STEP) and schedule[-1][1] == 1400
        summary = parent.PUBLIC/"summary.json"
        reference = next(r["result"] for r in read(summary)["records"]
                         if r["controller"] == "197" and r["kind"] == "zero")
        definitions.append(dict(side=side, task=task, reference=reference))
        for path in (parent.CONFIG, summary, ROOT/reference["folder"]/"states.npz"):
            inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for path in (RUNTIME, PLAN, ROOT/"scripts/go2_compare.py", ROOT/"scripts/go2_task.py",
                 ROOT/"scripts/go2_sequence.py", ROOT/"scripts/run_go2_reversal_push_task.py",
                 ROOT/"scripts/run_go2_full_heading_new_conditions.py",
                 ROOT/"docs/data/go2-current/selection.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(CONFIG, dict(experiment=207, definitions=definitions, controller="197", max_physical=4,
        training=0, fits=0, inputs=inputs, steps=1400,
        exit_gate=dict(max_seconds=2, planar_error_m_s=.08, yaw_error_rad_s=.15, consecutive=25)))


def exit_metrics(result, gate):
    folder = ROOT/result["folder"]
    with (folder/"steps.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))[EXIT_STEP:STOP_STEP]
    band = [hypot(float(r["vx"])-.4, float(r["vy"])) <= gate["planar_error_m_s"]
            and abs(float(r["yaw_rate"])) <= gate["yaw_error_rad_s"] for r in rows]
    count = gate["consecutive"]
    stable = next((i*.02 for i in range(len(band)-count+1) if all(band[i:i+count])), None)
    drift = None
    with np.load(folder/"states.npz") as states:
        if len(rows) == STOP_STEP-EXIT_STEP:
            heading = states["heading"][EXIT_STEP-1:STOP_STEP]
            angle = np.unwrap(np.arctan2(heading[:, 1], heading[:, 0]))
            drift = float(np.degrees(angle[-1]-angle[0]))
    return dict(stable_s=stable, complete=len(rows) == STOP_STEP-EXIT_STEP, heading_change_deg=drift,
                passed=stable is not None and stable <= gate["max_seconds"] and len(rows) == STOP_STEP-EXIT_STEP)


def run():
    config = read(CONFIG)
    previous.verify_inputs(config)
    RUN.mkdir(exist_ok=False)
    for path in (CONFIG, PLAN, RUNTIME):
        shutil.copy2(path, RUN/path.name)
    records = []
    try:
        for definition in config["definitions"]:
            side, task, reference = (definition[key] for key in ("side", "task", "reference"))
            zero = None
            for kind in previous.KINDS:
                result = go2_compare.run(config["controller"], task, kind, RUN/f"{side}-{kind}")
                folder = ROOT/result["folder"]
                previous.equal_npz(folder, ROOT/reference["folder"], "states.npz", EXIT_STEP)
                if kind == "zero":
                    zero = folder
                else:
                    for filename in ("initial-state.npz", "physics.npz"):
                        previous.equal_npz(folder, zero, filename)
                    previous.equal_npz(folder, zero, "states.npz", round(task["push"]["start_s"]*50))
                metrics = previous.measure(result, task)
                transition = exit_metrics(result, config["exit_gate"])
                passed = result["passed"] and transition["passed"]
                records.append(dict(side=side, kind=kind, result=result, metrics=metrics, exit=transition, passed=passed))
                print("207", side, kind, passed, result["peak_tilt"], result["recovery_seconds"], transition,
                      [(c["segment"], c["metric"], c["value"]) for c in result["checks"] if not c["passed"]],
                      flush=True)
        previous.verify_inputs(config)
        summary = dict(experiment=207, records=records, qualified=all(r["passed"] for r in records),
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
