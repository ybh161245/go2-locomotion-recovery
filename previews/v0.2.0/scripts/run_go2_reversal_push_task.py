"""205: one direct reversal task, paired by controller and push."""

import argparse
import pickle
import shutil
import traceback

import numpy as np

import go2_compare
import go2_task
from go2_157 import ROOT, read, sha, write_new
from run_go2_full_heading_new_conditions import measure


CONFIG = ROOT / "configs/go2-reversal-push-task.json"
PLAN = ROOT / "docs/experiments/2026-11-06-reversal-push-task.md"
RUN = ROOT / "artifacts/runs/20261106-reversal-push-task"
PUBLIC = ROOT / "docs/data/go2-reversal-push-task"
ARMS = ("157", "197")
KINDS = ("zero", "pressure")


def prepare():
    assert not any(path.exists() for path in (CONFIG, RUN, PUBLIC))
    segments = [dict(name=name, motion=motion, seconds=seconds)
                for name, motion, seconds in (
                    ("initial stand", "stand", 2), ("walk", "walk", 2), ("trot", "trot", 4),
                    ("left", "left", 6), ("reverse right", "right", 6), ("final stand", "stand", 4))]
    task = dict(sequence=dict(seed=505101, segments=segments),
                push=dict(start_s=14.06, duration_s=.2, body_y_n=-30))
    schedule = go2_task.validate_task(task)
    assert schedule[-1][1] == 1200
    inputs = go2_compare.candidate_inputs()
    for name in ("scripts/run_go2_reversal_push_task.py", "scripts/go2_compare.py",
                 "scripts/go2_task.py", "scripts/go2_sequence.py",
                 "scripts/run_go2_full_heading_new_conditions.py",
                 "docs/experiments/2026-11-06-reversal-push-task.md",
                 "docs/data/go2-current/selection.json"):
        inputs[name] = sha(ROOT/name)
    write_new(CONFIG, dict(experiment=205, task=task, arms=ARMS, kinds=KINDS, max_physical=4,
        training=0, fits=0, inputs=inputs, steps=1200,
        gate="197 both pass original 8deg, 2s, 25 consecutive steps, per-segment quality and both 6s angles; no losses or new terminations"))


def verify_inputs(config):
    for name, digest in config["inputs"].items():
        assert sha(ROOT/name) == digest, name


def equal_npz(folder_a, folder_b, filename, until=None):
    with np.load(folder_a/filename) as a, np.load(folder_b/filename) as b:
        assert set(a.files) == set(b.files)
        if filename == "states.npz":
            assert len(a.files) == 8
        for key in a.files:
            u, v = a[key], b[key]
            if until is not None:
                u, v = u[:until], v[:until]
            np.testing.assert_array_equal(u, v, err_msg=f"{filename}: {key}")


def compare(old, new):
    baseline, candidate = (ROOT/r["folder"] for r in (old["result"], new["result"]))
    for filename in ("initial-state.npz", "physics.npz"):
        equal_npz(baseline, candidate, filename)
    with (candidate/"full-heading-calls.pkl").open("rb") as stream:
        calls = pickle.load(stream)
    first = next((i for i, c in enumerate(calls)
                  if not np.array_equal(c["shadow"], c["actual"])), len(calls))
    equal_npz(baseline, candidate, "states.npz", first)
    # The two commands switch directly at 14 s. Record observed scope, rather than assume it.
    reverse_calls = calls[700:1000]
    a, b = old["result"], new["result"]
    return dict(kind=a["kind"], old_pass=a["passed"], new_pass=b["passed"],
                added_termination=not a["terminated"] and b["terminated"],
                first_changed_step=first, reverse_feedback_frames=sum(c["extra"] != 0 for c in reverse_calls),
                angle_delta_deg=[n-o for o, n in zip(a["turn_angles"], b["turn_angles"])],
                peak_delta_deg=b["peak_tilt"]-a["peak_tilt"],
                old_recovery_s=a["recovery_seconds"], new_recovery_s=b["recovery_seconds"],
                metrics_delta={key: new["metrics"][key]-old["metrics"][key] for key in new["metrics"]})


def run():
    config = read(CONFIG)
    verify_inputs(config)
    assert config["arms"] == list(ARMS) and config["kinds"] == list(KINDS)
    RUN.mkdir(exist_ok=False)
    for path in (CONFIG, PLAN, ROOT/"scripts/run_go2_reversal_push_task.py"):
        shutil.copy2(path, RUN/path.name)
    task = config["task"]
    records = {}
    try:
        for arm in ARMS:
            for kind in KINDS:
                result = go2_compare.run(arm, task, kind, RUN/f"{arm}-{kind}")
                records[arm, kind] = dict(controller=arm, kind=kind, result=result, metrics=measure(result, task))
                print("205", arm, kind, result["passed"], result["turn_angles"],
                      [(c["segment"], c["metric"], c["value"]) for c in result["checks"] if not c["passed"]],
                      flush=True)
            zero, push = (ROOT/records[arm, k]["result"]["folder"] for k in KINDS)
            for filename in ("initial-state.npz", "physics.npz"):
                equal_npz(zero, push, filename)
            equal_npz(zero, push, "states.npz", round(task["push"]["start_s"]*50))
        pairs = [compare(records["157", kind], records["197", kind]) for kind in KINDS]
        verify_inputs(config)
        qualified = all(p["new_pass"] and not p["added_termination"] for p in pairs)
        summary = dict(experiment=205, records=list(records.values()), pairs=pairs,
            qualified=qualified, gains=sum(not p["old_pass"] and p["new_pass"] for p in pairs),
            old_pass_losses=sum(p["old_pass"] and not p["new_pass"] for p in pairs),
            new_physical=4, training=0, fits=0, visual="pending")
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
