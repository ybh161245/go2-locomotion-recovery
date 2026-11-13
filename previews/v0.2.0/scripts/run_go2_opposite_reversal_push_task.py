"""206: reverse the turn and push signs in 205, using the same evaluation seed."""

import argparse
import shutil
import traceback

import go2_compare
import go2_task
import run_go2_reversal_push_task as previous
from go2_157 import ROOT, read, sha, write_new


CONFIG = ROOT / "configs/go2-opposite-reversal-push-task.json"
PLAN = ROOT / "docs/experiments/2026-11-06-opposite-reversal-push-task.md"
RUN = ROOT / "artifacts/runs/20261106-opposite-reversal-push-task"
PUBLIC = ROOT / "docs/data/go2-opposite-reversal-push-task"
RUNTIME = ROOT / "scripts/run_go2_opposite_reversal_push_task.py"


def prepare():
    assert not any(path.exists() for path in (CONFIG, RUN, PUBLIC))
    parent = read(previous.CONFIG)
    previous.verify_inputs(parent)
    task = parent["task"]
    for segment in task["sequence"]["segments"]:
        if segment["motion"] == "left":
            segment.update(name="right", motion="right")
        elif segment["motion"] == "right":
            segment.update(name="reverse left", motion="left")
    task["push"]["body_y_n"] *= -1
    assert go2_task.validate_task(task)[-1][1] == parent["steps"]
    inputs = dict(parent["inputs"])
    for path in (previous.CONFIG, PLAN, RUNTIME):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    write_new(CONFIG, dict(experiment=206, task=task, arms=parent["arms"], kinds=parent["kinds"],
        max_physical=4, training=0, fits=0, inputs=inputs, steps=parent["steps"], gate=parent["gate"]))


def run():
    config = read(CONFIG)
    previous.verify_inputs(config)
    assert config["arms"] == list(previous.ARMS) and config["kinds"] == list(previous.KINDS)
    RUN.mkdir(exist_ok=False)
    for path in (CONFIG, PLAN, RUNTIME):
        shutil.copy2(path, RUN/path.name)
    task = config["task"]
    records = {}
    try:
        for arm in config["arms"]:
            for kind in config["kinds"]:
                result = go2_compare.run(arm, task, kind, RUN/f"{arm}-{kind}")
                records[arm, kind] = dict(controller=arm, kind=kind, result=result,
                                          metrics=previous.measure(result, task))
                print("206", arm, kind, result["passed"], result["turn_angles"],
                      [(c["segment"], c["metric"], c["value"]) for c in result["checks"] if not c["passed"]],
                      flush=True)
            zero, push = (ROOT/records[arm, kind]["result"]["folder"] for kind in config["kinds"])
            for filename in ("initial-state.npz", "physics.npz"):
                previous.equal_npz(zero, push, filename)
            previous.equal_npz(zero, push, "states.npz", round(task["push"]["start_s"]*50))
        pairs = [previous.compare(records["157", kind], records["197", kind]) for kind in config["kinds"]]
        previous.verify_inputs(config)
        summary = dict(experiment=206, records=list(records.values()), pairs=pairs,
            qualified=all(p["new_pass"] and not p["added_termination"] for p in pairs),
            gains=sum(not p["old_pass"] and p["new_pass"] for p in pairs),
            old_pass_losses=sum(p["old_pass"] and not p["new_pass"] for p in pairs),
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
