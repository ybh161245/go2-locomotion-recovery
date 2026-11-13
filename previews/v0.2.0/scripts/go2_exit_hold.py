"""Run an explicitly selected 197 or 208 controller on a local task."""

import argparse
import pickle
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import numpy as np

import go2_compare
import go2_task
from go2_157 import ROOT, read, sha, write_new


FREEZE = ROOT/"docs/data/go2-exit-hold-new-conditions/frozen208.json"
VALIDATION = ROOT/"docs/data/go2-exit-hold-new-conditions/summary.json"
SCENARIOS = ROOT/"configs/go2-exit-hold-new-conditions.json"


def candidate_inputs():
    receipt = read(FREEZE)
    validation = read(VALIDATION)
    if receipt["candidate"] != 208 or validation["experiment"] != 209 or not validation["qualified"]:
        raise ValueError("Candidate208 requires its completed209 qualification")
    inputs = dict(receipt["sha256"])
    for path in (FREEZE, VALIDATION, Path(__file__)):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for path, digest in inputs.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Frozen208 input changed: {path}")
    return inputs, receipt["mechanism"]


def heading_segments(calls, states):
    intervals = []
    start = 0
    while start < len(calls):
        if calls[start]["anchor"] is None:
            start += 1
            continue
        end = start+1
        while end < len(calls) and calls[end]["anchor"] == calls[start]["anchor"]:
            end += 1
        anchor = calls[start]["anchor"]
        points = states["heading"][[min(start+49, end-1), end-1]]
        yaw = np.arctan2(points[:, 1], points[:, 0])
        changes = np.degrees((yaw-anchor+np.pi) % (2*np.pi)-np.pi)
        intervals.append(dict(start_s=start*.02, end_s=end*.02,
            first_second_deg=float(changes[0]) if end-start >= 50 else None,
            end_deg=float(changes[1]), feedback_frames=sum(c["extra"] != 0 for c in calls[start:end])))
        start = end
    return intervals


def run(controller, task, kind, output):
    if controller not in ("197", "208"):
        raise ValueError("Select controller197 or208 explicitly")
    if kind not in ("zero", "pressure"):
        raise ValueError("Select kind zero or pressure")
    go2_task.validate_task(task)
    if kind == "pressure" and "push" not in task:
        raise ValueError("pressure requires a task with push")
    runs = (ROOT/"artifacts/runs").resolve()
    output = Path(output).resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Use a new output directory below artifacts/runs")
    if output.exists():
        raise FileExistsError(output)
    if controller == "197":
        return go2_compare.run(controller, task, kind, output)

    inputs, mechanism = candidate_inputs()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"controller.json", dict(controller=208, inputs_sha256=inputs, mechanism=mechanism))
    shutil.copy2(__file__, output/"entry-runtime.py")
    from run_go2_turn_exit_heading_hold import attach, attach197

    for path in ("scripts/run_go2_turn_exit_heading_hold.py", "scripts/run_go2_full_heading_feedback.py"):
        shutil.copy2(ROOT/path, output/Path(path).name)
    load_stack = go2_task.load_stack
    instances = []

    def load_candidate(spec):
        config, stack, factory, evaluator = load_stack(spec)

        def candidate_factory(reference):
            plain, make = factory(reference)

            def make_candidate(*args, **kwargs):
                env = attach(attach197(make(*args, **kwargs), stack), mechanism)
                instances.append(env)
                return env

            return plain, make_candidate

        return config, stack, candidate_factory, evaluator

    try:
        with patch.object(go2_task, "load_stack", load_candidate):
            result = go2_task.run_task(task, output/"simulation", kind)
        env = instances.pop()
        folder = ROOT/result["folder"]
        with (folder/"exit-hold-calls.pkl").open("xb") as stream:
            pickle.dump(env.exit_hold_calls, stream, protocol=5)
        with (folder/"full-heading-calls.pkl").open("xb") as stream:
            pickle.dump(env.full_heading_calls, stream, protocol=5)
        with np.load(folder/"states.npz") as states:
            result["heading_hold_segments"] = heading_segments(env.exit_hold_calls, states)
        result["controller"] = 208
        candidate_inputs()
        write_new(output/"result.json", result)
        return result
    except Exception:
        (output/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["197", "208"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--side", choices=["left", "right"], help="Use a saved209 turn-exit task")
    source.add_argument("--config", type=Path, help="Use a go2_task JSON")
    parser.add_argument("--kind", choices=["zero", "pressure"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        task = read(args.config) if args.config else next(c["task"] for c in read(SCENARIOS)["cases"]
                                                         if c["side"] == args.side)
        result = run(args.controller, task, args.kind, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Controller {result['controller']}")
    go2_task.show([result])
    for segment in result.get("heading_hold_segments", []):
        print(f"Heading hold {segment['start_s']:.2f}-{segment['end_s']:.2f}s, end change {segment['end_deg']:.3f}deg")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
