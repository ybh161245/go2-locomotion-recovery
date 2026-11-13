"""Explicit208/212 local entry for holding direction during speed changes."""

import argparse
import pickle
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

import go2_exit_hold as parent
import go2_task
from go2_157 import ROOT, read, sha, write_new


FREEZE = ROOT/"docs/data/go2-variable-speed-new-conditions/frozen212.json"
VALIDATION = ROOT/"docs/data/go2-variable-speed-new-conditions/summary.json"
SCENARIOS = {"left": ROOT/"configs/go2-variable-speed-new-conditions.json",
             "right": ROOT/"configs/go2-variable-speed-heading-hold.json"}


def candidate_inputs():
    receipt, validation = read(FREEZE), read(VALIDATION)
    if receipt["candidate"] != 212 or validation["experiment"] != 213 or not validation["qualified"]:
        raise ValueError("Candidate212 requires completed213 qualification")
    inputs = dict(receipt["sha256"])
    for path in (FREEZE, VALIDATION, Path(__file__)):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for path, digest in inputs.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Frozen212 input changed: {path}")
    return inputs, receipt["mechanism"]


def run(controller, task, kind, output):
    if controller not in ("208", "212"):
        raise ValueError("Select controller208 or212 explicitly")
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
    if controller == "208":
        return parent.run(controller, task, kind, output)
    inputs, mechanism = candidate_inputs()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"controller.json", dict(controller=212, inputs_sha256=inputs, mechanism=mechanism))
    for path in (Path(__file__), ROOT/"scripts/go2_variable_speed_heading_hold.py",
                 ROOT/"scripts/run_go2_full_heading_feedback.py"):
        shutil.copy2(path, output/path.name)
    from go2_variable_speed_heading_hold import attach
    from run_go2_full_heading_feedback import attach as attach197
    load_stack = go2_task.load_stack
    instances = []

    def load(spec):
        config, stack, factory, evaluator = load_stack(spec)

        def wrapped(reference):
            plain, make = factory(reference)

            def make212(*args, **kwargs):
                env = attach(attach197(make(*args, **kwargs), stack), mechanism)
                instances.append(env)
                return env

            return plain, make212

        return config, stack, wrapped, evaluator

    try:
        with patch.object(go2_task, "load_stack", load):
            result = go2_task.run_task(task, output/"simulation", kind)
        env = instances.pop()
        folder = ROOT/result["folder"]
        with (folder/"exit-hold-calls.pkl").open("xb") as stream:
            pickle.dump(env.exit_hold_calls, stream, protocol=5)
        with (folder/"full-heading-calls.pkl").open("xb") as stream:
            pickle.dump(env.full_heading_calls, stream, protocol=5)
        import numpy as np
        with np.load(folder/"states.npz") as states:
            result["heading_hold_segments"] = parent.heading_segments(env.exit_hold_calls, states)
        result["controller"] = 212
        candidate_inputs()
        write_new(output/"result.json", result)
        return result
    except Exception:
        (output/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["208", "212"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--side", choices=SCENARIOS, help="Use a tested left or right speed-change task")
    source.add_argument("--config", type=Path, help="Use a go2_task JSON")
    parser.add_argument("--kind", choices=["zero", "pressure"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        task = read(args.config) if args.config else read(SCENARIOS[args.side])["task"]
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
