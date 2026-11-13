"""Run an explicit local comparison controller on a saved or custom task."""
import argparse
import pickle
from pathlib import Path
import shutil
from unittest.mock import patch

import go2_task
from go2_157 import ROOT, read, sha, write_new


SCENARIOS = ROOT / "configs/go2-full-heading-new-conditions.json"
RECEIPTS = [
    "docs/data/go2-full-heading-matrix/retention-qualified-candidate.json",
    "docs/data/go2-full-heading-new-conditions/new-scope-candidate-freeze.json",
    "docs/data/go2-full-heading-retention/protected-candidate-freeze.json",
    "docs/data/go2-full-heading-feedback/frozen-development-candidate.json",
]


def candidate_inputs():
    inputs = {}
    for name in RECEIPTS:
        receipt = read(ROOT / name)
        if receipt["candidate"] != 197:
            raise ValueError(f"Expected candidate197 in {name}")
        inputs.setdefault(name, sha(ROOT / name))
        for path, digest in receipt["sha256"].items():
            if inputs.setdefault(path, digest) != digest:
                raise ValueError(f"Candidate receipts disagree: {path}")
    for path, digest in inputs.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen candidate input changed: {path}")
    return inputs


def run(controller, task, kind, output):
    go2_task.validate_task(task)
    if kind == "pressure" and "push" not in task:
        raise ValueError("pressure requires a task with push")
    runs = (ROOT / "artifacts/runs").resolve()
    output = output.resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Use a new output directory below artifacts/runs")
    inputs = candidate_inputs() if controller == "197" else {}
    inputs[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "controller.json", {"controller": int(controller), "inputs_sha256": inputs})
    shutil.copy2(__file__, output / "comparison-runtime.py")
    if controller == "197":
        from run_go2_full_heading_feedback import attach

        load_stack = go2_task.load_stack
        instances = []

        def load_candidate(spec):
            config, stack, factory, evaluator = load_stack(spec)

            def candidate_factory(reference):
                plain, make = factory(reference)

                def make_candidate(*args, **kwargs):
                    env = attach(make(*args, **kwargs), stack)
                    instances.append(env)
                    return env

                return plain, make_candidate

            return config, stack, candidate_factory, evaluator

        with patch.object(go2_task, "load_stack", load_candidate):
            result = go2_task.run_task(task, output / "simulation", kind)
        calls = instances[0].full_heading_calls
        folder = ROOT / result["folder"]
        with (folder / "full-heading-calls.pkl").open("xb") as stream:
            pickle.dump(calls, stream, protocol=5)
        result["heading_feedback_frames"] = sum(call["extra"] != 0 for call in calls)
        candidate_inputs()
    else:
        result = go2_task.run_task(task, output / "simulation", kind)
    result["controller"] = int(controller)
    write_new(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=["157", "197"], required=True)
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument("--side", choices=["left", "right"], help="Use the saved199 comparison task")
    task_source.add_argument("--config", type=Path, help="Use a go2_task JSON")
    parser.add_argument("--kind", choices=["zero", "pressure"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.config:
            task = read(args.config)
        else:
            task = next(record["task"] for record in read(SCENARIOS)["definitions"] if record["side"] == args.side)
        result = run(args.controller, task, args.kind, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Controller {result['controller']}")
    go2_task.show([result])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
