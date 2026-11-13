"""Configure, run and inspect local frozen157 motion tasks."""
import argparse
import math
from pathlib import Path
import shutil
import traceback

from go2_157 import ROOT, RUNTIME, load_stack, read, sha, write_new


PRESETS = {
    "sequence": (186, "go2-command-sequence", "24-second mixed motion"),
    "push-stop": (187, "go2-sequence-push", "Turn, side push and stop"),
    "push-restart": (188, "go2-sequence-restart", "Turn, side push, stop and restart"),
}


def preset_config(preset, side):
    if preset == "sequence":
        return {"sequence": read(ROOT / "configs/go2-command-sequence-demo.json")}
    _, name, _ = PRESETS[preset]
    source = read(ROOT / "configs" / f"{name}.json")
    definition = next(row for row in source["definitions"] if row["id"] == side)
    task = {
        "sequence": definition["sequence"],
        "push": {
            "start_s": source["onset_step"] / 50,
            "duration_s": source["duration_steps"] / 50,
            "body_y_n": source["force_N"] * definition["body_y_sign"],
        },
    }
    if preset == "push-restart":
        task["restart_segment"] = "restart"
    return task


def ticks(value, label):
    if type(value) not in (int, float) or not 0 <= value <= 46 or not math.isfinite(value):
        raise ValueError(f"{label} must be a number from 0 to 46 seconds")
    step = round(value * 50)
    if abs(step - value * 50) > 1e-8:
        raise ValueError(f"{label} must be on the 0.02-second grid")
    return step


def validate_task(task):
    from go2_sequence import validate

    if not isinstance(task, dict) or "sequence" not in task or set(task) - {"sequence", "push", "restart_segment"}:
        raise ValueError("Task requires sequence and may include push and restart_segment")
    schedule = validate(task["sequence"])
    if "push" in task:
        push = task["push"]
        if not isinstance(push, dict) or set(push) != {"start_s", "duration_s", "body_y_n"}:
            raise ValueError("push requires start_s, duration_s and body_y_n")
        start = ticks(push["start_s"], "Push start")
        duration = ticks(push["duration_s"], "Push duration")
        force = push["body_y_n"]
        if type(force) not in (int, float) or not 0 < abs(force) <= 30 or not 1 <= duration <= 10:
            raise ValueError("Use one side push of up to 30 N for 0.02 to 0.20 seconds")
        if start + duration + 125 > schedule[-1][1]:
            raise ValueError("Allow at least 2.5 seconds after push withdrawal")
    if "restart_segment" in task:
        restart = task["restart_segment"]
        segments = task["sequence"]["segments"]
        if not any(seg["name"] == restart and seg["motion"] == "walk"
                   and seg["seconds"] >= 2.5 and i > 0 and segments[i-1]["motion"] == "stand"
                   for i, seg in enumerate(segments)):
            raise ValueError("restart_segment must name a walk of at least 2.5 seconds after stand")
    return schedule


def restart_metrics(task, schedule, rows):
    from math import hypot

    name = task.get("restart_segment")
    if name is None:
        return {}
    start, end, _, _ = next(segment for segment in schedule if segment[3] == name)
    band = [hypot(float(row["vx"]) - .2, float(row["vy"])) <= .08 and abs(float(row["yaw_rate"])) <= .15
            for row in rows[start:end]]
    seconds = next((i * .02 for i in range(len(band) - 24) if all(band[i:i+25])), None)
    return {"restart_seconds": seconds, "restart_pass": seconds is not None and seconds <= 2}


def simulate(task, schedule, out, kind):
    import numpy as np
    from evaluate_go2_robust import save_run
    from go2_contact_residual_learning import save_calls
    from go2_motion_metrics import sample_row
    from go2_parameter_probe import save_probe
    from go2_sequence import evaluate
    from validate_go2_context_states import local_force, pressure_metrics

    end_step = schedule[-1][1]
    duration = end_step * .02
    push = task.get("push")
    events = [dict(start=push["start_s"], duration=push["duration_s"], force=[0, push["body_y_n"], 0])] if push else []
    spec = read(RUNTIME)
    cfg, stack, factory, _ = load_stack(spec)
    env = None
    try:
        reference = read(ROOT / spec["cases"]["normal"]["reference_record"])
        plain, make = factory(reference)
        # The environment floors seconds / dt. Keep rounding from truncating one step early.
        env = make("external", episode_seconds=math.nextafter(duration, math.inf))
        env.reset(seed=task["sequence"]["seed"], options={"command": schedule[0][2].copy()})
        rows = []
        arrays = {key: [] for key in ("qpos", "qvel", "targets", "observations", "actions",
                                     "world_wrench", "feet_world", "heading")}
        for step in range(end_step):
            _, _, command, phase = next(segment for segment in schedule if segment[0] <= step < segment[1])
            env.command[:] = command
            rotation = env.data.xmat[env.base].reshape(3, 3)
            local = local_force(events, step, kind == "zero")
            wrench = np.r_[rotation @ local[:3], rotation @ local[3:]]
            env.data.xfrc_applied[:] = 0
            env.data.xfrc_applied[env.base] = wrench
            observation = env.observe()
            model, norm = stack.adapter
            action, _ = model.predict(norm.normalize_obs(observation.copy()), deterministic=True)
            _, _, terminated, truncated, info = env.step(action)
            values = (env.data.qpos, env.data.qvel, env.data.ctrl, observation, action, wrench,
                      env.data.geom_xpos[env.meter.feet], env.data.xmat[env.base].reshape(3, 3)[:, 0])
            for key, value in zip(arrays, values):
                arrays[key].append(value.copy())
            row = sample_row(env, env.meter, info, env.data.time, phase)
            row.update(vy=float(info["vy"]), terminated=bool(terminated),
                       command_vx=float(command[0]), command_vy=float(command[1]), command_yaw=float(command[2]))
            rows.append(row)
            if terminated or truncated:
                break
        packed = {key: np.asarray(values) for key, values in arrays.items()}
        complete = len(rows) == end_step and not rows[-1]["terminated"]
        result = dict(kind=kind, complete=complete, steps=len(rows), planned_steps=end_step,
                      terminated=rows[-1]["terminated"])
        if push:
            result.update(pressure_metrics(rows, {"events": events, "seconds": duration}))
            result["recovery_pass"] = result["passed"]
        segments, angles, checks = evaluate(schedule, rows, packed, env.validation_initial["qpos"], cfg["normal_budget"])
        result.update(segments=segments, turn_angles=angles, checks=checks,
                      quality_pass=complete and all(check["passed"] for check in checks))
        result.update(restart_metrics(task, schedule, rows))
        result["passed"] = result["quality_pass"] and result.get("recovery_pass", True) and result.get("restart_pass", True)
        result = save_run(out, kind, "task", task["sequence"]["seed"], result, packed, rows)
        folder = ROOT / result["folder"]
        save_probe(plain, folder)
        save_calls(env, folder)
        write_new(folder / "selector.json", env.recovery_selection_log)
        np.savez_compressed(folder / "initial-state.npz", **env.validation_initial)
        return result
    finally:
        if env is not None:
            env.data.xfrc_applied[:] = 0
            env.close()
        stack.close()


def run_task(task, out, kind):
    schedule = validate_task(task)
    if kind == "pressure" and "push" not in task:
        raise ValueError("pressure requires a task with push")
    out = out.resolve()
    runs = (ROOT / "artifacts/runs").resolve()
    if out == runs or not out.is_relative_to(runs):
        raise ValueError("Use a fresh output directory below artifacts/runs")
    out.mkdir(parents=True, exist_ok=False)
    try:
        write_new(out / "task.json", task)
        shutil.copy2(__file__, out / "task-runtime.py")
        frozen = read(ROOT / "docs/data/go2-contact-success-transfer/frozen-research-candidate.json")
        inputs = {**frozen["runtime_sha256"], frozen["model"]: frozen["model_sha256"]}
        for name in ["scripts/go2_task.py", "scripts/go2_sequence.py", "scripts/validate_go2_context_states.py",
                     "configs/go2-157-runtime.json", read(RUNTIME)["controller_config"]]:
            inputs.setdefault(name, sha(ROOT / name))
        for name, digest in inputs.items():
            if sha(ROOT / name) != digest:
                raise ValueError(f"Frozen input changed: {name}")
        write_new(out / "input-sha256.json", inputs)
        result = simulate(task, schedule, out, kind)
        write_new(out / "result.json", result)
        return result
    except Exception:
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def show(records):
    for row in records:
        checks = row.get("checks", row.get("absolute_checks", []))
        quality = f"{sum(check['passed'] for check in checks)}/{len(checks)}"
        passed = row.get("passed", row.get("absolute_pass"))
        metrics = " ".join(f"{key}={row[key]}" for key in ["peak_tilt", "recovery_seconds", "restart_seconds"] if key in row)
        print(f"{'PASS' if passed else 'FAIL'} seed={row['seed']} steps={row['steps']} "
              f"kind={row.get('kind', 'zero')} quality={quality} {metrics}\n  {row['folder']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List built-in tasks")
    archive = commands.add_parser("show", help="Read saved results without running physics")
    source = archive.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESETS)
    source.add_argument("--result", type=Path)
    export = commands.add_parser("export", help="Write an editable task JSON")
    export.add_argument("preset", choices=PRESETS)
    export.add_argument("--side", choices=["left", "right"], default="left")
    export.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render", help="Export a saved result as a video")
    render.add_argument("--result", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Execute one task")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=PRESETS)
    source.add_argument("--config", type=Path)
    run.add_argument("--side", choices=["left", "right"], default="left")
    run.add_argument("--kind", choices=["zero", "pressure"], required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "list":
            for name, (number, _, description) in PRESETS.items():
                print(f"{name:14} {number}  {description}")
        elif args.command == "show":
            result_path = args.result
            if args.preset:
                _, name, _ = PRESETS[args.preset]
                result_path = ROOT / "docs/data" / name / "summary.json"
            data = read(result_path)
            show(data["records"] if "records" in data else [data])
        elif args.command == "export":
            write_new(args.output, preset_config(args.preset, args.side))
            print(args.output)
        elif args.command == "render":
            from go2_task_video import render_result
            print(render_result(args.result, args.output))
        else:
            task = read(args.config) if args.config else preset_config(args.preset, args.side)
            result = run_task(task, args.output, args.kind)
            show([result])
            return 0 if result["passed"] else 1
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
