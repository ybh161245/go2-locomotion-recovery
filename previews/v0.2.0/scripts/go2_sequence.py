"""Run a JSON sequence of bounded flat-ground commands on frozen157 at 50 Hz."""
import argparse
import csv
import math
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from go2_157 import RUNTIME, load_stack, read, write_new, sha
from go2_motion_metrics import LEGS, sample_row, summarize

COMMANDS = {
    "stand": (0, 0, 0), "walk": (.2, 0, 0), "trot": (.4, 0, 0),
    "left": (.2, 0, .5), "right": (.2, 0, -.5),
}
RATE = 50


def validate(data):
    if not isinstance(data, dict) or set(data) != {"seed", "segments"}:
        raise ValueError("Expected exactly seed and segments.")
    if type(data["seed"]) is not int or not (0 <= data["seed"] < 2**32):
        raise ValueError("seed must be an unsigned 32-bit integer.")
    if not isinstance(data["segments"], list) or not data["segments"]:
        raise ValueError("Provide at least one segment.")
    start = 0
    schedule = []
    used = set()
    for segment in data["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"name", "motion", "seconds"}:
            raise ValueError("Each segment requires exactly name, motion and seconds.")
        label, motion, seconds = (segment[key] for key in ("name", "motion", "seconds"))
        if not isinstance(label, str) or not label.strip() or label in used:
            raise ValueError("Use nonempty unique names for segments.")
        if not isinstance(motion, str) or motion not in COMMANDS:
            raise ValueError(f"motion must be one of {list(COMMANDS)}.")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 2 <= seconds <= 46:
            raise ValueError("Each segment must last 2 to 46 seconds.")
        ticks = round(seconds * RATE)
        if abs(ticks - seconds * RATE) > 1e-8:
            raise ValueError("Segment durations must be multiples of 0.02 seconds.")
        used.add(label)
        schedule.append((start, start + ticks, np.asarray(COMMANDS[motion], float), label))
        start += ticks
    if start > 46 * RATE:
        raise ValueError("Maximum total duration is 46 seconds.")
    return schedule


def stride_median(indices, rows, states):
    lengths = []
    for leg_index, leg in enumerate(LEGS):
        armed, last_contact, streak, last = False, True, 0, None
        for i in indices:
            contact = bool(rows[i][leg + "_contact"])
            streak = streak + 1 if contact == last_contact else 1
            if not contact and streak >= 2:
                armed = True
            if contact and streak == 2 and armed:
                if last is not None and .2 < (i - last) * .02 < 1.5:
                    direction = states["heading"][i] + states["heading"][last]
                    direction[2] = 0
                    direction /= np.linalg.norm(direction)
                    lengths.append(float(np.dot(states["feet_world"][i, leg_index] -
                                                states["feet_world"][last, leg_index], direction)))
                last, armed = i, False
            last_contact = contact
    return float(np.median(lengths)) if lengths else None


def evaluate(schedule, rows, states, initial_qpos, budget):
    segments, angles, checks = {}, [], []
    def check(name, metric, value, limit, minimum=False):
        passed = value is not None and (value >= limit if minimum else value <= limit)
        checks.append(dict(segment=name, metric=metric, value=value, limit=limit, passed=bool(passed)))
    def yaw(q):
        w, x, y, z = q.T
        return np.unwrap(np.arctan2(2*(w*z + x*y), 1-2*(y*y + z*z)))
    headings = yaw(np.vstack((initial_qpos[3:7], states["qpos"][:, 3:7])))
    for start, end, command, name in schedule:
        # Original protocol uses native sampled times and excludes the first second for tracking.
        ids = [i for i, row in enumerate(rows) if start / RATE + 1 < row["time"] <= end / RATE + 1e-8]
        group = [rows[i] for i in ids]
        stats = summarize(group)
        stats["command"] = command.tolist()
        segments[name] = stats
        if not group:
            check(name, "missing_segment", 1, 0)
            continue
        stats["peak_tilt"] = max(max(abs(r["roll_deg"]), abs(r["pitch_deg"])) for r in rows
                                 if start / RATE < r["time"] <= end / RATE + 1e-8)
        stats["velocity_rmse"] = float(np.sqrt(np.mean([(r["vx"] - command[0])**2 + r["vy"]**2 for r in group])))
        stats["yaw_rmse"] = float(np.sqrt(np.mean([(r["yaw_rate"] - command[2])**2 for r in group])))
        stats["stride_median_m"] = stride_median(ids, rows, states)
        check(name, "peak_tilt", stats["peak_tilt"], 8)
        for metric, cap in [("velocity_rmse", "per_segment_velocity_rmse_max"), ("yaw_rmse", "per_segment_yaw_rmse_max")]:
            check(name, metric, stats[metric], budget[cap])
        if np.any(command):
            for leg in LEGS:
                check(name, leg + "_clearance", stats[leg]["clearance_p95_m"], budget["foot_clearance_p95_min_m"], minimum=True)
        else:
            for metric, cap in [("joint_speed_rms_mean", "stop_joint_speed_rms_max"),
                                ("roll_deg_std", "stop_roll_pitch_std_deg_max"), ("pitch_deg_std", "stop_roll_pitch_std_deg_max")]:
                check(name, metric, stats[metric], budget[cap])
        if command[2] and end < len(headings):
            angle = float(np.degrees(headings[end] - headings[start]))
            angles.append(angle)
            stats["angle_deg"] = angle
            if end - start == 6 * RATE:
                check(name, "angle_error_deg", abs(angle - np.degrees(6 * command[2])), budget["turn_angle_abs_error_max_deg"])
    return segments, angles, checks


def run(sequence_path, out):
    data = read(sequence_path)
    schedule = validate(data)
    out = out.resolve()
    if not out.is_relative_to((ROOT / "artifacts/runs").resolve()) or out == (ROOT / "artifacts/runs").resolve():
        raise ValueError("Use a fresh output directory below artifacts/runs.")
    out.mkdir(parents=True, exist_ok=False)
    folder = out / "trajectory"
    folder.mkdir()
    write_new(out / "sequence.json", data)
    shutil.copy2(__file__, out / "runtime.py")
    write_new(out / "input-sha256.json", {
        str(path.resolve().relative_to(ROOT)): sha(path)
        for path in [Path(__file__), RUNTIME, ROOT / read(RUNTIME)["controller_config"]]
    })
    stack = env = None
    try:
        runtime = read(RUNTIME)
        cfg, stack, factory, _ = load_stack(runtime)
        reference = read(ROOT / runtime["cases"]["normal"]["reference_record"])
        plain, make = factory(reference)
        end_tick = schedule[-1][1]
        env = make("external", episode_seconds=end_tick / RATE)
        env.reset(seed=data["seed"], options={"command": schedule[0][2].copy()})
        initial_qpos = env.data.qpos.copy()
        np.savez_compressed(folder / "initial-state.npz", **env.validation_initial)
        rows = []
        states = {k: [] for k in ["qpos", "qvel", "targets", "observations", "actions", "world_wrench", "feet_world", "heading"]}
        for start, end, cmd, name in schedule:
            for _tick in range(start, end):
                env.command[:] = cmd
                obs = env.observe()
                action, _ = stack.adapter[0].predict(stack.adapter[1].normalize_obs(obs.copy()), deterministic=True)
                wrench = env.data.xfrc_applied[env.base].copy()
                assert not env.data.xfrc_applied.any()
                _, _, term, trunc, info = env.step(action)
                for k,v in [("qpos",env.data.qpos),("qvel",env.data.qvel),("targets",env.data.ctrl),("observations",obs),
                            ("actions",action),("world_wrench",wrench),("feet_world",env.data.geom_xpos[env.meter.feet]),
                            ("heading",env.data.xmat[env.base].reshape(3,3)[:,0])]:
                    states[k].append(v.copy())
                row = sample_row(env, env.meter, info, env.data.time, name)
                row.update(vy=info["vy"], terminated=bool(term), command_vx=cmd[0], command_vy=cmd[1], command_yaw=cmd[2])
                rows.append(row)
                if term or trunc:
                    break
            if term or trunc:
                break
        packed = {k: np.asarray(v) for k,v in states.items()}
        np.savez_compressed(folder / "states.npz", **packed)
        with (folder / "steps.csv").open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        from go2_parameter_probe import save_probe
        from go2_contact_residual_learning import save_calls
        save_probe(plain, folder)
        save_calls(env, folder)
        write_new(folder / "selector.json", env.recovery_selection_log)
        segments, angles, checks = evaluate(schedule, rows, packed, initial_qpos, cfg["normal_budget"])
        complete = len(rows) == end_tick and not term
        result = dict(controller="frozen157", seed=data["seed"], steps=len(rows), planned_steps=end_tick,
                      complete=bool(complete), terminated=bool(term), segments=segments, turn_angles=angles,
                      absolute_checks=checks, absolute_pass=bool(complete and all(c["passed"] for c in checks)),
                      folder=folder.relative_to(ROOT).as_posix(), force_events=0)
        write_new(out / "result.json", result)
        return result
    except BaseException:
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if env is not None:
            env.close()
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True, help="JSON with seed and named motion segments.")
    parser.add_argument("--output", type=Path, required=True, help="New directory under artifacts/runs.")
    args = parser.parse_args()
    result = run(args.sequence, args.output)
    print(f'{result["steps"]}/{result["planned_steps"]} control steps, absolute quality pass={result["absolute_pass"]}, {result["folder"]}')
