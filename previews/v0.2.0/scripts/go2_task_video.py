"""Export one saved Go2 task as an annotated video."""
import csv
import math
from pathlib import Path
import shutil
import traceback
from unittest.mock import patch

from go2_157 import ROOT, read, sha, write_new


def render_result(result_path, output):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from go2_env import Go2VelocityEnv

    result_path = result_path.resolve()
    result = read(result_path)
    if "records" in result or "folder" not in result:
        raise ValueError("Choose one run's result.json containing folder")
    folder = (ROOT / result["folder"]).resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Trajectory folder must be inside this workspace")
    output = output.resolve()
    if not output.is_relative_to(ROOT / "artifacts") or output == ROOT / "artifacts":
        raise ValueError("Use a new output directory below artifacts")
    if output.exists():
        raise FileExistsError(output)
    states_path, rows_path = folder / "states.npz", folder / "steps.csv"
    with rows_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Trajectory contains no saved rows")
    times = np.asarray([float(row["time"]) for row in rows])
    np.testing.assert_allclose(times, np.arange(1, len(rows)+1)*.02, atol=1e-8, rtol=0)
    for row in rows:
        for key in ("command_vx", "command_yaw", "vx", "yaw_rate", "roll_deg", "pitch_deg"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"Non-finite trajectory value: {key}")
    passed = result.get("passed", result.get("absolute_pass"))
    terminated = result.get("terminated", False)
    if type(passed) is not bool or type(terminated) is not bool:
        raise ValueError("Result must declare boolean passed/absolute_pass and terminated")
    if result.get("steps") != len(rows) or (rows[-1]["terminated"] == "True") != terminated:
        raise ValueError("Saved trajectory and result length/termination disagree")
    segments = []
    for index, row in enumerate(rows):
        command = (float(row["command_vx"]), float(row["command_yaw"]))
        if not segments or segments[-1]["command"] != command:
            vx, yaw = command
            label = "左转" if yaw > 0 else "右转" if yaw < 0 else "前进" if vx else "停止"
            segments.append(dict(start=index, command=command, label=label))
    for index, segment in enumerate(segments):
        segment["end"] = segments[index+1]["start"] if index+1 < len(segments) else len(rows)
    small = ImageFont.truetype("msyh.ttc", 20)
    title = ImageFont.truetype("msyhbd.ttc", 26)
    sources = {str(path.relative_to(ROOT)): sha(path) for path in (result_path, states_path, rows_path)}
    env = None
    with np.load(states_path, allow_pickle=False) as states:
        shapes = {"qpos": (len(rows), 19), "qvel": (len(rows), 18),
                  "targets": (len(rows), 12), "world_wrench": (len(rows), 6)}
        for key, shape in shapes.items():
            if states[key].shape != shape or not np.isfinite(states[key]).all():
                raise ValueError(f"Invalid saved {key} shape or values")
        output.mkdir(parents=True, exist_ok=False)
        try:
            shutil.copy2(__file__, output / "renderer.py")
            write_new(output / "source-sha256.json", sources)
            with patch.object(mujoco, "mj_step", side_effect=AssertionError("Video export advanced physics")):
                env = Go2VelocityEnv("external")
                frames = math.ceil(len(rows)/2)
                with imageio.get_writer(str(output / "video.mp4"), fps=25, codec="libx264",
                                        quality=7, macro_block_size=16, pixelformat="yuv420p") as writer:
                    for frame in range(frames):
                        index = min(frame*2+1, len(rows)-1)
                        row = rows[index]
                        env.data.qpos[:] = states["qpos"][index]
                        env.data.qvel[:] = states["qvel"][index]
                        env.data.ctrl[:] = states["targets"][index]
                        env.data.time = times[index]
                        mujoco.mj_forward(env.model, env.data)
                        picture = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(picture)
                        segment = next(part for part in segments if part["start"] <= index < part["end"])
                        outcome = "指标通过" if passed else "指标未通过"
                        draw.rectangle((0, 0, picture.width, 137), fill=(19, 32, 48))
                        draw.text((16, 8), f"Go2 任务回放　{segment['label']}　{outcome}", font=title, fill="white")
                        draw.text((768, 13), f"{times[index]:.2f} 秒", font=small, fill="#cbd9e7")
                        vx, yaw = segment["command"]
                        draw.text((16, 49), f"指令  {vx:+.2f} m/s　{yaw:+.2f} rad/s", font=small, fill="#78dabe")
                        draw.text((480, 49), f"实测  {float(row['vx']):+.2f} m/s　{float(row['yaw_rate']):+.2f} rad/s",
                                  font=small, fill="#f1c179")
                        force = np.linalg.norm(states["world_wrench"][index, :3])
                        tilt = max(abs(float(row["roll_deg"])), abs(float(row["pitch_deg"])))
                        draw.text((16, 85), f"实际推力 {force:.1f} N　当前倾角 {tilt:.2f}°", font=small, fill="#cbd9e7")
                        if terminated:
                            draw.text((480, 85), f"本记录于 {times[-1]:.2f} 秒提前终止", font=small, fill="#ff8989")
                        y = picture.height-54
                        draw.rectangle((0, y, picture.width, picture.height), fill=(19, 32, 48))
                        for part in segments:
                            left = picture.width*part["start"]/len(rows)
                            right = picture.width*part["end"]/len(rows)
                            draw.rectangle((left, y, max(left, right-1), y+5),
                                           fill="#72d7bc" if part is segment else "#596c7d")
                            if right-left > 60:
                                draw.text(((left+right)/2, y+27), part["label"], anchor="mm", font=small, fill="white")
                        if terminated and frame == frames-1:
                            draw.rectangle((0, y-50, picture.width, y), fill="#6c2730")
                            draw.text((16, y-42), "运行提前终止 · 记录结束", font=title, fill="white")
                        writer.append_data(np.asarray(picture))
                        if frame == frames-1:
                            picture.save(output / "poster.png")
            with imageio.get_reader(str(output / "video.mp4")) as video:
                decoded = sum(1 for _ in video)
            if decoded != frames:
                raise ValueError("Encoded video frame count disagrees with saved trajectory")
            for name, digest in sources.items():
                if sha(ROOT / name) != digest:
                    raise ValueError(f"Saved source changed during export: {name}")
            manifest = dict(source_result=str(result_path.relative_to(ROOT)), source_folder=str(folder.relative_to(ROOT)),
                            sources_sha256=sources, frames=frames, fps=25, saved_steps=len(rows),
                            saved_duration_s=float(times[-1]), encoded_duration_s=frames/25,
                            maximum_time_rounding_s=.02, passed=passed, terminated=terminated,
                            segments=segments, physics_steps=0, visual_review="pending",
                            video_sha256=sha(output / "video.mp4"), renderer_sha256=sha(Path(__file__)))
            write_new(output / "manifest.json", manifest)
            return output / "video.mp4"
        except Exception:
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            if env is not None:
                env.close()
