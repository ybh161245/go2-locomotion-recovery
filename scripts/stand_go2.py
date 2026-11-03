"""Record free-base Go2 standing/crouching with joint PD, not an RL policy."""
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from check_go2 import audit
from fetch_go2 import DEST, verify_assets
from go2_model import LEGS, load_model, joint_map, set_home, pd_torques

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(duration=14.0, kp=60.0, kd=3.0, fps=25, width=960, height=640,
              controller='joint torque PD; no learned policy', physics_dt=0.002,
              test_seeds=[0,101,102,103,104], initial_joint_noise_rad=0.01,
              final_hold_start=10.0,
              thresholds=dict(min_height=0.16, max_height=0.38, max_tilt_deg=10.0,
                              final_min_height=0.24, final_max_height=0.35,
                              final_max_tilt_deg=5.0, min_three_feet_contact_fraction=0.95,
                              max_total_horizontal_drift_m=0.15))


def target_at(t):
    if t < 2:
        angle, phase = .9, 'settle at home'
    elif t < 4:
        u = (t-2)/2
        angle, phase = .9+.2*(3*u*u-2*u*u*u), 'slow crouch'
    elif t < 6:
        angle, phase = 1.1, 'hold crouch'
    elif t < 8:
        u = (t-6)/2
        angle, phase = 1.1-.3*(3*u*u-2*u*u*u), 'rise to stand'
    else:
        angle, phase = .8, 'hold stand'
    return np.tile([0.0, angle, -2*angle], 4), phase


def roll_pitch(quat):
    w,x,y,z = quat
    return np.degrees([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                       np.arcsin(np.clip(2*(w*y-z*x),-1,1))])


def contact_status(model, data):
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg): leg for leg in LEGS}
    touched, nonfoot = set(), False
    for c in data.contact[:data.ncon]:
        a,b = int(c.geom1), int(c.geom2)
        if floor not in (a,b):
            continue
        other = b if a == floor else a
        if other in feet:
            touched.add(feet[other])
        else:
            nonfoot = True
    return len(touched), nonfoot


def init_state(model, seed, mapping):
    data = mujoco.MjData(model)
    set_home(model, data)
    if seed:
        rng = np.random.default_rng(seed)
        for row in mapping:
            data.qpos[row['qpos_adr']] += rng.uniform(-CONFIG['initial_joint_noise_rad'], CONFIG['initial_joint_noise_rad'])
        mujoco.mj_forward(model, data)
    return data


def run(model, seed, writer=None, renderer=None, preview=None):
    mapping = joint_map(model)
    data = init_state(model, seed, mapping)
    steps = int(round(CONFIG['duration'] / model.opt.timestep))
    stride = int(round(1 / (model.opt.timestep * CONFIG['fps'])))
    if not np.isclose(stride*model.opt.timestep, 1/CONFIG['fps']):
        raise RuntimeError('Video FPS incompatible with physics timestep')
    rows, states, velocities, controls, targets = [], [], [], [], []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base')
    camera.distance, camera.azimuth, camera.elevation = 1.25, 135, -20
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    limits = np.max(np.abs(model.actuator_ctrlrange), axis=1)
    for step in range(steps):
        target, phase = target_at(step*model.opt.timestep)
        tau = pd_torques(model, data, mapping, target, kp=CONFIG['kp'], kd=CONFIG['kd'])
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            raise RuntimeError('Non-finite Go2 state')
        if np.any(data.xfrc_applied) or np.any(data.qfrc_applied):
            raise RuntimeError('Unexpected external support force')
        roll, pitch = roll_pitch(data.qpos[3:7])
        contacts, nonfoot = contact_status(model, data)
        rows.append(dict(step=step, time=float(data.time), phase=phase,
                         x=float(data.qpos[0]), y=float(data.qpos[1]), height=float(data.qpos[2]),
                         roll_deg=float(roll), pitch_deg=float(pitch),
                         foot_contacts=contacts, nonfoot_ground_contact=nonfoot,
                         max_abs_torque=float(np.max(np.abs(tau))),
                         max_torque_fraction=float(np.max(np.abs(tau)/limits)),
                         target_thigh=float(target[1])))
        states.append(data.qpos.copy()); velocities.append(data.qvel.copy())
        controls.append(tau.copy()); targets.append(target.copy())
        if writer is not None and (step+1) % stride == 0:
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = renderer.render()
            if frame.std() < 2:
                raise RuntimeError('Blank Go2 rendering')
            image = Image.fromarray(frame)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0,0,960,106), fill=(12,22,38))
            draw.text((18,10), 'STAGE 3 | UNITREE GO2 | 12 JOINTS | FREE BASE', font=font, fill='white')
            draw.text((18,37), f'Joint PD controller (NOT RL) | {phase} | t={data.time:.2f}s', font=font, fill='#59dfcb')
            draw.text((18,64), f'height={data.qpos[2]:.3f}m | roll={roll:+.2f}deg | pitch={pitch:+.2f}deg | feet={contacts}/4', font=font, fill='white')
            writer.append_data(np.asarray(image))
            if preview is not None and step+1 == int(10/model.opt.timestep):
                image.save(preview)
    return rows, dict(qpos=np.array(states), qvel=np.array(velocities), torques=np.array(controls), targets=np.array(targets))


def metrics(rows, seed):
    threshold = CONFIG['thresholds']
    active = [r for r in rows if r['time'] >= 2]
    hold = [r for r in rows if r['time'] >= CONFIG['final_hold_start']]
    result = dict(seed=seed, min_height=min(r['height'] for r in active), max_height=max(r['height'] for r in active),
                  max_tilt_deg=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in active),
                  nonfoot_ground_contacts=sum(r['nonfoot_ground_contact'] for r in rows),
                  final_mean_height=float(np.mean([r['height'] for r in hold])),
                  final_max_tilt_deg=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in hold),
                  final_three_feet_contact_fraction=float(np.mean([r['foot_contacts'] >= 3 for r in hold])),
                  horizontal_drift=float(np.linalg.norm([rows[-1]['x'],rows[-1]['y']])),
                  max_torque_fraction=max(r['max_torque_fraction'] for r in rows))
    result['passed'] = bool(result['min_height'] >= threshold['min_height'] and result['max_height'] <= threshold['max_height']
                            and result['max_tilt_deg'] <= threshold['max_tilt_deg'] and result['nonfoot_ground_contacts'] == 0
                            and threshold['final_min_height'] <= result['final_mean_height'] <= threshold['final_max_height']
                            and result['final_max_tilt_deg'] <= threshold['final_max_tilt_deg']
                            and result['final_three_feet_contact_fraction'] >= threshold['min_three_feet_contact_fraction']
                            and result['horizontal_drift'] <= threshold['max_total_horizontal_drift_m']
                            and result['max_torque_fraction'] <= 1.000001)
    return result


def main():
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / 'artifacts/runs' / f'{stamp}-go2-standing'
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps(CONFIG, indent=2), encoding='utf-8')
    model = load_model()
    if not np.isclose(model.opt.timestep, CONFIG['physics_dt']):
        raise RuntimeError('Physics timestep changed')
    report = audit(model)
    (out / 'model-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    model.vis.global_.offwidth = CONFIG['width']; model.vis.global_.offheight = CONFIG['height']
    renderer = mujoco.Renderer(model, height=CONFIG['height'], width=CONFIG['width'])
    try:
        with imageio.get_writer(str(out / 'go2-standing.mp4'), fps=CONFIG['fps'], codec='libx264',
                                quality=8, macro_block_size=16, pixelformat='yuv420p') as writer:
            rows, trajectory = run(model, 0, writer, renderer, out / 'preview.png')
    finally:
        renderer.close()
    with (out / 'steps.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    np.savez_compressed(out / 'trajectory.npz', **trajectory)
    results = [metrics(rows, 0)]
    for seed in CONFIG['test_seeds'][1:]:
        trial_rows, trial = run(model, seed)
        results.append(metrics(trial_rows, seed))
        np.savez_compressed(out / f'trajectory-seed-{seed}.npz', **trial)
    with (out / 'initial-state-tests.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with imageio.get_reader(str(out / 'go2-standing.mp4')) as reader:
        frames = sum(1 for _ in reader)
    if frames != int(CONFIG['duration']*CONFIG['fps']):
        raise RuntimeError('Go2 video frame count mismatch')
    (out / 'source').mkdir()
    for name in ('stand_go2.py','go2_model.py','fetch_go2.py','check_go2.py'):
        shutil.copy2(Path(__file__).with_name(name), out / 'source' / name)
    shutil.copy2(DEST / 'SOURCE.json', out / 'model-SOURCE.json')
    shutil.copy2(DEST / 'LICENSE', out / 'model-LICENSE')
    (out / 'requirements-lock.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    summary = dict(stage=3, controller='joint PD, not RL', steps=len(rows), video_frames=frames,
                   seconds=CONFIG['duration'], mass_kg=report['mass_kg'], all_initial_state_tests_passed=all(r['passed'] for r in results),
                   initial_state_test_count=len(results), nominal=results[0], tests=results,
                   note='Small initial-joint perturbations only; not an external-push robustness test.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    (ROOT / 'artifacts/latest-go2-standing.txt').write_text(out.relative_to(ROOT).as_posix(),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'RUN {out}',flush=True)
    if not summary['all_initial_state_tests_passed']:
        raise RuntimeError('Go2 standing acceptance thresholds not satisfied')


if __name__ == '__main__':
    main()
