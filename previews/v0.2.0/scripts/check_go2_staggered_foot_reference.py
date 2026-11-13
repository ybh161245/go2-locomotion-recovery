"""255 fixed synchronized/staggered foot references, evaluated in kinematics."""

import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from go2_157 import ROOT, read, sha, write_new
from go2_env import servo_model, ENV_CONFIG
from go2_model import LEGS, joint_map


OUT = ROOT / 'docs/data/go2-staggered-foot-reference'
PLAN = ROOT / 'docs/experiments/2026-11-11-staggered-foot-reference.md'
BASELINE = ROOT / 'artifacts/runs/20261108-dual-heading-handoff/mixed/states.npz'
ASSETS = ROOT / 'third_party/unitree_go2/SOURCE.json'
SPEED = .8
FREQUENCY = 1.75
DUTY = .6
LIFT = .065
STEPS = 200
DT = .02
MAX_ERROR_M = .002
JUMP_RATIO = 1.10
PLANS = {'synchronized': [0., .5, .5, 0.], 'staggered': [0., .5, .6, .1]}


def make_reference(phase, neutral, radii):
    stance = phase < DUTY
    progress = np.clip((phase - DUTY) / (1 - DUTY), 0, 1)
    stride = SPEED / FREQUENCY
    smooth = progress**3 * (10 + progress * (-15 + 6 * progress))
    desired = np.broadcast_to(neutral, (len(phase), *neutral.shape)).copy()
    desired[:, :, 0] += stride * (DUTY / 2 - phase + smooth)
    desired[:, :, 2] = radii + LIFT * np.sin(np.pi * progress)**2
    return desired, stance


def solve_leg(model, data, geom, q_indices, v_indices, nominal, low, high, goal):
    q = nominal.copy()
    jacobian = np.zeros((3, model.nv))
    for _ in range(50):
        data.qpos[q_indices] = q
        mujoco.mj_kinematics(model, data)
        error = goal - data.geom_xpos[geom]
        if np.linalg.norm(error) <= 1e-6:
            break
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacGeom(model, data, jacobian, None, geom)
        local = jacobian[:, v_indices]
        gradient = local.T @ error
        blocked = ((q <= low + 1e-12) & (gradient < 0)) | ((q >= high - 1e-12) & (gradient > 0))
        local[:, blocked] = 0
        delta = np.clip(local.T @ np.linalg.solve(local @ local.T + 1e-6 * np.eye(3), error), -.10, .10)
        for fraction in (1., .5, .25, .125):
            trial = np.clip(q + fraction * delta, low, high)
            data.qpos[q_indices] = trial
            mujoco.mj_kinematics(model, data)
            if np.linalg.norm(goal - data.geom_xpos[geom]) < np.linalg.norm(error) - 1e-12:
                q = trial
                break
        else:
            break
    return q


def solve_reference(model, data, mapping, geoms, desired, nominal, low, high):
    targets = np.empty((len(desired), 12))
    actual = np.empty_like(desired)
    for frame, goals in enumerate(desired):
        for leg, (geom, goal) in enumerate(zip(geoms, goals, strict=True)):
            joint_slice = slice(3 * leg, 3 * leg + 3)
            rows = mapping[joint_slice]
            q_indices = [row['qpos_adr'] for row in rows]
            v_indices = [row['dof_adr'] for row in rows]
            targets[frame, joint_slice] = solve_leg(
                model, data, geom, q_indices, v_indices, nominal[joint_slice],
                low[joint_slice], high[joint_slice], goal,
            )
        data.qpos[7:] = targets[frame]
        mujoco.mj_kinematics(model, data)
        actual[frame] = data.geom_xpos[geoms]
    return targets, actual


def draw(plans, neutral, radii):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), layout='constrained')
    for col, (name, values) in enumerate(plans.items()):
        desired, achieved, stance, phases = (values[k] for k in ('desired', 'achieved', 'stance', 'phase'))
        axes[0, col].set_title(name.capitalize())
        for leg, label in enumerate(LEGS):
            axes[0, col].scatter(phases[:, 0], leg + .8 * stance[:, leg], s=5, label=label)
            order = np.argsort(phases[:, leg])
            x = (desired[order, leg, 0] - neutral[leg, 0]) * 1000
            z = (desired[order, leg, 2] - radii[leg]) * 1000
            line, = axes[1, col].plot(x, z, alpha=.5, label=label)
            axes[1, col].scatter((achieved[order, leg, 0] - neutral[leg, 0]) * 1000,
                                 (achieved[order, leg, 2] - radii[leg]) * 1000,
                                 s=5, color=line.get_color())
        axes[0, col].set(yticks=np.arange(4) + .4, yticklabels=LEGS, xlabel='Reference cycle phase', ylabel='Planned support')
        axes[1, col].set(xlabel='Fore–aft target offset (mm)', ylabel='Foot sole height (mm)')
        axes[1, col].legend(loc='upper right', ncol=2)
    fig.suptitle('Kinematics only · lines: desired paths · dots: bounded joint solution')
    fig.savefig(OUT / 'reference-comparison.png', dpi=145)
    plt.close(fig)


def main():
    files = [Path(__file__), PLAN, BASELINE, ASSETS, ROOT / 'scripts/go2_model.py', ROOT / 'scripts/go2_env.py']
    inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    OUT.mkdir(exist_ok=False)
    write_new(OUT / 'contract.json', dict(experiment=255, inputs=inputs, speed_m_s=SPEED,
              frequency_hz=FREQUENCY, stance_fraction=DUTY, lift_m=LIFT, phase_offsets=PLANS,
              samples_per_plan=STEPS, dt_s=DT, base_height_source_steps=[400, 600],
              max_error_m=MAX_ERROR_M, jump_ratio_limit=JUMP_RATIO,
              max_ik_iterations=50, damping=1e-6, iteration_limit_rad=.1,
              new_physical=0, new_training=0, new_model_fits=0))
    shutil.copy2(__file__, OUT / Path(__file__).name)
    with np.load(BASELINE, allow_pickle=False) as old:
        height = float(np.median(old['qpos'][400:600, 2]))
        jumps = np.linalg.norm(np.diff(old['targets'], axis=0), axis=1)
    jump_limits = [float(jumps.max() * JUMP_RATIO), float(np.percentile(jumps, 95) * JUMP_RATIO)]
    nominal = np.asarray(ENV_CONFIG['nominal_angles'], dtype=float)
    phase = (np.arange(STEPS) * DT * FREQUENCY + .0025) % 1
    plans, rows, results = {}, [], {}
    with patch.object(mujoco, 'mj_step', side_effect=AssertionError('Kinematics only')):
        model = servo_model()
        mapping = joint_map(model)
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[0., 0., height, 1., 0., 0., 0., nominal]
        mujoco.mj_kinematics(model, data)
        geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS]
        neutral = data.geom_xpos[geoms].copy()
        radii = model.geom_size[geoms, 0].copy()
        low = np.maximum(nominal - .35, model.actuator_ctrlrange[:, 0])
        high = np.minimum(nominal + .35, model.actuator_ctrlrange[:, 1])
        for name, offsets in PLANS.items():
            local_phase = (phase[:, None] + offsets) % 1
            desired, stance = make_reference(local_phase, neutral, radii)
            targets, achieved = solve_reference(model, data, mapping, geoms, desired, nominal, low, high)
            error = np.linalg.norm(achieved - desired, axis=2)
            jump = np.linalg.norm(np.diff(np.vstack([targets, targets[0]]), axis=0), axis=1)
            support = stance.sum(axis=1)
            within = bool(np.all(targets >= low) and np.all(targets <= high))
            # Reapply the existing normalized action conversion and compare the geometry on new MjData.
            action = (targets - nominal) / .35
            assert np.max(abs(action)) <= 1 + 1e-14
            replay = mujoco.MjData(model)
            replay.qpos[:7] = data.qpos[:7]
            for target, position in zip(action, achieved, strict=True):
                replay.qpos[7:] = np.clip(nominal + .35 * np.clip(target, -1, 1), *model.actuator_ctrlrange.T)
                mujoco.mj_kinematics(model, replay)
                np.testing.assert_allclose(replay.geom_xpos[geoms], position, rtol=0, atol=1e-12)
            results[name] = dict(min_planned_support=int(support.min()), four_planned_support_frames=int((support == 4).sum()),
                                 max_error_mm=float(error.max() * 1000), p95_error_mm=float(np.percentile(error, 95) * 1000),
                                 jump_max_rad=float(jump.max()), jump_p95_rad=float(np.percentile(jump, 95)),
                                 within_original_joint_range=within,
                                 geometry_passed=bool(within and error.max() <= MAX_ERROR_M and jump.max() <= jump_limits[0]
                                                      and np.percentile(jump, 95) <= jump_limits[1]), legs={})
            for leg, label in enumerate(LEGS):
                joint_slice = slice(3 * leg, 3 * leg + 3)
                on_bound = np.any(np.isclose(targets[:, joint_slice], low[joint_slice], rtol=0, atol=1e-8) |
                                  np.isclose(targets[:, joint_slice], high[joint_slice], rtol=0, atol=1e-8), axis=1)
                results[name]['legs'][label] = dict(max_error_mm=float(error[:, leg].max() * 1000),
                    frames_over_2mm=int((error[:, leg] > MAX_ERROR_M).sum()), frames_at_joint_boundary=int(on_bound.sum()),
                    maximum_reconstructed_clearance_mm=float((achieved[:, leg, 2].max() - radii[leg]) * 1000))
                for i in range(STEPS):
                    rows.append(dict(plan=name, step=i, leg=label, phase=float(local_phase[i, leg]),
                                     planned_support=bool(stance[i, leg]), error_mm=float(error[i, leg] * 1000),
                                     at_joint_boundary=bool(on_bound[i])))
            plans[name] = dict(phase=local_phase, desired=desired, achieved=achieved, stance=stance,
                               targets=targets, action=action, error=error, target_jump=jump)
    reduced_overlap = (results['staggered']['four_planned_support_frames'] < results['synchronized']['four_planned_support_frames']
                       and results['staggered']['min_planned_support'] >= 2)
    np.savez_compressed(OUT / 'reference-arrays.npz', neutral=neutral, radii=radii, low=low, high=high,
                        **{f'{name}_{key}': array for name, values in plans.items() for key, array in values.items()})
    with (OUT / 'reference-errors.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draw(plans, neutral, radii)
    assert all(sha(ROOT / p) == expected for p, expected in inputs.items())
    summary = dict(experiment=255, records=results, base_height_m=height, stride_m=SPEED / FREQUENCY,
                   support_target_travel_m=SPEED / FREQUENCY * DUTY, rear_phase_lead_s=.1 / FREQUENCY,
                   jump_limits_rad=jump_limits, overlap_reduced=reduced_overlap,
                   preflight_passed=bool(reduced_overlap and all(r['geometry_passed'] for r in results.values())),
                   input_hashes_unchanged=True, new_physics=0, training_steps=0, model_fits=0)
    write_new(OUT / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
