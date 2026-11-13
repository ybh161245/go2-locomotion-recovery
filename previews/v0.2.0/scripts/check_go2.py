"""Audit Go2 dimensions, motor signs, named-leg geometry and joint limits."""
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from go2_model import LEGS, load_model, joint_map, set_home


def audit(model):
    data = mujoco.MjData(model)
    set_home(model, data)
    mapping = joint_map(model)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError('Root must be free')
    if not np.allclose(model.opt.gravity, [0, 0, -9.81]):
        raise RuntimeError('Unexpected world gravity')
    if any(model.jnt_type[r['joint_id']] != mujoco.mjtJoint.mjJNT_HINGE for r in mapping):
        raise RuntimeError('Expected hinge actuators only')
    feet = {leg: data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, leg)].copy() for leg in LEGS}
    if not (min(feet[l][0] for l in ('FL','FR')) > max(feet[l][0] for l in ('RL','RR')) and
            min(feet[l][1] for l in ('FL','RL')) > max(feet[l][1] for l in ('FR','RR'))):
        raise RuntimeError('Named feet do not match expected forward/left axes')
    home = data.qpos.copy()
    for row in mapping:
        angle = home[row['qpos_adr']]
        if not row['range'][0] <= angle <= row['range'][1]:
            raise RuntimeError('Home joint angle outside limits')
        part = row['name'].split('_')[1]
        expected_axis = [1,0,0] if part == 'hip' else [0,1,0]
        if not np.allclose(row['axis'], expected_axis):
            raise RuntimeError('Joint axis differs from expected Go2 convention')
        set_home(model, data)
        data.qpos[2] = 1.0
        mujoco.mj_forward(model, data)
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, row['name'][:2])
        before = data.geom_xpos[foot_id].copy()
        data.qpos[row['qpos_adr']] += 1e-6
        mujoco.mj_forward(model, data)
        row['foot_position_derivative_per_rad'] = ((data.geom_xpos[foot_id] - before)/1e-6).tolist()
        # Subtract +1/-1 Nm accelerations to remove gravity bias and check the
        # motor transmission sign without integrating or using ground contact.
        data.ctrl[row['actuator_id']] = 1.0
        mujoco.mj_forward(model, data)
        positive = float(data.qacc[row['dof_adr']])
        data.ctrl[row['actuator_id']] = -1.0
        mujoco.mj_forward(model, data)
        negative = float(data.qacc[row['dof_adr']])
        row['positive_minus_negative_acceleration'] = positive - negative
        row['home_angle'] = float(angle)
        if positive <= negative:
            raise RuntimeError('Motor sign audit failed')
    return dict(passed=True, nq=model.nq, nv=model.nv, nu=model.nu,
                mass_kg=float(model.body_mass.sum()), physics_dt=float(model.opt.timestep),
                root_joint='free; no root actuator', quaternion_order='wxyz',
                nominal_body_axes='x forward, y left, z up; identity home root orientation',
                home_foot_positions={l: p.tolist() for l,p in feet.items()}, joints=mapping)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('artifacts/go2-audit.json'))
    args = p.parse_args()
    report = audit(load_model())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('joints','home_foot_positions')}, indent=2))


if __name__ == '__main__':
    main()
