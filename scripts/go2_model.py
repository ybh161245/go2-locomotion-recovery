"""Unicode-safe Go2 model loading and joint-name-based PD control."""
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np
from fetch_go2 import DEST, verify_assets

LEGS = ('FL', 'FR', 'RL', 'RR')
JOINTS = tuple(f'{leg}_{part}_joint' for leg in LEGS for part in ('hip', 'thigh', 'calf'))


def load_model():
    verify_assets()
    # The model and its meshes are copied unchanged to an ASCII temp path for
    # MuJoCo's native Windows loader. The persistent original stays vendored.
    with tempfile.TemporaryDirectory(prefix='quadruped-go2-') as temp:
        target = Path(temp) / 'model'
        shutil.copytree(DEST, target)
        model = mujoco.MjModel.from_xml_path(str(target / 'scene.xml'))
    return model


def joint_map(model):
    if model.nu != 12 or model.nq != 19 or model.nv != 18:
        raise RuntimeError(f'Unexpected Go2 dimensions: {model.nq}, {model.nv}, {model.nu}')
    rows = []
    for name in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f'Missing joint: {name}')
        actuators = np.where(model.actuator_trnid[:, 0] == jid)[0]
        if len(actuators) != 1:
            raise RuntimeError(f'Joint must have exactly one actuator: {name}')
        aid = int(actuators[0])
        gear = float(model.actuator_gear[aid, 0])
        if not np.isclose(gear, 1.0):
            raise RuntimeError('PD assumes unit motor gear')
        rows.append(dict(name=name, joint_id=jid, actuator_id=aid,
                         qpos_adr=int(model.jnt_qposadr[jid]), dof_adr=int(model.jnt_dofadr[jid]),
                         axis=model.jnt_axis[jid].tolist(), range=model.jnt_range[jid].tolist(),
                         torque_range=model.actuator_ctrlrange[aid].tolist()))
    return rows


def set_home(model, data):
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    if key < 0:
        raise RuntimeError('Missing upstream home keyframe')
    mujoco.mj_resetDataKeyframe(model, data, key)
    # The upstream keyframe's ctrl holds nominal joint angles, while the model
    # uses torque motors. Always replace it with actual computed torques.
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)


def pd_torques(model, data, mapping, target, kp=40.0, kd=2.0):
    tau = np.zeros(model.nu)
    for row, desired in zip(mapping, target, strict=True):
        lo, hi = row['range']
        if not lo <= desired <= hi:
            raise ValueError(f'Target outside joint limit: {row["name"]}')
        value = kp * (desired - data.qpos[row['qpos_adr']]) - kd * data.qvel[row['dof_adr']]
        tau[row['actuator_id']] = np.clip(value, *row['torque_range'])
    return tau
