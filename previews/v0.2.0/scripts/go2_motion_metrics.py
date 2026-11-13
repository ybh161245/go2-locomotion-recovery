"""Foot/pose measurements, sampled at 50 Hz; no visual inference of contact."""
import mujoco
import numpy as np

LEGS = ('FL', 'FR', 'RL', 'RR')


class MotionMeter:
    def __init__(self, model):
        self.model = model
        self.feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in LEGS]
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.lookup = {g: i for i, g in enumerate(self.feet)}

    def sample(self, data):
        force = np.zeros(4)
        for c in range(data.ncon):
            contact = data.contact[c]
            a, b = int(contact.geom1), int(contact.geom2)
            other = b if a == self.floor else a if b == self.floor else -1
            if other in self.lookup:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, c, wrench)
                force[self.lookup[other]] += max(0., wrench[0])
        velocities = []
        for g in self.feet:
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_GEOM, g, spatial, 0)
            velocities.append(spatial[3:])
        velocity = np.array(velocities)
        height = data.geom_xpos[self.feet, 2] - self.model.geom_size[self.feet, 0]
        return dict(force=force, contact=force > 5., height=height,
                    speed_xy=np.linalg.norm(velocity[:, :2], axis=1), velocity=velocity)


def summarize(rows, dt=.02):
    """Rows must be one continuous interval, with no reset inside it."""
    result = dict(samples=len(rows), seconds=len(rows)*dt)
    if not rows:
        return result
    for key in ('vx', 'yaw_rate', 'height', 'roll_deg', 'pitch_deg', 'joint_speed_rms'):
        values = np.array([r[key] for r in rows])
        result[key+'_mean'] = float(values.mean())
        result[key+'_std'] = float(values.std())
    for leg in LEGS:
        contacts = np.array([r[leg+'_contact'] for r in rows], dtype=bool)
        heights = np.array([r[leg+'_clearance'] for r in rows])
        slips = np.array([r[leg+'_speed_xy'] for r in rows])
        longest = streak = 0
        for grounded in contacts:
            streak = 0 if grounded else streak+1
            longest = max(longest, streak)
        result[leg] = dict(contact_fraction=float(contacts.mean()),
                           longest_unloaded_seconds=longest*dt,
                           clearance_p95_m=float(np.percentile(heights,95)),
                           peak_clearance_m=float(heights.max()),
                           loaded_foot_speed_mean_m_s=float(slips[contacts].mean()) if contacts.any() else None)
    return result


def sample_row(env, meter, info, time, phase):
    from stand_go2 import roll_pitch
    feet = meter.sample(env.data)
    roll, pitch = roll_pitch(env.data.qpos[3:7])
    row = dict(time=float(time),phase=phase,vx=info['vx'],yaw_rate=info['yaw_rate'],
               height=float(env.data.qpos[2]),roll_deg=float(roll),pitch_deg=float(pitch),
               joint_speed_rms=float(np.sqrt(np.mean(env.data.qvel[6:]**2))))
    for i, leg in enumerate(LEGS):
        row.update({leg+'_contact':bool(feet['contact'][i]),leg+'_force':float(feet['force'][i]),
                    leg+'_clearance':float(feet['height'][i]),leg+'_speed_xy':float(feet['speed_xy'][i])})
    return row
