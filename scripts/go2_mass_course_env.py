"""Single mass/density randomization over continuous normal and side-push courses."""
import copy
import hashlib
import mujoco
import numpy as np
from go2_sequence_env import Go2SequenceEnv
from go2_motion_metrics import MotionMeter


def mass_assignment(config, worker, episode, external=False, options=None):
    """Independent four-episode balanced assignment; never consume task RNG."""
    spec = config['mass_randomization']
    if float(spec['nominal_probability']) != .5:
        raise ValueError('Balanced four-episode assignment requires nominal_probability=0.5')
    block_seed = [int(spec['seed']), int(worker), int(episode // 4)]
    sample_seed = [int(spec['seed']), int(worker), int(episode), 1]
    flags = np.array([False, False, True, True])
    np.random.default_rng(np.random.SeedSequence(block_seed)).shuffle(flags)
    planned = bool(flags[episode % 4])
    proposed = float(np.random.default_rng(np.random.SeedSequence(sample_seed)).uniform(*spec['range']))
    scale = proposed if planned and spec['enabled'] and not external else 1.
    explicit = bool(options is not None and 'mass_scale' in options)
    if explicit:
        scale = float(options['mass_scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Mass scale must be finite and positive')
    return scale, dict(mass_scale=scale, planned_randomized_flag=planned,
        planned_mass_scale=proposed if planned else 1., mass_block_seed=block_seed,
        mass_sample_seed=sample_seed, mass_randomization_enabled=bool(spec['enabled']),
        mass_assignment_external=bool(external), mass_assignment_explicit=explicit)


def course_events(config, worker, episode, slot, options=None):
    """Fixed independent force stream, common to nominal and randomized arms."""
    if options is not None and 'events' in options:
        return [dict(event, local_force=list(event['local_force'])) for event in options['events']]
    if slot is None or slot < config['sequence_course']['normal_slots']:
        return []
    spec = config['course_force']
    rng = np.random.default_rng(np.random.SeedSequence([int(spec['seed']), int(worker), int(episode)]))
    force = float(rng.uniform(*spec['force_N']))
    start = int(rng.integers(spec['start_steps'][0], spec['start_steps'][1] + 1))
    sign = int(rng.choice([-1, 1]))
    return [dict(start_step=start, duration_steps=int(spec['duration_steps']),
        local_force=[0., sign * force, 0.])]


_UNCHANGED = ('body_ipos', 'body_iquat', 'geom_friction', 'geom_priority',
    'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange',
    'actuator_forcelimited', 'actuator_ctrlrange', 'actuator_ctrllimited',
    'jnt_range', 'dof_damping', 'dof_armature', 'dof_frictionloss')


class Go2MassCourseEnv(Go2SequenceEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = copy.copy(self.model)
        self.data = mujoco.MjData(self.model)
        self.meter = MotionMeter(self.model)
        self.native_body_mass = self.model.body_mass.copy()
        self.native_body_inertia = self.model.body_inertia.copy()
        self.native_unchanged = {key: getattr(self.model, key).copy() for key in _UNCHANGED}
        self.native_gravity = self.model.opt.gravity.copy()
        self.native_timestep = float(self.model.opt.timestep)
        self.native_fingerprints = {key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in self.native_unchanged.items()}
        self.mass_scale = 1.
        self.previous_mass_scale = 1.

    def reset(self, *, seed=None, options=None):
        episode = self.episode + 1
        scale, mass_meta = mass_assignment(self.config, self.worker, episode,
            external=not self.use_sequence, options=options)
        # Restore exact native inputs before each assignment; never compound scales.
        self.model.body_mass[:] = self.native_body_mass * scale
        self.model.body_inertia[:] = self.native_body_inertia * scale
        constants_recomputed = scale != 1. or self.previous_mass_scale != 1.
        if constants_recomputed:
            mujoco.mj_setConst(self.model, self.data)
        self.previous_mass_scale = self.mass_scale = scale
        for key, value in self.native_unchanged.items():
            np.testing.assert_array_equal(getattr(self.model, key), value, err_msg=key)
        np.testing.assert_array_equal(self.model.opt.gravity, self.native_gravity)
        assert self.model.opt.timestep == self.native_timestep
        np.testing.assert_allclose(self.model.body_subtreemass[self.base],
            self.model.body_mass.sum(), atol=1e-12, rtol=0)
        # Parent flushes any prior episode before replacing its trace and metadata.
        _, info = super().reset(seed=seed, options=options)
        self.events = course_events(self.config, self.worker, episode,
            self.episode_meta['slot'], options)
        profile = 'explicit-preflight' if not self.use_sequence else (
            'continuous-normal' if len(self.schedule) > 1 else 'straight-push')
        self.episode_meta.update(mass_meta, profile=profile, events=self.events,
            course_force_seed=[int(self.config['course_force']['seed']), int(self.worker), int(episode)],
            native_body_mass=self.native_body_mass.tolist(),
            native_body_inertia=self.native_body_inertia.tolist(),
            actual_body_mass=self.model.body_mass.tolist(),
            actual_body_inertia=self.model.body_inertia.tolist(),
            native_total_mass_kg=float(self.native_body_mass.sum()),
            actual_total_mass_kg=float(self.model.body_mass.sum()),
            derived_constants_recomputed=bool(constants_recomputed),
            unchanged_physics_arrays_verified=True,
            unchanged_physics_fingerprints=self.native_fingerprints,
            initial_qpos=self.data.qpos.tolist(), initial_qvel=self.data.qvel.tolist())
        if self.trace is not None:
            self.trace['mass_scale'] = []
        return self.observe(), info

    def adjust_reward(self, reward, info):
        value = super().adjust_reward(reward, info)
        if self.trace is not None:
            self.trace['mass_scale'].append(self.mass_scale)
        info['mass_scale'] = self.mass_scale
        return value
