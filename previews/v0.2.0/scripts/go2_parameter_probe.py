"""Single-factor physical parameter probes over the unchanged frozen stack."""
import json
from pathlib import Path
import mujoco
import numpy as np
import evaluate_go2_robust as ev
from go2_env import servo_model


# Explicit primary physical inputs: avoid comparing renderer/cached/derived state.
PRIMARY_FIELDS = (
    'body_mass', 'body_inertia', 'body_ipos', 'body_iquat',
    'geom_friction', 'geom_priority', 'geom_condim',
    'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange',
    'actuator_forcelimited', 'actuator_ctrlrange', 'actuator_ctrllimited',
    'jnt_range', 'dof_damping', 'dof_armature', 'dof_frictionloss',
)


def snapshot(model):
    result = {key: getattr(model, key).copy() for key in PRIMARY_FIELDS}
    result['gravity'] = model.opt.gravity.copy()
    result['timestep'] = np.asarray(model.opt.timestep)
    return result


def assert_snapshot(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def expected_inputs(before, spec, floor, feet):
    expected = {key: value.copy() for key, value in before.items()}
    factor, level = spec['factor'], float(spec['level'])
    if factor == 'nominal':
        assert level == 1.
    elif factor == 'mass':
        expected['body_mass'] *= level
        expected['body_inertia'] *= level
    elif factor == 'friction':
        expected['geom_friction'][[floor] + sorted(feet), 0] = level
    elif factor == 'gain':
        expected['actuator_gainprm'][:, 0] *= level
        expected['actuator_biasprm'][:, 1:3] *= level
    else:
        raise ValueError(f'Unknown probe factor: {factor}')
    return expected


def factory(joint, spec):
    """Each fresh instance owns its model; no policy/actions/observations changed."""
    original = ev.stack_factory(joint)
    instances = []
    cached = snapshot(servo_model())

    def make(*args, **kwargs):
        assert_snapshot(snapshot(servo_model()), cached)
        env = original(*args, **kwargs)
        # stack_factory already copies model and creates new MjData/MotionMeter.
        assert env.model is not servo_model()
        before = snapshot(env.model)
        assert_snapshot(before, cached)
        expected = expected_inputs(before, spec, env.floor, env.feet)
        factor, level = spec['factor'], float(spec['level'])
        if factor == 'mass':
            env.model.body_mass[:] *= level
            env.model.body_inertia[:] *= level
            mujoco.mj_setConst(env.model, env.data)
        elif factor == 'friction':
            env.model.geom_friction[[env.floor] + sorted(env.feet), 0] = level
        elif factor == 'gain':
            env.model.actuator_gainprm[:, 0] *= level
            env.model.actuator_biasprm[:, 1:3] *= level
        elif factor != 'nominal':
            raise ValueError(f'Unknown factor: {factor}')
        after = snapshot(env.model)
        assert_snapshot(after, expected)
        assert_snapshot(snapshot(servo_model()), cached)
        if factor == 'mass':
            np.testing.assert_allclose(env.model.body_subtreemass[env.base], env.model.body_mass.sum(), atol=1e-12, rtol=0)
        env.probe_spec = dict(spec)
        env.probe_physics_before, env.probe_physics_after = before, after
        env.probe_telemetry = []
        env.probe_cached_snapshot = cached
        native_step = env.step

        def step(action):
            transition = native_step(action)
            coefficients = [contact.friction[:2].copy()
                for contact in env.data.contact[:env.data.ncon]
                if (int(contact.geom1) == env.floor and int(contact.geom2) in env.feet)
                or (int(contact.geom2) == env.floor and int(contact.geom1) in env.feet)]
            values = np.asarray(coefficients)
            env.probe_telemetry.append(dict(
                contact_count=len(coefficients),
                effective_sliding_friction_min=float(values.min()) if coefficients else -1.,
                effective_sliding_friction_max=float(values.max()) if coefficients else -1.,
                actuator_force=env.data.actuator_force.copy(),
                targets=env.data.ctrl.copy(),
            ))
            return transition

        env.step = step
        instances.append(env)
        return env

    make.instances = instances
    make.spec = dict(spec)
    return make


def save_probe(make, folder):
    """Save raw physics and sampled telemetry, and verify the declared intervention."""
    folder = Path(folder)
    env = make.instances[-1]
    before, after, spec = env.probe_physics_before, env.probe_physics_after, env.probe_spec
    assert_snapshot(snapshot(env.model), after)
    assert_snapshot(after, expected_inputs(before, spec, env.floor, env.feet))
    assert_snapshot(snapshot(servo_model()), env.probe_cached_snapshot)
    records = env.probe_telemetry
    assert records
    physics = {f'before_{key}': value for key, value in before.items()}
    physics.update({f'after_{key}': value for key, value in after.items()})
    data = {key: np.asarray([record[key] for record in records]) for key in records[0]}
    assert all(np.isfinite(value).all() for value in data.values())
    limits = before['actuator_forcerange']
    assert (data['actuator_force'] >= limits[:, 0] - 1e-9).all()
    assert (data['actuator_force'] <= limits[:, 1] + 1e-9).all()
    target_limits = before['actuator_ctrlrange']
    assert (data['targets'] >= target_limits[:, 0] - 1e-12).all()
    assert (data['targets'] <= target_limits[:, 1] + 1e-12).all()
    with np.load(folder/'states.npz') as states:
        np.testing.assert_array_equal(data['targets'], states['targets'])
        assert len(data['targets']) == len(states['qpos'])
    present = data['contact_count'] > 0
    assert present.any()
    expected_friction = float(spec['level']) if spec['factor'] == 'friction' else .8
    np.testing.assert_allclose(data['effective_sliding_friction_min'][present], expected_friction, atol=1e-12, rtol=0)
    np.testing.assert_allclose(data['effective_sliding_friction_max'][present], expected_friction, atol=1e-12, rtol=0)
    peak_force_ratio = np.abs(data['actuator_force']) / np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
    audit = dict(spec=spec, steps=len(records), single_factor_arrays_verified=True,
        native_motor_and_target_limits_preserved=True, all_targets_within_native_limits=True,
        all_sampled_motor_forces_within_native_limits=True, cached_model_unmodified=True,
        effective_contact_sliding_friction_verified=True, expected_effective_friction=expected_friction,
        contact_control_steps=int(present.sum()), total_mass_kg=float(after['body_mass'].sum()),
        kp=after['actuator_gainprm'][:, 0].tolist(), kd=(-after['actuator_biasprm'][:, 2]).tolist(),
        max_sampled_motor_limit_ratio=float(peak_force_ratio.max()),
        caveat='Motor forces sampled at control50Hz; not a500Hz torque saturation audit.')
    for name in ['physics.npz', 'telemetry.npz', 'probe-audit.json']:
        assert not (folder/name).exists(), folder/name
    np.savez_compressed(folder/'physics.npz', **physics)
    np.savez_compressed(folder/'telemetry.npz', **data)
    (folder/'probe-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    return audit
