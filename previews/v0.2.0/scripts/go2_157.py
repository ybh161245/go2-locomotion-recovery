"""Portable frozen157 entry: verify, zero-step load, or one explicit reference replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))  # Also works with python -I (isolated imports).
RUNTIME = ROOT / 'configs/go2-157-runtime.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def audit_reads(forbidden):
    opened = set()
    blocked = []
    def hook(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if forbidden and p.is_relative_to(forbidden) and not p.is_relative_to(ROOT):
            blocked.append(str(p))
            raise PermissionError('Standalone run attempted original-workspace access: ' + str(p))
        opened.add(p)
    sys.addaudithook(hook)
    return opened, blocked


def load_stack(spec):
    import torch
    from stable_baselines3 import PPO
    from go2_bilateral_pilot import Stack
    from go2_action_transfer import norm_load
    from go2_random_feature_recovery import RandomFeatureActor
    from go2_contact_success_transfer import ContactSuccessActor
    from go2_contact_residual_learning import attach_contact_residual
    from run_go2_random_feature_recovery import make_factory
    import evaluate_go2_robust as ev
    cfg = read(ROOT / spec['controller_config'])
    ev.CONFIG = cfg
    torch.set_num_threads(1)
    stack = Stack(cfg)
    student = PPO.load(ROOT / cfg['observed_entry_burst']['student_model'], device='cpu')
    actor = RandomFeatureActor(ROOT / cfg['random_feature_recovery']['parent_model'],
                              ROOT / cfg['contact_success_transfer']['actor'])
    norm = norm_load(ROOT / cfg['random_feature_recovery']['parent_norm'])
    model = ContactSuccessActor(ROOT / spec['residual_model'])
    def factory(source):
        plain, base = make_factory(cfg, stack, student, actor, source)
        def make(*args, **kwargs):
            return attach_contact_residual(base(*args, **kwargs), norm,
                                           cfg['contact_success_transfer']['residual_scale'], model)
        return plain, make
    assert all(not n.training and not n.norm_reward for _, n in
               [stack.D, stack.formal, stack.default, stack.low, stack.adapter])
    assert not norm.training and not norm.norm_reward and len(stack.bank) == 20
    return cfg, stack, factory, ev


def verify_manifest():
    path = ROOT / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError('Run --check only in the exported package containing manifest.json')
    manifest = read(path)
    for name, digest in manifest['files_sha256'].items():
        p = (ROOT / name).resolve()
        assert p.is_relative_to(ROOT) and p.is_file(), name
        assert sha(p) == digest, name
    return len(manifest['files_sha256'])


def replay(spec, case, out, stack, factory, ev):
    import numpy as np
    from validate_go2_context_states import run_transient
    from go2_parameter_probe import save_probe
    from go2_traction_protocol import save_traction_probe
    from go2_contact_residual_learning import save_calls
    source = spec['cases'][case]
    reference = read(ROOT / source['reference_record'])
    plain, make = factory(reference)
    if case == 'normal':
        result = ev.normal_run(out, 'replay157', reference['seed'], make, stack.adapter)
    else:
        result = run_transient(out, 'replay157', reference['protocol'], reference['seed'], make,
                               stack.adapter, reference['kind'] == 'zero')
    folder = ROOT / result['folder']
    env = plain.instances[-1]
    if case == 'normal':
        save_probe(plain, folder)
        baseline = read(ROOT / spec['normal_budget_reference'])
        checks = ev.normal_checks(result, baseline)
        rl = result['segments']['reverse right']['RL']['clearance_p95_m'] * 1000
        result['passed'] = bool(all(c['passed'] for c in checks) and rl >= 29.5702276)
        write_new(folder / 'normal-quality.json', dict(checks=checks, RL_mm=rl, passed=result['passed']))
        assert checks == reference['quality']['checks'] and rl == reference['quality']['RL_mm']
    else:
        save_traction_probe(plain, folder, reference['protocol'])
    save_calls(env, folder)
    write_new(folder / 'selector.json', env.recovery_selection_log)
    np.savez_compressed(folder / 'initial-state.npz', **env.validation_initial)
    matches = {}
    with np.load(ROOT / source['reference_states']) as old, np.load(folder / 'states.npz') as new:
        assert set(old.files) == set(new.files) and len(old.files) == 8
        for key in old.files:
            matches[key] = bool(old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key]))
        for i, call in enumerate(env.contact_calls):
            target = np.clip(env.nominal + 0.35 * call['actual'], env.model.actuator_ctrlrange[:, 0],
                             env.model.actuator_ctrlrange[:, 1])
            np.testing.assert_array_equal(target, new['targets'][i])
    assert all(matches.values()), matches
    for key in ['steps', 'terminated', 'passed']:
        assert result[key] == reference[key], key
    if case != 'normal':
        for key in ['peak_tilt', 'recovery_seconds', 'complete', 'last_withdrawal']:
            assert result[key] == reference[key], key
    result.update(kind=reference['kind'], profile=reference.get('profile'), controller='frozen157-portable')
    return dict(case=case, result=result, reference_run=reference['folder'], array_equality=matches,
                all8_arrays_exact=True, new_physical_runs=1, new_training=0, new_model_fits=0,
                expected_failure_reproduced=case == 'failure' and result['terminated'] and not result['passed'],
                formal_selection_changed=False, qualification='Replay only; not new independent scope or training seed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--check', action='store_true')
    action.add_argument('--load-only', action='store_true')
    action.add_argument('--replay', action='store_true')
    p.add_argument('--case', choices=['normal', 'recovery', 'failure'], default='recovery')
    p.add_argument('--output', help='Fresh relative path below artifacts/runs; only for --replay.')
    p.add_argument('--report', help='Fresh relative JSON report path.')
    p.add_argument('--trace-inputs', help='Fresh relative dependency-read inventory; zero-step loading only.')
    p.add_argument('--forbid-root', help='Deny Python file opens into this original workspace outside the bundle.')
    args = p.parse_args()
    forbidden = Path(args.forbid_root).resolve() if args.forbid_root else None
    opened, blocked = audit_reads(forbidden)
    started = time.perf_counter()
    report = dict(operation='check' if args.check else 'load-only' if args.load_only else 'replay',
                  bundle_root=str(ROOT), executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix,
                  isolated_python=bool(sys.flags.isolated), new_physical_runs=0, new_training=0, new_model_fits=0)
    stack = None
    out = None
    try:
        if (ROOT / 'manifest.json').exists() or args.check:
            report['files_verified'] = verify_manifest()
        if not args.check:
            spec = read(RUNTIME)
            cfg, stack, factory, ev = load_stack(spec)
            if args.load_only:
                import mujoco
                original_step = mujoco.mj_step
                def no_integration(*a, **kw):
                    raise AssertionError('No physical integration allowed during --load-only')
                mujoco.mj_step = no_integration
                try:
                    for name, source in spec['cases'].items():
                        ref = read(ROOT / source['reference_record'])
                        plain, make = factory(ref)
                        seconds = 46 if name == 'normal' else ref['protocol']['seconds']
                        env = make('external', episode_seconds=seconds)
                        command = [0, 0, 0] if name == 'normal' else ref['protocol']['schedule'][0]['command']
                        env.reset(seed=ref['seed'], options={'command': command})
                        assert env.step_count == 0
                        env.close()
                finally:
                    mujoco.mj_step = original_step
                report.update(cases_loaded=list(spec['cases']), physics_control_steps=0,
                              all_normalization_frozen=True, bank_entries=len(stack.bank))
            else:
                if not args.output:
                    p.error('--replay requires a new --output')
                out = (ROOT / args.output).resolve()
                allowed = ROOT / 'artifacts/runs'
                assert out.is_relative_to(allowed) and out != allowed and not out.exists()
                out.mkdir(parents=True, exist_ok=False)
                write_new(out / 'start.json', dict(case=args.case, new_physical_run_attempt=1,
                          model_sha256=sha(ROOT / spec['residual_model']), runtime_sha256=sha(RUNTIME)))
                report.update(replay(spec, args.case, out, stack, factory, ev))
                write_new(out / 'comparison.json', report)
        report.update(forbidden_workspace_reads=blocked, elapsed_seconds=time.perf_counter()-started)
        local_files = sorted(p.relative_to(ROOT).as_posix() for p in opened
                             if p.is_file() and p.is_relative_to(ROOT) and '.venv' not in p.relative_to(ROOT).parts)
        loaded_modules = sorted({Path(m.__file__).resolve().relative_to(ROOT).as_posix()
                                 for m in list(sys.modules.values()) if getattr(m, '__file__', None)
                                 and Path(m.__file__).resolve().is_relative_to(ROOT)
                                 and '.venv' not in Path(m.__file__).resolve().relative_to(ROOT).parts})
        report['local_files_read'] = len(local_files)
        if args.trace_inputs:
            assert args.load_only, 'Input discovery is only allowed before physics'
            write_new(ROOT / args.trace_inputs, dict(local_files=local_files, loaded_modules=loaded_modules,
                      report=report, source_config=str(RUNTIME.relative_to(ROOT))))
        if args.report:
            target = (ROOT / args.report).resolve()
            assert target.is_relative_to(ROOT)
            write_new(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException:
        if out is not None:
            (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        if stack is not None:
            stack.close()


if __name__ == '__main__':
    main()
