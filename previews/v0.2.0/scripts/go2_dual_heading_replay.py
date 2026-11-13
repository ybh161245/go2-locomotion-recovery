"""Local entry for the fixed 249 straight speed-change experiment."""

import argparse
import json
import math
import pickle
import shutil
import sys
import traceback
from pathlib import Path

from go2_157 import ROOT, read, sha, write_new


CONFIG = ROOT / 'configs/go2-dual-heading-local-entry.json'


def verify_inputs(config):
    for name, expected in config['inputs'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Archived input changed: {name}')


def compare_arrays(path, actual):
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(actual):
            raise ValueError(f'Array fields differ: {path.name}')
        for name, array in actual.items():
            if array.dtype != saved[name].dtype:
                raise ValueError(f'Array dtype differs: {path.name}/{name}')
            np.testing.assert_array_equal(array, saved[name], err_msg=f'{path.name}/{name}')
        return len(saved.files)


def replay(config, output):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from go2_157 import load_stack
    from go2_action_transfer import norm_load
    from go2_dual_heading_handoff import attach as attach249
    from go2_dual_speed import attach as attach248
    from go2_speed_hold import candidate_inputs
    from go2_variable_speed_heading_hold import attach as attach212
    from run_go2_dual_speed_handoff import metrics, run
    from run_go2_full_heading_feedback import attach as attach197

    torch.set_num_threads(1)
    archived = ROOT / config['archived_run']
    model_path = ROOT / config['high_model']
    actor = PPO.load(model_path, device='cpu')
    norm = norm_load(ROOT / config['high_norm'])
    _, mechanism = candidate_inputs()
    spec = read(ROOT / config['runtime'])
    reference = read(ROOT / spec['cases']['normal']['reference_record'])
    schedule = config['schedule']
    seed = config['seed']
    _, stack, factory, _ = load_stack(spec)
    try:
        _, make = factory(reference)
        env = make('external', episode_seconds=math.nextafter(22., math.inf))
        try:
            attach249(attach248(attach212(attach197(env, stack), mechanism), actor, norm))
            env.reset(seed=seed, options={'command': schedule[0][2]})
            initial = {name: getattr(env.data, name).copy() for name in config['initial_fields']}
            physics = {name: getattr(env.model, name).copy() for name in config['physics_fields']}
            trace = output / 'simulation'
            states, telemetry = run(env, stack, schedule, seed, trace)
            np.savez_compressed(trace / 'initial-full.npz', **initial)
            np.savez_compressed(trace / 'physics.npz', **physics)
            with (trace / 'handoff-calls.pkl').open('xb') as stream:
                pickle.dump(env.handoff_calls, stream)

            score = metrics(states, telemetry)
            stable_steps = (np.linalg.norm(telemetry['velocity'][300:500, :2] - [.8, 0], axis=1) <= .08)
            stable_steps &= abs(telemetry['angular'][300:500, 2]) <= .15
            changes = np.diff(np.r_[False, stable_steps, False].astype(int))
            streaks = np.flatnonzero(changes < 0) - np.flatnonzero(changes > 0)
            write_new(output / 'score.json', dict(score, longest_acceleration_streak=int(max(streaks, default=0))))

            comparisons = {
                'states': compare_arrays(archived / 'states.npz', states),
                'telemetry': compare_arrays(archived / 'telemetry.npz', telemetry),
                'initial': compare_arrays(archived / 'initial-full.npz', initial),
                'physics': compare_arrays(archived / 'physics.npz', physics),
            }
            original_score = read(archived / 'result.json')
            if any(value != original_score[key] for key, value in score.items()):
                raise ValueError('Archived task score differs')
            with (archived / 'handoff-calls.pkl').open('rb') as stream:
                archived_calls = pickle.load(stream)
            if len(archived_calls) != len(env.handoff_calls):
                raise ValueError('Archived handoff call count differs')
            for index, (actual, expected) in enumerate(zip(env.handoff_calls, archived_calls)):
                if actual.keys() != expected.keys():
                    raise ValueError(f'Handoff fields differ at step {index}')
                for name in actual:
                    np.testing.assert_array_equal(actual[name], expected[name], err_msg=f'handoff/{index}/{name}')
            verify_inputs(config)
            result = dict(experiment=254, baseline=249, replay_exact=True, steps=len(states['qpos']),
                          compared_arrays=comparisons, handoff_calls=len(archived_calls),
                          task_passed=score['passed'], score='score.json', new_replay_runs=1, training_steps=0)
            write_new(output / 'result.json', result)
            print(json.dumps(result, indent=2))
        finally:
            env.close()
    finally:
        stack.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('show', help='Read the archived score and video path without simulation')
    run_parser = commands.add_parser('run', help='Replay the fixed 22-second simulation')
    run_parser.add_argument('--output', type=Path, required=True, help='New folder under artifacts/runs')
    args = parser.parse_args()
    config = read(CONFIG)
    if args.command == 'show':
        verify_inputs(config)
        summary = read(ROOT / config['archived_run'] / 'result.json')
        print(json.dumps(dict(baseline=249, scope='fixed flat-ground experimental replay',
                              archived_score=summary, video=config['video'], new_replay_runs=0), indent=2))
        return

    output = args.output.resolve()
    runs = (ROOT / 'artifacts/runs').resolve()
    if output == runs or not output.is_relative_to(runs) or output.exists():
        parser.error('Choose a new output directory below artifacts/runs')
    verify_inputs(config)
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(CONFIG, output / CONFIG.name)
        shutil.copy2(__file__, output / Path(__file__).name)
        replay(config, output)
    except Exception:
        (output / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    sys.exit(main())
