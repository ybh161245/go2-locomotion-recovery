"""Summarize boundary hits for the saved 255 kinematic trajectories."""

import numpy as np

from go2_157 import ROOT, read, sha, write_new
from go2_model import LEGS


def main():
    directory = ROOT / 'docs/data/go2-staggered-foot-reference'
    source = directory / 'reference-arrays.npz'
    contract = read(directory / 'contract.json')
    assert all(sha(ROOT / name) == expected for name, expected in contract['inputs'].items())
    rows = {}
    with np.load(source, allow_pickle=False) as data:
        low = data['low'].reshape(4, 3)
        high = data['high'].reshape(4, 3)
        for name in contract['phase_offsets']:
            delta = data[f'{name}_achieved'] - data[f'{name}_desired']
            error = np.linalg.norm(delta, axis=2)
            np.testing.assert_array_equal(error, data[f'{name}_error'])
            targets = data[f'{name}_targets'].reshape(-1, 4, 3)
            at_bound = np.isclose(targets, low, atol=1e-8, rtol=0) | np.isclose(targets, high, atol=1e-8, rtol=0)
            stance = data[f'{name}_stance']
            exceeded = error > contract['max_error_m']
            frame, leg = np.unravel_index(error.argmax(), error.shape)
            rows[name] = dict(foot_frame_pairs_over_budget=int(exceeded.sum()),
                over_budget_with_joint_at_boundary=int((exceeded & at_bound.any(axis=2)).sum()),
                stance_max_error_mm=float(error[stance].max() * 1000), swing_max_error_mm=float(error[~stance].max() * 1000),
                worst_frame=int(frame), worst_leg=LEGS[leg], worst_local_phase=float(data[f'{name}_phase'][frame, leg]),
                worst_error_xyz_mm=(delta[frame, leg] * 1000).tolist(), worst_joint_targets_rad=targets[frame, leg].tolist(),
                worst_joint_upper_bounds_rad=high[leg].tolist())
    write_new(directory / 'error-boundary.json', dict(source_sha256=sha(source),
        script_sha256=sha(ROOT / 'scripts/summarize_go2_staggered_foot_limits.py'), results=rows))


if __name__ == '__main__':
    main()
