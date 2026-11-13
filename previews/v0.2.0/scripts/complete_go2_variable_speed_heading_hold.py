"""Complete212 after its result-file collision, reusing all three completed simulations."""

import pickle
import shutil
from unittest.mock import patch

import numpy as np

import go2_task
import run_go2_variable_speed_heading_hold as original
from go2_157 import ROOT, read, sha, write_new
from run_go2_reversal_push_task import equal_npz, measure, verify_inputs


OUT = ROOT/"artifacts/runs/20261106-variable-speed-heading-hold-completion"
SOURCE = ROOT/"scripts/complete_go2_variable_speed_heading_hold.py"


def main():
    config = read(original.CONFIG)
    verify_inputs(config)
    OUT.mkdir(exist_ok=False)
    assert (original.RUN/"failure.txt").exists()
    prior = {}
    for controller, kind in (("208", "zero"), ("208", "pressure"), ("212", "zero")):
        path = original.RUN/f"{controller}-{kind}"/"result.json"
        result = read(path)
        assert result["steps"] == 1100
        prior[controller, kind] = result
    write_new(OUT/"prospective-completion.json", dict(experiment=212,
        change="Save candidate metadata only in new summary; preserve task result.json",
        prior_physical=3, remaining_physical=1, original_config_sha256=sha(original.CONFIG),
        completion_runtime_sha256=sha(SOURCE),
        prior_results={f"{c}-{k}":sha(original.RUN/f"{c}-{k}"/"result.json") for c,k in prior}))
    shutil.copy2(SOURCE, OUT/SOURCE.name)
    native_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, evaluator = native_load(spec)

        def wrapped(reference):
            plain, make = factory(reference)

            def make212(*args, **kwargs):
                env = original.attach(original.attach197(make(*args, **kwargs), stack), config["mechanism"])
                instances.append(env)
                return env

            return plain, make212

        return cfg, stack, wrapped, evaluator

    with patch.object(go2_task, "load_stack", load):
        result = go2_task.run_task(config["task"], OUT/"212-pressure", "pressure")
    env = instances.pop()
    folder = ROOT/result["folder"]
    with (folder/"exit-hold-calls.pkl").open("xb") as stream:
        pickle.dump(env.exit_hold_calls, stream, protocol=5)
    prior["212", "pressure"] = result
    records = []
    for (controller, kind), result in prior.items():
        folder = ROOT/result["folder"]
        if controller == "212":
            reference = ROOT/prior["208", kind]["folder"]
            equal_npz(folder, reference, "states.npz", 600)
            for name in ("initial-state.npz", "physics.npz"):
                equal_npz(folder, reference, name)
            with (folder/"exit-hold-calls.pkl").open("rb") as stream:
                calls = pickle.load(stream)
            assert all(c["anchor"] == calls[500]["anchor"] for c in calls[500:900])
            assert all(c["anchor"] is None and c["extra"] == 0 for c in calls[:500]+calls[900:])
            with np.load(folder/"states.npz") as state:
                actions = np.stack([c["actual"] for c in calls]).astype(np.float64)
                np.testing.assert_array_equal(np.clip(env.action_targets(actions), *env.model.actuator_ctrlrange.T), state["targets"])
                result["heading_hold_segments"] = original.entry.heading_segments(calls, state)
            result["controller"] = 212
        if kind == "pressure":
            zero = ROOT/prior[controller, "zero"]["folder"]
            equal_npz(folder, zero, "states.npz", 753)
        values = original.metrics(result, config["task"])
        records.append(dict(controller=controller, kind=kind, result=result,
            motion=values, metrics=measure(result, config["task"]), passed=values["passed"]))
        print("212 completed", controller, kind, values, flush=True)
    pairs = []
    for kind in ("zero", "pressure"):
        old, new = [next(r for r in records if r["controller"] == c and r["kind"] == kind) for c in ("208", "212")]
        pairs.append(dict(kind=kind, old_pass=old["passed"], new_pass=new["passed"],
            old_heading_deg=old["motion"]["heading_end_deg"], new_heading_deg=new["motion"]["heading_end_deg"],
            new_termination=not old["result"]["terminated"] and new["result"]["terminated"],
            deltas={k: new["metrics"][k]-v for k,v in old["metrics"].items()}))
    gain = max(abs(p["old_heading_deg"]) for p in pairs)-max(abs(p["new_heading_deg"]) for p in pairs)
    protected = all(p["new_pass"] and not p["new_termination"] for p in pairs)
    improved = all(abs(p["new_heading_deg"]) < abs(p["old_heading_deg"]) for p in pairs)
    verify_inputs(config)
    summary = dict(experiment=212, records=records, pairs=pairs, worst_gain_deg=gain,
        qualified=protected and improved and gain >= config["min_worst_heading_gain_deg"],
        new_physical=4, original_run_physical=3, completion_physical=1, training=0, fits=0, visual="pending",
        retained_failure=(original.RUN/"failure.txt").relative_to(ROOT).as_posix())
    write_new(OUT/"summary.json", summary)
    original.PUBLIC.mkdir(exist_ok=False)
    write_new(original.PUBLIC/"summary.json", summary)


if __name__ == "__main__":
    main()
