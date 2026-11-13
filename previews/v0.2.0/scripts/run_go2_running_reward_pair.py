"""225 matched training pair for the high-speed phase reward release."""
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from go2_157 import ROOT, read, sha, write_new
from go2_action_transfer import norm_load
from go2_quality_guard import equal_state
from go2_running_reward import RunningRewardEnv
import run_go2_speed_course_pilot_validated as pilot

RUN = ROOT / "artifacts/runs/20261107-running-reward-pair"
PUBLIC = ROOT / "docs/data/go2-running-reward-pair"
CONFIG = ROOT / "configs/go2-running-reward-pair.json"
PLAN = ROOT / "docs/experiments/2026-11-07-running-reward-pair.md"
PARENT = ROOT / "artifacts/runs/20261107-speed-course-pilot"


class Archive(pilot.Archive):
    def _on_training_start(self):
        self.started = time.perf_counter()

    def _on_step(self):
        if time.perf_counter() - self.started > 600:
            raise TimeoutError("225 fixed per-arm time limit")
        return True


def main():
    torch.set_num_threads(1)
    inputs = dict(read(ROOT / "docs/data/go2-speed-course-pilot/frozen-candidate.json")["sha256"])
    for path in (Path(__file__), PLAN, ROOT / "scripts/go2_running_reward.py",
                 ROOT / "docs/data/go2-running-reward-interface/summary.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, digest in inputs.items():
        assert sha(ROOT / name) == digest, name
    write_new(CONFIG, dict(experiment=225, inputs=inputs, seed=624101, per_arm_steps=65536,
                          arms=["original", "released"], max_evaluations=8))
    RUN.mkdir(exist_ok=False)
    PUBLIC.mkdir(exist_ok=False)
    for path in (Path(__file__), PLAN, CONFIG, ROOT / "scripts/go2_running_reward.py"):
        shutil.copy2(path, RUN / path.name)
    records = []
    first = None
    for released in (False, True):
        name = "released" if released else "original"
        out = RUN / name
        out.mkdir()
        pilot.RUN = out
        vec = DummyVecEnv([lambda flag=released: RunningRewardEnv(release_phase=flag) for _ in range(8)])
        norm = VecNormalize.load(PARENT / "checkpoint-65536-norm.pkl", vec)
        norm.training = True
        norm.norm_reward = True
        model = PPO.load(PARENT / "checkpoint-65536.zip", env=norm, device="cpu")
        model.set_random_seed(624101)
        model.learning_rate = 5e-5
        model.lr_schedule = lambda _: 5e-5
        for group in model.policy.optimizer.param_groups:
            group["lr"] = 5e-5
        state = dict(policy=model.policy.state_dict(), adam=model.policy.optimizer.state_dict(),
                     normalization=pickle.dumps((norm.obs_rms,norm.ret_rms),protocol=5),
                     python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
        if first is None:
            first = pickle.loads(pickle.dumps(state,protocol=5))
        else:
            assert equal_state(state["policy"],first["policy"]) and equal_state(state["adam"],first["adam"])
            assert state["normalization"] == first["normalization"] and state["python"] == first["python"]
            np.testing.assert_equal(state["numpy"],first["numpy"])
            assert torch.equal(state["torch"],first["torch"])
        with (out/"initial-state.pkl").open("xb") as stream:
            pickle.dump(state,stream,protocol=5)
        start = model.num_timesteps
        began = time.perf_counter()
        try:
            model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=Archive(start))
            assert model.num_timesteps-start == 65536
            model.save(out/"checkpoint-65536")
            norm.save(out/"checkpoint-65536-norm.pkl")
        finally:
            for worker,env in enumerate(vec.envs):
                if env.trace:
                    pilot.save_trace(out/f"remaining-worker-{worker}.npz",env.trace)
            norm.close()
        elapsed = time.perf_counter()-began
        result = pilot.evaluate(model,norm_load(out/"checkpoint-65536-norm.pkl"),"evaluation")
        record = dict(arm=name,released=released,training_interactions=model.num_timesteps-start,
                      elapsed_s=elapsed,evaluations=result)
        write_new(out/"summary.json",record)
        records.append(record)
        print("225 arm complete",name,elapsed,flush=True)
    old,new = [r["evaluations"] for r in records]
    qualification = (all(r["passed"] for r in old[:3]+new[:3]) and new[3]["passed"] and
                     (not old[3]["passed"] or old[3]["rmse"]-new[3]["rmse"]>=.02) and
                     all(not n["terminated"] or o["terminated"] for o,n in zip(old,new)))
    for name,digest in inputs.items():
        assert sha(ROOT/name)==digest,name
    write_new(PUBLIC/"summary.json",dict(experiment=225,records=records,qualified=qualification,
                                       training_interactions=131072,new_evaluations=8,paired_training_seeds=1))


if __name__ == "__main__":
    main()
