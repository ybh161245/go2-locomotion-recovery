"""228 finite adaptation to the frequency intervention tested in227."""
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from go2_157 import ROOT,read,sha,write_new
from go2_action_transfer import norm_load
from go2_running_reward import RunningRewardEnv
from run_go2_running_reward_pair import Archive
import run_go2_speed_course_pilot_validated as pilot

RUN=ROOT/"artifacts/runs/20261107-frequency-adaptation"
PUBLIC=ROOT/"docs/data/go2-frequency-adaptation"
CONFIG=ROOT/"configs/go2-frequency-adaptation.json"
PLAN=ROOT/"docs/experiments/2026-11-07-frequency-adaptation.md"


class FrequencyCourse(RunningRewardEnv):
    def __init__(self,mode="mixed",**kwargs):
        super().__init__(mode,release_phase=False,**kwargs)

    def frequency(self):
        return super().frequency()+1.25*max(0.,abs(float(self.filtered_command[0]))-.8)


def main():
    torch.set_num_threads(1)
    inputs=dict(read(ROOT/"docs/data/go2-running-reward-pair/frozen-study.json")["sha256"])
    for path in (Path(__file__),PLAN,ROOT/"docs/data/go2-speed-frequency-causal/summary.json"):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=228,inputs=inputs,training_steps=65536,seed=628101,max_evaluations=4))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for path in (Path(__file__),PLAN,CONFIG):shutil.copy2(path,RUN/path.name)
    pilot.RUN=RUN
    pilot.Course=FrequencyCourse
    parent=ROOT/"artifacts/runs/20261107-running-reward-pair/original"
    vec=DummyVecEnv([lambda:FrequencyCourse() for _ in range(8)])
    norm=VecNormalize.load(parent/"checkpoint-65536-norm.pkl",vec)
    norm.training=True;norm.norm_reward=True
    model=PPO.load(parent/"checkpoint-65536.zip",env=norm,device="cpu")
    model.set_random_seed(628101)
    model.learning_rate=5e-5;model.lr_schedule=lambda _:5e-5
    for group in model.policy.optimizer.param_groups:group["lr"]=5e-5
    model.save(RUN/"initial-model");norm.save(RUN/"initial-norm.pkl")
    start=model.num_timesteps
    try:
        model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=Archive(start))
        model.save(RUN/"checkpoint-65536");norm.save(RUN/"checkpoint-65536-norm.pkl")
    finally:
        for worker,env in enumerate(vec.envs):
            if env.trace:pilot.save_trace(RUN/f"remaining-{worker}.npz",env.trace)
        norm.close()
    assert model.num_timesteps-start==65536
    records=pilot.evaluate(model,norm_load(RUN/"checkpoint-65536-norm.pkl"),"adapted")
    prior=read(ROOT/"docs/data/go2-speed-frequency-causal/summary.json")["records"][1]["after"]
    qualified=all(r["passed"] for r in records) and records[-1]["rmse"]<prior["rmse"]
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(PUBLIC/"summary.json",dict(experiment=228,records=records,qualified=qualified,
        training_interactions=65536,evaluations=4,seed=628101,visual="pending"))


if __name__=="__main__":main()
