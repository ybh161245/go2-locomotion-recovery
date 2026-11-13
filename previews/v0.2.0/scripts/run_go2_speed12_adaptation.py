"""244 one bounded adaptation to the existing scale map and1.2 speed curriculum."""
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from go2_157 import ROOT,read,sha,write_new
from go2_action_transfer import norm_load
from go2_quality_guard import equal_state
from run_go2_high_speed_action_scale import Course
from run_go2_running_reward_pair import Archive
from run_go2_straight_path_control import path_metrics
import run_go2_speed_course_pilot_validated as pilot
import run_go2_acceleration_bias_reset as control

RUN=ROOT/"artifacts/runs/20261107-speed12-adaptation"
PUBLIC=ROOT/"docs/data/go2-speed12-adaptation"
CONFIG=ROOT/"configs/go2-speed12-adaptation.json"
PLAN=ROOT/"docs/experiments/2026-11-07-speed12-adaptation.md"


class SpeedCourse(Course):
    def _sample_command(self):
        if self.training_command_mode=="external":return
        self.command[:]=[self.np_random.choice([0.,.4,.8,1.,1.2],p=[.1,.2,.2,.2,.3]),0.,0.]


def main():
    torch.set_num_threads(1)
    inputs=dict(read(ROOT/"docs/data/go2-acceleration-bias-reset/frozen-candidate.json")["sha256"])
    for p in (Path(__file__),PLAN,ROOT/"docs/data/go2-speed-qualification-probe/summary.json"):
        inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=244,inputs=inputs,training_steps=65536,seed=644101,max_evaluations=5))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for p in (Path(__file__),PLAN,CONFIG):shutil.copy2(p,RUN/p.name)
    pilot.RUN=RUN
    control.RUN=RUN
    control.Course=SpeedCourse
    parent=ROOT/"artifacts/runs/20261107-frequency-adaptation"
    vec=DummyVecEnv([lambda:SpeedCourse() for _ in range(8)])
    norm=VecNormalize.load(parent/"checkpoint-65536-norm.pkl",vec)
    norm.training=True;norm.norm_reward=True
    model=PPO.load(parent/"checkpoint-65536.zip",env=norm,device="cpu")
    original=PPO.load(parent/"checkpoint-65536.zip",device="cpu")
    assert equal_state(model.policy.state_dict(),original.policy.state_dict())
    assert equal_state(model.policy.optimizer.state_dict(),original.policy.optimizer.state_dict())
    model.set_random_seed(644101)
    model.learning_rate=5e-5;model.lr_schedule=lambda _:5e-5
    for group in model.policy.optimizer.param_groups:group["lr"]=5e-5
    model.save(RUN/"initial-model");norm.save(RUN/"initial-norm.pkl")
    start=model.num_timesteps
    try:
        model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=Archive(start))
        assert model.num_timesteps-start==65536
        model.save(RUN/"checkpoint-65536");norm.save(RUN/"checkpoint-65536-norm.pkl")
    finally:
        for worker,env in enumerate(vec.envs):
            if env.trace:pilot.save_trace(RUN/f"remaining-worker-{worker}.npz",env.trace)
        norm.close()
    records=[]
    fixed_norm=norm_load(RUN/"checkpoint-65536-norm.pkl")
    for speed in (.4,.6,.8,1.,1.2):
        result=control.evaluate(model,fixed_norm,"adapted",622901,speed)[0]
        p=path_metrics(ROOT/result["folder"])
        passed=(result["passed"] and p["moving_heading_max_deg"]<=5 and p["all_heading_max_deg"]<=5 and p["moving_cross_max_m"]<=.2 and p["all_cross_max_m"]<=.25)
        records.append(dict(result=result,path=p,passed=passed))
        print("244",speed,p,passed,flush=True)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(PUBLIC/"summary.json",dict(experiment=244,records=records,qualified=all(r["passed"] for r in records),
        training_interactions=65536,new_evaluations=5,training_seed=644101,visual="pending"))


if __name__=="__main__":main()
