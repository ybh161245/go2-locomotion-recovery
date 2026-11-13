"""222 first bounded speed expansion of the existing base joint policy."""
import csv
import pickle
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from go2_157 import ROOT, read, sha, write_new
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_action_transfer import norm_load
from go2_quality_guard import preserve_rng

RUN=ROOT/"artifacts/runs/20261107-speed-course-pilot"
PUBLIC=ROOT/"docs/data/go2-speed-course-pilot"
CONFIG=ROOT/"configs/go2-speed-course-pilot.json"
PLAN=ROOT/"docs/experiments/2026-11-07-speed-course-pilot.md"
PARENT=ROOT/"artifacts/runs/20261023-140238-938755-go2-transition-v3-mixed-ppo"


class Course(Go2TransitionEnvV3):
    def __init__(self,mode="mixed",**kwargs):
        self.trace=[]
        super().__init__(mode,**kwargs)

    def _sample_command(self):
        if self.training_command_mode=="external":return
        self.command[:]=[self.np_random.choice([0.,.4,.6,.8],p=[.2,.2,.3,.3]),0.,0.]

    def step(self,action):
        command=self.command.copy()
        out=super().step(action)
        self.trace.append(dict(command=command,action=np.asarray(action).copy(),
            qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),targets=self.data.ctrl.copy(),
            observation=out[0].copy(),reward=out[1],terminated=out[2],truncated=out[3]))
        return out


def save_trace(path,trace):
    np.savez_compressed(path,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})


def evaluate(model,norm,label):
    records=[]
    for speed in (.4,.6,.8,1.):
        env=Course("external",episode_seconds=18)
        env.reset(seed=622901,options={"command":[0.,0.,0.]})
        folder=RUN/f"{label}-{speed:.1f}"
        folder.mkdir()
        velocities=[];angular=[];tilts=[];joints=[];feet=[];forces=[];power=[]
        with preserve_rng(model):
            for i in range(900):
                env.command[:]=[speed if 100<=i<600 else 0.,0.,0.]
                action=model.predict(norm.normalize_obs(env.observe().copy()),deterministic=True)[0]
                _,_,term,trunc,info=env.step(action)
                v,w=env.velocity()
                velocities.append(v.copy());angular.append(w.copy())
                qw,qx,qy,qz=env.data.qpos[3:7]
                tilts.append(np.degrees([np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)),np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))]))
                joints.append(float(np.sqrt(np.mean(env.data.qvel[6:]**2))))
                foot=env.meter.sample(env.data)
                feet.append({k:foot[k].copy() for k in ("force","contact","height","speed_xy")})
                forces.append(env.data.actuator_force.copy())
                power.append(float(np.abs(env.data.actuator_force*env.data.qvel[6:]).sum()))
                if term or trunc:break
        save_trace(folder/"states.npz",env.trace)
        n=len(env.trace)
        v=np.array(velocities);w=np.array(angular);angles=np.array(tilts)
        heights=np.array([f["height"] for f in feet])
        contacts=np.array([f["contact"] for f in feet])
        np.savez_compressed(folder/"telemetry.npz",velocity=v,angular=w,tilt=angles,
            joint_rms=joints,foot_height=heights,foot_contact=contacts,
            foot_force=np.array([f["force"] for f in feet]),
            foot_speed=np.array([f["speed_xy"] for f in feet]),actuator_force=forces,power=power)
        rmse=float(np.sqrt(np.mean(np.sum((v[200:min(600,n),:2]-[speed,0])**2,axis=1)))) if n>200 else None
        band=(np.linalg.norm(v[600:,:2],axis=1)<=.08)&(np.abs(w[600:,2])<=.15)
        stable=next((i*.02 for i in range(max(0,len(band)-24)) if all(band[i:i+25])),None)
        clearance=np.percentile(heights[200:min(600,n)],95,axis=0).tolist() if n>200 else [0]*4
        check=dict(complete=n==900 and not term,tilt=float(abs(angles).max())<=8,
            speed=rmse is not None and rmse<=(.1 if speed==.4 else .15),
            stop=stable is not None and stable<=2,
            quiet=n==900 and float(np.mean(joints[-100:]))<=.3,
            clearance=min(clearance)>=.02)
        result=dict(label=label,speed=speed,steps=n,terminated=term,mean_vx=float(v[200:min(600,n),0].mean()) if n>200 else None,
            rmse=rmse,peak_tilt=float(abs(angles).max()),stop_stable_s=stable,
            final_joint_rms=float(np.mean(joints[-100:])),clearance=clearance,
            max_torque_fraction_50Hz=float(np.max(np.abs(forces)/np.max(abs(env.model.actuator_forcerange),axis=1))),
            moving_mean_abs_mechanical_power=float(np.mean(power[200:min(600,n)])) if n>200 else None,
            airborne_fraction=float(np.mean(~contacts[200:min(600,n)].any(axis=1))) if n>200 else None,
            checks=check,passed=all(check.values()),folder=folder.relative_to(ROOT).as_posix())
        write_new(folder/"result.json",result)
        records.append(result)
        env.close()
        print(label,speed,result["mean_vx"],rmse,result["passed"],flush=True)
    return records


class Archive(BaseCallback):
    def __init__(self,start):
        super().__init__();self.start=start;self.rollout=0

    def _on_step(self):return True

    def _on_rollout_start(self):
        self.rollout+=1
        with (RUN/f"rng-{self.rollout:03}.pkl").open("xb") as f:
            pickle.dump(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state()),f)
        if self.num_timesteps-self.start==32768:
            self.model.save(RUN/"checkpoint-32768")
            self.model.get_vec_normalize_env().save(RUN/"checkpoint-32768-norm.pkl")
        print("train222",self.num_timesteps-self.start,flush=True)

    def _on_rollout_end(self):
        b=self.model.rollout_buffer
        np.savez_compressed(RUN/f"buffer-{self.rollout:03}.npz",
            **{k:getattr(b,k) for k in ("observations","actions","rewards","returns","advantages","values","log_probs","episode_starts")})
        for i,env in enumerate(self.training_env.venv.envs):
            save_trace(RUN/f"rollout-{self.rollout:03}-worker-{i}.npz",env.trace)
            env.trace.clear()


def main():
    torch.set_num_threads(1)
    paths=[Path(__file__),PLAN,PARENT/"final-model.zip",PARENT/"final-vecnormalize.pkl",
           ROOT/"scripts/go2_transition_env_v3.py",ROOT/"scripts/go2_transition_env_v2.py",
           ROOT/"scripts/go2_comfort_env.py",ROOT/"scripts/go2_env.py"]
    for name,digest in read(CONFIG)["inputs"].items():assert sha(ROOT/name)==digest,name
    for p in paths[:2]:shutil.copy2(p,RUN/p.name)
    parent=PPO.load(PARENT/"final-model.zip",device="cpu")
    base_norm=norm_load(PARENT/"final-vecnormalize.pkl")
    before=evaluate(parent,base_norm,"parent-validated")
    vec=DummyVecEnv([lambda:Course() for _ in range(8)])
    norm=VecNormalize.load(PARENT/"final-vecnormalize.pkl",vec)
    norm.training=True;norm.norm_reward=True
    model=PPO.load(PARENT/"final-model.zip",env=norm,device="cpu")
    model.set_random_seed(622101)
    model.learning_rate=5e-5;model.lr_schedule=lambda _:5e-5
    for group in model.policy.optimizer.param_groups:group["lr"]=5e-5
    start=model.num_timesteps
    model.save(RUN/"initial-model");norm.save(RUN/"initial-norm.pkl")
    model.learn(total_timesteps=65536,reset_num_timesteps=False,callback=Archive(start))
    assert model.num_timesteps-start==65536
    model.save(RUN/"checkpoint-65536");norm.save(RUN/"checkpoint-65536-norm.pkl")
    norm.close()
    after=evaluate(model,norm_load(RUN/"checkpoint-65536-norm.pkl"),"trained")
    qualified=(all(n["passed"] for n in after[:3]) and
        any((not o["passed"] and n["passed"]) or o["rmse"]-n["rmse"]>=.02 for o,n in zip(before[1:3],after[1:3])) and
        all(not n["terminated"] or o["terminated"] for o,n in zip(before,after)))
    for name,digest in read(CONFIG)["inputs"].items():assert sha(ROOT/name)==digest,name
    write_new(PUBLIC/"summary.json",dict(experiment=222,before=before,after=after,
        training_interactions=65536,evaluations=8,qualified=qualified,training_seed=622101,
        evaluation_seed=622901,visual="pending"))


if __name__=="__main__":main()
