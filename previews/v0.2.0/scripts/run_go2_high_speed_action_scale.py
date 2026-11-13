"""230 fixed-model high-speed target scale causal test."""
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from go2_157 import ROOT,read,sha,write_new
from go2_action_transfer import norm_load
from go2_quality_guard import preserve_rng
from run_go2_speed_course_pilot_validated import save_trace
from run_go2_frequency_adaptation import FrequencyCourse
RUN=ROOT/"artifacts/runs/20261107-high-speed-action-scale"
PUBLIC=ROOT/"docs/data/go2-high-speed-action-scale"
CONFIG=ROOT/"configs/go2-high-speed-action-scale.json"
PLAN=ROOT/"docs/experiments/2026-11-07-high-speed-action-scale.md"

class Course(FrequencyCourse):
    def action_targets(self,action):
        self.actual_target_scale=1.+.5*float(np.clip(abs(float(self.filtered_command[0]))-.8,0.,.2))
        if self.actual_target_scale==1.:
            return super().action_targets(action)
        return self.nominal+.35*self.actual_target_scale*action

    def step(self,action):
        result=super().step(action)
        self.trace[-1]["target_scale"]=self.actual_target_scale
        return result

def evaluate(model,norm,label,seed,speed):
    records=[]
    for speed in (speed,):
        env=Course("external",episode_seconds=18)
        env.reset(seed=seed,options={"command":[0.,0.,0.]})
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


def main():
    torch.set_num_threads(1)
    inputs=dict(read(ROOT/"docs/data/go2-frequency-adaptation/frozen-study.json")["sha256"])
    for path in (Path(__file__),PLAN,ROOT/"docs/data/go2-joint-action-headroom/summary.json"):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=230,inputs=inputs,max_physical=2,max_target_scale=1.1))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for path in (Path(__file__),PLAN,CONFIG):shutil.copy2(path,RUN/path.name)
    parent=ROOT/"artifacts/runs/20261107-frequency-adaptation"
    model=PPO.load(parent/"checkpoint-65536.zip",device="cpu")
    norm=norm_load(parent/"checkpoint-65536-norm.pkl")
    old=read(ROOT/"docs/data/go2-frequency-adaptation/summary.json")["records"]
    records=[]
    for speed in (.8,1.):
        after=evaluate(model,norm,"scaled",622901,speed)[0]
        before=next(r for r in old if r["speed"]==speed)
        if speed==.8:
            for filename in ("states.npz","telemetry.npz"):
                with np.load(ROOT/after["folder"]/filename) as a,np.load(ROOT/before["folder"]/filename) as b:
                    for key in b.files:np.testing.assert_array_equal(a[key],b[key])
                    if filename=="states.npz":assert np.all(a["target_scale"]==1.)
        records.append(dict(before=before,after=after))
    before,after=records[1]["before"],records[1]["after"]
    positive=(after["mean_vx"]-before["mean_vx"]>=.03 or before["rmse"]-after["rmse"]>=.02)
    positive=positive and records[0]["after"]["passed"] and not after["terminated"]
    write_new(PUBLIC/"summary.json",dict(experiment=230,records=records,positive_speed_evidence=positive,
        qualified=positive and after["passed"],new_physical=2,new_training=0,new_fits=0))
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name


if __name__=="__main__":main()
