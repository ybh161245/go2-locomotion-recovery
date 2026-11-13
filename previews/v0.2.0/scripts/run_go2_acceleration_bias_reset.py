"""242 discard opposing integral memory at nonzero speed increases."""
from pathlib import Path
import shutil
import torch
import numpy as np
from stable_baselines3 import PPO
from go2_157 import ROOT,read,sha,write_new
from go2_action_transfer import norm_load
from go2_quality_guard import preserve_rng
from run_go2_speed_course_pilot_validated import save_trace
from run_go2_high_speed_action_scale import Course
from run_go2_straight_path_control import path_metrics
RUN=ROOT/"artifacts/runs/20261107-acceleration-bias-reset"
PUBLIC=ROOT/"docs/data/go2-acceleration-bias-reset"
CONFIG=ROOT/"configs/go2-acceleration-bias-reset.json"
PLAN=ROOT/"docs/experiments/2026-11-07-acceleration-bias-reset.md"
class LineController:
    def __init__(self):
        self.reference=None
        self.extra=0.
        self.bias=0.
        self.velocity=np.zeros(2)
        self.qualification_count=0
        self.qualified=False
        self.previous_request=0.

    def observe(self,env,raw):
        self.velocity+=.1*(np.asarray(raw[:2],np.float64)/2-self.velocity)
        beta=float(np.arctan2(self.velocity[1],max(self.velocity[0],.2)))
        w,x,y,z=env.data.qpos[3:7]
        yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        if self.reference is None and env.command[0]>.025:
            self.reference=np.r_[env.data.qpos[:2],yaw]
        if env.command[0]==0 and abs(env.filtered_command[0])<=.025:
            self.reference=None
            self.bias=0.
        if self.reference is None:
            self.qualification_count=0
            self.qualified=False
        elif not self.qualified:
            self.qualification_count=self.qualification_count+1 if self.velocity[0]>=.5 else 0
            self.qualified=self.qualification_count>=5
        raw_beta=beta
        beta=beta if self.qualified else 0.
        bias_reset=False
        bias_before=self.bias
        heading=cross=target=0.
        reference=np.zeros(3) if self.reference is None else self.reference.copy()
        if self.reference is not None:
            theta=self.reference[2]
            heading=float(np.arctan2(np.sin(yaw-theta),np.cos(yaw-theta)))
            cross=float(np.dot(env.data.qpos[:2]-self.reference[:2],[-np.sin(theta),np.cos(theta)]))
            if env.command[0]>self.previous_request and self.previous_request>.025 and self.bias*cross>0:
                self.bias=0.
                bias_reset=True
            if env.command[0]==0:
                self.bias=0.
                beta=0.
                target=float(np.clip(-1.5*heading,-.3,.3))
            else:
                self.bias=float(np.clip(self.bias-.8*(heading+beta)*.02,-.15,.15))
                target=float(np.clip(-1.5*(heading+beta)-.6*cross+self.bias,-.3,.3))
        self.extra+=float(np.clip(target-self.extra,-.04,.04))
        self.previous_request=float(env.command[0])
        actual=raw.copy()
        actual[11]+=np.float32(.25*self.extra)
        return actual,dict(path_reference=reference,path_active=self.reference is not None,
                          path_heading_error=heading,path_cross_error=cross,path_yaw_extra=self.extra,path_integral_bias=self.bias,travel_beta=beta,travel_velocity=self.velocity.copy(),travel_raw_beta=raw_beta,travel_qualified=self.qualified,travel_qualification_count=self.qualification_count,braking_heading_only=self.reference is not None and env.command[0]==0,acceleration_bias_reset=bias_reset,previous_integral_bias=bias_before)

def evaluate(model,norm,label,seed,speed,prefix_seconds=0,prefix_speed=0):
    records=[]
    for speed in (speed,):
        env=Course("external",episode_seconds=18)
        env.reset(seed=seed,options={"command":[0.,0.,0.]})
        folder=RUN/f"{label}-{speed:.1f}"
        folder.mkdir()
        velocities=[];angular=[];tilts=[];joints=[];feet=[];forces=[];power=[]
        controller=LineController()
        with preserve_rng(model):
            for i in range(900):
                prefix_end=100+round(prefix_seconds*50)
                requested=prefix_speed if 100<=i<prefix_end else speed if prefix_end<=i<600 else 0.
                env.command[:]=[requested,0.,0.]
                raw=env.observe().copy()
                policy_input,log=controller.observe(env,raw)
                action=model.predict(norm.normalize_obs(policy_input.copy()),deterministic=True)[0]
                _,_,term,trunc,info=env.step(action)
                env.trace[-1].update(policy_input=policy_input,**log)
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
        rmse=float(np.sqrt(np.mean(np.sum((v[max(200,prefix_end+50):min(600,n),:2]-[speed,0])**2,axis=1)))) if n>200 else None
        band=(np.linalg.norm(v[600:,:2],axis=1)<=.08)&(np.abs(w[600:,2])<=.15)
        stable=next((i*.02 for i in range(max(0,len(band)-24)) if all(band[i:i+25])),None)
        clearance=np.percentile(heights[200:min(600,n)],95,axis=0).tolist() if n>200 else [0]*4
        check=dict(complete=n==900 and not term,tilt=float(abs(angles).max())<=8,
            speed=rmse is not None and rmse<=(.1 if speed==.4 else .15),
            stop=stable is not None and stable<=2,
            quiet=n==900 and float(np.mean(joints[-100:]))<=.3,
            clearance=min(clearance)>=.02)
        result=dict(label=label,speed=speed,steps=n,terminated=term,mean_vx=float(v[max(200,prefix_end+50):min(600,n),0].mean()) if n>200 else None,
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
    inputs=dict(read(ROOT/"docs/data/go2-braking-heading-control/frozen-component.json")["sha256"])
    for p in (Path(__file__),PLAN,ROOT/"docs/data/go2-acceleration-integral-history/summary.json"):
        inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=242,inputs=inputs,max_physical=2))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for p in (Path(__file__),PLAN,CONFIG):shutil.copy2(p,RUN/p.name)
    parent=ROOT/"artifacts/runs/20261107-frequency-adaptation"
    model=PPO.load(parent/"checkpoint-65536.zip",device="cpu");norm=norm_load(parent/"checkpoint-65536-norm.pkl")
    cases=read(ROOT/"configs/go2-path-new-histories.json")["cases"]
    before=read(ROOT/"docs/data/go2-braking-heading-control/summary.json")["records"]
    records=[]
    for case,old in zip(cases,before):
        result=evaluate(model,norm,case["name"],case["seed"],1.,case["prefix_seconds"],case["prefix_speed"])[0]
        p=path_metrics(ROOT/result["folder"])
        with np.load(ROOT/result["folder"]/"states.npz") as a,np.load(ROOT/old["result"]["folder"]/"states.npz") as b:
            resets=np.flatnonzero(a["acceleration_bias_reset"])
            first=int(resets[0]) if len(resets) else len(a["qpos"])
            for key in b.files:np.testing.assert_array_equal(a[key][:first],b[key][:first])
        passed=(result["passed"] and p["moving_heading_max_deg"]<=5 and p["all_heading_max_deg"]<=5 and p["moving_cross_max_m"]<=.2 and p["all_cross_max_m"]<=.25)
        records.append(dict(history=case["name"],result=result,path=p,passed=passed,reset_steps=resets.tolist(),parent_prefix_exact=True))
        print("242",case["name"],p,passed,resets,flush=True)
    write_new(PUBLIC/"summary.json",dict(experiment=242,records=records,qualified=all(r["passed"] for r in records),new_physical=2,new_training=0))
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name


if __name__=="__main__":main()
