"""232 bounded path feedback added to the frozen230 straight-speed task."""
import shutil
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from go2_157 import ROOT,read,sha,write_new
from go2_action_transfer import norm_load
from go2_quality_guard import preserve_rng
from run_go2_speed_course_pilot_validated import save_trace
from run_go2_high_speed_action_scale import Course
RUN=ROOT/"artifacts/runs/20261107-straight-path-control"
PUBLIC=ROOT/"docs/data/go2-straight-path-control"
CONFIG=ROOT/"configs/go2-straight-path-control.json"
PLAN=ROOT/"docs/experiments/2026-11-07-straight-path-control.md"

class LineController:
    def __init__(self):
        self.reference=None
        self.extra=0.

    def observe(self,env,raw):
        w,x,y,z=env.data.qpos[3:7]
        yaw=float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        if self.reference is None and env.command[0]>.025:
            self.reference=np.r_[env.data.qpos[:2],yaw]
        if env.command[0]==0 and abs(env.filtered_command[0])<=.025:
            self.reference=None
        heading=cross=target=0.
        reference=np.zeros(3) if self.reference is None else self.reference.copy()
        if self.reference is not None:
            theta=self.reference[2]
            heading=float(np.arctan2(np.sin(yaw-theta),np.cos(yaw-theta)))
            cross=float(np.dot(env.data.qpos[:2]-self.reference[:2],[-np.sin(theta),np.cos(theta)]))
            target=float(np.clip(-1.5*heading-.6*cross,-.3,.3))
        self.extra+=float(np.clip(target-self.extra,-.04,.04))
        actual=raw.copy()
        actual[11]+=np.float32(.25*self.extra)
        return actual,dict(path_reference=reference,path_active=self.reference is not None,
                          path_heading_error=heading,path_cross_error=cross,path_yaw_extra=self.extra)

def evaluate(model,norm,label,seed,speed):
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
                env.command[:]=[speed if 100<=i<600 else 0.,0.,0.]
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


def path_metrics(folder):
    with np.load(folder/"states.npz") as data:q=data["qpos"]
    w,x,y,z=q[:,3:7].T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    angle=np.degrees(yaw-yaw[99])
    cross=(q[:,:2]-q[99,:2])@np.array([-np.sin(yaw[99]),np.cos(yaw[99])])
    return dict(moving_heading_max_deg=float(abs(angle[100:600]).max()),moving_cross_max_m=float(abs(cross[100:600]).max()),
                all_heading_max_deg=float(abs(angle[100:]).max()),all_cross_max_m=float(abs(cross[100:]).max()))


def main():
    torch.set_num_threads(1)
    inputs=dict(read(ROOT/"docs/data/go2-high-speed-action-scale/frozen-candidate.json")["sha256"])
    for path in (Path(__file__),PLAN,ROOT/"docs/data/go2-straight-path-audit/summary.json"):
        inputs[path.relative_to(ROOT).as_posix()]=sha(path)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    write_new(CONFIG,dict(experiment=232,inputs=inputs,max_physical=2,
        feedback=dict(heading_gain=1.5,cross_gain=.6,limit=.3,slew=2.),
        path_gates=dict(heading_deg=5,moving_cross_m=.2,all_cross_m=.25)))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for path in (Path(__file__),PLAN,CONFIG):shutil.copy2(path,RUN/path.name)
    parent=ROOT/"artifacts/runs/20261107-frequency-adaptation"
    model=PPO.load(parent/"checkpoint-65536.zip",device="cpu")
    norm=norm_load(parent/"checkpoint-65536-norm.pkl")
    old=read(ROOT/"docs/data/go2-high-speed-action-scale/summary.json")["records"]
    records=[]
    for speed in (.8,1.):
        result=evaluate(model,norm,"line",622901,speed)[0]
        before=next(p["after"] for p in old if p["after"]["speed"]==speed)
        folder=ROOT/result["folder"]
        with np.load(folder/"states.npz") as a,np.load(ROOT/before["folder"]/"states.npz") as b:
            for key in b.files:np.testing.assert_array_equal(a[key][:100],b[key][:100])
        metrics=path_metrics(folder)
        passed=(result["passed"] and metrics["moving_heading_max_deg"]<=5 and metrics["all_heading_max_deg"]<=5
                and metrics["moving_cross_max_m"]<=.2 and metrics["all_cross_max_m"]<=.25)
        records.append(dict(result=result,before=before,path=metrics,passed=passed))
        print("path232",speed,metrics,passed,flush=True)
    write_new(PUBLIC/"summary.json",dict(experiment=232,records=records,qualified=all(r["passed"] for r in records),new_physical=2,new_training=0))
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name


if __name__=="__main__":main()
