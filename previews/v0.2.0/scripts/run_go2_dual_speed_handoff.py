"""248 two finite physical tasks, no training or model updates."""
from pathlib import Path
import csv,math,shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from go2_157 import ROOT,RUNTIME,read,sha,write_new,load_stack
from go2_speed_hold import candidate_inputs
from go2_variable_speed_heading_hold import attach as attach212
from run_go2_full_heading_feedback import attach as attach197
from go2_dual_speed import attach
from go2_action_transfer import norm_load
from go2_sequence import validate
from run_go2_speed12_adaptation import SpeedCourse

RUN=ROOT/'artifacts/runs/20261108-dual-speed-handoff'
PUBLIC=ROOT/'docs/data/go2-dual-speed-handoff'
PLAN=ROOT/'docs/experiments/2026-11-08-dual-speed-handoff.md'
CONFIG=ROOT/'configs/go2-dual-speed-handoff.json'
PARENT=ROOT/'artifacts/runs/20261107-speed12-adaptation'

def stable(band):return next((i*.02 for i in range(max(0,len(band)-24)) if all(band[i:i+25])),None)

def metrics(d,t):
    q=d['qpos'];w,x,y,z=q[:,3:7].T
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    theta=yaw[99];heading=np.degrees(yaw-theta)
    cross=(q[:,:2]-q[99,:2])@[-np.sin(theta),np.cos(theta)]
    n=len(q);segments=[]
    for start,end,v in ((200,300,.4),(400,600,.8),(700,800,.4)):
        available=n>=end
        rmse=float(np.sqrt(np.mean(np.sum((t['velocity'][start:min(end,n),:2]-[v,0])**2,axis=1)))) if n>start else None
        clear=np.percentile(t['foot_height'][start:end],95,axis=0).tolist() if available else None
        segments.append(dict(start_s=start*.02,end_s=end*.02,request=v,mean_vx=float(t['velocity'][start:end,0].mean()) if available else None,
            rmse=rmse,clearance=clear,passed=bool(available and rmse<=(.15 if v==.8 else .1) and min(clear)>=.02)))
    stop=stable((np.linalg.norm(t['velocity'][800:,:2],axis=1)<=.08)&(abs(t['angular'][800:,2])<=.15)) if n>800 else None
    transitions=[]
    for start,v in ((300,.8),(600,.4)):
        band=(np.linalg.norm(t['velocity'][start:min(start+200,n),:2]-[v,0],axis=1)<=.08)&(abs(t['angular'][start:min(start+200,n),2])<=.15)
        s=stable(band);transitions.append(dict(time_s=start*.02,stable_s=s,passed=s is not None and s<=2))
    checks=dict(complete=n==1100 and not bool(d['terminated'][-1]),tilt=float(abs(t['tilt']).max())<=8,
        segments=all(s['passed'] for s in segments),transitions=all(s['passed'] for s in transitions),
        stop=stop is not None and stop<=2,quiet=n==1100 and float(t['joint_rms'][-100:].mean())<=.3,
        path=float(abs(heading[100:]).max())<=5 and float(abs(cross[100:min(n,800)]).max())<=.2 and float(abs(cross[100:]).max())<=.25)
    return dict(checks=checks,passed=all(checks.values()),segments=segments,transitions=transitions,stop_stable_s=stop,
        peak_tilt=float(abs(t['tilt']).max()),all_heading_max_deg=float(abs(heading[100:]).max()),all_cross_max_m=float(abs(cross[100:]).max()))

def run(env,stack,schedule,seed,out):
    out.mkdir(exist_ok=False);env.reset(seed=seed,options={'command':schedule[0][2]})
    arrays={k:[] for k in ('qpos','qvel','targets','observations','actions','world_wrench','feet_world','heading','command','terminated','actual_action','weight','low','high','policy_input')}
    tel={k:[] for k in ('velocity','angular','tilt','joint_rms','foot_height','foot_contact','actuator_force','power')}
    initial=env.data.qpos.copy()
    for i in range(schedule[-1][1]):
        cmd=next(s[2] for s in schedule if s[0]<=i<s[1]);env.command[:]=cmd;env.data.xfrc_applied[:]=0
        obs=env.observe().copy();action=stack.adapter[0].predict(stack.adapter[1].normalize_obs(obs.copy()),deterministic=True)[0]
        _,_,term,trunc,info=env.step(action);call=env.dual_calls[-1]
        values=(env.data.qpos,env.data.qvel,env.data.ctrl,obs,action,np.zeros(6),env.data.geom_xpos[env.meter.feet],env.data.xmat[env.base].reshape(3,3)[:,0],cmd,term,call['actual'],call['weight'],call['low'],call['high'],call['policy_input'])
        for k,v in zip(arrays,values):arrays[k].append(np.asarray(v).copy())
        v,w=env.velocity();qw,qx,qy,qz=env.data.qpos[3:7];foot=env.meter.sample(env.data)
        tilt=np.degrees([np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)),np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1))])
        for k,value in zip(tel,(v,w,tilt,np.sqrt(np.mean(env.data.qvel[6:]**2)),foot['height'],foot['contact'],env.data.actuator_force,np.abs(env.data.actuator_force*env.data.qvel[6:]).sum())):tel[k].append(np.asarray(value).copy())
        np.testing.assert_array_equal(np.clip(env.action_targets(call['actual']),*env.model.actuator_ctrlrange.T),env.data.ctrl)
        if term or trunc:break
    d={k:np.asarray(v) for k,v in arrays.items()};t={k:np.asarray(v) for k,v in tel.items()}
    np.savez_compressed(out/'states.npz',**d);np.savez_compressed(out/'telemetry.npz',**t);np.savez_compressed(out/'initial.npz',qpos=initial)
    return d,t

def main():
    torch.set_num_threads(1)
    inputs,mechanism=candidate_inputs()
    inputs.update(read(ROOT/'docs/data/go2-speed12-adaptation/frozen-study.json')['sha256'])
    for p in (Path(__file__),ROOT/'scripts/go2_dual_speed.py',PLAN):inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    write_new(CONFIG,dict(experiment=248,inputs=inputs,max_physical=2,training=0,blend_steps=15))
    RUN.mkdir(exist_ok=False);PUBLIC.mkdir(exist_ok=False)
    for p in (Path(__file__),ROOT/'scripts/go2_dual_speed.py',PLAN,CONFIG):shutil.copy2(p,RUN/p.name)
    cfg,stack,factory,_=load_stack(read(RUNTIME));actor=PPO.load(PARENT/'checkpoint-65536.zip',device='cpu');norm=norm_load(PARENT/'checkpoint-65536-norm.pkl')
    reference=read(ROOT/read(RUNTIME)['cases']['normal']['reference_record'])
    def make():
        _,f=factory(reference)
        return attach(attach212(attach197(f('external',episode_seconds=math.nextafter(22.,math.inf)),stack),mechanism),actor,norm)
    probe=make();high=SpeedCourse('external')
    fields=('body_mass','body_inertia','geom_friction','jnt_range','actuator_gainprm','actuator_biasprm','actuator_forcerange')
    diff={k:float(abs(getattr(probe.model,k)-getattr(high.model,k)).max()) for k in fields}
    write_new(PUBLIC/'interface.json',dict(physics_max_differences=diff,adapter_observation=53,teacher_observation=50,high_observation=50,new_physical=0))
    probe.close();high.close()
    old_task=read(ROOT/'configs/go2-variable-speed-new-conditions.json')['task'];schedule=validate(old_task['sequence'])
    env=make()
    try:d,t=run(env,stack,schedule,old_task['sequence']['seed'],RUN/'low-replay')
    finally:env.close()
    old_summary=read(ROOT/'docs/data/go2-variable-speed-new-conditions/summary.json')
    # Locate the exact212 left-zero record in the saved, structured panel.
    def records(obj):
        if isinstance(obj,dict):
            if 'folder' in obj and 'seed' in obj:yield obj
            for v in obj.values():yield from records(v)
        elif isinstance(obj,list):
            for v in obj:yield from records(v)
    matches=[r for r in records(old_summary) if r.get('seed')==513101 and '212' in r['folder'] and 'zero' in r['folder']]
    assert matches,matches
    old=matches[0]
    with np.load(ROOT/old['folder']/'states.npz') as b:
        for k in b.files:np.testing.assert_array_equal(d[k],b[k])
    assert np.all(d['weight']==0)
    write_new(PUBLIC/'low-replay.json',dict(reference=old['folder'],steps=len(d['qpos']),all8_exact=True,high_calls=0))
    schedule=[(0,100,[0.,0.,0.],'stand'),(100,300,[.4,0.,0.],'low'),(300,600,[.8,0.,0.],'high'),(600,800,[.4,0.,0.],'low'),(800,1100,[0.,0.,0.],'stop')]
    env=make()
    try:d,t=run(env,stack,schedule,648101,RUN/'mixed')
    finally:env.close();stack.close();norm.close()
    result=metrics(d,t);result.update(experiment=248,folder=(RUN/'mixed').relative_to(ROOT).as_posix(),steps=len(d['qpos']),new_physical=2,new_training=0,
        high_weight_frames=int(np.count_nonzero(d['weight'])),maximum_target_jump=float(np.linalg.norm(np.diff(d['targets'],axis=0),axis=1).max()))
    write_new(PUBLIC/'summary.json',result)
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    print(result,flush=True)

if __name__=='__main__':main()
