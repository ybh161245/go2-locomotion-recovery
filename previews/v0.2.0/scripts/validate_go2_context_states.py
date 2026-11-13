"""090 frozen policies on actual new body states and transient command/push protocols."""
import copy,json,shutil,time,traceback
import mujoco
import numpy as np
import torch
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from screen_go2_mass_course import factory,save_probe,adapter_policy,norm_state,exact
from screen_go2_recovery_specialist import load_expert
from go2_support_context_router import attach_support_context
from go2_fast_capture_router import attach_fast_capture_router
from go2_motion_metrics import sample_row,summarize
from go2_quality_guard import equal_state
from audit_go2_fast_capture import same
CONFIG='configs/go2-context-state-validation.json'
ACTIVE=None

def euler_quaternion(degrees):
    r,p,y=np.radians(degrees)/2;cr,sr=np.cos(r),np.sin(r);cp,sp=np.cos(p),np.sin(p);cy,sy=np.cos(y),np.sin(y)
    return np.array([cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy])

def attach_initial_state(env,profile):
    native=env.reset
    def reset(*args,**kwargs):
        native(*args,**kwargs);env.data.qpos[3:7]=euler_quaternion(profile['rpy_deg']);rot=np.empty(9);mujoco.mju_quat2Mat(rot,env.data.qpos[3:7]);env.data.qvel[:3]=rot.reshape(3,3)@np.asarray(profile['body_velocity']);mujoco.mj_forward(env.model,env.data)
        feet=env.meter.sample(env.data);env.validation_initial=dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),foot_contact=feet['contact'].copy(),foot_speed=feet['speed_xy'].copy(),command=env.command.copy(),filtered_command=env.filtered_command.copy())
        return env.observe(),{}
    env.reset=reset;return env

def command_at(protocol,i):
    t=i*.02;s=next(s for s in protocol['schedule'] if s['start']<=t<s['end']);return np.array(s['command'],dtype=float),s['phase']

def local_force(events,i,zero=False):
    force=np.zeros(3);torque=np.zeros(3)
    if not zero:
        for e in events:
            if round(e['start']/.02)<=i<round((e['start']+e['duration'])/.02):force+=e['force'];torque+=e.get('torque',[0.,0.,0.])
    return np.r_[force,torque]

def pressure_metrics(rs,protocol):
    last=max(e['start']+e['duration'] for e in protocol['events']);start=round(last/.02)
    band=[np.hypot(r['vx']-r['command_vx'],r['vy'])<=.08 and abs(r['yaw_rate']-r['command_yaw'])<=.15 for r in rs[start:]]
    recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None);peak=max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rs);complete=len(rs)==round(protocol['seconds']/.02) and not rs[-1]['terminated']
    return dict(steps=len(rs),terminated=bool(rs[-1]['terminated']),peak_tilt=peak,recovery_seconds=recovery,passed=bool(complete and peak<=8 and recovery is not None and recovery<=2),complete=bool(complete),last_withdrawal=last)

def run_transient(out,label,protocol,seed,make,adapter,zero):
    env=make('external',episode_seconds=protocol['seconds']);env.reset(seed=seed,options={'command':protocol['schedule'][0]['command']});states={k:[] for k in ['observations','actions','qpos','qvel','targets','world_wrench','feet_world','heading']};rs=[]
    try:
        for i in range(round(protocol['seconds']/.02)):
            cmd,phase=command_at(protocol,i);env.command[:]=cmd;obs=env.observe();local=local_force(protocol['events'],i,zero);R=env.data.xmat[env.base].reshape(3,3);wrench=np.r_[R@local[:3],R@local[3:]];env.data.xfrc_applied[:]=0;env.data.xfrc_applied[env.base]=wrench
            action,_=adapter[0].predict(adapter[1].normalize_obs(obs.copy()),deterministic=True);_,_,term,trunc,info=env.step(action)
            for k,v in [('observations',obs),('actions',action),('qpos',env.data.qpos),('qvel',env.data.qvel),('targets',env.data.ctrl),('world_wrench',wrench),('feet_world',env.data.geom_xpos[env.meter.feet]),('heading',env.data.xmat[env.base].reshape(3,3)[:,0])]:states[k].append(v.copy())
            row=sample_row(env,env.meter,info,env.data.time,phase);row.update(vy=float(info['vy']),terminated=bool(term),command_vx=float(cmd[0]),command_yaw=float(cmd[2]));rs.append(row)
            if term or trunc:break
    finally:env.data.xfrc_applied[:]=0;env.close()
    z={k:np.array(v) for k,v in states.items()};r=pressure_metrics(rs,protocol);r.update(protocol=protocol,zero_force=zero,mass_scale=1.,mode='routed',mass_kg=float(env.model.body_mass.sum()),force_impulse_norm_Ns=float(np.linalg.norm(z['world_wrench'][:,:3],axis=1).sum()*.02),segments={p:summarize([x for x in rs if x['phase']==p and x['time']>s['start']+1]) for s in protocol['schedule'] for p in [s['phase']]})
    return ev.save_run(out,label,protocol['id'],seed,r,z,rs)

def main():
    global ACTIVE
    cfg=read(ROOT/'configs/go2-support-context.json');cfg.update(experiment_id='GO2-B-261026-090',run='artifacts/runs/20261026-context-state-validation',public='docs/data/go2-context-state-validation',predeclared_plan='docs/experiments/2026-10-26-context-state-validation.md')
    profiles=[dict(id='A',seed=190101,rpy_deg=[2,-1,0],body_velocity=[0,.04,0]),dict(id='B',seed=190102,rpy_deg=[-2,1,0],body_velocity=[0,-.04,0]),dict(id='C',seed=190103,rpy_deg=[0,0,5],body_velocity=[0,0,0])]
    straight=[dict(start=0,end=12,phase='straight',command=[.2,0,0])];transition=[dict(start=a,end=b,phase=p,command=c) for a,b,p,c in [(0,1,'stand',[0,0,0]),(1,4,'start',[.2,0,0]),(4,7,'accelerate',[.4,0,0]),(7,10,'brake',[0,0,0]),(10,13,'turn-left',[.2,0,.5]),(13,16,'turn-reverse',[.2,0,-.5]),(16,20,'stop',[0,0,0])]]
    def event(t,d,fy):return dict(start=t,duration=d,force=[0,fy,0],torque=[0,0,0])
    protocols=[dict(id='startup-A',profile='A',friction=None,seconds=12,schedule=straight,events=[event(.12,.18,-20),event(6,.6,-40)],core=False),dict(id='startup-B',profile='B',friction=None,seconds=12,schedule=straight,events=[event(.12,.18,20),event(6,.6,-40)],core=False),dict(id='startup-low',profile='C',friction=.2,seconds=12,schedule=straight,events=[event(.18,.2,-20),event(6.14,.6,-30)],core=False),dict(id='acceleration',profile='A',friction=None,seconds=20,schedule=transition,events=[event(4.12,.3,-30)],core=True),dict(id='braking',profile='B',friction=None,seconds=20,schedule=transition,events=[event(7.06,.3,-30)],core=True),dict(id='reversal',profile='C',friction=None,seconds=20,schedule=transition,events=[event(13.04,.3,20)],core=True)]
    cfg['state_validation']=dict(profiles=profiles,protocols=protocols,max_physical_runs=30,training_steps=0,pressure_peak_window='full_episode',normal_RL_min_mm=29.5702276)
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);ev.CONFIG=cfg;torch.set_num_threads(1);spec=cfg['support_context'];out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');names={CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json','docs/data/go2-sustained-A/screen.json',*cfg['command_router']['model_hashes'],'docs/data/go2-support-context-boundary/research-candidate.json'}
    for key,step in [('default_train',spec['default_steps']),('low_train',spec['low_steps'])]:names.update(spec[key]+f'/checkpoint-{step}'+suffix for suffix in ['.zip','-vecnormalize.pkl'])
    inputs={n:sha(ROOT/n) for n in names};dump(out/'input-sha256.json',inputs);descriptor=read(ROOT/'docs/data/go2-support-context-boundary/research-candidate.json')
    for n,h in descriptor['models'].items():assert sha(ROOT/n)==h,n
    for n,h in descriptor['code_sha256'].items():assert sha(ROOT/'scripts'/n)==h,n
    D=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);formal=ev.load_policy(ROOT/cfg['motion_composition']['formal_joint']);default=load_expert(ROOT/spec['default_train'],spec['default_steps']);low=load_expert(ROOT/spec['low_train'],spec['low_steps']);adapter=adapter_policy(cfg,D);policies=[D,formal,default,low,adapter];stats=[norm_state(n) for m,n in policies];weights=[copy.deepcopy(m.policy.state_dict()) for m,n in policies]
    records=[];normal=[];zero_pairs=[];start=time.perf_counter();working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
    def make_factory(controller,profile,friction,label):
        plain=factory(D,dict(id=label,factor='friction' if friction is not None else 'nominal',level=friction if friction is not None else 1.,scope='frozen-state-validation'))
        def make(*a,**kw):
            env=plain(*a,**kw)
            if controller=='context':attach_support_context(env,*formal,default[0],low[0],default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],spec)
            else:attach_fast_capture_router(env,*formal,*default,cfg['confirmed_selector']['selector'],cfg['fast_capture'],4)
            return attach_initial_state(env,profile)
        return plain,make
    def finish(plain,r,controller,profile,kind):
        folder=ROOT/r['folder'];save_probe(plain,folder);env=plain.instances[-1];dump(folder/'selector.json',env.recovery_selection_log);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
        logs=env.recovery_selection_log;r.update(controller=controller,profile=profile,kind=kind,first_low_step=next((i for i,l in enumerate(logs) if l.get('context_changed')),None),low_expert_steps=sum(l.get('support_context')=='low' and l['weight']>0 for l in logs));records.append(r);dump(out/'progress.json',records);print('VALIDATION '+json.dumps({k:r.get(k) for k in ['case','controller','kind','passed','terminated','peak_tilt','first_low_step','low_expert_steps']}),flush=True)
    try:
        for profile in profiles:
            for controller in ['default','context']:
                plain,make=make_factory(controller,profile,None,'normal-'+profile['id']);r=ev.normal_run(out,controller+'-'+profile['id'],profile['seed'],make,adapter);r.update(mass_scale=1.,mode='routed');finish(plain,r,controller,profile,'normal');checks=ev.normal_checks(r,working);rl=r['segments'].get('reverse right',{}).get('RL',{}).get('clearance_p95_m',0.)*1000;normal.append(dict(result=r,checks=checks,RL_mm=rl,passed=bool(all(c['passed'] for c in checks) and rl>=29.5702276)))
        for protocol in protocols:
            profile=next(p for p in profiles if p['id']==protocol['profile'])
            for controller in ['default','context']:
                own=[]
                for zero in [True,False]:
                    label=controller+('-zero' if zero else '-push');plain,make=make_factory(controller,profile,protocol['friction'],label);r=run_transient(out,label,protocol,profile['seed'],make,adapter,zero);finish(plain,r,controller,profile,'zero' if zero else 'pressure');own.append(r)
                same(own[0],own[1],round(protocol['events'][0]['start']/.02));zero_pairs.append(dict(zero=own[0],push=own[1],preforce_exact=True))
        assert all(exact(a,norm_state(n)) for a,(m,n) in zip(stats,policies)) and all(equal_state(w,m.policy.state_dict()) for w,(m,n) in zip(weights,policies))
    finally:
        for m,n in policies:n.close()
    comparisons=[];pass_losses=[];terms=[];normal_losses=[]
    for protocol in protocols:
        a=next(r for r in records if r['controller']=='default' and r['kind']=='pressure' and r['case']==protocol['id']);b=next(r for r in records if r['controller']=='context' and r['kind']=='pressure' and r['case']==protocol['id'])
        if a['passed'] and not b['passed']:pass_losses.append(protocol['id'])
        if not a['terminated'] and b['terminated']:terms.append(protocol['id'])
        comparisons.append(dict(default=a,context=b,delta_peak=b['peak_tilt']-a['peak_tilt'],delta_recovery=b['recovery_seconds']-a['recovery_seconds'] if b['recovery_seconds'] is not None and a['recovery_seconds'] is not None else None))
    for p in profiles:
        a=next(n for n in normal if n['result']['controller']=='default' and n['result']['profile']['id']==p['id']);b=next(n for n in normal if n['result']['controller']=='context' and n['result']['profile']['id']==p['id']);old={(c['segment'],c['metric']):c for c in a['checks']}
        normal_losses.extend(dict(profile=p['id'],check=c) for c in b['checks'] if not c['passed'] and old[(c['segment'],c['metric'])]['passed'])
    zeros=[r for r in records if r['kind']=='zero'];core=[r for r in records if r['controller']=='context' and r['kind']=='pressure' and r['protocol']['core']];qualified=all(n['passed'] for n in normal if n['result']['controller']=='context') and all(r['passed'] for r in zeros+core) and not(pass_losses or terms or normal_losses)
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    summary=dict(config=cfg,records=records,normals=normal,comparisons=comparisons,zero_pairs=zero_pairs,pressure_pass_losses=pass_losses,added_pressure_terminations=terms,normal_pass_losses=normal_losses,qualified=bool(qualified),new_physical_runs=len(records),training_samples=0,PPO_updates=0,formal_selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-start);assert len(records)==30
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()});public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary);print(json.dumps(dict(qualified=qualified,new_physical_runs=len(records),normals=[dict(profile=n['result']['profile']['id'],controller=n['result']['controller'],failures=sum(not c['passed'] for c in n['checks']),RL=n['RL_mm']) for n in normal],zero_passes=sum(r['passed'] for r in zeros),core_passes=sum(r['passed'] for r in core),pass_losses=pass_losses,added_terms=terms)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
