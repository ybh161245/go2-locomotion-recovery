"""078 independently reconstruct fast sensor entry, latched braking and native targets from raw state."""
import ast,copy,json,shutil
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest
from audit_go2_recovery_screen import rows
from audit_go2_routed_pressure import gate,physics,inference
from audit_go2_transfer_regression import vector_force_audit
from go2_mass_course_env import Go2MassCourseEnv
from go2_recovery_state_env import max_tilt
from go2_action_transfer import norm_load
from go2_quality_guard import check_limits

def audit_controller(cfg,r,mass):
    folder=ROOT/r['folder'];rs=rows(folder);logs=read(folder/'selector.json');spec=cfg['confirmed_selector']['selector'];fast=cfg['fast_capture']
    env=Go2MassCourseEnv('external',config=cfg);env.reset(seed=r['seed'],options={'command':[rs[0]['command_vx'],0,rs[0]['command_yaw']],'events':[],'mass_scale':mass})
    initialq=env.data.qpos.copy();initialv=env.data.qvel.copy();state=dict(active=False,risk=0,safe=0,age=0,cooldown=0,weight=0.,fw=0.,previous=None,latched=False);filtered=np.zeros(3);applied=[];events=[];action_jumps=[]
    assert len(logs)==len(rs)==r['steps']
    try:
        with np.load(folder/'states.npz') as z,np.load(folder/'physics.npz') as p:
            q=np.vstack([initialq[None],z['qpos'][:-1]]);v=np.vstack([initialv[None],z['qvel'][:-1]])
            for i,l in enumerate(logs):
                cmd=np.asarray([rs[i]['command_vx'],0.,rs[i]['command_yaw']]);np.testing.assert_array_equal(l['requested_command'],cmd);np.testing.assert_allclose(l['filtered_command'],filtered,atol=1e-12,rtol=0)
                state['fw']+=float(np.clip(float(abs(cmd[2])>spec['max_requested_yaw'])-state['fw'],-.25,.25));assert l['formal_weight']==state['fw']
                env.data.qpos[:]=q[i];env.data.qvel[:]=v[i];mujoco.mj_forward(env.model,env.data);lin,ang=env.velocity();tilt=np.degrees(max_tilt(env))
                assert abs(l['tilt']-tilt)<1e-10 and abs(l['vy']-lin[1])<1e-10 and abs(l['angular_xy']-np.linalg.norm(ang[:2]))<1e-10
                previous=state['previous'];acc=0. if previous is None else (float(lin[1])-previous)/fast['control_dt_s'];state['previous']=float(lin[1]);assert l['previous_vy']==previous and abs(l['lateral_acceleration']-acc)<1e-10
                old=abs(cmd[0]-spec['only_vx'])<1e-8 and abs(cmd[1])<1e-8 and abs(cmd[2])<=spec['max_requested_yaw'];domain=state['fw']==0 and abs(filtered[2])<=spec['max_requested_yaw'];stop=np.max(np.abs(cmd))<1e-8
                brake=stop and filtered[0]>=fast['brake_filtered_vx_min'];eligible=domain and (old or brake);signal=bool(eligible and lin[1]<=fast['vy_max'] and acc<=fast['lateral_acceleration_max'])
                state['cooldown']=max(0,state['cooldown']-1);entry=bool(signal and not state['active'] and not state['cooldown'])
                if entry:state.update(active=True,age=0,safe=0,latched=True,weight=1.);events.append(i*.02)
                permitted=bool(domain and (old or (state['latched'] and stop)));risk=bool(permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on']) and abs(lin[1])>=spec['vy_confirmation'])
                for key,value in [('old_permitted',old),('domain',domain),('braking',brake),('fast_eligible',eligible),('fast_signal',signal),('fast_entry',entry),('permitted',permitted),('risk',risk)]:assert l[key]==value,(r['folder'],i,key)
                state['risk']=state['risk']+1 if risk else 0
                if not state['active'] and not state['cooldown'] and state['risk']>=spec['risk_steps']:state.update(active=True,age=0,safe=0)
                if state['active']:
                    state['age']+=1;safe=tilt<=spec['safe_tilt_deg'] and np.linalg.norm(lin[:2]-cmd[:2])<=spec['safe_planar_error'] and abs(ang[2]-cmd[2])<=spec['safe_yaw_error'] and np.linalg.norm(ang[:2])<=spec['safe_angular_xy'];state['safe']=state['safe']+1 if safe else 0
                    if not permitted or state['age']>=spec['max_active_steps'] or (state['age']>=spec['min_active_steps'] and state['safe']>=spec['safe_steps']):state.update(active=False,risk=0,cooldown=spec['cooldown_steps'])
                state['weight']+=float(np.clip(float(state['active'])-state['weight'],-.25,.25));assert l['active']==state['active'] and l['weight']==state['weight'] and l['fast_latched']==state['latched']
                d=np.asarray(l['D_action'],dtype=np.float64);f=np.asarray(l['formal_action'],dtype=np.float32);routed=(1-state['fw'])*np.clip(d,-1,1)+state['fw']*np.clip(f,-1,1) if state['fw'] else d;base=np.clip(routed,-1,1);np.testing.assert_array_equal(base,l['base_action'])
                expert=np.asarray(l['expert_action'],dtype=np.float32);result=(1-state['weight'])*base+state['weight']*expert if state['weight'] else base;np.testing.assert_array_equal(result,l['applied_action']);applied.append(result)
                if entry:action_jumps.append(float(np.max(np.abs(result-(applied[-2] if len(applied)>1 else np.zeros(12))))))
                raw=np.asarray(l['base_observation'],dtype=np.float32);teacher=raw.copy();teacher[11]+=np.float32(.25*l['last_correction']);np.testing.assert_array_equal(teacher,np.asarray(l['teacher_observation'],dtype=np.float32))
                if 'observations' in z:np.testing.assert_array_equal(raw,z['observations'][i,:50])
                filtered+=np.clip(cmd-filtered,-np.array([.8,.8,2.])*.02,np.array([.8,.8,2.])*.02)
                if not state['active'] and state['weight']==0:state['latched']=False
            targets=np.clip(np.tile([0.,.8,-1.6],4)+.35*np.asarray(applied),p['after_actuator_ctrlrange'][:,0],p['after_actuator_ctrlrange'][:,1]);np.testing.assert_array_equal(targets,z['targets'])
    finally:env.close()
    return dict(folder=r['folder'],steps=len(rs),fast_entry_times_s=events,fast_entry_max_action_jumps=action_jumps,expert_steps=sum(l['weight']>0 for l in logs),
        max_target_step_rad=float(np.max(np.abs(np.diff(targets,axis=0)))) if len(targets)>1 else 0.,raw_sensors_state_machine_and_targets_exact=True)

def same(a,b,cut=None):
    with np.load(ROOT/a['folder']/'states.npz') as x,np.load(ROOT/b['folder']/'states.npz') as y:
        assert set(x.files)==set(y.files)
        for key in x.files:np.testing.assert_array_equal(x[key][:cut],y[key][:cut])

def main():
    cfg=read(ROOT/'configs/go2-fast-capture-entry.json');ev.CONFIG=cfg;run=ROOT/cfg['run'];s=read(run/'summary.json');verify_manifest(run);assert not (run/'audit.json').exists();torch.set_num_threads(2)
    source=(run/'source/go2_fast_capture_router.py').read_text(encoding='utf-8');tree=ast.parse(source);banned={'xfrc_applied','events','step_count','time','probe_spec','mass_scale','geom_friction'}
    assert not any(isinstance(n,ast.Attribute) and n.attr in banned for n in ast.walk(tree))
    parent=read(ROOT/cfg['fast_capture']['reference075']/'acceptance.json');assert parent['confirmed_selector']['selector']==cfg['confirmed_selector']['selector']
    for name in ['go2_turn_residual_env.py','go2_transition_env_v3.py','go2_transition_env_v2.py','go2_comfort_env.py','go2_env.py','go2_parameter_probe.py','stress_go2_frozen.py']:assert sha(run/'source'/name)==sha(ROOT/cfg['fast_capture']['reference075']/'source'/name)
    paths={'D':(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']),'formal':(ROOT/cfg['motion_composition']['formal_joint'],'final'),'expert':((ROOT/cfg['motion_composition']['expert']['model']).parent,'checkpoint-250')}
    policies={key:(PPO.load(f/('final-model.zip' if stem=='final' else stem+'.zip'),device='cpu'),norm_load(f/(stem+'-vecnormalize.pkl'))) for key,(f,stem) in paths.items()};adapter_cfg=read(ROOT/'artifacts/runs'/cfg['working_stack']/'config.json')
    audits=[];force=[];errors=[];comparisons=[]
    for b in s['results']+s['zero_controls']:
        r=b['result'];mass=r['mass_scale'];gate(r);physics(r);audits.append(audit_controller(cfg,r,mass));force.append(vector_force_audit(cfg,r,mass));errors.append(inference(r,policies,adapter_cfg))
        cut=round(r['condition']['pulses'][0]/.02);same(r,b['reference'],cut)
        if b['group']=='zero':same(r,b['reference']);assert audits[-1]['expert_steps']==0
        comparisons.append(dict(case=r['case'],seed=r['seed'],mass_scale=mass,group=b['group'],old=b['reference'],new=r,
            delta_peak=r['peak_tilt']-b['reference']['peak_tilt'],delta_recovery=r['recovery_seconds']-b['reference']['recovery_seconds'] if r['recovery_seconds'] is not None and b['reference']['recovery_seconds'] is not None else None))
        assert audits[-1]['fast_entry_times_s']==b['fast_entry_times_s']
    working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working'];normal_checks=0
    for n in s['normals']:
        r=n['result'];same(r,n['reference']);checks=ev.normal_checks(r,working);assert checks==n['checks'] and all(c['passed'] for c in checks);normal_checks+=len(checks);a=audit_controller(cfg,r,1.);assert a['expert_steps']==0;audits.append(a)
        errors.append(inference(r,policies,adapter_cfg))
    role=read(ROOT/cfg['fast_capture']['reference071']/'frozen-role-limits.json');internal_checks=0
    for cid,b in s['internals'].items():
        r=b['result'];same(r,b['reference']);checks=check_limits(r['metrics'],role[cid]);assert checks==b['checks'] and all(c['passed'] for c in checks);internal_checks+=len(checks);a=audit_controller(cfg,r,1.);assert a['expert_steps']==0;audits.append(a)
    for n in s['normals']+list(s['internals'].values()):
        r=n['result']
        with np.load(ROOT/r['folder']/'physics.npz') as p:mass=float(p['after_body_mass'].sum())
        wrapped=dict(r,mass_kg=mass,force_impulse_norm_Ns=0.,condition=dict(friction=None,pulses=[6.],duration=.2,force=0,direction=[0,1],torque=0))
        physics(wrapped);force.append(vector_force_audit(cfg,wrapped,1.))
    protected=[b for b in s['results'] if b['group']=='protected'];loss=[b['result']['folder'] for b in protected if b['reference']['passed'] and not b['result']['passed']];extras=[b['result']['folder'] for b in protected if not b['reference']['terminated'] and b['result']['terminated']]
    assert len(audits)==s['new_physical_runs']
    if s['qualified']:assert len(protected)==45 and not loss and not extras and len(s['normals'])==3 and len(s['internals'])==2 and all(b['result']['passed'] for b in s['results'] if b['group']=='primary')
    for name,h in read(run/'input-sha256.json').items():assert sha(ROOT/name)==h,name
    for name,h in cfg['command_router']['model_hashes'].items():assert sha(ROOT/name)==h,name
    report=dict(records=len(audits),raw_sensor_acceleration_entry_latch_and_exit_verified=True,all_native_targets_physics_contact_and_pressure_gates_verified=True,
        no_force_time_counter_or_parameter_oracle_in_router_AST=True,model_and_original_environment_hashes_unchanged=True,controller_is_combined_mechanism_not_single_factor=True,
        primary_passes=sum(b['result']['passed'] for b in s['results'] if b['group']=='primary'),protected_records=len(protected),protected_pass_losses=loss,protected_added_terminations=extras,
        normal_original_checks=normal_checks,internal_role_checks=internal_checks,normal_arrays_exact075=bool(len(s['normals'])==3),internal_arrays_exact070=bool(len(s['internals'])==2),
        max_force_vector_error=max(force or [0.]),max_batch_action_error=max(errors or [0.]),maximum_target_step_rad=max([a['max_target_step_rad'] for a in audits] or [0.]),
        qualified=s['qualified'],new_physics_integration_runs=0,training_samples=0,formal_selection_changed=False,visual_acceptance='pending',records_detail=audits)
    dest=run/'audit-source';dest.mkdir(exist_ok=False)
    for name in ['audit_go2_fast_capture.py','audit_go2_routed_pressure.py','audit_go2_transfer_regression.py']:shutil.copy2(ROOT/'scripts'/name,dest/name)
    dump(run/'audit.json',report);dump(ROOT/cfg['public']/'audit.json',report);dump(run/'paired-comparison.json',comparisons);dump(ROOT/cfg['public']/'paired-comparison.json',comparisons)
    print(json.dumps({k:v for k,v in report.items() if k!='records_detail'}),flush=True)

if __name__=='__main__':main()
