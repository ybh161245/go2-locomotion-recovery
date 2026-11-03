"""061 raw sensor/state-machine/actuation/gate audit, with no physics integration."""
import copy,json
import mujoco
import numpy as np
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest,pressure_metrics
from audit_go2_recovery_screen import rows
from go2_mass_course_env import Go2MassCourseEnv
from go2_recovery_state_env import max_tilt
from screen_go2_mass_course import annotate_mass
from evaluate_go2_confirmed_selector import trajectory_protection
from prepare_go2_confirmed_selector import verify_single_change


def audit_record(cfg,r,mass,reference,normal,protection_override=None):
    folder=ROOT/r['folder'];rs=rows(folder);logs=read(folder/'selector.json');spec=cfg['confirmed_selector']['selector']
    assert len(logs)==len(rs)==r['steps']
    if not normal:
        start=round((max(r['condition']['pulses'])+r['condition']['duration'])/.02)
        band=[np.hypot(x['vx']-.2,x['vy'])<=.08 and abs(x['yaw_rate'])<=.15 for x in rs[start:]]
        recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
        peak=max(max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in rs[300:])
        assert peak==r['peak_tilt'] and recovery==r['recovery_seconds']
        assert r['passed']==(r['steps']==600 and not r['terminated'] and peak<=8 and recovery is not None and recovery<=2)
    protection=trajectory_protection(r,reference,normal) if protection_override is None else protection_override
    env=Go2MassCourseEnv('external',config=cfg);env.reset(seed=r['seed'],options={'command':[rs[0]['command_vx'],0,rs[0]['command_yaw']],'events':[],'mass_scale':float(mass)})
    initialq=env.data.qpos.copy();initialv=env.data.qvel.copy();st=dict(active=False,risk=0,safe=0,age=0,cooldown=0,weight=0.);entries=[];exits=[]
    try:
        with np.load(folder/'states.npz') as z,np.load(folder/'physics.npz') as physics,np.load(folder/'telemetry.npz') as telemetry:
            q=np.vstack([initialq[None],z['qpos'][:-1]]);v=np.vstack([initialv[None],z['qvel'][:-1]])
            applied=np.asarray([l['applied_action'] for l in logs]);base=np.asarray([l['base_action'] for l in logs]);expert=np.asarray([l['expert_action'] for l in logs]);weights=np.asarray([l['weight'] for l in logs])
            mixed=np.asarray([(1-float(w))*base[i]+float(w)*expert[i].astype(np.float32) if w else base[i] for i,w in enumerate(weights)])
            np.testing.assert_array_equal(applied,mixed);np.testing.assert_array_equal(applied[weights==0],base[weights==0])
            target=np.clip(np.tile([0.,.8,-1.6],4)+.35*applied,physics['after_actuator_ctrlrange'][:,0],physics['after_actuator_ctrlrange'][:,1])
            np.testing.assert_array_equal(target,z['targets']);np.testing.assert_array_equal(telemetry['targets'],z['targets'])
            expected_force=np.zeros(len(rs))
            if not normal:
                for onset in r['condition']['pulses']:
                    expected_force[round(onset/.02):min(len(rs),round((onset+r['condition']['duration'])/.02))]=r['condition']['force']
            np.testing.assert_allclose(np.linalg.norm(z['world_wrench'][:,:3],axis=1),expected_force,atol=1e-8,rtol=0)
            for i,l in enumerate(logs):
                env.data.qpos[:]=q[i];env.data.qvel[:]=v[i];mujoco.mj_forward(env.model,env.data)
                lin,ang=env.velocity();tilt=np.degrees(max_tilt(env));cmd=np.array([rs[i]['command_vx'],0.,rs[i]['command_yaw']])
                assert abs(tilt-l['tilt'])<1e-10 and abs(lin[1]-l['vy'])<1e-10 and abs(np.linalg.norm(ang[:2])-l['angular_xy'])<1e-10
                permitted=abs(cmd[0]-spec['only_vx'])<1e-8 and abs(cmd[1])<1e-8 and abs(cmd[2])<=spec['max_requested_yaw']
                risk=permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on']) and abs(lin[1])>=spec['vy_confirmation']
                st['cooldown']=max(0,st['cooldown']-1);st['risk']=st['risk']+1 if risk else 0
                if not st['active'] and not st['cooldown'] and st['risk']>=spec['risk_steps']:
                    st.update(active=True,age=0,safe=0);entries.append(i*.02)
                if st['active']:
                    st['age']+=1
                    safe=tilt<=spec['safe_tilt_deg'] and np.linalg.norm(lin[:2]-cmd[:2])<=spec['safe_planar_error'] and abs(ang[2]-cmd[2])<=spec['safe_yaw_error'] and np.linalg.norm(ang[:2])<=spec['safe_angular_xy']
                    st['safe']=st['safe']+1 if safe else 0
                    if not permitted or st['age']>=spec['max_active_steps'] or (st['age']>=spec['min_active_steps'] and st['safe']>=spec['safe_steps']):
                        reason='command' if not permitted else 'duration_limit' if st['age']>=spec['max_active_steps'] else 'safe_hold'
                        later=rs[i:min(i+25,len(rs))]
                        exits.append(dict(time_s=i*.02,reason=reason,following_seconds=len(later)*.02,
                            following_peak_tilt=max(max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in later)))
                        st.update(active=False,risk=0,cooldown=spec['cooldown_steps'])
                st['weight']+=float(np.clip(float(st['active'])-st['weight'],-1/spec['blend_steps'],1/spec['blend_steps']))
                assert l['risk']==risk and l['permitted']==permitted and l['active']==st['active'] and l['weight']==st['weight']
    finally:env.close()
    return dict(mass=mass,case=r['case'],raw_sensor_state_machine_and_actuation_exact=True,trajectory_protection=protection,
        entries_s=entries,exits=exits,max_action_change=float(np.abs(np.diff(applied,axis=0)).max()) if len(applied)>1 else 0.)


def main():
    cfg=read(ROOT/'configs/go2-confirmed-selector.json');ev.CONFIG=cfg;run=ROOT/cfg['run'];spec=cfg['confirmed_selector'];s=read(run/'summary.json');verify_manifest(run)
    assert verify_single_change()==s['change_proof']
    for n,h in spec['model_hashes'].items():assert sha(ROOT/n)==h,n
    old=s['reference']['by_mass'];working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working'];audits=[]
    for m,b in s['by_mass'].items():
        again=annotate_mass(copy.deepcopy(b),old[m],cfg,working)
        for k in ['normal_checks','new_normal_failures','support_checks','rl_retention_pass','pressure_pass_losses','additional_terminations','retention_pass']:assert b[k]==again[k]
        for r in [b['normal']]+b['pressure']:
            normal=r['case']=='normal';prior=old[m]['normal'] if normal else next(x for x in old[m]['pressure'] if x['case']==r['case'])
            a=audit_record(cfg,r,m,prior,normal);assert a['trajectory_protection']==s['trajectory_protections'][r['folder']];audits.append(a)
    assert pressure_metrics(s['by_mass'])==s['pressure_metrics']
    if s['followup']['executed']:
        previous=read(ROOT/spec['followup']/'summary.json');ref={r['case']:r for r in previous['results']}
        cases={c['condition']['id']:c for c in previous['config']['internal_pressure']['cases']}
        for r in s['followup']['results']:
            a=audit_record(cfg,r,f'{cases[r["case"]]["mass_scale"]:.2f}',ref[r['case']],False)
            assert a['trajectory_protection']==s['trajectory_protections'][r['folder']];audits.append(a)
        assert s['followup']['passed']==(not s['followup']['pass_losses'] and not s['followup']['additional_terminations'] and all(a['trajectory_protection']['passed'] for a in audits))
    for n,h in read(run/'input-sha256.json').items():assert sha(ROOT/n)==h,n
    report=dict(change_proof=s['change_proof'],models_and_normalization_hashes_unchanged=True,
        raw_normal_and_pressure_gates_recomputed=True,all_selector_sensors_state_machine_and_targets_verified=True,
        all_protections_passed=all(a['trajectory_protection']['passed'] for a in audits),
        normal_complete_trajectories_exact_D=sum(a['case']=='normal' and a['trajectory_protection']['passed'] for a in audits),
        pressure_pre_push_trajectories_exact_D=sum(a['case']!='normal' and a['trajectory_protection']['passed'] for a in audits),
        records=audits,training_samples=0,new_physics_integration_runs=0)
    p=run/'selector-audit.json';assert not p.exists();dump(p,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','change_proof']}),flush=True)


if __name__=='__main__':main()
