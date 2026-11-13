"""Independently check060 raw gates, selector actuation and intervention timing."""
import csv,json
import mujoco
import numpy as np
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest,pressure_metrics
from go2_mass_course_env import Go2MassCourseEnv
from screen_go2_mass_course import annotate_mass
from go2_recovery_state_env import suffix_metrics,max_tilt


def rows(folder):
    with (folder/'steps.csv').open(encoding='utf-8') as f:
        result=[]
        for row in csv.DictReader(f):
            result.append({k:(v=='True' if v in ['True','False'] else float(v) if k not in ['phase'] else v) for k,v in row.items()})
    return result


def main():
    cfg=read(ROOT/'configs/go2-recovery-specialist-r2.json');ev.CONFIG=cfg;run=ROOT/cfg['screen_run'];s=read(run/'summary.json');verify_manifest(run)
    base=read(ROOT/cfg['recovery_specialist']['normal_reference']/'summary.json');working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
    snapshot_checks=[]
    for e in s['checkpoints']:
        for r in e['results']:
            data=rows(ROOT/r['folder']);actual=suffix_metrics(data,r['start_step'],r['prefix_peak'],len(data)==150)
            for k,v in actual.items():assert v==r[k],(r['bank_id'],k,v,r[k])
            snapshot_checks.append(dict(checkpoint=e['steps'],bank_id=r['bank_id'],raw_gates_recomputed=True))
    selector=[];normalfail=0;newfail=0
    for mass,b in s['by_mass'].items():
        import copy
        annotated=annotate_mass(copy.deepcopy(b),base['by_mass'][mass],cfg,working)
        for k in ['normal_checks','new_normal_failures','support_checks','rl_retention_pass','pressure_pass_losses','additional_terminations','retention_pass']:assert annotated[k]==b[k]
        normalfail+=b['normal_failures'];newfail+=len(b['new_normal_failures'])
        for r in [b['normal']]+b['pressure']:
            folder=ROOT/r['folder'];rs=rows(folder);logs=read(folder/'selector.json');assert len(logs)==len(rs)==r['steps']
            if r['case']!='normal':
                start=round((max(r['condition']['pulses'])+r['condition']['duration'])/.02)
                band=[np.hypot(x['vx']-.2,x['vy'])<=.08 and abs(x['yaw_rate'])<=.15 for x in rs[start:]]
                recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
                peak=max(max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in rs[300:])
                assert peak==r['peak_tilt'] and recovery==r['recovery_seconds']
                assert r['passed']==(r['steps']==600 and not r['terminated'] and peak<=8 and recovery is not None and recovery<=2)
            env=Go2MassCourseEnv('external',config=cfg);env.reset(seed=r['seed'],options={'command':[rs[0]['command_vx'],0,rs[0]['command_yaw']],'events':[],'mass_scale':float(mass)})
            initialq=env.data.qpos.copy();initialv=env.data.qvel.copy()
            with np.load(folder/'states.npz') as z,np.load(folder/'physics.npz') as physics:
                q=np.vstack([initialq[None],z['qpos'][:-1]]);v=np.vstack([initialv[None],z['qvel'][:-1]])
                applied=np.asarray([l['applied_action'] for l in logs]);baseaction=np.asarray([l['base_action'] for l in logs]);expert=np.asarray([l['expert_action'] for l in logs]);weight=np.asarray([l['weight'] for l in logs])
                # Match NumPy's native float32 expert term and float64 baseline term.
                mixed=np.asarray([(1-float(w))*baseaction[i]+float(w)*expert[i].astype(np.float32) if w else baseaction[i] for i,w in enumerate(weight)])
                np.testing.assert_array_equal(applied,mixed)
                np.testing.assert_array_equal(applied[weight==0],baseaction[weight==0])
                expected=np.clip(np.tile([0.,.8,-1.6],4)+.35*applied,physics['after_actuator_ctrlrange'][:,0],physics['after_actuator_ctrlrange'][:,1])
                np.testing.assert_array_equal(expected,z['targets'])
                for i,l in enumerate(logs):
                    env.data.qpos[:]=q[i];env.data.qvel[:]=v[i];mujoco.mj_forward(env.model,env.data)
                    lin,ang=env.velocity();tilt=np.degrees(max_tilt(env))
                    assert abs(tilt-l['tilt'])<1e-10 and abs(lin[1]-l['vy'])<1e-10 and abs(np.linalg.norm(ang[:2])-l['angular_xy'])<1e-10
                    spec=cfg['recovery_specialist']['selector'];permitted=abs(rs[i]['command_vx']-.2)<1e-8 and abs(rs[i]['command_yaw'])<=.025
                    risk=permitted and (tilt>=spec['tilt_on_deg'] or abs(lin[1])>=spec['vy_on'] or np.linalg.norm(ang[:2])>=spec['angular_xy_on'])
                    assert l['permitted']==permitted and l['risk']==risk
                ids=np.flatnonzero(weight>0);reference=base['by_mass'][mass]['normal'] if r['case']=='normal' else next(x for x in base['by_mass'][mass]['pressure'] if x['case']==r['case'])
                with np.load(ROOT/reference['folder']/'states.npz') as old:
                    prefix=int(ids[0]) if len(ids) else len(weight)
                    for k in ['qpos','qvel','targets']:np.testing.assert_array_equal(z[k][:prefix],old[k][:prefix])
                    prefixdiff=float(np.abs(z['qpos'][:min(300,len(z['qpos']),len(old['qpos']))]-old['qpos'][:min(300,len(z['qpos']),len(old['qpos']))]).max())
            env.close()
            selector.append(dict(mass=mass,case=r['case'],expert_steps=len(ids),first_expert_time_s=float(ids[0]*.02) if len(ids) else None,
                before_push_expert_steps=int((ids<300).sum()),prefix_before_first_intervention_exact_D=True,max_first6s_qpos_difference_from_D=prefixdiff))
    assert s['full_pressure_metrics']==pressure_metrics(s['by_mass'])
    report=dict(snapshot_raw_gate_checks=snapshot_checks,full_raw_pressure_gates_recomputed=True,all_normal_gates_recomputed=True,
        selector_sensors_and_action_blending_targets_verified=True,zero_weight_action_exact_D=True,
        selector_uses_only_command_velocity_and_orientation=True,normal_failures=normalfail,new_normal_failures=newfail,
        early_intervention_present=any(x['before_push_expert_steps'] for x in selector if x['case']!='normal'),
        full_protocol_has_no_reset=True,selector=selector,
        limitation='Prototype changes the pre-push trajectory;9/9 is end-to-end prototype performance, not isolated causal evidence for post-push recovery.')
    path=run/'screen-audit.json';assert not path.exists();dump(path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['snapshot_raw_gate_checks','selector']}),flush=True)


if __name__=='__main__':main()
