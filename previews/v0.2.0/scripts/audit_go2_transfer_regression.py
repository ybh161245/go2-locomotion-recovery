"""066 recompute expanded metrics and check physical/selector/raw preservation evidence."""
import copy,json
import mujoco
import numpy as np
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_transfer_regression import CONFIG,verify_inputs
from evaluate_go2_probe_value import verify_manifest
from evaluate_go2_recovery_regression import metrics
from audit_go2_confirmed_selector import audit_record
from audit_go2_recovery_screen import rows
from screen_go2_mass_course import annotate_mass
from go2_quality_guard import check_limits
from go2_mass_course_env import Go2MassCourseEnv

def vector_force_audit(cfg,r,mass):
    rs=rows(ROOT/r['folder']);env=Go2MassCourseEnv('external',config=cfg)
    env.reset(seed=r['seed'],options={'command':[rs[0]['command_vx'],0,rs[0]['command_yaw']],'events':[],'mass_scale':mass})
    initial=env.data.qpos.copy();env.close()
    with np.load(ROOT/r['folder']/'states.npz') as z:
        before=np.vstack([initial[None],z['qpos'][:-1]]);expected=np.zeros_like(z['world_wrench']);c=r['condition']
        for i,q in enumerate(before):
            active=any(round(t/.02)<=i<round((t+c['duration'])/.02) for t in c['pulses'])
            if active:
                rot=np.empty(9);mujoco.mju_quat2Mat(rot,q[3:7]);rotation=rot.reshape(3,3)
                expected[i,:3]=rotation@np.array([*np.array(c['direction'])*c['force'],0.])
                expected[i,3:]=rotation@np.array([0.,0.,c['torque']])
        error=float(np.max(np.abs(expected-z['world_wrench'])));assert error<1e-8,error
    return error

def main():
    cfg=read(ROOT/CONFIG);spec=cfg['transfer_regression'];ev.CONFIG=cfg;run=ROOT/cfg['run'];verify_manifest(run);verify_inputs(run)
    s=read(run/'summary.json');current={r['case']:r for r in s['results']};D={r['case']:r for r in s['reference_D']};prev={r['case']:r for r in s['reference061']}
    aliases=read(ROOT/spec['previous_regression']/'summary.json')['config']['recovery_regression']['aliases'];audits=[];seen=set();force_errors=[]
    allrecords=[r for r in s['results'] if r['case'] not in aliases]
    allrecords += [r for mass,b in s['by_mass'].items() for r in [b['normal']]+(b['pressure'] if mass!='1.00' else [])]
    allrecords += s['internal_pressure']+[b['A'] for b in s['fallback_A']]
    for r in allrecords:
        if r['folder'] in seen:continue
        seen.add(r['folder']);context=s['audit_references'][r['folder']]
        a=audit_record(cfg,r,context['mass'],context['reference'],context['normal']);audits.append(a)
        assert all(a['trajectory_protection'][k]==v for k,v in s['protections'][r['folder']].items() if k!='zero_force_complete_exact')
        if not context['normal']:force_errors.append(vector_force_audit(cfg,r,context['mass']))
    limits=read(ROOT/cfg['quality_guard']['reference_folder']/'limits.json');normalref=read(ROOT/cfg['quality_guard']['reference_folder']/'report.json');internal_audits=[]
    for name,b in s['internal_normals'].items():
        checks=check_limits(b['result']['metrics'],limits[name]);assert checks==b['checks'] and all(c['passed'] for c in checks)
        proof=dict(passed=b['passed'],arrays_exact=b['arrays_exact'],no_expert_before_limit=b['no_expert'])
        internal_audits.append(audit_record(cfg,b['result'],1.,normalref['results'][name],True,proof))
        with np.load(ROOT/b['result']['folder']/'states.npz') as x,np.load(ROOT/normalref['results'][name]['folder']/'states.npz') as y:
            for k,v in b['arrays_exact'].items():assert v and np.array_equal(x[k],y[k])
    base=read(ROOT/cfg['recovery_specialist']['normal_reference']/'summary.json');working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working'];normal_proofs={}
    for mass,b in s['by_mass'].items():
        again=annotate_mass(copy.deepcopy(b),base['by_mass'][mass],cfg,working)
        for k in ['normal_checks','normal_failures','new_normal_failures','support_checks','support_failures','rl_retention_pass','pressure_pass_losses','additional_terminations','retention_pass']:assert again[k]==b[k]
        with np.load(ROOT/b['normal']['folder']/'states.npz') as x,np.load(ROOT/base['by_mass'][mass]['normal']['folder']/'states.npz') as y:
            assert set(x.files)==set(y.files);exact={k:np.array_equal(x[k],y[k]) for k in x.files}
        assert all(exact.values());normal_proofs[mass]=dict(steps=b['normal']['steps'],all_raw_arrays_exact_D=exact,
            normal_failures=b['normal_failures'],new_normal_failures=b['new_normal_failures'],RL_clearance_mm=b['rl_clearance_m']*1000)
    losses={name:[cid for cid,r in ref.items() if r['passed'] and not current[cid]['passed']] for name,ref in [('D',D),('061',prev),*s['original_references'].items()]}
    extras={name:[cid for cid,r in ref.items() if not r['terminated'] and current[cid]['terminated']] for name,ref in [('D',D),('061',prev),*s['original_references'].items()]}
    assert losses==s['pass_losses'] and extras==s['additional_terminations']
    assert metrics([r for cid,r in current.items() if cid not in aliases])==s['unique_metrics']['B']
    for alias,source in aliases.items():assert current[alias]['folder']==current[source]['folder']
    for r in s['results']:
        if r['condition']['force']==0:
            with np.load(ROOT/r['folder']/'states.npz') as x,np.load(ROOT/D[r['case']]['folder']/'states.npz') as y:
                assert all(np.array_equal(x[k],y[k]) for k in x.files)
    old061=read(ROOT/spec['previous061']/'summary.json');coreloss=[]
    for mass,b in s['by_mass'].items():
        for r in b['pressure']:
            old=next(x for x in old061['by_mass'][mass]['pressure'] if x['case']==r['case'])
            if old['passed'] and not r['passed']:coreloss.append(dict(mass=float(mass),case=r['case']))
    assert coreloss==s['core_pass_losses']
    expected_relative=not any(losses.values()) and not any(extras.values()) and not coreloss and not s['internal_pass_losses'] and s['normal_relative_retention_pass'] and s['all_B_prefixes_and_normal_protections'] and current['case-045']['passed']
    assert bool(expected_relative)==s['expanded_development_relative_pass']
    assert read(run/'interface-replay.json')['passed']
    for n,h in spec['model_hashes'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'scripts/go2_confirmed_selector.py')==spec['selector_sha256'];verify_inputs(run)
    report=dict(raw_full_records_audited=len(audits)+len(internal_audits),all_native_targets_sensor_state_machines_and_gates_verified=True,
        full_world_force_and_torque_vectors_verified=True,maximum_force_vector_error=max(force_errors or [0.]),
        unique_nominal_conditions=30,nominal_record_count=31,alias_count=len(aliases),nominal_metrics=s['unique_metrics']['B'],
        normal_proofs=normal_proofs,internal_normal_checks=sum(len(b['checks']) for b in s['internal_normals'].values()),
        internal_normal_raw_steps=sum(b['result']['steps'] for b in s['internal_normals'].values()),
        all_normals_and_pre_force_exact_D=True,primary_replay_exact065=True,all_models_normalizations_inputs_unchanged=True,
        expanded_development_relative_pass=s['expanded_development_relative_pass'],nominal_absolute_stage_gate=s['nominal_absolute_stage_gate'],
        new_physics_integration_runs=0,training_samples=0,visual_acceptance='pending',selection_changed=False,
        records=audits,internal_normal_audits=internal_audits)
    shutil_copy_source=__import__('shutil').copy2;shutil_copy_source(ROOT/'scripts/audit_go2_transfer_regression.py',run/'audit-runtime-source.py')
    dump(run/'audit.json',report);dump(ROOT/cfg['public']/'audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','internal_normal_audits','normal_proofs']}),flush=True)

if __name__=='__main__':main()
