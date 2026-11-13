"""Bounded frozen-policy diagnosis; no training, no adaptive grid or fallback."""
import csv,hashlib,json,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import evaluate_go2_robust as ev
from go2_headroom_env import factory,save_trace
from go2_quality_guard import QualityEvaluator,check_limits
from go2_motion_metrics import LEGS
from go2_transition_protocol import SCHEDULE
from stress_go2_frozen import run_case
ROOT=ev.ROOT
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def support_metrics(result):
    with (ROOT/result['folder']/'steps.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    metrics={}
    for start,end,cmd,phase in SCHEDULE:
        if np.linalg.norm(cmd)==0:continue
        group=[r for r in rows if start<float(r['time'])<=end+1e-8]
        for leg in LEGS:
            contacts=[str(r[leg+'_contact']).lower() in ['1','true','1.0'] for r in group]
            longest=run=0
            for contact in contacts:
                run=0 if contact else run+1;longest=max(longest,run)
            metrics[phase+'/'+leg]=dict(contact_fraction=float(np.mean(contacts)) if contacts else 0.,longest_unloaded_seconds=longest*.02)
    return metrics

def support_checks(result,baseline,spec):
    values,old=support_metrics(result),support_metrics(baseline);checks=[]
    for key,v in values.items():
        lo,hi=spec['moving_contact_fraction'];value=v['contact_fraction']
        checks.append(dict(metric=key+'/contact_fraction',value=value,limit=[lo,hi],passed=lo<=value<=hi))
        limit=max(spec['unloaded_floor_s'],old[key]['longest_unloaded_seconds']+spec['unloaded_increase_s']);value=v['longest_unloaded_seconds']
        checks.append(dict(metric=key+'/longest_unloaded_seconds',value=value,limit=limit,passed=value<=limit))
    return checks

def audit_mapping(record,delta,baseline_physics):
    folder=ROOT/record['folder']
    with np.load(folder/'mapping.npz') as z,np.load(folder/'states.npz') as state,np.load(folder/'physics.npz') as physics:
        assert len(z['raw_mean'])==record['steps']
        np.testing.assert_array_equal(z['clipped_action'],z['last_action'])
        np.testing.assert_allclose(np.clip(z['raw_mean'],-1,1),z['clipped_action'],atol=2e-6,rtol=0)
        residual=np.clip(.35*(z['raw_mean'][:,8].astype(np.float64)+1),-delta,0)
        np.testing.assert_array_equal(z['residual_rad'],residual)
        expected=z['original_targets'].copy();expected[:,8]+=residual
        np.testing.assert_array_equal(expected,z['mapped_targets'])
        expected=np.clip(expected,physics['actuator_ctrlrange'][:,0],physics['actuator_ctrlrange'][:,1])
        np.testing.assert_array_equal(expected,z['actual_targets']);np.testing.assert_array_equal(expected,state['targets'])
        for key in physics.files:np.testing.assert_array_equal(physics[key],baseline_physics[key])
        assert np.isfinite(state['qpos']).all()
        return dict(steps=len(residual),nonzero_steps=int(np.count_nonzero(residual)),min_residual_rad=float(residual.min()),
            mean_absolute_residual_rad=float(np.abs(residual).mean()),all_mapping_equations_verified=True,last_action_unchanged=True,physical_joint_PD_torque_arrays_unchanged=True)

def main():
    torch.set_num_threads(1);cfg=read(ROOT/'configs/go2-action-headroom.json');ev.CONFIG=cfg
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);(out/'source').mkdir()
    for f in (ROOT/'scripts').glob('*.py'):shutil.copy2(f,out/'source'/f.name)
    ev.dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/'docs/experiments/2026-10-25-action-headroom.md',out/'predeclared-plan.md')
    inputs=['configs/go2-action-headroom.json','docs/data/go2-current/selection.json',cfg['reference_screen'],'docs/data/go2-sustained-A/screen.json','docs/data/go2-capability-boundary/summary.json']
    for folder,stem in [(cfg['parent'],cfg['parent_checkpoint']),(cfg['working_stack'],'final'),('20261023-140238-938755-go2-transition-v3-mixed-ppo','final')]:
        for name in ['config.json',('final-model.zip' if stem=='final' else stem+'.zip'),stem+'-vecnormalize.pkl']:inputs.append('artifacts/runs/'+folder+'/'+name)
    reference=read(ROOT/cfg['reference_screen']);old=reference['normals'][cfg['reference_variant']]
    inputs += [old['folder']+'/'+name for name in ['states.npz','steps.csv','result.json']]
    inputs += [cfg['quality_guard']['reference_folder']+'/'+name for name in ['limits.json','report.json','acceptance.json','manifest.json','model-inputs.json']]
    inputs += [p.relative_to(ROOT).as_posix() for p in (ROOT/'third_party/unitree_go2').rglob('*') if p.is_file()]
    if (ROOT/'requirements-lock.txt').exists():inputs.append('requirements-lock.txt')
    hashes={n:sha(ROOT/n) for n in inputs};ev.dump(out/'input-sha256.json',hashes)
    ev.dump(out/'preflight-support.json',support_checks(old,old,cfg['support']));assert all(c['passed'] for c in support_checks(old,old,cfg['support']))
    for n,h in read(ROOT/cfg['quality_guard']['reference_folder']/'manifest.json').items():assert sha(ROOT/cfg['quality_guard']['reference_folder']/n)==h
    start=time.perf_counter();joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint'])
    work=ROOT/'artifacts/runs'/cfg['working_stack'];adapter_model=PPO.load(work/'final-model.zip',device='cpu')
    adapter_norm=VecNormalize.load(work/'final-vecnormalize.pkl',DummyVecEnv([lambda:ev.stack_factory(joint)('external')]))
    adapter_norm.training=False;adapter_norm.norm_reward=False;adapter=(adapter_model,adapter_norm)
    stack_cfg=read(ROOT/'artifacts/runs'/cfg['working_stack']/'config.json');normalbase=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
    reports=[];normals={};audits={};pressure=[];internal=None
    initial_norms=[(n.obs_rms.mean.copy(),n.obs_rms.var.copy(),n.obs_rms.count) for n in [joint[1],adapter[1]]]
    grid=np.linspace(-3,3,1201)
    for d in cfg['headroom_rad']:
        r=np.clip(.35*(grid+1),-d,0);assert (r>=-d).all() and (r<=0).all() and np.all(r[grid>=-1]==0)
    baseline_physics=None
    for d in cfg['headroom_rad']:
        label=f'delta-{round(d*1000):03d}mrad';cls=factory(stack_cfg,joint,d)
        record=ev.normal_run(out,label,cfg['normal_seed'],cls,adapter);save_trace(cls,ROOT/record['folder']);normals[label]=record
        if d==0:
            with np.load(ROOT/record['folder']/'states.npz') as current,np.load(ROOT/old['folder']/'states.npz') as previous:
                assert set(current.files)==set(previous.files)
                for key in current.files:np.testing.assert_array_equal(current[key],previous[key])
            with np.load(ROOT/record['folder']/'physics.npz') as z:baseline_physics={k:z[k].copy() for k in z.files}
            ev.dump(out/'zero-identity.json',dict(reference=old['folder'],steps=record['steps'],all_saved_state_arrays_exact=True))
        audits[label]=audit_mapping(record,d,baseline_physics)
        checks=ev.normal_checks(record,normalbase);supports=support_checks(record,old,cfg['support'])
        rl=record['segments'].get('reverse right',{}).get('RL',{}).get('clearance_p95_m');oldrl=old['segments']['reverse right']['RL']['clearance_p95_m']
        known=cfg['normal_known_failure'];newfails=[c for c in checks if not c['passed'] and not(c['segment']==known['segment'] and c['metric']==known['metric'])]
        gain=rl-oldrl if rl is not None else None
        signal=bool(d>0 and gain is not None and gain>=cfg['minimum_gain_m'] and not newfails and all(c['passed'] for c in supports))
        report=dict(variant=label,delta_rad=d,rl_clearance_m=rl,gain_m=gain,normal_checks=checks,support_checks=supports,
            normal_failures=sum(not c['passed'] for c in checks),new_normal_failures=len(newfails),support_failures=sum(not c['passed'] for c in supports),screen_signal=signal)
        reports.append(report);ev.dump(out/'progress.json',dict(ranking=reports,normals=normals,mapping_audits=audits))
        print(json.dumps({k:v for k,v in report.items() if k not in ['normal_checks','support_checks']}),flush=True)
    selected=next((r for r in reports if r['screen_signal']),None);diagnosis_pass=False
    if selected:
        label=selected['variant'];cls=factory(stack_cfg,joint,selected['delta_rad']);evaluator=QualityEvaluator(joint[0],cfg)
        evaluator.make_env=lambda seconds=12:cls('external',episode_seconds=seconds)
        limits=read(ROOT/cfg['quality_guard']['reference_folder']/'limits.json');cases={};checks={}
        for case in cfg['quality_guard']['cases']:
            result=evaluator.run_case(out,label,case);save_trace(cls,ROOT/result['folder']);audits[label+'/'+case['id']]=audit_mapping(result,selected['delta_rad'],baseline_physics)
            cases[case['id']]=result;checks[case['id']]=check_limits(result['metrics'],limits[case['id']])
        evaluator.close();internal=dict(results=cases,checks=checks,passed=all(c['passed'] for group in checks.values() for c in group));ev.dump(out/'internal.json',internal)
        if internal['passed']:
            conditions=ev.development_cases()
            for force in [20,25,30]:
                for sign in [1,-1]:
                    for offset in [0,.2,.4]:conditions.append(dict(id=f'F{force}-D0.6-S{sign}-O{offset}',phase='straight',force=force,direction=[0,sign],duration=.6,pulses=[6+offset],torque=0,friction=None))
            for case in conditions:
                result,packed,rows=run_case(case,cfg['development_seed'],cls,adapter[0],adapter[1]);result['condition']=result.pop('case')
                record=ev.save_run(out,label,case['id'],cfg['development_seed'],result,packed,rows);save_trace(cls,ROOT/record['folder'])
                audits[label+'/'+case['id']]=audit_mapping(record,selected['delta_rad'],baseline_physics);pressure.append(record)
            current={r['case']:r for r in pressure};previous=[r for r in reference['results'] if r['variant']==cfg['reference_variant']]
            previous += [r for r in read(ROOT/'docs/data/go2-sustained-A/screen.json')['results'] if r['variant']=='previous-stack']
            previous += [r for r in read(ROOT/'docs/data/go2-capability-boundary/summary.json')['results'] if r['variant']=='research' and r['condition']['force'] in [20,30] and r['condition']['duration']==.6]
            losses=sorted({r['case'] for r in previous if r['passed'] and not current[r['case']]['passed']})
            target=[r for r in pressure if r['case'].startswith('F30-D0.6-S1-')];diagnosis_pass=not losses and not any(r['terminated'] for r in target)
            selected.update(pressure_pass_losses=losses,target_passes=sum(r['passed'] for r in target),target_terminations=sum(r['terminated'] for r in target))
    for n,h in hashes.items():assert sha(ROOT/n)==h,n
    for n,previous in zip([joint[1],adapter[1]],initial_norms):
        for value,expected in zip([n.obs_rms.mean,n.obs_rms.var,n.obs_rms.count],previous):np.testing.assert_array_equal(value,expected)
    joint[1].close();adapter[1].close()
    summary=dict(config=cfg,ranking=reports,normals=normals,selected=selected,internal=internal,pressure=pressure,mapping_audits=audits,
        diagnostic_pass=bool(diagnosis_pass),branch_status='one_confirmation_eligible' if diagnosis_pass else 'closed_no_qualified_signal',training_steps_added=0,
        physical_runs=len(normals)+(len(internal['results']) if internal else 0)+len(pressure),zero_dose_exact=True,input_hashes_unchanged=True,
        normalization_unchanged=True,working_selection_changed=False,held_out_executed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-start)
    ev.dump(out/'summary.json',summary);ev.dump(public/'summary.json',summary);ev.dump(public/'input-sha256.json',hashes)
    ev.dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(dict(diagnostic_pass=diagnosis_pass,physical_runs=summary['physical_runs'],seconds=summary['elapsed_seconds'])),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        folder=ROOT/'artifacts/runs/20261025-action-headroom'
        if folder.exists() and not (folder/'failure.txt').exists():(folder/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
