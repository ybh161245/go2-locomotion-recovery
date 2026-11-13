"""107 fifteen actual regressions; all original quality limits and complete raw references retained."""
import copy,json,shutil,time,traceback
import numpy as np
import torch
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from screen_go2_mass_course import factory,save_probe,adapter_policy,norm_state,exact
from screen_go2_recovery_specialist import load_expert
from go2_unified_recovery_selection import attach_unified_recovery_selection
from go2_release_regression import attach_release_capture
from go2_traction_protocol import attach_traction_schedule,save_traction_probe,friction_event_metrics
from validate_go2_context_states import attach_initial_state,run_transient
from audit_go2_recovery_screen import rows
from go2_quality_guard import QualityEvaluator,check_limits,equal_state
CONFIG='configs/go2-release-regression.json';ACTIVE=None


def raw_equal(a,b,cut=None):
    with np.load(ROOT/a['folder']/'states.npz') as x,np.load(ROOT/b['folder']/'states.npz') as y:return bool(set(x.files)==set(y.files) and all(np.array_equal(x[k][:cut],y[k][:cut]) for k in x.files))


def main():
    global ACTIVE
    cfg=read(ROOT/CONFIG);ev.CONFIG=cfg;task=cfg['release_regression'];spec=cfg['support_context'];setting=cfg['unified_recovery_selection'];torch.set_num_threads(1)
    for n,h in task['runtime_sha256'].items():assert sha(ROOT/n)==h,n
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');descriptor=read(ROOT/'docs/data/go2-support-context-boundary/research-candidate.json');names={CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json',*descriptor['models'],'artifacts/runs/'+cfg['working_stack']+'/config.json',spec['default_train']+'/acceptance.json',spec['low_train']+'/acceptance.json',task['normal_reference'],task['internal_limits'],*task['runtime_sha256']}
    for r in [x['reference'] for x in task['quality']]+[r for p in task['pressure_cases'] for r in [p['push_reference'],p['zero_reference']]]:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/r['folder']).iterdir() if p.is_file())
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/setting['bank']).rglob('*') if p.is_file());inputs={n:sha(ROOT/n) for n in names};dump(out/'input-sha256.json',inputs)
    with np.load(ROOT/setting['bank']/'bank.npz') as z:bank=z['prototypes'].copy()
    D=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);formal=ev.load_policy(ROOT/cfg['motion_composition']['formal_joint']);default=load_expert(ROOT/spec['default_train'],spec['default_steps']);low=load_expert(ROOT/spec['low_train'],spec['low_steps']);adapter=adapter_policy(cfg,D);quality=QualityEvaluator(D[0],cfg);policies=[D,formal,default,low,adapter,(quality.adapter,quality.adapter_norm)];stats=[norm_state(n) for m,n in policies];weights=[copy.deepcopy(m.policy.state_dict()) for m,n in policies];records=[];qualities=[];comparisons=[];zero_pairs=[];start=time.perf_counter();working=read(ROOT/task['normal_reference'])['normals']['working'];limits=read(ROOT/task['internal_limits'])
    def make_factory(label,protocol=None,profile=None):
        plain=factory(D,dict(id=label,factor='friction' if protocol else 'nominal',level=protocol['friction'] if protocol else 1.,scope='frozen-release-regression'))
        def make(*args,**kwargs):
            env=attach_unified_recovery_selection(plain(*args,**kwargs),*formal,default[0],low[0],default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],bank,setting)
            if protocol:env=attach_initial_state(attach_traction_schedule(env,protocol),profile)
            return attach_release_capture(env)
        return plain,make
    def finish(plain,r,kind,reference,profile=None,protocol=None):
        folder=ROOT/r['folder'];env=plain.instances[-1]
        if protocol:save_traction_probe(plain,folder,protocol)
        else:save_probe(plain,folder)
        logs=env.recovery_selection_log;dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial);np.savez_compressed(folder/'adapter-io.npz',**{k:np.asarray(v) for k,v in env.release_io.items()});r.update(controller='original098',kind=kind,profile=profile or dict(id='nominal',seed=r['seed'],rpy_deg=[0,0,0],body_velocity=[0,0,0]),zero_force=kind!='pressure',mass_scale=1.,mode='routed',mass_kg=float(env.model.body_mass.sum()),force_impulse_norm_Ns=r.get('force_impulse_norm_Ns',0.),low_expert_steps=sum(l['actual_expert']=='low' for l in logs),default_expert_steps=sum(l['actual_expert']=='default' for l in logs),selection_entries=[dict(step=i,choice=l['selection_choice'],distance=l['selection_distance']) for i,l in enumerate(logs) if l['selection_entry']])
        if protocol:r['friction_event_recoveries']=friction_event_metrics(rows(folder),protocol,kind=='zero')
        same=raw_equal(r,reference);records.append(r);comparisons.append(dict(result=r,reference=reference,raw_all_arrays_exact=same));dump(out/'progress.json',records);print('RELEASE '+json.dumps({k:r.get(k) for k in ['case','seed','kind','passed','peak_tilt','recovery_seconds','steps','terminated','low_expert_steps']}|dict(raw_all_arrays_exact=same)),flush=True);return r
    try:
        for entry in task['quality']:
            ref=entry['reference'];kind=entry['kind'];label='release-'+kind;plain,make=make_factory(label)
            if kind=='normal':r=ev.normal_run(out,label,ref['seed'],make,adapter)
            else:
                case=next(c for c in cfg['quality_guard']['cases'] if c['id']==ref['case']);quality.make_env=lambda seconds=12,make=make:make('external',episode_seconds=seconds);r=quality.run_case(out,label,case)
            r=finish(plain,r,kind,ref);checks=ev.normal_checks(r,working) if kind=='normal' else check_limits(r['metrics'],limits[r['case']]);rl=r['segments']['reverse right']['RL']['clearance_p95_m']*1000 if kind=='normal' else None;passed=all(c['passed'] for c in checks) and (rl is None or rl>=task['RL_min_mm']);qualities.append(dict(result=r,reference=ref,checks=checks,RL_reverse_right_mm=rl,passed=bool(passed)))
        for case in task['pressure_cases']:
            protocol=case['protocol'];profile=case['profile'];own=[]
            for zero in [True,False]:
                label='release'+('-zero' if zero else '-push');plain,make=make_factory(label,protocol,profile);r=run_transient(out,label,protocol,profile['seed'],make,adapter,zero);ref=case['zero_reference'] if zero else case['push_reference'];r=finish(plain,r,'zero' if zero else 'pressure',ref,profile,protocol);own.append(r)
            zero_pairs.append(dict(zero=own[0],push=own[1],prefix_steps=round(protocol['events'][0]['start']/.02),preforce_exact=raw_equal(own[0],own[1],round(protocol['events'][0]['start']/.02))))
        assert all(exact(a,norm_state(n)) for a,(m,n) in zip(stats,policies)) and all(equal_state(w,m.policy.state_dict()) for w,(m,n) in zip(weights,policies))
    finally:
        quality.close()
        for m,n in policies[:5]:n.close()
    pressure=[r for r in records if r['kind'] in ['zero','pressure']];qualified=bool(len(records)==15 and all(q['passed'] for q in qualities) and all(r['passed'] for r in pressure) and all(c['raw_all_arrays_exact'] for c in comparisons) and all(p['preforce_exact'] for p in zero_pairs) and all(w['passed'] for r in pressure if r['kind']=='zero' for w in r['friction_event_recoveries']));s=dict(config=cfg,records=records,quality=qualities,comparisons=comparisons,zero_pairs=zero_pairs,qualified=qualified,qualification_scope='107 finite opened regression only;not broad stage4 or formal adoption',new_physical_runs=15,new_normal_46s_runs=3,new_internal_runs=2,new_zero_runs=5,new_pressure_runs=5,conditionally_reused_runs=0,training_samples=0,PPO_updates=0,positive_memory_entries_added=0,negative_memory_entries_added=0,formal_selection_changed=False,old_qualifications_unchanged=True,visual_acceptance='pending_for_new107_full_regression_videos',scope=task['scope'],elapsed_seconds=time.perf_counter()-start)
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    dump(out/'summary.json',s);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()});public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',s);print(json.dumps(dict(qualified=qualified,new_physical_runs=15,quality_checks=sum(len(q['checks']) for q in qualities),quality_passes=sum(q['passed'] for q in qualities),pressure_and_zero_passes=sum(r['passed'] for r in pressure),raw_exact=sum(c['raw_all_arrays_exact'] for c in comparisons))),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
