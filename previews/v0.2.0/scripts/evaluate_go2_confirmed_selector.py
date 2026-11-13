"""061 complete fixed evaluation: no learned tensor or handback changes."""
import copy,json,shutil,time,traceback
import numpy as np
import torch
import evaluate_go2_robust as ev
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest,pressure_metrics,benefit_signal
from prepare_go2_confirmed_selector import verify_single_change
from screen_go2_recovery_specialist import load_expert
from screen_go2_mass_course import factory,save_probe,conditions,annotate_mass,adapter_policy,norm_state,exact
from go2_confirmed_selector import attach_confirmed_selector
from go2_quality_guard import equal_state
from stress_go2_frozen import run_case
ROOT=ev.ROOT
ACTIVE=None


def trajectory_protection(record,reference,normal=False):
    ls=read(ROOT/record['folder']/'selector.json')
    limit=record['steps'] if normal else min(record['steps'],round(min(record['condition']['pulses'])/.02))
    no_expert=all(x['weight']==0 for x in ls[:limit]);equal={}
    with np.load(ROOT/record['folder']/'states.npz') as a,np.load(ROOT/reference['folder']/'states.npz') as b:
        keys=['qpos','qvel','targets','observations','actions','world_wrench']
        if normal:keys+=['feet_world','heading']
        for k in keys:equal[k]=bool(a[k][:limit].shape==b[k][:limit].shape and np.array_equal(a[k][:limit],b[k][:limit]))
    return dict(no_expert_before_limit=no_expert,limit_control_steps=limit,arrays_exact=equal,passed=bool(no_expert and all(equal.values())),
        expert_weight_steps=sum(x['weight']>0 for x in ls),first_expert_step=next((i for i,x in enumerate(ls) if x['weight']>0),None))


def main():
    global ACTIVE
    cfg=read(ROOT/'configs/go2-confirmed-selector.json');spec=cfg['confirmed_selector'];ev.CONFIG=cfg;torch.set_num_threads(1)
    assert verify_single_change()==spec['change_proof']
    for n,h in spec['model_hashes'].items():assert sha(ROOT/n)==h,n
    baseline=ROOT/spec['baseline'];priorroot=ROOT/spec['prior_screen'];verify_manifest(baseline);verify_manifest(priorroot)
    reference=read(baseline/'summary.json');prior=read(priorroot/'summary.json');out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={'configs/go2-confirmed-selector.json',cfg['predeclared_plan'],'docs/data/go2-current/selection.json','docs/data/go2-sustained-A/screen.json',*spec['model_hashes']}
    for root in [baseline,priorroot,ROOT/spec['followup']]:names.update(p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file())
    for bundles in [reference['by_mass'],prior['by_mass']]:
        for b in bundles.values():
            for r in [b['normal']]+b['pressure']:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/r['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs)
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);expert,enorm=load_expert(ROOT/spec['specialist_train'],spec['specialist_checkpoint']);adapter=adapter_policy(cfg,joint)
    norms=[joint[1],adapter[1],enorm];before=[norm_state(n) for n in norms]
    weights=[{k:v.clone() for k,v in m.policy.state_dict().items()} for m in [joint[0],adapter[0],expert]]
    def make_factory(scale,label):
        plain=factory(joint,dict(id=label,factor='nominal' if scale==1 else 'mass',level=scale,scope='confirmed-lateral-entry'))
        def make(*args,**kwargs):return attach_confirmed_selector(plain(*args,**kwargs),expert,enorm,spec['selector'])
        return plain,make
    def finish_probe(plain,r):
        a=save_probe(plain,ROOT/r['folder']);dump(ROOT/r['folder']/'selector.json',plain.instances[-1].recovery_selection_log);return a
    by_mass={};protections={};actual=0;started=time.perf_counter();follow=dict(executed=False,results=[],reason='Core qualification not achieved.',new_physical_runs=0)
    try:
        working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
        for mass in cfg['mass_scales']:
            key=f'{mass:.2f}';label='confirmed-mass-'+key;plain,make=make_factory(mass,label)
            normal=ev.normal_run(out,label,cfg['normal_seed'],make,adapter);normal_audit=finish_probe(plain,normal);actual+=1
            protections[normal['folder']]=trajectory_protection(normal,reference['by_mass'][key]['normal'],True);pressure=[];audits={'normal':normal_audit}
            for c in conditions():
                r,packed,rows=run_case(c,cfg['development_seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
                r=ev.save_run(out,label,c['id'],cfg['development_seed'],r,packed,rows);audits[c['id']]=finish_probe(plain,r);actual+=1
                old=next(x for x in reference['by_mass'][key]['pressure'] if x['case']==r['case'])
                protections[r['folder']]=trajectory_protection(r,old);pressure.append(r)
            by_mass[key]=annotate_mass(dict(normal=normal,pressure=pressure,physics_audits=audits),reference['by_mass'][key],cfg,working)
            dump(out/'progress.json',dict(by_mass=by_mass,protections=protections,new_physical_runs=actual))
        assert actual==12
        metrics=pressure_metrics(by_mass);retention=all(b['retention_pass'] for b in by_mass.values());prefix=all(p['passed'] for p in protections.values())
        primary=bool(retention and prefix and metrics['passes']==9 and metrics['terminations']==0)
        if primary:
            oldroot=ROOT/spec['followup'];old=read(oldroot/'summary.json');previous={r['case']:r for r in old['results']};results=[];losses=[];extra=[]
            for item in old['config']['internal_pressure']['cases']:
                mass=item['mass_scale'];c=item['condition'];label='old-development-'+c['id'];plain,make=make_factory(mass,label)
                r,packed,rows=run_case(c,item['seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
                r=ev.save_run(out,label,c['id'],item['seed'],r,packed,rows);finish_probe(plain,r);results.append(r);actual+=1
                protections[r['folder']]=trajectory_protection(r,previous[r['case']])
                if previous[r['case']]['passed'] and not r['passed']:losses.append(r['case'])
                if not previous[r['case']]['terminated'] and r['terminated']:extra.append(r['case'])
            follow=dict(executed=True,results=results,pass_losses=losses,additional_terminations=extra,
                passed=not losses and not extra and all(protections[r['folder']]['passed'] for r in results),new_physical_runs=6,
                independent_generalization_claim=False)
        assert all(exact(a,norm_state(n)) for a,n in zip(before,norms))
        assert all(equal_state(w,m.policy.state_dict()) for w,m in zip(weights,[joint[0],adapter[0],expert]))
    finally:
        for n in norms:n.close()
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    oldpassed={(m,r['case']) for m,b in prior['by_mass'].items() for r in b['pressure'] if r['passed']}
    priorloss=[dict(mass=m,case=r['case']) for m,b in by_mass.items() for r in b['pressure'] if (m,r['case']) in oldpassed and not r['passed']]
    decision=('frozen_expert_entry_development_qualified_requires_stage_and_independent_tests' if primary and follow.get('passed') else
        'core_fixed_but_old_development_retention_failed' if primary else 'entry_change_did_not_preserve_all_primary_conditions')
    summary=dict(config=cfg,change_proof=spec['change_proof'],by_mass=by_mass,reference=reference,prior060=prior,
        pressure_metrics=metrics,benefit_vs_D=bool(benefit_signal(metrics,pressure_metrics(reference['by_mass']),cfg['benefit'])),
        retention_pass=bool(retention),trajectory_protections=protections,all_core_normal_and_pre_push_exact_D=bool(prefix),
        primary_pass=primary,prior060_pressure_pass_losses=priorloss,followup=follow,decision=decision,
        new_physical_runs=actual,training_samples=0,models_and_normalization_unchanged=True,input_hashes_unchanged=True,
        nominal_absolute_stage_pass=by_mass['1.00']['normal_failures']==0 and all(r['passed'] for r in by_mass['1.00']['pressure']),
        held_out_executed=False,selection_changed=False,independent_generalization_claim=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['config','by_mass','reference','prior060','trajectory_protections','followup']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
