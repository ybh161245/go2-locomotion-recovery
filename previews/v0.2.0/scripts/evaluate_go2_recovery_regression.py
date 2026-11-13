"""062 bounded known-condition regression over frozen061 controller."""
import copy,json,shutil,time,traceback
import numpy as np
import torch
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest
from prepare_go2_recovery_regression import case_key
from go2_confirmed_selector import attach_confirmed_selector
from screen_go2_recovery_specialist import load_expert
from screen_go2_mass_course import factory,save_probe,adapter_policy,norm_state,exact
from go2_quality_guard import QualityEvaluator,equal_state
from evaluate_go2_confirmed_selector import trajectory_protection
from stress_go2_frozen import run_case
ACTIVE=None


def metrics(records):
    return dict(samples=len(records),passes=sum(r['passed'] for r in records),terminations=sum(r['terminated'] for r in records),
        mean_peak=float(np.mean([r['peak_tilt'] for r in records])),worst_peak=max(r['peak_tilt'] for r in records))


def main():
    global ACTIVE
    cfg=read(ROOT/'configs/go2-recovery-regression.json');spec=cfg['recovery_regression'];ev.CONFIG=cfg;torch.set_num_threads(1)
    for n,h in spec['models'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'scripts/go2_confirmed_selector.py')==spec['selector_source_sha256']
    assert sha(ROOT/'configs/go2-confirmed-selector.json')==spec['candidate_config_sha256']
    candidate_root=ROOT/spec['candidate_run'];verify_manifest(candidate_root);candidate=read(candidate_root/'summary.json')
    dscreen=read(ROOT/spec['reference_screen']);d={r['case']:r for r in dscreen['results'] if r['variant']==spec['reference_variant']}
    oscreen=read(ROOT/spec['original_screen']);original={v:{r['case']:r for r in oscreen['results'] if r['variant']==v} for v in spec['original_variants']}
    for c in spec['known_cases']:
        assert case_key(c)==case_key(d[c['id']]['condition']) and d[c['id']]['seed']==cfg['development_seed']
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={'configs/go2-recovery-regression.json',cfg['predeclared_plan'],'docs/data/go2-current/selection.json',spec['reference_screen'],spec['original_screen'],*spec['models']}
    names.add('configs/go2-confirmed-selector.json')
    for root in [candidate_root,ROOT/spec['normal_reference']]:names.update(p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file())
    for r in list(d.values())+[r for group in original.values() for r in group.values()]:
        assert r['seed']==cfg['development_seed']
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/r['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs)
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);expert,enorm=load_expert(ROOT/cfg['confirmed_selector']['specialist_train'],131072);adapter=adapter_policy(cfg,joint)
    norms=[joint[1],adapter[1],enorm];stats=[norm_state(n) for n in norms];weights=[{k:v.clone() for k,v in m.policy.state_dict().items()} for m in [joint[0],adapter[0],expert]]
    def make_factory(label):
        plain=factory(joint,dict(id=label,factor='nominal',level=1.,scope='known-recovery-regression'))
        def make(*args,**kwargs):return attach_confirmed_selector(plain(*args,**kwargs),expert,enorm,cfg['confirmed_selector']['selector'])
        return plain,make
    records={};aliases=[];protections={};physical=0;started=time.perf_counter();normal_report=None
    try:
        for c in spec['known_cases']:
            cid=c['id']
            if cid in spec['aliases']:
                origin=spec['aliases'][cid];r=copy.deepcopy(records[origin]);r.update(case=cid,condition=c,reused_from_case=origin)
                aliases.append(dict(case=cid,source_case=origin,reason='Same physical condition/seed; oldD raw arrays exact.'))
            elif cid in spec['reuse_cases']:
                r=copy.deepcopy(next(x for x in candidate['by_mass']['1.00']['pressure'] if x['case']==cid));r['reused_from']='061 nominal same frozen policy/normalization/condition'
            else:
                label='regression-'+cid;plain,make=make_factory(label)
                raw,packed,rows=run_case(c,cfg['development_seed'],make,adapter[0],adapter[1]);raw['condition']=raw.pop('case')
                r=ev.save_run(out,label,cid,cfg['development_seed'],raw,packed,rows);save_probe(plain,ROOT/r['folder']);dump(ROOT/r['folder']/'selector.json',plain.instances[-1].recovery_selection_log);physical+=1
            assert case_key(r['condition'])==case_key(c)
            records[cid]=r;protections[cid]=trajectory_protection(r,d[cid])
            if c['force']==0:
                logs=read(ROOT/r['folder']/'selector.json');equal={}
                with np.load(ROOT/r['folder']/'states.npz') as a,np.load(ROOT/d[cid]['folder']/'states.npz') as b:
                    for k in a.files:equal[k]=bool(a[k].shape==b[k].shape and np.array_equal(a[k],b[k]))
                protections[cid]['zero_force_complete_exact']=all(equal.values()) and all(x['weight']==0 for x in logs)
                protections[cid]['zero_force_arrays_exact']=equal
            dump(out/'progress.json',dict(results=list(records.values()),protections=protections,new_physical_runs=physical))
        assert physical==27
        evaluator=QualityEvaluator(joint[0],cfg);reference=read(ROOT/spec['normal_reference']/'report.json');limits=read(ROOT/spec['normal_reference']/'limits.json')
        normal_proofs={}
        try:
            for c in cfg['quality_guard']['cases']:
                plain,make=make_factory('normal-'+c['id']);evaluator.make_env=lambda seconds=12,make=make:make('external',episode_seconds=seconds)
                r=evaluator.run_case(out,'confirmed-normal',c);save_probe(plain,ROOT/r['folder']);logs=plain.instances[-1].recovery_selection_log;dump(ROOT/r['folder']/'selector.json',logs);physical+=1
                old=reference['results'][c['id']];same={}
                with np.load(ROOT/r['folder']/'states.npz') as a,np.load(ROOT/old['folder']/'states.npz') as b:
                    assert set(a.files)==set(b.files)
                    for k in a.files:same[k]=bool(a[k].shape==b[k].shape and np.array_equal(a[k],b[k]))
                from go2_quality_guard import check_limits
                checks=check_limits(r['metrics'],limits[c['id']])
                normal_proofs[c['id']]=dict(result=r,checks=checks,passed=all(x['passed'] for x in checks),arrays_exact=same,
                    no_expert=all(x['weight']==0 for x in logs),complete_exact_D=all(same.values()) and all(x['weight']==0 for x in logs))
        finally:evaluator.close()
        normal_report=dict(cases=normal_proofs,passed=all(x['passed'] and x['complete_exact_D'] for x in normal_proofs.values()))
        assert all(exact(a,norm_state(n)) for a,n in zip(stats,norms)) and all(equal_state(w,m.policy.state_dict()) for w,m in zip(weights,[joint[0],adapter[0],expert]))
    finally:
        for n in norms:n.close()
    unique=[r for cid,r in records.items() if cid not in spec['aliases']];du=[d[cid] for cid in records if cid not in spec['aliases']]
    losses={'D':[cid for cid,r in d.items() if r['passed'] and not records[cid]['passed']]}
    losses.update({v:[cid for cid,r in group.items() if r['passed'] and not records[cid]['passed']] for v,group in original.items()})
    extra=[cid for cid,r in d.items() if not r['terminated'] and records[cid]['terminated']]
    prefix=all(p['passed'] and p.get('zero_force_complete_exact',True) for p in protections.values())
    regression_pass=not any(losses.values()) and not extra and normal_report['passed'] and prefix
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    summary=dict(config=cfg,results=list(records.values()),aliases=aliases,reference_D=list(d.values()),original_references=original,
        unique_metrics=dict(candidate=metrics(unique),D=metrics(du)),record_metrics=dict(candidate=metrics(list(records.values())),D=metrics(list(d.values()))),
        pass_losses=losses,additional_terminations_vs_D=extra,trajectory_protections=protections,all_pressure_prefixes_exact=prefix,
        normals=normal_report,regression_pass=bool(regression_pass),new_physical_runs=physical,new_pressure_runs=27,new_normal_runs=2,
        reused_pressure_cases=spec['reuse_cases'],training_samples=0,decision='known_regression_pass_requires_original_quality_and_independent_tests' if regression_pass else 'known_regression_losses_require_review_before_foot_change',
        models_and_normalization_unchanged=True,input_hashes_unchanged=True,independent_generalization_claim=False,
        held_out_executed=False,selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    assert physical==29
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['config','results','reference_D','original_references','trajectory_protections','normals']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
