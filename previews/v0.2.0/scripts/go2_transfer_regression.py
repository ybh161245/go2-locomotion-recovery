"""066 frozen B wider development; conditional A probes never imply global A qualification."""
import argparse,copy,json,shutil,time,traceback
import numpy as np
import torch
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest
from screen_go2_recovery_specialist import load_expert
from screen_go2_mass_course import factory,save_probe,adapter_policy,conditions,annotate_mass,norm_state,exact
from go2_confirmed_selector import attach_confirmed_selector
from stress_go2_frozen import run_case
from evaluate_go2_confirmed_selector import trajectory_protection
from evaluate_go2_recovery_regression import metrics
from go2_quality_guard import QualityEvaluator,check_limits,equal_state
from prepare_go2_recovery_regression import case_key
CONFIG='configs/go2-transfer-expanded-regression.json'
ACTIVE=None

def prepare():
    cfg=read(ROOT/'configs/go2-action-transfer-objective.json');candidates=read(ROOT/cfg['public']/'research-candidates.json')['candidates']
    cfg.update(experiment_id='GO2-B-261026-066',run='artifacts/runs/20261026-transfer-expanded-regression',
        public='docs/data/go2-transfer-expanded-regression',predeclared_plan='docs/experiments/2026-10-26-transfer-expanded-regression.md')
    cfg['transfer_regression']=dict(candidate_run='artifacts/runs/20261026-action-transfer-screen',
        train='artifacts/runs/20261026-action-transfer-train',previous_regression=cfg['early_recovery']['regression'],
        previous061='artifacts/runs/20261026-confirmed-selector',candidates=candidates,primary='B',comparison='A',
        max_primary_runs=43,max_A_runs=12,selector_sha256=sha(ROOT/'scripts/go2_confirmed_selector.py'),
        model_hashes=copy.deepcopy(cfg['action_transfer']['models']))
    for candidate in candidates.values():
        cfg['transfer_regression']['model_hashes'][candidate['model']]=candidate['model_sha256']
        cfg['transfer_regression']['model_hashes'][candidate['normalization']]=candidate['normalization_sha256']
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);print(CONFIG)

def verify_inputs(out):
    for n,h in read(out/'input-sha256.json').items():assert sha(ROOT/n)==h,n

def evaluate():
    global ACTIVE
    cfg=read(ROOT/CONFIG);spec=cfg['transfer_regression'];ev.CONFIG=cfg;torch.set_num_threads(1)
    for n,h in spec['model_hashes'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'scripts/go2_confirmed_selector.py')==spec['selector_sha256']
    oldroot=ROOT/spec['candidate_run'];verify_manifest(oldroot);prior=read(oldroot/'summary.json')
    reg=read(ROOT/spec['previous_regression']/'summary.json');old061=read(ROOT/spec['previous061']/'summary.json')
    base=read(ROOT/cfg['recovery_specialist']['normal_reference']/'summary.json');d={r['case']:r for r in reg['reference_D']};previous={r['case']:r for r in reg['results']}
    protocols=reg['config']['recovery_regression'];aliases=protocols['aliases'];known=protocols['known_cases']
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json',*spec['model_hashes']}
    for root in [oldroot,ROOT/spec['previous_regression'],ROOT/spec['previous061'],ROOT/cfg['recovery_specialist']['normal_reference'],ROOT/cfg['quality_guard']['reference_folder'],ROOT/cfg['confirmed_selector']['followup']]:
        names.update(p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file())
    for arm in ['A','B']:
        for r in prior['arms'][arm]['full_pressure']+[prior['arms'][arm]['normal']]:
            names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/r['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs)
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);adapter=adapter_policy(cfg,joint)
    experts={arm:load_expert(ROOT/spec['train']/arm,c['supervised_updates']) for arm,c in spec['candidates'].items()}
    policies=[joint,adapter,*experts.values()];norms=[n for m,n in policies];weights=[copy.deepcopy(m.policy.state_dict()) for m,n in policies];statistics=[norm_state(n) for n in norms]
    physical=dict(B=0,A=0);started=time.perf_counter();proofs={};audit_references={};records={};reuse=[]
    def make_factory(arm,mass,label):
        plain=factory(joint,dict(id=label,factor='nominal' if mass==1 else 'mass',level=mass,scope='transfer-expanded-development'))
        def make(*args,**kwargs):return attach_confirmed_selector(plain(*args,**kwargs),*experts[arm],cfg['confirmed_selector']['selector'])
        return plain,make
    def finish(arm,plain,r):
        save_probe(plain,ROOT/r['folder']);dump(ROOT/r['folder']/'selector.json',plain.instances[-1].recovery_selection_log);physical[arm]+=1
    def pressure(arm,c,mass,seed,label):
        plain,make=make_factory(arm,mass,label);r,z,rows=run_case(c,seed,make,adapter[0],adapter[1]);r['condition']=r.pop('case')
        r=ev.save_run(out,label,c['id'],seed,r,z,rows);finish(arm,plain,r);return r
    def protect(r,ref,mass,normal=False):
        proof=trajectory_protection(r,ref,normal);proofs[r['folder']]=proof;audit_references[r['folder']]=dict(reference=ref,mass=mass,normal=normal)
        return proof
    try:
        focus=next(c for c in known if c['id']=='case-045');new=pressure('B',focus,1.,cfg['development_seed'],'B-replay')
        old=next(r for r in prior['arms']['B']['full_pressure'] if r['case']=='case-045');same={}
        with np.load(ROOT/new['folder']/'states.npz') as x,np.load(ROOT/old['folder']/'states.npz') as y:
            assert set(x.files)==set(y.files)
            for k in x.files:same[k]=np.array_equal(x[k],y[k])
        assert all(same.values()) and read(ROOT/new['folder']/'selector.json')==read(ROOT/old['folder']/'selector.json')
        assert all(new[k]==old[k] for k in ['steps','terminated','peak_tilt','recovery_seconds','passed'])
        dump(out/'interface-replay.json',dict(case='case-045',arrays_exact=same,selector_log_exact=True,metrics_exact=True,passed=True))
        records['case-045']=new;old_pressure={r['case']:r for r in prior['arms']['B']['full_pressure']}
        for c in known:
            cid=c['id']
            if cid=='case-045':r=records[cid]
            elif cid in aliases:
                r=copy.deepcopy(records[aliases[cid]]);r.update(case=cid,condition=c,reused_from_case=aliases[cid])
            elif cid in old_pressure:
                r=copy.deepcopy(old_pressure[cid]);r['reused_from']='065 same frozen B/condition/seed';reuse.append(dict(kind='pressure',case=cid,folder=r['folder']))
            else:r=pressure('B',c,1.,cfg['development_seed'],'B-'+cid)
            assert case_key(c)==case_key(r['condition']) and r['seed']==cfg['development_seed'];records[cid]=r;proof=protect(r,d[cid],1.)
            if c['force']==0:
                with np.load(ROOT/r['folder']/'states.npz') as x,np.load(ROOT/d[cid]['folder']/'states.npz') as y:
                    proof['zero_force_complete_exact']=set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            dump(out/'progress.json',dict(results=list(records.values()),new_physical_runs=physical))
        working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working'];by_mass={}
        for mass in cfg['mass_scales']:
            key=f'{mass:.2f}';label='B-mass-'+key
            if mass==1.:
                normal=copy.deepcopy(prior['arms']['B']['normal']);normal['reused_from']='065 same frozen B/normal protocol/seed';reuse.append(dict(kind='normal',mass=mass,folder=normal['folder']))
            else:
                plain,make=make_factory('B',mass,label);normal=ev.normal_run(out,label,cfg['normal_seed'],make,adapter);finish('B',plain,normal)
            protect(normal,base['by_mass'][key]['normal'],mass,True);prs=[]
            for c in conditions():
                r=records[c['id']] if mass==1. else pressure('B',c,mass,cfg['development_seed'],label)
                prs.append(r);protect(r,next(x for x in base['by_mass'][key]['pressure'] if x['case']==r['case']),mass)
            by_mass[key]=annotate_mass(dict(normal=normal,pressure=prs),base['by_mass'][key],cfg,working)
        intref=read(ROOT/cfg['confirmed_selector']['followup']/'summary.json');internal=[];internal_conditions={}
        for item in intref['config']['internal_pressure']['cases']:
            c=item['condition'];r=pressure('B',c,item['mass_scale'],item['seed'],'B-old-'+c['id']);internal.append(r);internal_conditions[r['case']]=item
            protect(r,next(x for x in intref['results'] if x['case']==r['case']),item['mass_scale'])
        evaluator=QualityEvaluator(joint[0],cfg);normalref=read(ROOT/cfg['quality_guard']['reference_folder']/'report.json');limits=read(ROOT/cfg['quality_guard']['reference_folder']/'limits.json');internal_normals={}
        try:
            for c in cfg['quality_guard']['cases']:
                plain,make=make_factory('B',1.,'B-normal-'+c['id']);evaluator.make_env=lambda seconds=12,make=make:make('external',episode_seconds=seconds)
                r=evaluator.run_case(out,'B-internal-normal',c);finish('B',plain,r);old=normalref['results'][c['id']];same={}
                with np.load(ROOT/r['folder']/'states.npz') as x,np.load(ROOT/old['folder']/'states.npz') as y:
                    assert set(x.files)==set(y.files)
                    for k in x.files:same[k]=np.array_equal(x[k],y[k])
                checks=check_limits(r['metrics'],limits[c['id']]);logs=read(ROOT/r['folder']/'selector.json');noexpert=all(l['weight']==0 for l in logs)
                internal_normals[c['id']]=dict(result=r,checks=checks,arrays_exact=same,no_expert=noexpert,
                    passed=all(t['passed'] for t in checks) and all(same.values()) and noexpert)
        finally:evaluator.close()
        assert physical['B']==43
        # Build protected-loss A probe list in the exact predeclared context order.
        losses={name:[cid for cid,r in ref.items() if r['passed'] and not records[cid]['passed']] for name,ref in [('D',d),('061',previous),*reg['original_references'].items()]}
        extras={name:[cid for cid,r in ref.items() if not r['terminated'] and records[cid]['terminated']] for name,ref in [('D',d),('061',previous),*reg['original_references'].items()]}
        core_losses=[];internal_losses=[];probes=[]
        def add_probe(c,mass,seed,current,ref,context):
            if (ref['passed'] and not current['passed']) or (not ref['terminated'] and current['terminated']):
                probes.append(dict(condition=c,mass_scale=mass,seed=seed,B=current,reference=ref,context=context))
        for mass in [.95,1.05,1.]:
            key=f'{mass:.2f}'
            for c in conditions():
                r=next(x for x in by_mass[key]['pressure'] if x['case']==c['id']);ref=next(x for x in old061['by_mass'][key]['pressure'] if x['case']==c['id'])
                if ref['passed'] and not r['passed']:core_losses.append(dict(mass=mass,case=c['id']))
                add_probe(c,mass,cfg['development_seed'],r,ref,'core')
        protected_nominal=set(cid for group in list(losses.values())+list(extras.values()) for cid in group)
        for c in [focus]+[c for c in known if c['id']!='case-045' and c['id'] not in aliases]:
            if c['id'] in protected_nominal:probes.append(dict(condition=c,mass_scale=1.,seed=cfg['development_seed'],B=records[c['id']],reference=previous[c['id']],context='nominal'))
        previous_internal={r['case']:r for r in old061['followup']['results']}
        for r in internal:
            item=internal_conditions[r['case']];ref=previous_internal[r['case']]
            if ref['passed'] and not r['passed']:internal_losses.append(r['case'])
            add_probe(item['condition'],item['mass_scale'],item['seed'],r,ref,'internal')
        distinct=[];seen=set()
        for item in probes:
            key=(item['mass_scale'],item['seed'],case_key(item['condition']))
            if key not in seen:distinct.append(item);seen.add(key)
        fallback=[];oldA={r['case']:r for r in prior['arms']['A']['full_pressure']}
        for item in distinct[:spec['max_A_runs']]:
            c=item['condition'];cid=c['id']
            if item['mass_scale']==1. and item['seed']==cfg['development_seed'] and cid in oldA:
                r=copy.deepcopy(oldA[cid]);r['reused_from']='065 frozen A same condition'
            else:r=pressure('A',c,item['mass_scale'],item['seed'],'A-probe-'+item['context']+'-'+cid+'-'+str(item['mass_scale']))
            fallback.append(dict(**item,A=r));protect(r,item['reference'],item['mass_scale'])
        unique=[r for cid,r in records.items() if cid not in aliases]
        normal_relative=all(b['retention_pass'] and not b['new_normal_failures'] for b in by_mass.values()) and all(x['passed'] for x in internal_normals.values())
        prefix=all(p['passed'] and p.get('zero_force_complete_exact',True) for p in proofs.values())
        Bprefix=all(proofs[r['folder']]['passed'] for r in unique+[b['normal'] for b in by_mass.values()]+[r for b in by_mass.values() for r in b['pressure']]+internal)
        relative=not any(losses.values()) and not any(extras.values()) and not core_losses and not internal_losses and normal_relative and Bprefix and records['case-045']['passed']
        assert physical['A']<=12 and sum(physical.values())<=55
        assert all(exact(a,norm_state(n)) for a,n in zip(statistics,norms)) and all(equal_state(w,m.policy.state_dict()) for w,(m,n) in zip(weights,policies))
    finally:
        for n in norms:n.close()
    verify_inputs(out)
    summary=dict(config=cfg,results=list(records.values()),by_mass=by_mass,internal_pressure=internal,internal_normals=internal_normals,
        reference_D=list(d.values()),reference061=list(previous.values()),original_references=reg['original_references'],
        unique_metrics=dict(B=metrics(unique),D=reg['unique_metrics']['D'],previous061=reg['unique_metrics']['candidate']),
        pass_losses=losses,additional_terminations=extras,core_pass_losses=core_losses,internal_pass_losses=internal_losses,
        normal_relative_retention_pass=normal_relative,all_B_prefixes_and_normal_protections=Bprefix,all_including_A_protections=prefix,
        expanded_development_relative_pass=bool(relative),nominal_absolute_stage_gate=by_mass['1.00']['normal_failures']==0 and all(r['passed'] for r in by_mass['1.00']['pressure']),
        fallback_A=fallback,fallback_conditions_total=len(distinct),fallback_conditions_not_tested=max(0,len(distinct)-len(fallback)),
        protections=proofs,audit_references=audit_references,reused=reuse,new_physical_runs=sum(physical.values()),physical_by_arm=physical,
        training_samples=0,PPO_updates=0,models_and_normalizations_frozen=True,selection_changed=False,held_out_executed=False,
        independent_generalization_claim=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['unique_metrics','pass_losses','core_pass_losses','internal_pass_losses','expanded_development_relative_pass','nominal_absolute_stage_gate','new_physical_runs','physical_by_arm','fallback_conditions_total']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('step',choices=['prepare','evaluate']);args=parser.parse_args();globals()[args.step]()

if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
