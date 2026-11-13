"""064 fixed two-checkpoint selection and bounded known-condition retention."""
import copy,json,pickle,shutil,time,traceback
import numpy as np
import torch
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest
from prepare_go2_early_recovery import CONFIG
from screen_go2_recovery_specialist import load_expert
from go2_mass_course_env import Go2MassCourseEnv
from go2_recovery_state_env import restore_state,row_and_state
from go2_confirmed_selector import attach_confirmed_selector
from screen_go2_mass_course import factory,save_probe,adapter_policy,conditions,annotate_mass,norm_state,exact
from stress_go2_frozen import run_case
from evaluate_go2_confirmed_selector import trajectory_protection
from evaluate_go2_recovery_regression import metrics
from go2_quality_guard import QualityEvaluator,check_limits,equal_state
ACTIVE=None

def snapshot_metrics(rows,entry):
    cut=entry['snapshot']['control']['step_count'];events=entry['snapshot']['control']['events']
    withdrawal=max(e['start_step']+e['duration_steps'] for e in events)
    after=rows[max(0,withdrawal-cut):]
    band=[np.hypot(r['vx']-.2,r['vy'])<=.08 and abs(r['yaw_rate'])<=.15 for r in after]
    recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
    peak=max([entry['prefix_peak']]+[max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows])
    term=bool(rows[-1]['terminated']);complete=len(rows)==150 and not term
    return dict(steps=len(rows),terminated=term,peak_tilt=float(peak),recovery_seconds=recovery,
        passed=bool(complete and peak<=8 and recovery is not None and recovery<=2),cut_step=cut,withdrawal_s=withdrawal*.02,
        prefix_peak=entry['prefix_peak'],course=entry['course'],bank_id=entry['id'],mass_scale=entry['snapshot']['mass'])

def main():
    global ACTIVE
    cfg=read(ROOT/CONFIG);spec=cfg['early_recovery'];ev.CONFIG=cfg;torch.set_num_threads(1)
    train=ROOT/cfg['run'];verify_manifest(train,'training-manifest.json');bankroot=ROOT/spec['bank'];verify_manifest(bankroot)
    out=ROOT/cfg['screen_run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    inputs={n:sha(ROOT/n) for n in [CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json',*spec['model_hashes']]}
    for root in [train,bankroot,ROOT/spec['regression']]:
        for p in root.rglob('*'):
            if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    dump(out/'input-sha256.json',inputs)
    bank=pickle.loads((bankroot/'bank.pkl').read_bytes());oldids=set(read(bankroot/'summary.json')['old_successful_ids'])
    reg=read(ROOT/spec['regression']/'summary.json');old061=read(ROOT/'artifacts/runs/20261026-confirmed-selector/summary.json')
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);adapter=adapter_policy(cfg,joint)
    before=[norm_state(n) for n in [joint[1],adapter[1]]];weights=[copy.deepcopy(m.policy.state_dict()) for m in [joint[0],adapter[0]]]
    actual=0;checkpoints=[];started=time.perf_counter()
    def make_factory(expert,enorm,mass,label):
        plain=factory(joint,dict(id=label,factor='nominal' if mass==1 else 'mass',level=mass,scope='early-recovery-transfer'))
        def make(*args,**kwargs):return attach_confirmed_selector(plain(*args,**kwargs),expert,enorm,cfg['confirmed_selector']['selector'])
        return plain,make
    def pressure(expert,enorm,c,mass,label,seed):
        nonlocal actual
        plain,make=make_factory(expert,enorm,mass,label)
        r,z,rows=run_case(c,seed,make,adapter[0],adapter[1]);r['condition']=r.pop('case')
        r=ev.save_run(out,label,c['id'],seed,r,z,rows);save_probe(plain,ROOT/r['folder'])
        dump(ROOT/r['folder']/'selector.json',plain.instances[-1].recovery_selection_log);actual+=1
        return r
    try:
        focus=next(c for c in reg['config']['recovery_regression']['known_cases'] if c['id']=='case-045')
        for steps in spec['checkpoints']:
            expert,enorm=load_expert(train,steps);stats=norm_state(enorm);records=[]
            try:
                for entry in bank:
                    env=Go2MassCourseEnv('external',config=cfg);restore_state(env,entry['snapshot']);rows=[];states={}
                    try:
                        for i in range(150):
                            obs=env.observe();action,_=expert.predict(enorm.normalize_obs(obs.copy()),deterministic=True)
                            _,reward,term,trunc,info=env.step(action);row,values=row_and_state(env,obs,action,info,term)
                            values['actuator_force']=env.data.actuator_force.copy();values['rewards']=reward;rows.append(row)
                            for k,v in values.items():states.setdefault(k,[]).append(v)
                            if term or trunc:break
                    finally:env.close()
                    r=snapshot_metrics(rows,entry);r['seed']=cfg['development_seed']
                    records.append(ev.save_run(out,f'expert-{steps}',entry['id'],cfg['development_seed'],r,{k:np.asarray(v) for k,v in states.items()},rows));actual+=1
                full=pressure(expert,enorm,focus,1.,f'focus-{steps}',cfg['development_seed'])
                assert exact(stats,norm_state(enorm))
            finally:enorm.close()
            oldloss=[r['bank_id'] for r in records if r['bank_id'] in oldids and not r['passed']]
            e=dict(steps=steps,results=records,focus=full,old_pass_losses=oldloss,old_retention=not oldloss,
                new_passes=sum(r['passed'] for r in records if r['course']=='new40'),passes=sum(r['passed'] for r in records),
                terminations=sum(r['terminated'] for r in records),new_mean_peak=float(np.mean([r['peak_tilt'] for r in records if r['course']=='new40'])))
            checkpoints.append(e);dump(out/'checkpoint-results.json',checkpoints)
            print(json.dumps({k:v for k,v in e.items() if k not in ['results','focus']}|{'full_pass':full['passed'],'full_peak':full['peak_tilt']}),flush=True)
        rank=lambda e:(not(e['focus']['passed'] and e['old_retention']),not e['focus']['passed'],not e['old_retention'],-e['new_passes'],-e['passes'],e['terminations'],e['new_mean_peak'],e['steps'])
        selected=min(checkpoints,key=rank);expert,enorm=load_expert(train,selected['steps']);stats=norm_state(enorm)
        records={};proofs={};ds={r['case']:r for r in reg['reference_D']};previous={r['case']:r for r in reg['results']}
        aliases=reg['config']['recovery_regression']['aliases']
        for c in reg['config']['recovery_regression']['known_cases']:
            cid=c['id']
            if cid in aliases:
                r=copy.deepcopy(records[aliases[cid]]);r.update(case=cid,condition=c,reused_from_case=aliases[cid])
            elif cid=='case-045':r=selected['focus']
            else:r=pressure(expert,enorm,c,1.,'selected-'+cid,cfg['development_seed'])
            records[cid]=r;proofs[cid]=trajectory_protection(r,ds[cid])
            if c['force']==0:
                with np.load(ROOT/r['folder']/'states.npz') as a,np.load(ROOT/ds[cid]['folder']/'states.npz') as b:
                    assert set(a.files)==set(b.files)
                    proofs[cid]['zero_force_complete_exact']=all(np.array_equal(a[k],b[k]) for k in a.files)
        by_mass={};baseline=read(ROOT/cfg['recovery_specialist']['normal_reference']/'summary.json')
        working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
        for mass in cfg['mass_scales']:
            key=f'{mass:.2f}';label='selected-mass-'+key;plain,make=make_factory(expert,enorm,mass,label)
            normal=ev.normal_run(out,label,cfg['normal_seed'],make,adapter);save_probe(plain,ROOT/normal['folder'])
            dump(ROOT/normal['folder']/'selector.json',plain.instances[-1].recovery_selection_log);actual+=1
            proofs['normal-'+key]=trajectory_protection(normal,baseline['by_mass'][key]['normal'],True)
            pressure_records=[]
            for c in conditions():
                if mass==1:r=records[c['id']]
                else:r=pressure(expert,enorm,c,mass,label,cfg['development_seed'])
                pressure_records.append(r)
                old=next(x for x in baseline['by_mass'][key]['pressure'] if x['case']==r['case'])
                proofs[key+'-'+c['id']]=trajectory_protection(r,old)
            by_mass[key]=annotate_mass(dict(normal=normal,pressure=pressure_records),baseline['by_mass'][key],cfg,working)
        internal=[];intref=read(ROOT/cfg['confirmed_selector']['followup']/'summary.json')
        for item in intref['config']['internal_pressure']['cases']:
            r=pressure(expert,enorm,item['condition'],item['mass_scale'],'selected-old-'+item['condition']['id'],item['seed']);internal.append(r)
            old=next(x for x in intref['results'] if x['case']==r['case']);proofs['internal-'+r['case']]=trajectory_protection(r,old)
        evaluator=QualityEvaluator(joint[0],cfg);reference=read(ROOT/cfg['quality_guard']['reference_folder']/'report.json');limits=read(ROOT/cfg['quality_guard']['reference_folder']/'limits.json');normal_report={}
        try:
            for c in cfg['quality_guard']['cases']:
                plain,make=make_factory(expert,enorm,1.,'selected-normal-'+c['id']);evaluator.make_env=lambda seconds=12,make=make:make('external',episode_seconds=seconds)
                r=evaluator.run_case(out,'selected-normal',c);save_probe(plain,ROOT/r['folder']);logs=plain.instances[-1].recovery_selection_log
                dump(ROOT/r['folder']/'selector.json',logs);actual+=1;old=reference['results'][c['id']];same={}
                with np.load(ROOT/r['folder']/'states.npz') as a,np.load(ROOT/old['folder']/'states.npz') as b:
                    assert set(a.files)==set(b.files)
                    for k in a.files:same[k]=np.array_equal(a[k],b[k])
                checks=check_limits(r['metrics'],limits[c['id']]);normal_report[c['id']]=dict(result=r,checks=checks,
                    arrays_exact=same,no_expert=all(x['weight']==0 for x in logs),passed=all(x['passed'] for x in checks) and all(same.values()) and all(x['weight']==0 for x in logs))
        finally:evaluator.close()
        assert exact(stats,norm_state(enorm));enorm.close()
        assert all(exact(a,norm_state(n)) for a,n in zip(before,[joint[1],adapter[1]]))
        assert all(equal_state(w,m.policy.state_dict()) for w,m in zip(weights,[joint[0],adapter[0]]))
    finally:joint[1].close();adapter[1].close()
    unique=[r for cid,r in records.items() if cid not in aliases];losses={}
    for name,ref in [('D',ds),('061',previous),*reg['original_references'].items()]:
        losses[name]=[cid for cid,r in ref.items() if r['passed'] and not records[cid]['passed']]
    extra=[cid for cid,r in ds.items() if not r['terminated'] and records[cid]['terminated']]
    old_internal={r['case']:r for r in old061['followup']['results']}
    internal_losses=[r['case'] for r in internal if old_internal[r['case']]['passed'] and not r['passed']]
    core_losses=[dict(mass=m,case=r['case']) for m,b in by_mass.items() for r in b['pressure']
        if next(x for x in old061['by_mass'][m]['pressure'] if x['case']==r['case'])['passed'] and not r['passed']]
    prefix=all(p['passed'] and p.get('zero_force_complete_exact',True) for p in proofs.values())
    normal_ok=all(p['passed'] for p in normal_report.values()) and all(not b['new_normal_failures'] and b['retention_pass'] for b in by_mass.values())
    qualified=bool(selected['focus']['passed'] and selected['old_retention'] and not any(losses.values()) and not extra and not internal_losses and not core_losses and prefix and normal_ok)
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    assert actual<=75
    summary=dict(config=cfg,checkpoints=checkpoints,selected_steps=selected['steps'],results=list(records.values()),by_mass=by_mass,
        internal_pressure=internal,internal_normals=normal_report,protections=proofs,all_prefixes_and_normals_exact_D=prefix,
        unique_metrics=dict(candidate=metrics(unique),D=reg['unique_metrics']['D'],previous061=reg['unique_metrics']['candidate']),
        pass_losses=losses,additional_terminations_vs_D=extra,internal_pass_losses=internal_losses,core_pass_losses=core_losses,
        normal_retention_pass=normal_ok,development_qualified=qualified,new_physical_runs=actual,training_samples=spec['training_steps'],
        held_out_executed=False,selection_changed=False,independent_generalization_claim=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ['selected_steps','unique_metrics','pass_losses','internal_pass_losses','core_pass_losses','normal_retention_pass','development_qualified','new_physical_runs']}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
