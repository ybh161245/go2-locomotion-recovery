"""Separate060 snapshot recovery skill from original full-protocol selector capability."""
import argparse,copy,json,pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import evaluate_go2_robust as ev
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import verify_manifest,pressure_metrics,benefit_signal
from go2_mass_course_env import Go2MassCourseEnv
from go2_recovery_state_env import restore_state,row_and_state,suffix_metrics
from go2_recovery_selector import attach_selector
from go2_parameter_probe import factory,save_probe
from screen_go2_mass_course import conditions,annotate_mass,adapter_policy,norm_state,exact
from stress_go2_frozen import run_case
ROOT=ev.ROOT
ACTIVE=None


def load_expert(train,steps):
    model=PPO.load(train/f'checkpoint-{steps}.zip',device='cpu')
    norm=VecNormalize.load(train/f'checkpoint-{steps}-vecnormalize.pkl',DummyVecEnv([lambda:Go2MassCourseEnv('external',config=read(train/'acceptance.json'))]))
    norm.training=False;norm.norm_reward=False
    return model,norm


def main():
    global ACTIVE
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/go2-recovery-specialist.json');args=parser.parse_args()
    cfg=read(ROOT/args.config);spec=cfg['recovery_specialist'];ev.CONFIG=cfg;torch.set_num_threads(1)
    train=ROOT/cfg['run'];bankroot=ROOT/spec['bank'];verify_manifest(train,'training-manifest.json');verify_manifest(bankroot)
    assert read(train/'summary.json')['training_samples']==spec['training_steps']
    out=ROOT/cfg['screen_run'];out.mkdir(exist_ok=False);ACTIVE=out;(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    inputs={n:sha(ROOT/n) for n in [args.config,cfg['predeclared_plan'],'docs/data/go2-current/selection.json']}
    for root in [train,bankroot,ROOT/spec['normal_reference']]:
        for p in root.rglob('*'):
            if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    reuse=spec.get('screen_reuse')
    if reuse:
        for p in (ROOT/reuse['source_run']).rglob('*'):
            if p.is_file():inputs[p.relative_to(ROOT).as_posix()]=sha(p)
    dump(out/'input-sha256.json',inputs)
    with (bankroot/'bank.pkl').open('rb') as f:bank=pickle.load(f)
    checkpoints=[];started=time.perf_counter()
    for steps in spec['checkpoints']:
        if reuse:
            prior=next(e for e in read(ROOT/reuse['source_run']/'snapshot-results.json') if e['steps']==steps)
            for r in prior['results']:
                actual=read(ROOT/r['folder']/'result.json')
                assert all(actual[k]==v for k,v in r.items() if k!='folder')
            checkpoints.append(copy.deepcopy(prior));continue
        model,norm=load_expert(train,steps);before=norm_state(norm);records=[]
        try:
            for entry in bank:
                env=Go2MassCourseEnv('external',config=cfg);restore_state(env,entry['snapshot']);cut=env.step_count
                rows=[];states={}
                try:
                    for i in range(spec['episode_steps']):
                        obs=env.observe();action,_=model.predict(norm.normalize_obs(obs.copy()),deterministic=True)
                        _,reward,term,trunc,info=env.step(action);row,values=row_and_state(env,obs,action,info,term)
                        rows.append(row)
                        for k,v in values.items():states.setdefault(k,[]).append(v)
                        if term or trunc:break
                finally:env.close()
                result=suffix_metrics(rows,cut,entry['prefix_peak'],complete=len(rows)==spec['episode_steps'])
                result.update(case=entry['id'],bank_id=entry['id'],seed=cfg['development_seed'],snapshot_time_s=cut*.02,
                    mass_scale=entry['snapshot']['mass'],checkpoint_steps=steps,variant=f'expert-{steps}',prefix_peak=entry['prefix_peak'])
                records.append(ev.save_run(out,f'expert-{steps}',entry['id'],cfg['development_seed'],result,{k:np.asarray(v) for k,v in states.items()},rows))
            assert exact(before,norm_state(norm))
        finally:norm.close()
        entry=dict(steps=steps,results=records,passes=sum(r['passed'] for r in records),terminations=sum(r['terminated'] for r in records),
            mean_peak=float(np.mean([r['peak_tilt'] for r in records])))
        checkpoints.append(entry);dump(out/'snapshot-results.json',checkpoints)
        print(json.dumps({k:v for k,v in entry.items() if k!='results'}),flush=True)
    ranked=sorted(checkpoints,key=lambda e:(-e['passes'],e['terminations'],e['mean_peak'],e['steps']));selected=ranked[0]
    joint=ev.load_policy(ROOT/'artifacts/runs'/cfg['parent'],cfg['parent_checkpoint']);expert,expert_norm=load_expert(train,selected['steps'])
    adapter=adapter_policy(cfg,joint);before=[norm_state(n) for n in [joint[1],adapter[1],expert_norm]]
    records={};normals={};audits={};selector_logs={};base=read(ROOT/spec['normal_reference']/'summary.json')
    def make_for(scale,label):
        plain=factory(joint,dict(id=label,factor='nominal' if scale==1 else 'mass',level=scale,scope='recovery-selector-prototype'))
        def make(*args,**kwargs):return attach_selector(plain(*args,**kwargs),expert,expert_norm,spec['selector'])
        # The probe recorder needs its original factory attributes.
        for key in ['instances','cached','spec']:
            if hasattr(plain,key):setattr(make,key,getattr(plain,key))
        return plain,make
    def save_selection(plain,folder):
        # factory exposes the list used by save_probe (its documented get_last callback).
        env=plain.instances[-1] if hasattr(plain,'instances') else plain.last_instance
        path=ROOT/folder/'selector.json';dump(path,env.recovery_selection_log);selector_logs[folder]=path.relative_to(ROOT).as_posix()
    # Accessor is supplied by the native physical wrapper's .instances attribute.
    complete2=[]
    try:
        for mass in spec['source_masses']:
            key=f'{mass:.2f}';label=f'prototype-mass-{key}';plain,make=make_for(mass,label);c=conditions()[1]
            r,packed,rows=run_case(c,cfg['development_seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
            r=ev.save_run(out,label,c['id'],cfg['development_seed'],r,packed,rows)
            if reuse and mass==1.:
                prior=ROOT/reuse['source_run']/f'{label}-{c["id"]}-seed-{cfg["development_seed"]}'
                with np.load(prior/'states.npz') as old:
                    assert set(old.files)==set(packed)
                    for k in old.files:np.testing.assert_array_equal(old[k],packed[k])
            audits[r['folder']]=save_probe(plain,ROOT/r['folder']);save_selection(plain,r['folder'])
            records[(key,c['id'])]=r;complete2.append(r)
        old2=[next(r for r in base['by_mass'][f'{mass:.2f}']['pressure'] if r['case']=='F30-D0.6-S1-O0.2') for mass in spec['source_masses']]
        useful2=sum(r['passed'] for r in complete2)>sum(r['passed'] for r in old2)
        extra2=any(r['terminated'] and not old['terminated'] for r,old in zip(complete2,old2))
        extended=bool(useful2 and not extra2);by_mass={}
        if extended:
            working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
            for mass in cfg['mass_scales']:
                key=f'{mass:.2f}';label=f'prototype-mass-{key}';plain,make=make_for(mass,label)
                n=ev.normal_run(out,label,cfg['normal_seed'],make,adapter);audits[n['folder']]=save_probe(plain,ROOT/n['folder']);save_selection(plain,n['folder']);normals[key]=n
                pressure=[]
                for c in conditions():
                    if (key,c['id']) in records:r=records[(key,c['id'])]
                    else:
                        r,packed,rows=run_case(c,cfg['development_seed'],make,adapter[0],adapter[1]);r['condition']=r.pop('case')
                        r=ev.save_run(out,label,c['id'],cfg['development_seed'],r,packed,rows)
                        audits[r['folder']]=save_probe(plain,ROOT/r['folder']);save_selection(plain,r['folder']);records[(key,c['id'])]=r
                    pressure.append(r)
                by_mass[key]=annotate_mass(dict(normal=n,pressure=pressure),base['by_mass'][key],cfg,working)
        assert all(exact(a,norm_state(n)) for a,n in zip(before,[joint[1],adapter[1],expert_norm]))
    finally:joint[1].close();adapter[1].close();expert_norm.close()
    if by_mass:
        full_metrics=pressure_metrics(by_mass);benefit=benefit_signal(full_metrics,pressure_metrics(base['by_mass']),cfg['benefit']);retention=all(b['retention_pass'] for b in by_mass.values())
    else:full_metrics=None;benefit=False;retention=False
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    decision=('recovery_prototype_development_benefit_requires_more_qualification' if benefit and retention else
        'snapshot_skill_without_qualified_full_system_benefit' if selected['passes'] else 'no_snapshot_recovery_pass_at_fixed_budget')
    summary=dict(config=cfg,bank=read(bankroot/'summary.json'),checkpoints=checkpoints,selected_steps=selected['steps'],
        snapshot_passes=selected['passes'],full_initial_two=complete2,full_extension_executed=extended,by_mass=by_mass,
        full_pressure_metrics=full_metrics,benefit_signal=bool(benefit),retention_pass=bool(retention),
        physics_audits=audits,selector_logs=selector_logs,decision=decision,new_snapshot_evaluations=0 if reuse else 12,
        reused_snapshot_evaluations=12 if reuse else 0,serialization_failure_prior_full_runs=1 if reuse else 0,
        prior_nominal_trajectory_exact_after_logging_fix=bool(reuse),
        new_full_evaluations=len(records)+len(normals),normalization_unchanged=True,input_hashes_unchanged=True,
        independent_generalization_claim=False,selection_changed=False,held_out_executed=False,visual_acceptance='pending',
        elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
    public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['config','bank','checkpoints','by_mass','physics_audits','selector_logs','full_initial_two']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE is not None:(ACTIVE/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
