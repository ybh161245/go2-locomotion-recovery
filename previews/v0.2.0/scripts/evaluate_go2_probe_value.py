"""Measure the rejected052 candidate's value with unchanged physical and quality gates."""
import argparse,copy,hashlib,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import evaluate_go2_robust as ev
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_parameter_probe import factory,save_probe
from screen_go2_mass_course import evaluate_mass,annotate_mass,adapter_policy,norm_state,exact,mass_key
from stress_go2_frozen import run_case
ROOT=ev.ROOT
ACTIVE_RUN=None

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_manifest(folder,name='manifest.json'):
    manifest=read(folder/name)
    for key,value in manifest.items():assert sha(folder/key)==value,(folder,key)
    return manifest

def pressure_metrics(by_mass):
    records=[r for bundle in by_mass.values() for r in bundle['pressure']]
    return dict(passes=sum(r['passed'] for r in records),terminations=sum(r['terminated'] for r in records),
        mean_peak=float(np.mean([r['peak_tilt'] for r in records])),worst_peak=max(r['peak_tilt'] for r in records),samples=len(records))

def benefit_signal(candidate,baseline,spec):
    return (candidate['passes']-baseline['passes']>=spec['extra_passes_min'] or
        baseline['terminations']-candidate['terminations']>=spec['fewer_terminations_min'] or
        baseline['mean_peak']-candidate['mean_peak']>=spec['mean_peak_decrease_deg_min'])

def extension_cases():
    result=ev.development_cases()
    for force in [20,25,30]:
        for sign in [1,-1]:
            for offset in [0,.2,.4]:
                result.append(dict(id=f'F{force}-D0.6-S{sign}-O{offset}',phase='straight',force=force,direction=[0,sign],duration=.6,pulses=[6+offset],torque=0,friction=None))
    assert len(result)==31 and len({r['id'] for r in result})==31
    return result

def extend_nominal(out,joint,adapter,cfg,nominal):
    current={r['case']:r for r in nominal['pressure']};reused=list(current);new=[]
    make=factory(joint,dict(id='probe-value-mass-1.00',factor='nominal',level=1.,scope='probe-value-extension'))
    audits={}
    for condition in extension_cases():
        if condition['id'] in current:continue
        result,packed,rows=run_case(condition,cfg['development_seed'],make,adapter[0],adapter[1]);result['condition']=result.pop('case')
        record=ev.save_run(out,'probe-value-mass-1.00',condition['id'],cfg['development_seed'],result,packed,rows)
        audits[record['case']]=save_probe(make,ROOT/record['folder']);current[record['case']]=record;new.append(record)
    assert len(new)==28 and len(current)==31
    dscreen=read(ROOT/cfg['reference_screen']);d={r['case']:r for r in dscreen['results'] if r['variant']==cfg['reference_variant']}
    assert set(d)==set(current)
    oldscreen=read(ROOT/'docs/data/go2-sustained-A/screen.json')
    core={r['case']:r for r in oldscreen['results'] if r['variant']=='previous-stack'}
    boundary=read(ROOT/'docs/data/go2-capability-boundary/summary.json')
    old={r['case']:r for r in boundary['results'] if r['variant']=='research' and r['condition']['force'] in [20,30] and r['condition']['duration']==.6}
    losses={name:[key for key,record in reference.items() if record['passed'] and not current[key]['passed']]
        for name,reference in [('D',d),('original_core',core),('original_20_30N_boundary',old)]}
    additional=[key for key,record in d.items() if not record['terminated'] and current[key]['terminated']]
    return dict(executed=True,results=list(current.values()),new_results=new,reused_cases=reused,physics_audits=audits,
        pass_losses=losses,additional_terminations=additional,passed=not any(losses.values()) and not additional,new_physical_runs=len(new))

def main():
    global ACTIVE_RUN
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/go2-probe-value.json');args=parser.parse_args()
    config_path=Path(args.config);config_path=config_path if config_path.is_absolute() else ROOT/config_path
    cfg=read(config_path);ev.CONFIG=cfg;torch.set_num_threads(1)
    basefolder=ROOT/cfg['baseline'];verify_manifest(basefolder);reference=read(basefolder/'summary.json')
    for n,h in read(basefolder/'input-sha256.json').items():assert sha(ROOT/n)==h,n
    for key in ['parent','parent_checkpoint','working_stack','normal_seed','development_seed','normal_budget','stress_gates','support','endpoint_existing_clearance_retention','mass_scales']:
        assert cfg[key]==reference['config'][key],key
    source=ROOT/'artifacts/runs/20261025-exposure-sampling';manifest=verify_manifest(source)
    for name in [cfg['candidate_model'],cfg['candidate_norm']]:
        path=ROOT/name;relative=path.relative_to(source).as_posix();assert sha(path)==manifest[relative],name
    probe=read(source/'summary.json')['probe'];assert probe['executed'] and not probe['accepted']
    assert (source/probe['candidate']).resolve()==(ROOT/cfg['candidate_model']).resolve()
    internal_failures=[dict(c,case=case) for case,checks in probe['assessment']['checks'].items() for c in checks if not c['passed']]
    assert len(internal_failures)==1 and internal_failures[0]['metric']=='turn-a/angle_error'
    out=ROOT/cfg['run'];out.mkdir(parents=True,exist_ok=False);ACTIVE_RUN=out
    public=ROOT/cfg['public'];public.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    for path in (ROOT/'scripts').glob('*.py'):shutil.copy2(path,out/'source'/path.name)
    ev.dump(out/'acceptance.json',cfg)
    plan=ROOT/'docs/experiments/2026-10-25-probe-value.md'
    if plan.exists():shutil.copy2(plan,out/'predeclared-plan.md')
    names={config_path.relative_to(ROOT).as_posix(),cfg['candidate_model'],cfg['candidate_norm'],cfg['reference_screen'],
        cfg['baseline']+'/summary.json',cfg['baseline']+'/manifest.json',cfg['baseline']+'/input-sha256.json',
        'docs/data/go2-current/selection.json','docs/data/go2-sustained-A/screen.json','docs/data/go2-capability-boundary/summary.json',
        'artifacts/runs/20261025-exposure-sampling/manifest.json','artifacts/runs/20261025-exposure-sampling/summary.json'}
    names.update(read(basefolder/'input-sha256.json'))
    for bundle in reference['by_mass'].values():
        for record in [bundle['normal']]+bundle['pressure']:
            names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/record['folder']).iterdir() if p.is_file())
    for folder in [cfg['working_stack'],'20261023-140238-938755-go2-transition-v3-mixed-ppo']:
        names.update(f'artifacts/runs/{folder}/'+n for n in ['config.json','final-model.zip','final-vecnormalize.pkl'])
    hashes={n:sha(ROOT/n) for n in sorted(names)};ev.dump(out/'input-sha256.json',hashes)
    started=time.perf_counter();model=PPO.load(ROOT/cfg['candidate_model'],device='cpu')
    norm=VecNormalize.load(ROOT/cfg['candidate_norm'],DummyVecEnv([lambda:Go2TransitionEnvV3('external')]))
    norm.training=False;norm.norm_reward=False;joint=(model,norm);adapter=adapter_policy(cfg,joint)
    before=[norm_state(n) for n in [joint[1],adapter[1]]];by_mass={};actual=0
    try:
        working=read(ROOT/'docs/data/go2-sustained-A/screen.json')['normals']['working']
        for scale in cfg['mass_scales']:
            key=mass_key(scale);bundle=evaluate_mass(out,'probe-value-mass-'+key,scale,joint,adapter,cfg)
            by_mass[key]=annotate_mass(bundle,reference['by_mass'][key],cfg,working);actual+=4
            ev.dump(out/'progress.json',dict(by_mass=by_mass,new_physical_runs=actual))
            print(json.dumps(dict(mass=key,normal_failures=bundle['normal_failures'],new_normal_failures=bundle['new_normal_failures'],
                pressure_passes=sum(r['passed'] for r in bundle['pressure']),pressure_terminations=sum(r['terminated'] for r in bundle['pressure']),retention_pass=bundle['retention_pass'])),flush=True)
        comparison=dict(D=pressure_metrics(reference['by_mass']),candidate=pressure_metrics(by_mass))
        useful=benefit_signal(comparison['candidate'],comparison['D'],cfg['benefit']);retention=all(b['retention_pass'] for b in by_mass.values())
        extended=dict(executed=False,results=[],new_results=[],pass_losses={},additional_terminations=[],new_physical_runs=0,
            reason='No useful signal or relative quality retention failed')
        if useful and retention:
            extended=extend_nominal(out,joint,adapter,cfg,by_mass['1.00']);actual+=extended['new_physical_runs']
        assert actual in [12,40]
        assert all(exact(old,norm_state(n)) for old,n in zip(before,[joint[1],adapter[1]]))
        for name,h in hashes.items():assert sha(ROOT/name)==h,name
        if useful and retention and extended['passed']:decision='robustness_benefit_with_known_internal_angle_cost_requires_tradeoff_review'
        elif useful:decision='partial_pressure_signal_without_global_retention'
        else:decision='no_predeclared_pressure_benefit'
        summary=dict(config=cfg,reference=reference,by_mass=by_mass,pressure_comparison=comparison,benefit_signal=bool(useful),retention_pass=bool(retention),
            extended=extended,decision=decision,new_physical_runs=actual,reused_D_baseline_runs=12,training_steps_added=0,
            internal_angle_gate_still_failed=True,known_internal_failures=internal_failures,normalization_unchanged=True,input_hashes_unchanged=True,
            selection_changed=False,held_out_executed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
        ev.dump(out/'summary.json',summary);ev.dump(public/'summary.json',summary);ev.dump(public/'input-sha256.json',hashes)
        ev.dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()})
        print(json.dumps(dict(pressure_comparison=comparison,benefit_signal=useful,retention_pass=retention,decision=decision,new_physical_runs=actual)),flush=True)
    finally:
        joint[1].close();adapter[1].close()

if __name__=='__main__':
    try:main()
    except BaseException:
        if ACTIVE_RUN is not None and not (ACTIVE_RUN/'failure.txt').exists():(ACTIVE_RUN/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
