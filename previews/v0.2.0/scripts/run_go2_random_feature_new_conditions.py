"""142 immutable141 versus its identity parent on32 prospectively declared physical runs."""
import pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from replay_go2_random_feature_recovery import verify_frozen141
from go2_random_feature_recovery import RandomFeatureActor,features
from go2_action_transfer import norm_load
from go2_bilateral_pilot import Stack
from go2_training_tape_replay import pack_native_cache
from run_go2_random_feature_recovery import make_factory as original_factory
from validate_go2_context_states import run_transient
from go2_traction_protocol import save_traction_probe
from evaluate_go2_release_regression import raw_equal
from screen_go2_mass_course import exact
import evaluate_go2_robust as ev
CONFIG='configs/go2-random-feature-new-conditions.json'
CONFIG_SHA='8e20fd20e45fcd0d405655527570e91c3b807d2c98db2eab83cc468cad0c9a3b'
PLAN_SHA='3a8ae13b1442e614a7acf5f543b7957c2cb1f89e702d42a0d041b73b98dd8176'

class TracedActor:
    def __init__(self,actor):self.actor=actor;self.trace=[]
    def predict(self,observation,deterministic=True):
        result=self.actor.predict(observation,deterministic=deterministic)
        base=np.asarray(self.actor.parent.predict(observation.copy(),deterministic=True)[0],np.float32).astype(np.float64)
        residual=features(observation,self.actor.frequency,self.actor.phase)@self.actor.coefficient
        np.testing.assert_array_equal(np.clip(base+residual,-1,1),result[0])
        self.trace.append(dict(normalized=np.asarray(observation).copy(),parent=base,residual=residual,actual=np.asarray(result[0]).copy()))
        return result

def instrument(env,model,norm):
    modify=env.modify_joint_action;step=env.step;reset=env.reset
    env.decomposition=[];env.entry_snapshots={}
    def wrapped_modify(action):
        applied=modify(action);log=env.recovery_selection_log[-1];used=log['learner_applied'];teacher=np.asarray(log['teacher_observation'],np.float32);x=norm.normalize_obs(teacher.copy())
        trace=model.trace[-1] if used else None
        if used:np.testing.assert_array_equal(trace['normalized'],x);np.testing.assert_array_equal(trace['actual'],applied)
        env.decomposition.append(dict(step=env.step_count,normalized=x,evaluated=used,parent=np.zeros(12) if trace is None else trace['parent'],
            residual=np.zeros(12) if trace is None else trace['residual'],actual=np.asarray(applied).copy(),dtype=str(np.asarray(applied).dtype)))
        return applied
    def wrapped_step(action):
        i=env.step_count;previous=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0
        before=pack_native_cache(env) if previous==0 or i==0 else None
        warm=env.data.qacc_warmstart.copy() if before is not None else None
        result=step(action);log=env.recovery_selection_log[-1]
        if i==0 or log['selection_entry']:
            assert before is not None
            env.entry_snapshots[i]=dict(native=before,qacc_warmstart=warm)
        return result
    def wrapped_reset(*args,**kwargs):
        env.decomposition=[];env.entry_snapshots={};model.trace=[];return reset(*args,**kwargs)
    env.modify_joint_action=wrapped_modify;env.step=wrapped_step;env.reset=wrapped_reset;return env

def make_factory(cfg,stack,student,actor,source,norm):
    traced=TracedActor(actor);plain,base=original_factory(cfg,stack,student,traced,source)
    def make(*args,**kwargs):return instrument(base(*args,**kwargs),traced,norm)
    return plain,make

def costs(folder):
    with np.load(folder/'states.npz') as z,np.load(folder/'telemetry.npz') as t,np.load(folder/'physics.npz') as p:
        jumps=np.abs(np.diff(z['targets'],axis=0));limit=np.max(abs(p['after_actuator_forcerange']),axis=1)
        return dict(max_joint_target_step_rad=float(jumps.max()),p95_joint_target_step_rad=float(np.quantile(jumps,.95)),
            max_50Hz_torque_limit_fraction=float(np.max(abs(t['actuator_force'])/limit)))

def main():
    cfg=read(ROOT/CONFIG);task=cfg['random_feature_new_conditions'];assert sha(ROOT/CONFIG)==CONFIG_SHA and sha(ROOT/cfg['predeclared_plan'])==PLAN_SHA
    verify_frozen141();assert sha(ROOT/task['frozen_candidate'])==task['frozen_candidate_sha256'];ev.CONFIG=cfg;torch.set_num_threads(1)
    out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names=set(read(ROOT/'artifacts/runs/20261029-random-feature-recovery/input-sha256.json'))
    names.update([CONFIG,cfg['predeclared_plan'],task['frozen_candidate'],task['fitted_actor'],'scripts/run_go2_random_feature_new_conditions.py','scripts/go2_training_tape_replay.py'])
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').glob('*.py'))
    dump(out/'input-sha256.json',{n:sha(ROOT/n) for n in sorted(names)})
    dump(out/'runtime-freeze.json',dict(config_sha256=CONFIG_SHA,plan_sha256=PLAN_SHA,max_new_physical_runs=32,model_fits=0,optimizer_updates=0,PPO_interactions=0,
        source_sha256={n:sha(ROOT/n) for n in sorted(names) if n.startswith('scripts/')},saved_before_any142_physics=True))
    spec=cfg['random_feature_recovery'];actor=RandomFeatureActor(ROOT/spec['parent_model'],ROOT/task['fitted_actor']);full=actor.coefficient.copy();norm=norm_load(ROOT/spec['parent_norm'])
    with np.load(ROOT/spec['reference']) as z:x=z['normalized'].copy()
    actor.coefficient=np.zeros_like(full)
    for row in x:
        np.testing.assert_array_equal(actor.predict(row.copy())[0],actor.parent.predict(row.copy(),deterministic=True)[0].astype(np.float64))
    np.savez_compressed(out/'identity-parent-actor.npz',frequency=actor.frequency,phase=actor.phase,coefficient=actor.coefficient)
    dump(out/'identity-preflight.json',dict(all471_parent_actual_endpoints_bitwise_exact=True,zero_coefficient_deterministic_conversion_not_fit=True,new_physical_runs=0,new_model_fits=0))
    stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu');records=[];pairs=[];snapshots={};started=time.perf_counter()
    try:
        for c in task['conditions']:
            own={}
            for variant in task['variants']:
                actor.coefficient=np.zeros_like(full) if variant=='parent135' else full.copy()
                for kind in task['kinds']:
                    source=dict(protocol=c['protocol'],profile=c['profile'],kind=kind)
                    plain,make=make_factory(cfg,stack,student,actor,source,norm)
                    r=run_transient(out,variant+'-'+kind,c['protocol'],c['profile']['seed'],make,stack.adapter,kind=='zero')
                    env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,c['protocol']);logs=env.recovery_selection_log
                    dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
                    for name,calls in [('learner-actions',env.learner_calls),('learner-decomposition',env.decomposition)]:np.savez_compressed(folder/(name+'.npz'),**{k:np.asarray([x[k] for x in calls]) for k in calls[0]})
                    (folder/'entry-snapshots.pkl').write_bytes(pickle.dumps(env.entry_snapshots,protocol=5))
                    (folder/'queries.pkl').write_bytes(pickle.dumps(dict(native=env.scoped_query_snapshots,scope=env.scoped_extra_before,burst=env.observed_burst_forks),protocol=5))
                    first=next((i for i,l in enumerate(logs) if l['learner_applied']),None)
                    r.update(kind=kind,profile=c['profile'],controller=variant,first_learner_step=first,learner_calls=sum(l['learner_applied'] for l in logs),
                        costs=costs(folder),old_pass_lost=False,added_termination=False)
                    snapshots[(c['id'],variant,kind)]=env.entry_snapshots
                    if variant=='random-feature141':
                        parent=own[('parent135',kind)];r['old_pass_lost']=bool(parent['passed'] and not r['passed']);r['added_termination']=bool(not parent['terminated'] and r['terminated'])
                        assert raw_equal(r,parent,first)
                        assert exact(env.entry_snapshots[0],snapshots[(c['id'],'parent135',kind)][0])
                        if first is not None:
                            assert first in env.entry_snapshots and first in snapshots[(c['id'],'parent135',kind)]
                            assert exact(env.entry_snapshots[first],snapshots[(c['id'],'parent135',kind)][first])
                    own[(variant,kind)]=r;dump(folder/'result.json',r);records.append(r);dump(out/'progress.json',records)
                    print('NEW142',c['id'],variant,kind,r['passed'],r['peak_tilt'],r['recovery_seconds'],r['steps'],flush=True)
                assert raw_equal(own[(variant,'zero')],own[(variant,'pressure')],round(c['protocol']['events'][0]['start']/.02))
            pairs.append(dict(case=c['id'],parent=own[('parent135','pressure')],candidate=own[('random-feature141','pressure')]))
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()
    for n,h in read(out/'input-sha256.json').items():assert sha(ROOT/n)==h,n
    verify_frozen141();new=[r for r in records if r['controller']=='random-feature141'];parent=[r for r in records if r['controller']=='parent135']
    losses=[r['case']+'/'+r['kind'] for r in new if r['old_pass_lost']];added=[r['case']+'/'+r['kind'] for r in new if r['added_termination']]
    benefits=[p['case'] for p in pairs if not p['parent']['passed'] and p['candidate']['passed']]
    summary=dict(config=cfg,records=records,pressure_pairs=pairs,new_physical_runs=len(records),new_model_fits=0,optimizer_updates=0,PPO_interactions=0,new_normal_runs=0,new_training_seeds=0,
        candidate_pressure_passes=sum(r['passed'] for r in new if r['kind']=='pressure'),candidate_zero_passes=sum(r['passed'] for r in new if r['kind']=='zero'),
        parent_pressure_passes=sum(r['passed'] for r in parent if r['kind']=='pressure'),parent_zero_passes=sum(r['passed'] for r in parent if r['kind']=='zero'),
        candidate_terminations=sum(r['terminated'] for r in new),parent_terminations=sum(r['terminated'] for r in parent),old_pass_losses=losses,added_terminations=added,benefit_conditions=benefits,
        qualified=bool(len(records)==32 and all(r['passed'] for r in new) and not losses and not added),formal_selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    dump(out/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print('COMPLETE142',summary['candidate_pressure_passes'],summary['parent_pressure_passes'],losses,summary['qualified'],flush=True)

if __name__=='__main__':main()
