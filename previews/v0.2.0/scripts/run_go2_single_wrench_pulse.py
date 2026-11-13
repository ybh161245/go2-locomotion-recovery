"""153 two full protocols with one fixed20ms offline wrench pulse, then145.

Lean development checks: source freezing, original-prefix/known20ms equivalence,
action bounds, and the existing full-protocol acceptance. No repeated full-stack
audit, archive extraction, normal suite or parameter sweep for this screening.
"""
import argparse,copy,json,pickle,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_random_feature_recovery import RandomFeatureActor
from go2_action_transfer import norm_load
from go2_bilateral_pilot import Stack
from go2_training_tape_replay import pack_native_cache
from run_go2_random_feature_new_conditions import make_factory as evidence_factory,costs
from validate_go2_context_states import run_transient,pressure_metrics
from go2_traction_protocol import save_traction_probe
from evaluate_go2_release_regression import raw_equal
from audit_go2_recovery_screen import rows
from screen_go2_mass_course import exact
import evaluate_go2_robust as ev

CONFIG='configs/go2-single-wrench-pulse.json'

def prepare():
    cfg=read(ROOT/'configs/go2-contact-support.json');previous=read(ROOT/'configs/go2-wrench-execution.json');short=read(ROOT/previous['run']/'summary.json');sources=[]
    for s in previous['sources']:
        r=s['preflight'];origin=s['snapshot_source'];assert r['converged']
        sources.append(dict(source=origin['source'],step=origin['step'],solution=r['folder']+'/solution.npz',snapshot=r['folder']+'/snapshot.pkl',
                            short_reference=next(r for r in short['records'] if r['case']==origin['case'] and r['variant']=='QP_native_PD_target')))
    cfg.update(experiment_id='GO2-B-261030-153',run='artifacts/runs/20261030-single-wrench-pulse',public='docs/data/go2-single-wrench-pulse',predeclared_plan='docs/experiments/2026-10-30-single-wrench-pulse.md')
    cfg['single_wrench_pulse']=dict(sources=sources,actor=cfg['contact_support']['actor'],pulse_control_frames=1,max_new_protocol_runs=2,new_short_forks=0,new_QP_solves=0,new_training_updates=0,
         acceptance='Original12s/no termination/peak<=8deg/lastwithdrawal2s planar.08yaw.15continuous25steps; fix old failure and preserve old success. One20ms pulse only, no duration/gain/weight sweep.',
         scope='Offline known-entry causal pulse, generated with151 diagnostic oracle; not autonomous controller, no new holdout/normal qualification.',
         development_checks=['freeze source/models/config/inputs once','8-array prefix exact old146','complete native pre-pulse state exact151','pulse-end qpos/qvel/ctrl exact152','actual target bounds and only one pulse','original full protocol metrics'],
         deferred_checks=['repeated whole-stack inference audit','full matrix and normal regression before benefit','ZIP/copied-tree rebuild each rejected screen','overview regeneration'])
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg)
    p=ROOT/cfg['predeclared_plan'];assert not p.exists();p.write_text('''# 153 单控制周期分配脉冲的完整协议验证（事前）

用户要求加快实验、删去重复防御性检验。本轮保留不可覆盖原数据、事前方案、同历史对照与原验收门槛；暂缓重复网络全量重推、反复ZIP/复制树重建、无收益前的正常/180回归与交付总览。

固定两个原历史：phase626/seed246101/入口326，以及duration050/seed246107/入口340。从起始状态完整运行原12秒指令/外力协议，仅在该入口一次使用151已收敛目标20ms，下一控制步完全交回原145。目标和时刻沿已验证152读取，不重新求解QP，不扫时长/增益/权重。最多2条新完整协议尝试，0新基线、0短分叉、0训练/拟合。

该脉冲是离线已知入口干预，151目标生成使用过真实外力/材料，不能称自主修复。原145原始轨迹作复用对照，不算本轮新物理。脉冲之后全部观测与last_action来自实际轨迹，策略正常继续闭环，不重放旧动作带。

原PD、力矩和动作界限不变。脉冲归一化动作=(target-nominal)/.35，并在同一控制步的action_targets入口返回原target字节，以避免除法/乘回的浮点重建造成与152差异；这只是精确供给同一个目标，不改变PD。其余步使用原action_targets和dtype。RF影子预测与实际脉冲分别保存。

必要有效性检查：脉冲前8组数组对原146精确；入口native integration/warmstart/CONTROL/原闭包对151已存cache精确；脉冲末qpos/qvel/target对152短分叉逐位精确。最多只改一个控制步，原8度/2秒/25连续步/完整12秒门槛按保存CSV计算。若两例没有净收益，关闭此单步干预，不追加时长扫描。连续峰/恢复/目标跳变/50Hz力矩退步全部保留。

开发筛选只做上述关键检查，候选真的有收益后再另立正常保护、新条件和必要独立复算。不把精简检查描述为已做过全量审计。153不替换正式选择，不更新每轮交付总览。\n''',encoding='utf8')
    print('PREPARED153',sha(ROOT/CONFIG),sha(p),flush=True)

def attach_pulse(env,task):
    native_modify=env.modify_joint_action;native_targets=env.action_targets;native_step=env.step;native_reset=env.reset
    with np.load(ROOT/task['solution']) as z:target=z['implied_PD_targets'].copy()
    action=(target-env.nominal)/.35;assert np.max(abs(action))<=1+1e-12
    assert np.all(target>=env.nominal-.35-1e-12) and np.all(target<=env.nominal+.35+1e-12)
    pulse=dict(active=False);env.pulse_calls=[];env.pulse_snapshot=None;env.pulse_end=None
    def modify(x):
        shadow=native_modify(x);active=env.step_count==task['step'];pulse['active']=active;actual=action.copy() if active else np.asarray(shadow).copy();log=env.recovery_selection_log[-1]
        log.update(wrench_pulse_applied=active,learner_shadow_applied=log['learner_applied'],learner_actual_applied=bool(log['learner_applied'] and not active),
                   model_shadow_action=np.asarray(shadow).tolist(),model_shadow_dtype=str(np.asarray(shadow).dtype),wrench_pulse_action=action.tolist() if active else None,actual_output_dtype=str(actual.dtype))
        log['applied_action']=actual.tolist();env.pulse_calls.append(dict(step=env.step_count,applied=active,shadow=np.asarray(shadow).copy(),actual=actual.copy(),target=target.copy() if active else None))
        return actual
    def targets(x):return target.copy() if pulse['active'] else native_targets(x)
    def step(x):
        i=env.step_count
        if i==task['step']:env.pulse_snapshot=dict(native=pack_native_cache(env),qacc_warmstart=env.data.qacc_warmstart.copy())
        result=native_step(x)
        if pulse['active']:
            np.testing.assert_array_equal(env.data.ctrl,target);env.pulse_end=dict(qpos=env.data.qpos.copy(),qvel=env.data.qvel.copy(),targets=env.data.ctrl.copy())
        pulse['active']=False;return result
    def reset(*a,**kw):
        pulse['active']=False;env.pulse_calls=[];env.pulse_snapshot=None;env.pulse_end=None;return native_reset(*a,**kw)
    env.modify_joint_action=modify;env.action_targets=targets;env.step=step;env.reset=reset;return env

def runtime_files():
    result=set()
    for module in list(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if value:
            p=Path(value).resolve()
            if p.is_relative_to(ROOT/'scripts') and p.suffix=='.py':result.add(p.relative_to(ROOT).as_posix())
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    if a.prepare:prepare();return
    cfg=read(ROOT/CONFIG);task=cfg['single_wrench_pulse'];ev.CONFIG=cfg;torch.set_num_threads(1);run=ROOT/cfg['run'];run.mkdir(exist_ok=False);(run/'source').mkdir()
    stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu');actor=RandomFeatureActor(ROOT/cfg['random_feature_recovery']['parent_model'],ROOT/task['actor']);norm=norm_load(ROOT/cfg['random_feature_recovery']['parent_norm'])
    code=runtime_files();names=set(code)|{CONFIG,cfg['predeclared_plan'],'docs/data/go2-retention-protected-recovery/frozen-candidate.json','docs/data/go2-current/selection.json'}
    names.update(read(ROOT/'docs/data/go2-retention-protected-recovery/frozen-candidate.json')['models'])
    for s in task['sources']:
        names.update([s['solution'],s['snapshot'],s['short_reference']['folder']+'/native-substeps.npz',s['source']['folder']+'/states.npz'])
    for n in code:shutil.copy2(ROOT/n,run/'source'/Path(n).name)
    hashes={n:sha(ROOT/n) for n in sorted(names)};dump(run/'input-sha256.json',hashes);dump(run/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],run/'predeclared-plan.md')
    dump(run/'runtime-freeze.json',dict(config_sha256=sha(ROOT/CONFIG),plan_sha256=sha(ROOT/cfg['predeclared_plan']),runtime_modules=len(code),saved_before_physics=True,max_new_protocol_runs=2,validation_level='lean development screen'))
    records=[];checks=[];started=time.perf_counter()
    try:
        for s in task['sources']:
            source=s['source'];plain,base=evidence_factory(cfg,stack,student,actor,source,norm)
            def make(*args,**kw):return attach_pulse(base(*args,**kw),s)
            r=run_transient(run,'single-wrench-pulse',source['protocol'],source['seed'],make,stack.adapter,False);env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,source['protocol'])
            dump(folder/'selector.json',env.recovery_selection_log);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
            (folder/'pulse-calls.pkl').write_bytes(pickle.dumps(env.pulse_calls,protocol=5));(folder/'pulse-snapshot.pkl').write_bytes(pickle.dumps(env.pulse_snapshot,protocol=5));np.savez_compressed(folder/'pulse-end.npz',**env.pulse_end)
            for label,calls in [('learner-shadow-actions',env.learner_calls),('learner-shadow-decomposition',env.decomposition)]:np.savez_compressed(folder/(label+'.npz'),**{key:np.asarray([c[key] for c in calls]) for key in calls[0]})
            r.update(kind='pressure',profile=source['profile'],source=source,controller='145-plus-one-offline-wrench-pulse',pulse_step=s['step'],pulse_calls=sum(c['applied'] for c in env.pulse_calls),
                     costs=costs(folder),old_pass_lost=bool(source['passed'] and not r['passed']),added_termination=bool(not source['terminated'] and r['terminated']),validation_level='lean development screen')
            dump(folder/'result.json',r);records.append(r);dump(run/'progress.json',records)
            assert r['pulse_calls']==1 and raw_equal(r,source,s['step'])
            old=pickle.loads((ROOT/s['snapshot']).read_bytes());assert exact(env.pulse_snapshot,{key:old[key] for key in ['native','qacc_warmstart']})
            with np.load(ROOT/s['short_reference']['folder']/'native-substeps.npz') as z:
                for key,short in [('qpos','qpos_after'),('qvel','qvel_after'),('targets','target')]:np.testing.assert_array_equal(env.pulse_end[key],z[short][-1])
            recomputed=pressure_metrics(rows(folder),source['protocol'])
            for key in recomputed:assert recomputed[key]==r[key],key
            checks.append(dict(case=r['case'],prefix8_exact=True,complete_native_fork_exact=True,pulse_end_exact152=True,pulse_frames=1,original_acceptance_recomputed=True))
            print('PULSE153',r['case'],r['passed'],r['peak_tilt'],r['recovery_seconds'],r['steps'],flush=True)
    except BaseException:(run/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()
    for n in code:assert sha(ROOT/n)==hashes[n],n
    assert sha(ROOT/'docs/data/go2-current/selection.json')==hashes['docs/data/go2-current/selection.json']
    summary=dict(config=cfg,records=records,new_protocol_runs=len(records),new_short_forks=0,new_training_updates=0,new_QP_solves=0,passes=sum(r['passed'] for r in records),
                 qualified=all(r['passed'] for r in records),old_pass_losses=[r['case'] for r in records if r['old_pass_lost']],added_terminations=[r['case'] for r in records if r['added_termination']],formal_selection_changed=False,
                 checks=checks,deferred_checks=task['deferred_checks'],visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
    dump(run/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary);dump(public/'development-checks.json',dict(passed=True,checks=checks,validation_level='lean development',no_claim_of_full_independent_audit=True))
    print('COMPLETE153',summary['passes'],summary['qualified'],'seconds',summary['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
