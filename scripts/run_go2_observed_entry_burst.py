"""130 predeclared self-observed pulse development plus conditional normal protection."""
import argparse,copy,json,pickle,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from run_go2_supply_new_conditions import verify_supply_frozen,FROZEN
from go2_observed_entry_burst import attach_observed_entry_burst,trigger_condition
from go2_bilateral_pilot import Stack
from go2_release_regression import attach_release_capture
from go2_traction_protocol import attach_traction_schedule,save_traction_probe
from validate_go2_context_states import attach_initial_state,run_transient
from evaluate_go2_release_regression import raw_equal
from screen_go2_mass_course import factory
import evaluate_go2_robust as ev
CONFIG='configs/go2-observed-entry-burst.json'

def prepare():
    verify_supply_frozen();oldcfg=read(ROOT/'configs/go2-entry-burst-causal.json');old=oldcfg['entry_burst_causal'];assert read(ROOT/oldcfg['run']/'audit.json')['evidence_valid']
    cfg=copy.deepcopy(oldcfg);cfg.pop('entry_burst_causal');r129=read(ROOT/oldcfg['run']/'summary.json');target=next(r for r in r129['records'] if r['direct_variant']=='burst-only');assert target['passed']
    selected=[r for r in read(ROOT/'docs/data/go2-successful-supply-student/summary.json')['records'] if r['checkpoint']==250]
    names=['mu02-B-positive30','revisit-B-positive30','mu02-B-negative30','revisit-B-negative30','mu04-B-core30','mu08-B-core30','nominal-40','low-94001','constant-early']
    core=[dict(role='target',source=target,reference_kind='129-offline-burst')]+[dict(role='old-protection',source=next(r for r in selected if r['case']==n),reference_kind='123-frozen') for n in names]
    matrix=read(ROOT/'docs/data/go2-supply-full-matrix/summary.json');short=next(r for r in matrix['records'] if r['case']=='case-011' and r['seed']==94001 and r['controller']=='student123');assert short['passed'];core.append(dict(role='old-short-negative',source=short,reference_kind='126-frozen'))
    normals=[dict(role='normal',source=r,reference_kind='123-frozen') for r in selected if r['kind']=='normal'];assert len(core)==11 and len(normals)==3
    gate=dict(entry_weight=.25,scaled_vy_max=-.16,scaled_filtered_vx=.4,command_atol=1e-6,scaled_max_yaw=.00625,frames=3)
    cfg.update(experiment_id='GO2-B-261028-130',run='artifacts/runs/20261028-observed-entry-burst',public='docs/data/go2-observed-entry-burst',predeclared_plan='docs/experiments/2026-10-28-observed-entry-burst.md')
    cfg['observed_entry_burst']=dict(core=core,normals=normals,gate=gate,student_model=old['student_model'],student_norm=old['student_norm'],max_physical_runs=14,training_samples=0,
        hypothesis='The successful three-frame intervention can be triggered by self-observed signed lateral velocity at an original default recovery entry,while preserving old successes and normal quality.',
        gate_origin='Existing118 positive self-observed command/velocity thresholds with negative sign,existing native .25 entry weight and129 fixed3frames. No fit or threshold search.',
        qualification='All11 core complete original8deg/2s/25stable with no old pass loss/added termination;target exact129 burst-only. Only then3 fresh46s normals with324 original checks and RL>=29.5702276mm. Any failure rejects mechanism;no retuning this round.',
        stop='At most14 actual;all11 core retained even if failed,normal only if11/11. No training or new conditions. Freeze after development/protection only;future heldout qualification separate.')
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);p=ROOT/cfg['predeclared_plan'];assert not p.exists();p.write_text('''# 130：自身观测触发三步短接管（事前）

单一机制：原098新查询且选择default、权重恰为.25时，自身raw50满足vy缩放<=-.16（实际-.08m/s）、滤波vx缩放.4±1e-6、横向指令0±1e-6、yaw缩放绝对<=.00625，即进入3个实际控制步formal全权限；之后完全回原123。原本weight1快速入口、low路由和正向scope不变。阈值来源既有118自身观测判据的负向与129已验证三帧，不拟合新阈值。

控制逻辑只读本体观测、原查询/实际专家/权重及自身剩余帧数；不读真实外力、摩擦、协议、已知步号或时间。步号只给证据快照索引，不能参与触发。formal使用其自身冻结归一化及teacher观测，动作float32转float64，经原native目标/PD。每次查询最多3步，不靠已知304/307触发。

最多14次新实际、0训练：11核心（129负向持续目标、123六核心正反侧/材料保护、nominal40、low94001、constant-early零力、126负向40短推）全部跑完。仅11/11且无旧通过损失/新增终止时，才开3条46秒正常，共324原质量预算及RL>=29.5702276mm。核心仍为完整12秒、不终止、全程8度、撤力后2秒内平面.08/yaw.15连续25步稳定。允许预算内连续取舍，但逐项报告所有退步，不能平均抵消失格。

目标必须全8数组精确129 burst-only；其他来源保持首实际修改前原数组、原查询完整native/scope状态精确，无触发则全轨迹精确。保存触发前/结束后完整integration/warmstart/ctrl/当前力/材料/fast/selector/scope/本体CONTROL/自身counter，原请求和真实动作分开。先0新物理只读检验已存来源上触发规则/动作语义，再实际运行；影子结果不当闭环通过。

失败则拒绝此规则，不在本轮扫阈值/时长或增加训练；成功仅是已打开开发保护，另轮冻结后新初态/相位资格才能扩展结论。正式/107/118/123不换，不回填126分数。3正常若运行均是真实新物理，不用复用代替。全部模型/种子/配置/运行源码/21原生资产/raw/失败/可视保留，新视觉待用户。
''',encoding='utf8');print('PLAN130_FROZEN',flush=True)

def make_factory(cfg,stack,student,source):
    normal=source['kind']=='normal';p=source.get('protocol');plain=factory(stack.D,dict(id='130-observed-burst',factor='friction',level=.8 if normal else p['friction'],scope='130-opened-development'))
    def make(*args,**kwargs):
        env=attach_observed_entry_burst(plain(*args,**kwargs),stack,student,cfg)
        if p:attach_traction_schedule(env,p)
        return attach_release_capture(attach_initial_state(env,source['profile']))
    return plain,make

def execute(cfg,stack,student,out,item,index):
    source=item['source'];normal=source['kind']=='normal';plain,make=make_factory(cfg,stack,student,source)
    r=ev.normal_run(out,f'normal-{index}',source['seed'],make,stack.adapter) if normal else run_transient(out,f'core-{index}',source['protocol'],source['seed'],make,stack.adapter,source['kind']=='zero')
    env=plain.instances[-1];folder=ROOT/r['folder'];logs=env.recovery_selection_log;dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
    calls=env.scoped_calls;keys=['normalized_observation','raw_observation','parent_action','learned_action','learned_evaluated','applied','output_is_float64','scope_entry'];np.savez_compressed(folder/'expert-pair.npz',**{k:np.asarray([x[k] for x in calls]) for k in keys})
    calls=env.observed_burst_calls;np.savez_compressed(folder/'direct-actions.npz',**{k:np.asarray([x[k] for x in calls]) for k in calls[0]});(folder/'snapshots.pkl').write_bytes(pickle.dumps(dict(queries=env.scoped_query_snapshots,extra_before=env.scoped_extra_before,forks=env.observed_burst_forks),protocol=5))
    quality=None
    if normal:
        from go2_parameter_probe import save_probe
        save_probe(plain,folder);ref=next(q for q in read(ROOT/'artifacts/runs/20261027-release-regression/summary.json')['quality'] if q['result']['seed']==r['seed']);checks=ev.normal_checks(r,ref['reference']);rl=r['segments']['reverse right']['RL']['clearance_p95_m']*1000;r['passed']=all(c['passed'] for c in checks) and rl>=29.5702276;quality=dict(result_folder=r['folder'],checks=checks,RL_mm=rl,passed=bool(r['passed']))
    else:save_traction_probe(plain,folder,source['protocol'])
    first=next((i for i,l in enumerate(logs) if l['direct_override']),None);assert raw_equal(r,source,first),source['case']
    r.update(kind=source['kind'],controller='observed-entry-burst',profile=source['profile'],source=source,role=item['role'],first_burst_step=first,burst_entries=[i for i,l in enumerate(logs) if l['burst_entry']],direct_calls=sum(l['direct_override'] for l in logs),default_calls=len(env.scoped_calls),scope_learned_steps=sum(l['scope_learned'] for l in logs),all_raw_exact_source=raw_equal(r,source),delta_peak=None if normal else r['peak_tilt']-source['peak_tilt'],delta_recovery=None if normal or r['recovery_seconds'] is None or source['recovery_seconds'] is None else r['recovery_seconds']-source['recovery_seconds'])
    if item['role']=='target':assert r['all_raw_exact_source']
    dump(folder/'result.json',r);print('OBSERVED130 '+json.dumps({k:r.get(k) for k in ['case','role','passed','peak_tilt','recovery_seconds','terminated','burst_entries','direct_calls','all_raw_exact_source']}),flush=True);return r,quality

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.prepare_only:prepare();return
    cfg=read(ROOT/CONFIG);task=cfg['observed_entry_burst'];f=verify_supply_frozen();ev.CONFIG=cfg;out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');names={CONFIG,cfg['predeclared_plan'],FROZEN,'scripts/run_go2_observed_entry_burst.py','scripts/go2_observed_entry_burst.py',*f['models'],*f['runtime_code_sha256'],'artifacts/runs/20261027-release-regression/summary.json'}
    names.update(p.relative_to(ROOT).as_posix() for m in list(sys.modules.values()) if getattr(m,'__file__',None) for p in [Path(m.__file__).resolve()] if p.parent==(ROOT/'scripts').resolve() and p.suffix=='.py')
    for directory in ['third_party/unitree_go2',cfg['unified_recovery_selection']['bank']]:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/directory).rglob('*') if p.is_file())
    for item in task['core']+task['normals']:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/item['source']['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs);pre=[]
    for item in task['core']+task['normals']:
        r=item['source'];logs=read(ROOT/r['folder']/'selector.json');pre.append(dict(case=r['case'],folder=r['folder'],source_trigger_steps=[i for i,l in enumerate(logs) if trigger_condition(l,task['gate'])],note='Read-only source shadow predicates,not counterfactual performance.'))
    assert pre[0]['source_trigger_steps']==[304];dump(out/'preflight.json',dict(new_physical_runs=0,gate=task['gate'],source_predicates=pre))
    torch.set_num_threads(1);stack=Stack(cfg);student=PPO.load(ROOT/task['student_model'],device='cpu');records=[];quality=[];started=time.perf_counter()
    try:
        for i,item in enumerate(task['core']):r,_=execute(cfg,stack,student,out,item,i);records.append(r);dump(out/'progress.json',records)
        core_passed=all(r['passed'] for r in records);dump(out/'core-selection.json',dict(core_passed=core_passed,core_passes=sum(r['passed'] for r in records),normal_opened=core_passed))
        if core_passed:
            for i,item in enumerate(task['normals']):r,q=execute(cfg,stack,student,out,item,i);records.append(r);quality.append(q);dump(out/'progress.json',records)
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    verify_supply_frozen();qualified=len(records)==14 and all(r['passed'] for r in records);summary=dict(config=cfg,records=records,quality=quality,new_physical_runs=len(records),training_samples=0,PPO_updates=0,supervised_updates=0,new_normal_runs=len(quality),normal_checks=sum(len(q['checks']) for q in quality),core_passes=sum(r['passed'] for r in records if r['kind']!='normal'),qualified=qualified,old_pass_losses=[r['case'] for r in records if r['role']!='target' and r['source']['passed'] and not r['passed']],added_terminations=[r['case'] for r in records if not r['source']['terminated'] and r['terminated']],qualification_scope='Self-observed pulse opened development and old protection only;not frozen new conditions or full-matrix claim.',formal_selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started);assert len(records) in [11,14];dump(out/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});print('COMPLETE130 '+json.dumps(dict(new_physical_runs=len(records),core_passes=summary['core_passes'],normal_checks=summary['normal_checks'],qualified=qualified,old_pass_losses=summary['old_pass_losses'])),flush=True)

if __name__=='__main__':main()
