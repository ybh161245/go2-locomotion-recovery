"""141 one different function class, one closed-form fit, at most12 physical validations."""
import argparse,copy,pickle,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_action_transfer import norm_load
from go2_observed_burst_frozen import verify_frozen130,FROZEN130
from go2_random_feature_recovery import features,RandomFeatureActor,attach_rf_learner
from go2_bilateral_pilot import Stack
from run_go2_observed_burst_new_conditions import make_factory as observed_factory
from run_go2_observed_entry_burst import make_factory as normal_factory
from validate_go2_context_states import run_transient
from go2_traction_protocol import save_traction_probe
from evaluate_go2_release_regression import raw_equal
import evaluate_go2_robust as ev
CONFIG='configs/go2-random-feature-recovery.json'

def prepare():
    verify_frozen130();cfg=read(ROOT/'configs/go2-native-action-loss.json');old=cfg.pop('protected_native_transfer');cfg.pop('native_action_loss')
    cfg.update(experiment_id='GO2-B-261029-141',run='artifacts/runs/20261029-random-feature-recovery',
        public='docs/data/go2-random-feature-recovery',predeclared_plan='docs/experiments/2026-10-29-random-feature-recovery.md')
    cfg['random_feature_recovery']=dict(parent_model=old['initial_model'],parent_norm=old['initial_norm'],
        reference='artifacts/runs/20261029-protected-native-transfer/reference.npz',
        reference_audit='docs/data/go2-native-action-loss/mechanism-audit.json',
        core_sources=old['core_sources'],protection_sources=old['protection_sources'],
        seed=241101,features=2048,frequency_scale=4/np.sqrt(50),ridge=1e-8,
        reference_group_native_RMSE_deg_max=.05,reference_max_joint_error_deg_max=.1,
        max_new_physical_runs=12,core_physical_runs=6,optional_protection=6,new_model_fits=1,optimizer_updates=0,PPO_interactions=0,
        mechanism='Different function class: frozen135clipped parent + fixed independent cosine random features linear residual;one deterministic ridge solve on471actualclip labels. Original raw50/frozen formal norm/observablehold/native supply.',
        stop='One width/scale/ridge/seed;no coefficient/length/threshold or PPO sweeps. If fit interface fails 0physics; if6core not allpass no protection. No new conditions within141.',
        prospective_runtime_sha256={n:sha(ROOT/n) for n in ['scripts/go2_random_feature_recovery.py','scripts/run_go2_random_feature_recovery.py']})
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);plan=ROOT/cfg['predeclared_plan'];assert not plan.exists()
    plan.write_text('''# 141 新函数表达：固定随机特征残差恢复策略（事前）

用户已授权睡眠期间自主实验。140前三帧成功后现有模型仍失败；139共享成功输入上仍有约4.65度目标误差。现有证据支持先检验“换一种可更充分拟合实际成功动作的函数表达，闭环能否恢复”，但不证明表达能力是唯一瓶颈。本轮不是续训137/138或增加它们更新预算，也不改变已关闭短前缀方案。

原13516384确定性parent冻结。输入仍50维与原formal冻结规范化；新策略为clip(parent_actual_clip + Phi(x) B,-1,1)，Phi=√(2/2048)cos(xW+b)，固定seed241101，W各项N(0,(4/√50)^2)、b在[0,2π]均匀。只一次B=Phi.T(Phi Phi.T+1e-8 I)^(-1)(desired_actual_clip-parent_actual_clip)解析拟合，无优化器/PPO更新。471个旧已打开输入、152C成功实际样本及319旧成功clip标签全部保留；部署仅保存W/b/B与冻结parent，不查询训练样本/近邻/协议步号/材料/真实力，不加20银行原型或半径。它是一次新模型拟合，不能称零学习；seed是随机特征种子，不是独立PPO种子。

先0物理预检：原raw/normalized/实际标签与137/138审计输入精确；zero-coefficient在全部471输入逐位等于原parent实际clip；所有actual native目标语义精确。独立重算Gram/线性方程残差/拟合输出，每源native目标RMSE<=.05度、任一关节误差<=.1度只是执行预检，不代替闭环资格。预检失败停，不扫宽度/频率/正则/种子。新输出float64直接沿原native接口，关闭残差则与原float32clip转float64逐位相同。

预检通过才6次原完整核心C/A/目标/D/E/G，原自身入口触发/hold、lowerfast/20银行/scope/burst全部保持，无已知步覆盖。完整12秒不终止/全程8度/撤力后2秒内平面.08、yaw.15连续25步，原info口径不变，G真零力。C过且其余五旧通过保持、无新增终止，才冻结这个唯一模型再开6旧保护：nominal40/low94001受推、constant-early完整零力、三个46秒正常324原预算及RL29.5702276mm。最多12新实际，0额外基线物理；原135/130基线明确复用不当新运行。拟合好也不能用来冒充闭环或泛化。

核心失败关闭此一次表达试验，不追加同拟合包或PPO。核心+保护全过也仅已打开开发，另轮冻结新初态/相位验证，正式不自动换。保存前后模型/规范化/随机特征/解析中间量/所有raw失败/运行时源代码/21原生资产/种子/配置/视频，独立审计与归档。普通动作舒展/停止/足端质量依然按原保护与用户视觉验收，不用恢复速度抵消。当前自动任务已删除，本轮未另建定时任务。
''',encoding='utf8');print('PLAN141_FROZEN',sha(ROOT/CONFIG),sha(plan),flush=True)

def fit(cfg):
    spec=cfg['random_feature_recovery'];out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');f=verify_frozen130()
    names={CONFIG,cfg['predeclared_plan'],FROZEN130,*f['models'],*f['runtime_code_sha256'],*f['native_assets_sha256'],*f['bank_sha256'],spec['reference'],spec['reference_audit'],spec['parent_model'],spec['parent_norm'],'docs/data/go2-current/selection.json'}
    names.update(spec['prospective_runtime_sha256']);names.update(read(ROOT/'artifacts/runs/20261029-native-action-loss/input-sha256.json'))
    for source in spec['core_sources']+spec['protection_sources']:
        names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/source['folder']).iterdir() if p.is_file())
    names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'scripts').glob('*.py'))
    dump(out/'input-sha256.json',{n:sha(ROOT/n) for n in sorted(names)})
    for n,h in spec['prospective_runtime_sha256'].items():assert sha(ROOT/n)==h
    with np.load(ROOT/spec['reference']) as z:ref={k:z[k].copy() for k in z.files}
    norm=norm_load(ROOT/spec['parent_norm']);parent=PPO.load(ROOT/spec['parent_model'],device='cpu')
    np.testing.assert_array_equal(norm.normalize_obs(ref['observations'].copy()),ref['normalized'])
    x=ref['normalized'];desired=np.clip(ref['labels'],-1,1).astype(np.float64)
    base=np.asarray([parent.predict(row.copy(),deterministic=True)[0] for row in x],np.float32).astype(np.float64)
    # Old471 labels were already paired with real native targets in137/138, not fabricated counterfactuals.
    for group in range(1,6):np.testing.assert_array_equal(base[ref['source_id']==group],desired[ref['source_id']==group])
    rng=np.random.default_rng(spec['seed']);frequency=rng.normal(0,spec['frequency_scale'],(50,spec['features']));phase=rng.uniform(0,2*np.pi,spec['features'])
    phi=features(x,frequency,phase);gram=phi@phi.T;delta=desired-base
    dual=np.linalg.solve(gram+spec['ridge']*np.eye(len(x)),delta);coefficient=phi.T@dual
    np.savez_compressed(out/'fitted-actor.npz',frequency=frequency,phase=phase,coefficient=coefficient)
    np.savez_compressed(out/'fit-evidence.npz',observations=ref['observations'],normalized=x,labels=desired,source_id=ref['source_id'],base=base,
        features=phi,gram=gram,dual=dual,residual_labels=delta)
    actor=RandomFeatureActor(ROOT/spec['parent_model'],out/'fitted-actor.npz')
    predicted=np.asarray([actor.predict(row.copy())[0] for row in x]);error=np.degrees(.35*(predicted-desired));metrics={}
    for g in range(6):
        e=error[ref['source_id']==g];metrics[str(g)]=dict(frames=len(e),native_RMSE_deg=float(np.sqrt(np.mean(e**2))),max_joint_error_deg=float(abs(e).max()))
    zero=np.clip(base+phi@np.zeros_like(coefficient),-1,1);np.testing.assert_array_equal(zero,base)
    passed=all(m['native_RMSE_deg']<=spec['reference_group_native_RMSE_deg_max'] for m in metrics.values()) and abs(error).max()<=spec['reference_max_joint_error_deg_max']
    residual=float(np.max(abs((gram+spec['ridge']*np.eye(len(x)))@dual-delta)))
    report=dict(passed=bool(passed),new_model_fits=1,optimizer_updates=0,PPO_interactions=0,new_physical_runs=0,unique_reference_states=len(x),
        labels_are_real_actual_clip=True,zero_residual_parent_endpoint_exact=True,normalization_exact=True,linear_equation_max_residual=residual,metrics=metrics,
        no_training_state_lookup_at_runtime=True,fit_not_closed_loop_qualification=True,coefficient_sha256=sha(out/'fitted-actor.npz'))
    dump(out/'interface-audit.json',report);print('FIT141',report,flush=True)
    return out,report

def make_factory(cfg,stack,student,actor,source):
    if source['kind']=='normal':plain,base=normal_factory(cfg,stack,student,source)
    else:plain,base=observed_factory(cfg,stack,student,dict(protocol=source['protocol'],profile=source['profile']),'observed130')
    def make(*args,**kwargs):return attach_rf_learner(base(*args,**kwargs),stack,actor)
    return plain,make

def execute(cfg,stack,student,actor,source,out,label,role):
    plain,make=make_factory(cfg,stack,student,actor,source);normal=source['kind']=='normal'
    r=ev.normal_run(out,label,source['seed'],make,stack.adapter) if normal else run_transient(out,label,source['protocol'],source['seed'],make,stack.adapter,source['kind']=='zero')
    env=plain.instances[-1];folder=ROOT/r['folder'];quality=None
    if normal:
        from go2_parameter_probe import save_probe
        save_probe(plain,folder);ref=next(q for q in read(ROOT/'artifacts/runs/20261027-release-regression/summary.json')['quality'] if q['result']['seed']==source['seed'])
        checks=ev.normal_checks(r,ref['reference']);rl=r['segments']['reverse right']['RL']['clearance_p95_m']*1000;r['passed']=bool(all(c['passed'] for c in checks) and rl>=29.5702276);quality=dict(result_folder=r['folder'],checks=checks,RL_mm=rl,passed=r['passed'])
    else:save_traction_probe(plain,folder,source['protocol'])
    dump(folder/'selector.json',env.recovery_selection_log);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
    np.savez_compressed(folder/'learner-actions.npz',**{k:np.asarray([x[k] for x in env.learner_calls]) for k in env.learner_calls[0]})
    (folder/'queries.pkl').write_bytes(pickle.dumps(dict(native=env.scoped_query_snapshots,scope=env.scoped_extra_before,burst=env.observed_burst_forks),protocol=5))
    first=next((i for i,l in enumerate(env.recovery_selection_log) if l['learner_applied']),None);assert raw_equal(r,source,first)
    r.update(kind=source['kind'],profile=source['profile'],source=source,role=role,first_learner_step=first,learner_calls=sum(l['learner_applied'] for l in env.recovery_selection_log),
        old_pass_lost=bool(source['passed'] and not r['passed']),added_termination=bool(not source['terminated'] and r['terminated']),all_raw_exact_source=raw_equal(r,source))
    dump(folder/'result.json',r);print('SCREEN141',role,r['case'],r['passed'],r.get('peak_tilt'),r.get('recovery_seconds'),r['steps'],flush=True)
    return r,quality

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.prepare_only:prepare();return
    torch.set_num_threads(1);cfg=read(ROOT/CONFIG);spec=cfg['random_feature_recovery'];ev.CONFIG=cfg
    out,interface=fit(cfg);records=[];quality=[];start=time.perf_counter();selected=False
    if interface['passed']:
        stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu');actor=RandomFeatureActor(ROOT/spec['parent_model'],out/'fitted-actor.npz')
        try:
            for i,source in enumerate(spec['core_sources']):
                r,q=execute(cfg,stack,student,actor,source,out,f'core-{i}','core');records.append(r);dump(out/'progress.json',records)
            selected=all(r['passed'] for r in records);dump(out/'core-selection.json',dict(core_all6_passed=selected,selected_before_optional_protection=True,model_sha256=sha(out/'fitted-actor.npz')))
            if selected:
                for i,source in enumerate(spec['protection_sources']):
                    r,q=execute(cfg,stack,student,actor,source,out,f'protection-{i}','protection');records.append(r)
                    if q:quality.append(q)
                    dump(out/'progress.json',records)
        except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
        finally:stack.close()
    for n,h in read(out/'input-sha256.json').items():assert sha(ROOT/n)==h,n
    verify_frozen130();summary=dict(config=cfg,records=records,quality=quality,interface_passed=interface['passed'],core_all6_passed=selected,
        qualified=bool(selected and len(records)==12 and all(r['passed'] for r in records)),optional_protection_opened=selected,
        new_physical_runs=len(records),new_model_fits=1,optimizer_updates=0,PPO_interactions=0,new_normal_runs=len(quality),new_normal_checks=sum(len(q['checks']) for q in quality),
        new_heldout_conditions=0,formal_selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-start)
    dump(out/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary)
    dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'})
    print('COMPLETE141',summary['qualified'],sum(r['passed'] for r in records),len(records),flush=True)

if __name__=='__main__':main()
