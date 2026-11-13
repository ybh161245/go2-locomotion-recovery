"""157 one new successful-data fit, four core tests and six conditional protections."""
import argparse,pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_bilateral_pilot import Stack
from go2_random_feature_recovery import RandomFeatureActor
from go2_action_transfer import norm_load
from go2_contact_residual_learning import attach_contact_residual,combine,save_calls
from go2_contact_success_transfer import ContactSuccessActor,feature
from run_go2_contact_residual_learning import CONFIG as CONFIG155
from run_go2_random_feature_recovery import make_factory as parent_factory
from run_go2_single_wrench_pulse import runtime_files
from run_go2_random_feature_new_conditions import costs
from validate_go2_context_states import run_transient,pressure_metrics
from go2_traction_protocol import save_traction_probe
from go2_parameter_probe import save_probe
from go2_env import servo_model
from evaluate_go2_release_regression import raw_equal
from audit_go2_recovery_screen import rows
import evaluate_go2_robust as ev

CONFIG='configs/go2-contact-success-transfer.json'


def prepare():
    cfg=read(ROOT/CONFIG155);old=cfg.pop('contact_residual_learning');s=read(ROOT/cfg['run']/'summary.json')
    retained=read(ROOT/'artifacts/runs/20261029-retention-protected-recovery/summary.json')
    teachers=[dict(id='first-phase626-success',kind='episode',path=cfg['run']+'/episodes/episode-00044.npz',
                   metadata=cfg['run']+'/episodes/episode-00044.json',frames=132)]
    for case in ['146-phase634','146-duration050']:
        r=next(r for r in s['records'] if r['case']==case and r['checkpoint']==32768 and r['passed'])
        teachers.append(dict(id='retained-'+case,kind='evaluation',path=r['folder']+'/contact-calls.pkl',states=r['folder']+'/states.npz',frames=r['residual_frames']))
    cfg.update(experiment_id='GO2-B-261030-157',run='artifacts/runs/20261030-contact-success-transfer',public='docs/data/go2-contact-success-transfer',predeclared_plan='docs/experiments/2026-10-30-contact-success-transfer.md')
    cfg['contact_success_transfer']=dict(actor=old['actor'],sources=old['sources'],teachers=teachers,
        protections=[r for r in retained['records'] if r.get('role')=='protection'],residual_scale=2.,feature_seed=257101,features=2048,frequency_scale=4/np.sqrt(70),ridge=1e-8,
        max_source_native_RMSE_deg=.05,max_joint_native_error_deg=.1,new_model_fits=1,PPO_interactions=0,optimizer_updates=0,max_new_physical_runs=10,
        gain_cases=['146-phase626','146-phase634','146-duration050'],boundary_case='146-duration070',
        gate='All three predeclared gain/protection cases pass unchanged12s8deg2s25steps; all four including longpush reported. Longpush is an already failed boundary, not secretly recategorized155 or146. Any old passed loss or added termination blocks protection.',
        qualification='Targeted component benefit only if gain gate and all6 original protections pass. All-four qualification separately recorded. Overall failure/boundary never averaged away, no formal replacement.',
        stop='One first-success episode44 + two fixed32768 successful traces, one seed/width/ridge/equalrow fit. No best-episode selection, extra teacher, weighting/feature/regularization scan or PPO. Core no gain -> no protection. No new conditions or180.')
    names={CONFIG155,'docs/data/go2-contact-residual-diagnosis/summary.json','scripts/go2_contact_success_transfer.py','scripts/run_go2_contact_success_transfer.py'}
    for t in teachers:
        names.add(t['path']);names.add(t.get('metadata',t.get('states')))
    cfg['contact_success_transfer']['prospective_inputs']={n:sha(ROOT/n) for n in sorted(names)}
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);p=ROOT/cfg['predeclared_plan'];assert not p.exists()
    p.write_text('''# 157：新成功经验的确定性接触残差迁移（事前）

156发现phase626已有7条真实随机成功但检查点全部失败，长推则0/299完整成功。本轮只检验是否能把新发现的一个可行行为保留成自主观测驱动策略，不是续训155，也不是重开137/138同数据监督。冻结145父链/原规范化/50+20传感/原恢复权限、PD和动作界限。

事前固定取首次成功episode44（非最小倾角或最好回报）：132真实eligible动作；另保留155checkpoint32768的phase634和duration050两个真实完整成功路径92/75次动作。共299行70维float32输入，标签是实际native动作，生成Gaussian均值不作成功动作。没有新物理采集标签，不平均七条共享初始输入的随机动作。若精确重复/冲突，原样报告并停，不删除/平均。

新函数residual70=clip(Phi70 B,-1,1)，Phi=√(2/2048)cos(xW+b)，seed257101，频率4/√70；只一次等权ridge1e-8求解B。标签residual=(actual-parent145_shadow)/2，使不可见的超限残差不占拟合目标。原父输出每源逐位核验，全部真实target与native nominal±.35/ctrl界限配对；新residual仍按155执行float32后与float64父合成。这个新解析头也是学习，不称零学习；0优化器/PPO。保存真实标签、W/b/Phi/Gram/dual/B和残差，不做重复整栈/全部梯度审计。

接口预检每源native RMSE<=.05度、任一关节误差<=.1度，低拟合误差不是闭环通过。首次运行前保存代码/配置/输入SHA；接口不过0物理停，不调参。

先四次146原已打开条件，原info完整12秒/全程8度/最后撤力后2秒平面.08/yaw.15连续25步不改。有限收益门：phase626新增通过且phase634/duration050原通过保住、任何原通过损失/新增终止均阻止继续。duration070已失败边界仍完整运行并逐项报告，整体4/4资格单独记录，不能把3/4叫全通过，也不能回填155、146或其他历史分数。对长推的连续恶化如实报告，不证明它无解。

仅三目标收益门通过后先冻结唯一模型SHA，再开6原145保护：nominal40/low94001受推、constant-early完整零力和三46秒正常324原预算/RL29.5702276mm。最多10新实际，其中3normal与1zero为条件打开后实际新跑，0新留出/训练种子。保护任一失败则整体不采用，即便phase626修复；保留全部退步，不加教师或扫描参数。全过也只针对已打开子目标的有限研究候选，边界失败不消失，新条件下一轮。

保留真实父链/运行时源码/成功失败/raw/视频，首次实际残差前8数组对原145精确，原门槛重算、实际动作与native目标校验。无重复ZIP/复制树/交付总览，无自动发布或正式替换。新的恢复视觉待用户，092认可不扩大。
''',encoding='utf8')
    print('PREPARED157',sha(ROOT/CONFIG),sha(p),flush=True)


def labels(task):
    xs=[];parents=[];actuals=[];targets=[];source_ids=[];source_steps=[]
    for sid,t in enumerate(task['teachers']):
        if t['kind']=='episode':
            with np.load(ROOT/t['path']) as z:
                mask=z['actor_applied'];x=z['observation'][mask];parent=z['parent_action'][mask];actual=z['actual_action'][mask];target=z['targets'][mask];steps=np.flatnonzero(mask)+read(ROOT/t['metadata'])['start_step']
        else:
            calls=pickle.loads((ROOT/t['path']).read_bytes());mask=np.asarray([c['actor_applied'] for c in calls]);cs=[c for c in calls if c['actor_applied']]
            x=np.asarray([c['observation'] for c in cs]);parent=np.asarray([c['parent'] for c in cs]);actual=np.asarray([c['actual'] for c in cs]);steps=np.asarray([c['step'] for c in cs])
            with np.load(ROOT/t['states']) as z:target=z['targets'][mask].copy()
        assert len(x)==t['frames'];xs.append(x);parents.append(parent);actuals.append(actual);targets.append(target);source_ids.extend([sid]*len(x));source_steps.extend(steps.tolist())
    return dict(normalized70=np.concatenate(xs),parent=np.concatenate(parents),actual=np.concatenate(actuals),native_targets=np.concatenate(targets),source_id=np.asarray(source_ids,np.int16),source_step=np.asarray(source_steps,np.int32))


def execute(cfg,task,stack,student,actor,norm,model,source,out,label):
    plain,base=parent_factory(cfg,stack,student,actor,source)
    def make(*a,**kw):return attach_contact_residual(base(*a,**kw),norm,task['residual_scale'],model)
    normal=source['kind']=='normal'
    r=ev.normal_run(out,label,source['seed'],make,stack.adapter) if normal else run_transient(out,label,source['protocol'],source['seed'],make,stack.adapter,source['kind']=='zero')
    env=plain.instances[-1];folder=ROOT/r['folder'];quality=None
    if normal:
        save_probe(plain,folder);ref=next(q for q in read(ROOT/'artifacts/runs/20261027-release-regression/summary.json')['quality'] if q['result']['seed']==source['seed'])
        checks=ev.normal_checks(r,ref['reference']);rl=r['segments']['reverse right']['RL']['clearance_p95_m']*1000
        r['passed']=bool(all(c['passed'] for c in checks) and rl>=29.5702276);quality=dict(case=r['case'],seed=source['seed'],checks=checks,RL_mm=rl,passed=r['passed'])
    else:
        save_traction_probe(plain,folder,source['protocol'])
        if source['kind']!='zero':
            for k,v in pressure_metrics(rows(folder),source['protocol']).items():assert r[k]==v,k
    save_calls(env,folder);dump(folder/'selector.json',env.recovery_selection_log);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
    first=next((c['step'] for c in env.contact_calls if c['changed']),None);assert raw_equal(r,source,first)
    r.update(controller='145-plus-contact-success-transfer',kind=source['kind'],profile=source.get('profile'),source=source,first_residual_change=first,
       residual_frames=sum(c['actor_applied'] for c in env.contact_calls),old_pass_lost=bool(source['passed'] and not r['passed']),added_termination=bool(not source['terminated'] and r['terminated']),
       full_raw_exact_source=bool(raw_equal(r,source)),quality=quality)
    if not normal:r['costs']=costs(folder)
    dump(folder/'result.json',r);print('EVAL157',source['case'],source['kind'],r['passed'],r.get('peak_tilt'),r.get('recovery_seconds'),r['terminated'],flush=True)
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.prepare:prepare();return
    cfg=read(ROOT/CONFIG);task=cfg['contact_success_transfer'];out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir();torch.set_num_threads(1);ev.CONFIG=cfg;started=time.perf_counter()
    stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu')
    actor=RandomFeatureActor(ROOT/cfg['random_feature_recovery']['parent_model'],ROOT/task['actor']);norm=norm_load(ROOT/cfg['random_feature_recovery']['parent_norm'])
    names=set(runtime_files())|{CONFIG,cfg['predeclared_plan'],'docs/data/go2-current/selection.json'}|set(task['prospective_inputs'])
    names.update(read(ROOT/'docs/data/go2-retention-protected-recovery/frozen-candidate.json')['models'])
    for r in task['sources']+task['protections']:names.add(r['folder']+'/states.npz')
    hashes={n:sha(ROOT/n) for n in sorted(names)}
    for n in task['prospective_inputs']:assert hashes[n]==task['prospective_inputs'][n],n
    for n in names:
        if n.startswith('scripts/'):shutil.copy2(ROOT/n,out/'source'/n.split('/')[-1])
    dump(out/'input-sha256.json',hashes);dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    dump(out/'runtime-freeze.json',dict(saved_before_fit_and_physics=True,config_sha256=sha(ROOT/CONFIG),plan_sha256=sha(ROOT/cfg['predeclared_plan']),validation_level='lean development'))
    try:
        ref=labels(task);x=ref['normalized70'];assert len(x)==299 and len(np.unique(x,axis=0))==len(x),'Exact repeated/conflicting source inputs; stop without deleting labels'
        predicted_parent=np.asarray([actor.predict(row[:50].copy(),deterministic=True)[0] for row in x]);np.testing.assert_array_equal(predicted_parent,ref['parent'])
        nominal=np.tile([0.,.8,-1.6],4);limits=servo_model().actuator_ctrlrange
        target=np.clip(nominal+.35*ref['actual'],limits[:,0],limits[:,1])
        np.testing.assert_array_equal(target,ref['native_targets'])
        desired=(ref['actual']-ref['parent'])/task['residual_scale'];assert np.max(abs(desired))<=1
        rng=np.random.default_rng(task['feature_seed']);W=rng.normal(0,task['frequency_scale'],(70,task['features']));phase=rng.uniform(0,2*np.pi,task['features'])
        phi=feature(x,W,phase);gram=phi@phi.T;dual=np.linalg.solve(gram+task['ridge']*np.eye(len(x)),desired);B=phi.T@dual
        np.savez_compressed(out/'fitted-residual.npz',frequency=W,phase=phase,coefficient=B)
        np.savez_compressed(out/'fit-evidence.npz',**ref,desired_residual=desired,features=phi,gram=gram,dual=dual)
        model=ContactSuccessActor(out/'fitted-residual.npz');pred=np.asarray([combine(p,np.clip(model.predict(row)[0],-1,1).astype(np.float32),task['residual_scale']) for row,p in zip(x,ref['parent'])])
        err=np.degrees(.35*(pred-ref['actual']));per_source=[float(np.sqrt(np.mean(err[ref['source_id']==i]**2))) for i in range(len(task['teachers']))]
        interface=bool(max(per_source)<=task['max_source_native_RMSE_deg'] and abs(err).max()<=task['max_joint_native_error_deg'])
        dump(out/'fit.json',dict(interface_passed=interface,source_native_RMSE_deg=per_source,max_joint_error_deg=float(abs(err).max()),unique_inputs=299,
          parent_exact=True,native_label_targets_exact=True,new_analytic_fits=1,linear_residual_max=float(abs((gram+task['ridge']*np.eye(len(x)))@dual-desired).max())))
        assert interface,'Fixed fit interface failed; no physical run or retuning permitted'
        records=[]
        for i,source in enumerate(task['sources']):
            r=execute(cfg,task,stack,student,actor,norm,model,source,out,f'core-{i}');r['role']='core';records.append(r);dump(out/'progress.json',records)
        gate=all(next(r['passed'] for r in records if r['case']==c) for c in task['gain_cases']) and not any(r['old_pass_lost'] or r['added_termination'] for r in records)
        if gate:
            dump(out/'core-frozen.json',dict(model_sha256=sha(out/'fitted-residual.npz'),fit_sha256=sha(out/'fit.json'),gate_cases=task['gain_cases'],all_four_passed=all(r['passed'] for r in records),saved_before_protections=True))
            for i,source in enumerate(task['protections']):
                r=execute(cfg,task,stack,student,actor,norm,model,source,out,f'protection-{i}');r['role']='protection';records.append(r);dump(out/'progress.json',records)
        assert sha(ROOT/'docs/data/go2-current/selection.json')==hashes['docs/data/go2-current/selection.json']
        report=dict(config=cfg,records=records,core_passes=sum(r['passed'] for r in records if r['role']=='core'),target_gain_gate=bool(gate),all_four_core_passed=all(r['passed'] for r in records if r['role']=='core'),
             protection_opened=bool(gate),protection_passes=sum(r['passed'] for r in records if r['role']=='protection'),
             targeted_component_qualified=bool(gate and all(r['passed'] for r in records if r['role']=='protection')),
             new_physical_runs=len(records),new_analytic_fits=1,PPO_interactions=0,optimizer_updates=0,new_feature_seeds=1,new_normal_runs=sum(r['kind']=='normal' for r in records),new_heldout_conditions=0,
             old_pass_losses=[r['case']+'/'+r['kind'] for r in records if r['old_pass_lost']],added_terminations=[r['case']+'/'+r['kind'] for r in records if r['added_termination']],
             formal_selection_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started)
        dump(out/'summary.json',report);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',report);shutil.copy2(out/'fit.json',public/'fit.json')
        print('COMPLETE157',report['core_passes'],report['protection_passes'],report['targeted_component_qualified'],'seconds',report['elapsed_seconds'],flush=True)
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()


if __name__=='__main__':main()
