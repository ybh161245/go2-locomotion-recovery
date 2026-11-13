"""200: frozen197 versus historical174,63 new turn runs/117 proven identity reuse."""
import argparse,copy,csv,json,pickle,shutil,time,traceback
from pathlib import Path
import numpy as np
from archive_project import ROOT
from diagnose_go2_update import read,dump,sha
from run_go2_frozen157_exit_matrix import load,make_factory
from run_go2_full_heading_feedback import attach
from run_go2_full_heading_new_conditions import verify as verify198
from go2_parameter_probe import save_probe
from go2_contact_residual_learning import save_calls
from stress_go2_frozen import run_case,cases,command
import evaluate_go2_robust as ev

CONFIG=ROOT/'configs/go2-full-heading-matrix.json'
RUN=ROOT/'artifacts/runs/20261106-full-heading-matrix'
PUBLIC=ROOT/'docs/data/go2-full-heading-matrix'
PLAN=ROOT/'docs/experiments/2026-11-06-full-heading-matrix.md'
BASE=ROOT/'docs/data/go2-frozen157-exit-matrix/summary.json'
FREEZE=ROOT/'docs/data/go2-full-heading-new-conditions/new-scope-candidate-freeze.json'

def verify():
    verify198()
    for n,h in read(FREEZE)['sha256'].items():assert sha(ROOT/n)==h,n

def prepare():
    verify();assert not CONFIG.exists() and not RUN.exists() and not PLAN.exists()
    base=read(BASE);definitions=base['config']['frozen157_exit_matrix']['definitions']
    assert json.dumps(cases(),sort_keys=True)==json.dumps(definitions,sort_keys=True)
    assert len(base['records'])==180 and sum(r['historical_case']['phase']=='turn' for r in base['records'])==63
    dump(CONFIG,dict(experiment='GO2-261106-200',base='configs/go2-frozen157-exit-matrix.json',baseline=BASE.relative_to(ROOT).as_posix(),baseline_sha256=sha(BASE),
        frozen=FREEZE.relative_to(ROOT).as_posix(),frozen_sha256=sha(FREEZE),runner_sha256=sha(Path(__file__)),
        seeds=[94001,94002,94003],definitions=definitions,max_new_physical=63,candidate_nonturn_reuse=117,baseline_reuse=180,new_training=0,new_fits=0,new_normal46=0,
        acceptance='Retention:no historical pass lost or added termination across original180. Whole8deg/full600steps/no termination/withdrawal2s/planar.08/yaw.15/25consecutive/originalinfo;also retain oldpost6 score.',
        known_boundaries=['174baseline33failures','198normalfirstturnerror1.205deg','newrecoveryvisualpending'],
        reuse='117nonturn only if all original rows and selector logs requested yaw0/formal_weight0 and frozen control induction proves new guard never permitted. 198three actual normal protocols/324 passed checks referenced;0newnormal.',
        stop='Complete63once,no rule/model/threshold/material/reset tuning. Reuse proof failure stops before new physics. Qualification failure blocks promotion. No automatic formal selection or publication.'))
    PLAN.write_text("""# 200　冻结197的原60×3适用压力矩阵回归

2026-11-06。199两条新受推角度收益已验证，集中检查候选197是否损伤174原矩阵的已有成功。原60定义×94001/2/3，沿原材料、原生reset、请求指令、力、时长与原info评分。原147/180历史基线引用。63条turn候选真实新运行，其余117仅在控制恒等证明成立时复用。0训练拟合、0新normal；正常质量证据引用198实际三46秒/324项。

主回归门槛为不损失原174任何通过、不增加终止。全程8度/完整600步/原info末撤力2秒内平面.08yaw.15连续25步判据不变，原post6评分次级另列。允许继承原失败并公开其变化。至少修复一格才称本矩阵新增收益；分数相同且无损失仅证明保留此范围，199的角度收益按199范围保留。

先核117非turn原日志requested yaw及formal_weight全0。原router初始fw0且非turn指令yaw恒0，由状态更新归纳fw恒0；197作用域需fw1且请求yaw非0，所以extra恒0、actual精确shadow且dtype相同。新补偿无其它物理或命令写入。源代码/模型/原物理字节一致方能复用，任何证据缺失停在0新物理。

全部63条一次完成，不依据早期结果改参数。保存实际输出和下层影子、8数组、初态物理/遥测和原命令评分。首个实际输出改变前8数组及原初态匹配，实际输出到native目标逐步核对。最终原180逐格收益/损失/终止及连续yaw/峰/恢复/目标跳变/50Hz力矩代价全部保留。无复建环境、全网络重算、ZIP或总览。
""",encoding="utf-8")
    print('PREPARED200 63new/117identityreuse/180historicalbaseline',flush=True)

def prove_reuse(task,base):
    evidence=[];cfg=read(ROOT/task['base']);assert cfg['confirmed_selector']['selector']['max_requested_yaw']>=0
    for r in base['records']:
        c=r['historical_case']
        if c['phase']=='turn':continue
        p=ROOT/r['folder'];logs=read(p/'selector.json');assert len(logs)==r['steps']
        rows=list(csv.DictReader((p/'steps.csv').open(encoding='utf-8-sig')))
        assert len(rows)==len(logs)
        assert all(x['formal_weight']==0. and x['requested_command'][2]==0. for x in logs)
        assert all(float(x['command_yaw'])==0. for x in rows)
        assert all(command(c['phase'],i*.02)[2]==0 for i in range(600))
        with np.load(p/'initial-state.npz') as z:assert z['command'][2]==0
        files={n:sha(p/n) for n in ['states.npz','steps.csv','selector.json','initial-state.npz','physics.npz','telemetry.npz','result.json']}
        evidence.append(dict(case=c['id'],seed=r['seed'],folder=r['folder'],frames=len(logs),formal_weight_always0=True,guard_permitted_always_false=True,source_sha256=files))
    assert len(evidence)==117
    # Frozen original runtime byte check, including every model/normalization input already bound in174.
    original_inputs=read(ROOT/'artifacts/runs/20261102-frozen157-exit-matrix/input-sha256.json')
    for n,h in original_inputs.items():assert sha(ROOT/n)==h,n
    normal=read(ROOT/'docs/data/go2-full-heading-retention/summary.json')
    assert normal['qualified'] and sum(r['result']['kind']=='normal' for r in normal['records'])==3
    return dict(nonturn_conditions=117,baseline_passes=sum(r['passed'] for r in base['records'] if r['historical_case']['phase']!='turn'),
        induction='newenv formal_weight0;original nonturncommand yaw0=>turnfalse=>fw0 forever;197 permitted requires fw1 and requested yaw!=0=>extra0=>actual exact shadow with same dtype;no downstream command/material/state changes.',
        unchanged_original174_inputs=len(original_inputs),source_logic_sha256={n:sha(ROOT/n) for n in ['scripts/go2_fast_capture_router.py','scripts/run_go2_full_heading_feedback.py','scripts/stress_go2_frozen.py']},
        evidence=evidence,normal198_reference='docs/data/go2-full-heading-retention/summary.json',normal_budget324_reference_passed=True,new_physical=0)

def run():
    task=read(CONFIG);verify();assert sha(BASE)==task['baseline_sha256'] and sha(FREEZE)==task['frozen_sha256'] and sha(Path(__file__))==task['runner_sha256']
    RUN.mkdir(exist_ok=False);(RUN/'source').mkdir();shutil.copy2(__file__,RUN/'source'/Path(__file__).name);shutil.copy2(PLAN,RUN/'predeclared-plan.md');dump(RUN/'prospective.json',task)
    base=read(BASE);tick=time.perf_counter()
    try:
        proof=prove_reuse(task,base);dump(RUN/'nonturn-normal-identity-proof.json',proof)
        cfg=read(ROOT/task['base']);names=[CONFIG.relative_to(ROOT).as_posix(),PLAN.relative_to(ROOT).as_posix(),task['base'],task['baseline'],task['frozen'],'docs/data/go2-current/selection.json',
            'scripts/run_go2_full_heading_feedback.py','scripts/go2_fast_capture_router.py','scripts/stress_go2_frozen.py']
        for r in base['records']:names.extend(r['folder']+'/'+n for n in ['states.npz','steps.csv','initial-state.npz','physics.npz','telemetry.npz'])
        inputs={n:sha(ROOT/n) for n in names};dump(RUN/'input-sha256.json',inputs)
        print('REUSE200 117nonturn identity proven,102historical passes;198normal324 actual evidence referenced',flush=True)
        loaded=load(cfg);stack=loaded[0];records=[];actual=0;start=time.perf_counter()
        try:
            for old in base['records']:
                c=old['historical_case'];seed=old['seed']
                if c['phase']!='turn':
                    r=copy.deepcopy(old);r.update(controller='fullheading197',reused=True,new_physical=0,reuse_proof='nonturn-normal-identity-proof.json');records.append(r);continue
                plain,original=make_factory(cfg,loaded,c['friction'])
                def make(*a,**kw):return attach(original(*a,**kw),stack)
                r,packed,rows=run_case(c,seed,make,*stack.adapter,False);env=plain.instances[-1]
                packed.update(feet_world=np.asarray(env.extra_feet),heading=np.asarray(env.extra_heading))
                legacy=dict(passed=bool(r['passed']),post6_peak_tilt=r['peak_tilt'],post6_available=len(rows)>300)
                whole=max(max(abs(x['roll_deg']),abs(x['pitch_deg'])) for x in rows)
                r.update(historical_case=c,case=c['id'],kind='zero' if c['force']==0 else 'pressure',controller='fullheading197',peak_tilt=whole,legacy=legacy,
                    passed=bool(r['steps']==600 and not r['terminated'] and whole<=8 and r['recovery_seconds'] is not None and r['recovery_seconds']<=2),new_physical=1,reused=False)
                r=ev.save_run(RUN,'fullheading197',c['id'],seed,r,packed,rows);p=ROOT/r['folder'];save_probe(plain,p);save_calls(env,p)
                dump(p/'selector-shadow.json',env.recovery_selection_log);np.savez_compressed(p/'initial-state.npz',**env.validation_initial)
                calls=env.full_heading_calls;(p/'full-heading-calls.pkl').write_bytes(pickle.dumps(calls,protocol=5))
                first=next((i for i,x in enumerate(calls) if not np.array_equal(x['actual'],x['shadow'])),len(calls))
                with np.load(ROOT/old['folder']/'states.npz') as z:
                    assert set(z.files)==set(packed)
                    for k,v in packed.items():np.testing.assert_array_equal(v[:first],z[k][:first],err_msg=k)
                with np.load(ROOT/old['folder']/'initial-state.npz') as z:
                    assert set(z.files)==set(env.validation_initial)
                    for k,v in env.validation_initial.items():np.testing.assert_array_equal(v,z[k])
                for i,x in enumerate(calls):np.testing.assert_array_equal(np.clip(env.nominal+.35*x['actual'],env.model.actuator_ctrlrange[:,0],env.model.actuator_ctrlrange[:,1]),packed['targets'][i])
                d=np.abs(np.diff(packed['targets'],axis=0));lim=np.max(np.abs(env.model.actuator_forcerange),axis=1)
                r.update(active_steps=sum(x['extra']!=0 for x in calls),active_at_final=bool(calls[-1]['extra']!=0),first_actual_change_step=None if first==len(calls) else first,
                    yaw_rmse=float(np.sqrt(np.mean([(x['yaw_rate']-x['command_yaw'])**2 for x in rows]))),max_target_step=float(d.max()),p95_target_step=float(np.percentile(d,95)),
                    max_50Hz_torque_fraction=float(np.max([np.max(np.abs(x['actuator_force'])/lim) for x in env.probe_telemetry])))
                dump(p/'result.json',r);records.append(r);actual+=1;dump(RUN/f'progress-{actual:02}.json',r)
                print('MATRIX200',actual,c['id'],seed,r['passed'],round(whole,6),r['recovery_seconds'],'term',r['terminated'],flush=True)
            assert actual==63 and len(records)==180
            pairs=list(zip(base['records'],records));assert all((a['case'],a['seed'])==(b['case'],b['seed']) for a,b in pairs)
            gains=sum(not a['passed'] and b['passed'] for a,b in pairs);losses=sum(a['passed'] and not b['passed'] for a,b in pairs);terms=sum(not a['terminated'] and b['terminated'] for a,b in pairs)
            verify()
            for n,h in inputs.items():assert sha(ROOT/n)==h,n
            s=dict(config=task,records=records,baseline_passes=sum(a['passed'] for a,b in pairs),candidate_passes=sum(b['passed'] for a,b in pairs),
                legacy_passes=sum(r['legacy']['passed'] for r in records),terminations=sum(r['terminated'] for r in records),gains=gains,old_pass_losses=losses,new_terminations=terms,
                qualified=bool(not losses and not terms),logical_matrix=180,new_physical=63,candidate_reuse=117,baseline_reuse=180,new_training=0,new_fits=0,new_normal46=0,
                new_turn_zero=3,reused_nonturn_zero=9,normal324_from198=True,formal_selection_changed=False,old174_not_updated=True,new_visual='pending',
                physical_and_save_seconds=time.perf_counter()-start,wall_seconds=time.perf_counter()-tick)
            dump(RUN/'summary.json',s);PUBLIC.mkdir(exist_ok=False);dump(PUBLIC/'summary.json',s);shutil.copy2(RUN/'nonturn-normal-identity-proof.json',PUBLIC/'nonturn-normal-identity-proof.json')
            print('COMPLETE200',s['candidate_passes'],'/180 vs',s['baseline_passes'],'gains',gains,'losses',losses,'newterms',terms,'qualified',s['qualified'],flush=True)
        finally:stack.close()
    except BaseException:(RUN/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args();prepare() if args.prepare else run()
