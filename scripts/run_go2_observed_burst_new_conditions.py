"""131 frozen123/130 on eight prospectively declared new evaluation combinations."""
import argparse,copy,inspect,json,pickle,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from go2_observed_burst_frozen import verify_frozen130,FROZEN130,FROZEN130_SHA
from run_go2_scoped_new_conditions import semantic_signature
from go2_observed_entry_burst import attach_observed_entry_burst
from go2_successful_supply_student import attach_successful_supply_student
from go2_neighbor_entry_causal import snapshot
from go2_bilateral_pilot import Stack,CONTROL
from go2_release_regression import attach_release_capture
from go2_traction_protocol import attach_traction_schedule,save_traction_probe
from validate_go2_context_states import attach_initial_state,run_transient
from evaluate_go2_release_regression import raw_equal
from screen_go2_mass_course import factory,exact,norm_state
from go2_action_transfer import norm_load
import evaluate_go2_robust as ev
CONFIG='configs/go2-observed-burst-new-conditions.json'


def prepare():
    f=verify_frozen130();cfg=read(ROOT/f['config']);old_task=cfg['observed_entry_burst']
    cfg['observed_entry_burst']={k:copy.deepcopy(old_task[k]) for k in ['gate','student_model','student_norm']}
    specs=[
        ('A-mu10-early',1.,[(5.94,.6,-40)],231101,[0,0,0],0,None),
        ('B-mu10-phase',1.,[(6.12,.6,-40)],231102,[-.75,.35,-6],-.025,None),
        ('C-mu10-late',1.,[(6.30,.6,-40)],231103,[.75,-.35,6],.025,None),
        ('D-mu08-sustained',.8,[(6.08,.6,-40)],231104,[-1,.5,-4],-.03,None),
        ('E-mu08-short',.8,[(6.24,.2,-40)],231105,[1,-.5,4],.03,None),
        ('F-mu02-repeat',.2,[(6.10,.3,-40),(7.60,.3,-40)],231106,[0,0,0],0,None),
        ('G-mu025-capture',.25,[(6.18,.6,-30)],231107,[.5,.75,3],.02,None),
        ('H-traction-revisit',.2,[(6.34,.6,-30)],231108,[-.5,-.75,-3],-.02,[dict(step=0,mu=.2),dict(step=160,mu=.8),dict(step=260,mu=.2)])]
    conditions=[]
    for name,mu,events,seed,rpy,vy,traction in specs:
        name='131-'+name;profile=dict(id=name,seed=seed,rpy_deg=rpy,body_velocity=[0,vy,0])
        protocol=dict(id=name,tier='frozen131',profile=name,seconds=12,friction=mu,
            friction_schedule=traction or [dict(step=0,mu=mu)],
            schedule=[dict(start=0,end=12,phase='straight',command=[.2,0,0.])],
            events=[dict(start=t,duration=d,force=[0,fy,0],torque=[0,0,0]) for t,d,fy in events])
        conditions.append(dict(id=name,protocol=protocol,profile=profile))
    signatures={semantic_signature(c['protocol'],c['profile']):c['id'] for c in conditions};assert len(signatures)==8
    history=[];matches=[];seed_matches=[];seeds={c['profile']['seed'] for c in conditions}
    def scan(obj,name):
        if isinstance(obj,dict):
            if isinstance(obj.get('protocol'),dict) and isinstance(obj.get('profile'),dict):
                sig=semantic_signature(obj['protocol'],obj['profile'])
                if sig in signatures:matches.append(dict(condition=signatures[sig],summary=name,folder=obj.get('folder')))
            if obj.get('seed') in seeds:seed_matches.append(dict(seed=obj['seed'],summary=name))
            for value in obj.values():scan(value,name)
        elif isinstance(obj,list):
            for value in obj:scan(value,name)
    for path in sorted((ROOT/'artifacts/runs').glob('*/summary.json')):
        name=path.relative_to(ROOT).as_posix();history.append(dict(path=name,sha256=sha(path)));scan(read(path),name)
    assert not matches and not seed_matches,(matches,seed_matches)
    cfg.update(experiment_id='GO2-B-261028-131',run='artifacts/runs/20261028-observed-burst-new-conditions',public='docs/data/go2-observed-burst-new-conditions',predeclared_plan='docs/experiments/2026-10-28-observed-burst-new-conditions.md')
    cfg['observed_burst_new_conditions']=dict(frozen_candidate=FROZEN130,frozen_candidate_sha256=FROZEN130_SHA,
        conditions=conditions,controllers=['student123','observed130'],kinds=['zero','pressure'],max_physical_runs=32,
        training_samples=0,new_normal_runs=0,
        hypothesis='A frozen own-observed three-frame negative recovery entry pulse generalizes to new initial states/onsets/material combinations without losses to original123.',
        historical_signature_check=dict(summaries=history,matches=matches,seed_matches=seed_matches,scope='Direct artifacts/runs/*/summary.json nested protocol/profile semantic tuples; not all conceivable historical files.'),
        qualification='New130 all8pressure+8zero full12s/no termination/maxabsrollpitch<=8deg/recovery<=2s after last withdrawal/25 stable steps;zero events explicitly suppressed;no old123 pass loss or added termination. At least one old-failure to new-pass before claiming new-condition benefit;otherwise coverage/protection only.',
        continuous_tradeoff='Continuous deltas allowed within original8deg/2s budgets and all reported;no averaging away failed conditions. No new pressure foot-quality certificate.',
        stop='Exactly at most32 actual/0 training/0 fresh normals. No gate/frame count/bank/model/normalization/physical change after opening. Failure is retained and diagnosed separately,not retuned or backfilled.',
        counting='8new evaluation protocol/profile combinations,8declared seed/initial inputs not independent training or guaranteed independent entry states;all32 actual,0 condition reuse;0 new normal certificate.')
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg)
    lines=['# 131：冻结自身观测三步接管后的新条件配对（事前）','',
        '单一问题：130的自主三帧接管能否在未运行的新初态/起推相位/材料组合保持收益，且不丢原123通过。固定130清单SHA '+FROZEN130_SHA+'，固定6实际网络、各冻结归一化、20银行/.1、原fast和PD、原观测门槛与3帧长度。0训练/新机制。','',
        '|条件|摩擦时程|推力事件(秒/时长/负y牛)|seed|姿态度|初始body速度|','|---|---|---|---:|---|---|']
    for c in conditions:lines.append(f"|{c['id']}|{c['protocol']['friction_schedule']}|{c['protocol']['events']}|{c['profile']['seed']}|{c['profile']['rpy_deg']}|{c['profile']['body_velocity']}|")
    lines+=['','8组合×旧123/新130×零力/受推=32新实际，完整12秒/straight .2，0训练/0新46秒正常。8seed仅显式评估初态输入，不是独立训练重复或已证8个独立入口状态；实际入口raw50/规范化33维与历史如实保存。材料是全局接触材料变化，不是空间地形。','',
        '新条件资格：130八受推与八零力全部12秒不终止，全程max(|roll|,|pitch|)<=8度，最后撤力后<=2秒平面误差<=.08/yaw<=.15连续25步稳定；配对旧123不得丢通过或新增终止。至少一条旧失败→新通过才称新条件收益；双方全过仅覆盖/保护。连续指标允许在原预算内取舍，逐项报告所有变化，失败不能平均抵消。零力必须explicit True，抑制协议中保留的事件；身体规则可能在零力启动触发，不能称外推检测。','',
        '首推前两版各自零力/受推8原始数组精确，两版同初态、首实际改变前原始及完整query状态精确。保留integration/warmstart/ctrl/当前世界力/材料、fast/selector/scope/fullCONTROL/自身counter，以及原请求和实际动作dtype。原生21资产、实际import runtime、全部模型/种子/配置/失败/raw事前SHA留存，审计/渲染不算新物理。','',
        '动作代价另作诊断、不增加资格门槛：root在配对raw targets/telemetry上统计全程及恢复段最大和95分位单步关节目标变化(rad)、50Hz采样最大电机力矩限额占比，逐项报告上升与下降。未超8度/2秒不能据此宣称所有动作质量改善，压力自然性仍待用户。', '',
        '全部32运行完成，失败不提前删除或挑结果；打开后不修改门槛/帧长/银行/模型/物理回填资格，不追加PPO/扫参。若失败，下一有限机制诊断；若全部合格，仍仅此8新组合，不自动替换正式或直接宣称全矩阵通过。130正常324检查仍属130原范围，0本轮新正常；126138/180及旧088等分数不移植。全部新视觉待用户。']
    plan=ROOT/cfg['predeclared_plan'];assert not plan.exists();plan.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('PLAN131_FROZEN '+json.dumps(dict(config_sha256=sha(ROOT/CONFIG),plan_sha256=sha(plan),historical_summaries=len(history),matches=0,max_physical_runs=32)),flush=True)


def instrument(env,controller):
    """Passive131 evidence only; old123 native return remains untouched."""
    observed=controller=='observed130';modify=env.modify_joint_action;step=env.step;reset=env.reset
    student_modify=inspect.getclosurevars(modify).nonlocals['native_modify'] if observed else modify
    closure=inspect.getclosurevars(student_modify).nonlocals;unified=closure['native'];scope=closure['state']
    state=inspect.getclosurevars(modify).nonlocals['state'] if observed else dict(remaining=0)
    env.validation_full_queries={}
    if not observed:
        env.observed_burst_calls=[];env.observed_burst_forks={}
        def passive_modify(action):
            applied=modify(action);log=env.recovery_selection_log[-1];teacher=np.asarray(log['teacher_observation'],np.float32);formal=np.zeros(12,np.float32)
            log.update(shadow_applied_action=log['applied_action'],direct_override=False,direct_formal_action=formal.tolist(),burst_entry=False,burst_remaining_before=0,burst_remaining_after=0)
            env.observed_burst_calls.append(dict(teacher_observation=teacher.copy(),formal_action=formal,requested=np.asarray(applied).copy(),applied=np.asarray(applied).copy(),direct_override=False,burst_entry=False,remaining_before=0,remaining_after=0))
            return applied
        env.modify_joint_action=passive_modify
    def wrapped_step(action):
        i=env.step_count;prior=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0.
        before=dict(native=snapshot(env,unified),scope=copy.deepcopy(scope),full_control={k:copy.deepcopy(getattr(env,k)) for k in CONTROL},burst=copy.deepcopy(state)) if prior==0 else None
        result=step(action)
        if env.recovery_selection_log[-1]['selection_entry']:
            assert before is not None;env.validation_full_queries[i]=before
        return result
    def wrapped_reset(*args,**kwargs):
        env.validation_full_queries={}
        if not observed:env.observed_burst_calls=[];env.observed_burst_forks={}
        return reset(*args,**kwargs)
    env.step=wrapped_step;env.reset=wrapped_reset;return env


def make_factory(cfg,stack,student,condition,controller):
    p=condition['protocol'];plain=factory(stack.D,dict(id='131-'+controller,factor='friction',level=p['friction'],scope='frozen131-new-combinations'))
    def make(*args,**kwargs):
        env=plain(*args,**kwargs)
        if controller=='observed130':attach_observed_entry_burst(env,stack,student,cfg)
        else:
            assert controller=='student123'
            attach_successful_supply_student(env,*stack.formal,stack.default[0],student,stack.low[0],stack.default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],stack.bank,cfg['unified_recovery_selection'],cfg['scoped_expert_blend']['gate'])
        instrument(env,controller)
        return attach_release_capture(attach_initial_state(attach_traction_schedule(env,p),condition['profile']))
    return plain,make


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();return
    cfg=read(ROOT/CONFIG);task=cfg['observed_burst_new_conditions'];f=verify_frozen130();ev.CONFIG=cfg;out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for p in (ROOT/'scripts').glob('*.py'):shutil.copy2(p,out/'source'/p.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md')
    names={CONFIG,cfg['predeclared_plan'],FROZEN130,*f['models'],*f['runtime_code_sha256'],f['base123_frozen_manifest'],f['config'],f['summary'],f['audit'],*f['native_assets_sha256'],*f['bank_sha256']}
    names.update(p.relative_to(ROOT).as_posix() for m in list(sys.modules.values()) if getattr(m,'__file__',None) for p in [Path(m.__file__).resolve()] if p.parent==(ROOT/'scripts').resolve() and p.suffix=='.py')
    inputs={name:sha(ROOT/name) for name in sorted(names)};dump(out/'input-sha256.json',inputs);dump(out/'novelty-audit.json',task['historical_signature_check'])
    torch.set_num_threads(1);stack=Stack(cfg);student=PPO.load(ROOT/cfg['observed_entry_burst']['student_model'],device='cpu');assert exact(norm_state(stack.default[1]),norm_state(norm_load(ROOT/cfg['observed_entry_burst']['student_norm'])))
    records=[];zero_pairs=[];started=time.perf_counter()
    try:
        for c in task['conditions']:
            for controller in task['controllers']:
                own=[]
                for kind in task['kinds']:
                    plain,make=make_factory(cfg,stack,student,c,controller);r=run_transient(out,controller+'-'+kind,c['protocol'],c['profile']['seed'],make,stack.adapter,kind=='zero');env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,c['protocol'])
                    logs=env.recovery_selection_log;dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial)
                    calls=env.scoped_calls;np.savez_compressed(folder/'expert-pair.npz',**{k:np.asarray([v[k] for v in calls]) for k in ['normalized_observation','raw_observation','parent_action','learned_action','learned_evaluated','applied','output_is_float64','scope_entry']})
                    calls=env.observed_burst_calls;np.savez_compressed(folder/'direct-actions.npz',**{k:np.asarray([x[k] for x in calls]) for k in calls[0]})
                    (folder/'snapshots.pkl').write_bytes(pickle.dumps(dict(queries=env.scoped_query_snapshots,extra_before=env.scoped_extra_before,forks=env.observed_burst_forks,full_queries=env.validation_full_queries),protocol=5))
                    r.update(controller=controller,kind=kind,condition_id=c['id'],profile=c['profile'],burst_entries=[i for i,l in enumerate(logs) if l['burst_entry']],direct_calls=sum(l['direct_override'] for l in logs),default_calls=len(env.scoped_calls),scope_learned_steps=sum(l['scope_learned'] for l in logs),low_expert_steps=sum(l['actual_expert']=='low' for l in logs))
                    dump(folder/'result.json',r);records.append(r);own.append(r);dump(out/'progress.json',records);print('NEW131 '+json.dumps({k:r[k] for k in ['case','controller','kind','passed','peak_tilt','recovery_seconds','terminated','burst_entries','direct_calls']}),flush=True)
                onset=round(c['protocol']['events'][0]['start']/.02);assert raw_equal(own[0],own[1],onset);zero_pairs.append(dict(zero=own[0],push=own[1],preforce_exact=True,preforce_steps=onset))
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf8');raise
    finally:stack.close()
    comparisons=[]
    for c in task['conditions']:
        for kind in task['kinds']:
            old=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='student123');new=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='observed130');first=new['burst_entries'][0] if new['burst_entries'] else None;assert raw_equal(old,new,first)
            comparisons.append(dict(case=c['id'],kind=kind,parent=old,scoped=new,first_changed_step=first,prefix_exact=True,full_raw_arrays_exact=raw_equal(old,new),delta_peak=new['peak_tilt']-old['peak_tilt'],delta_recovery=None if old['recovery_seconds'] is None or new['recovery_seconds'] is None else new['recovery_seconds']-old['recovery_seconds'],old_pass_lost=bool(old['passed'] and not new['passed']),added_termination=bool(not old['terminated'] and new['terminated']),new_pass=bool(not old['passed'] and new['passed'])))
    passes={controller:{kind:sum(r['passed'] for r in records if r['controller']==controller and r['kind']==kind) for kind in task['kinds']} for controller in task['controllers']};losses=sum(x['old_pass_lost'] for x in comparisons);terms=sum(x['added_termination'] for x in comparisons);new_passes=sum(x['new_pass'] for x in comparisons if x['kind']=='pressure');qualified=passes['observed130']==dict(zero=8,pressure=8) and not losses and not terms
    for name,digest in inputs.items():assert sha(ROOT/name)==digest,name
    verify_frozen130();summary=dict(config=cfg,records=records,comparisons=comparisons,zero_pairs=zero_pairs,passes=passes,qualified=bool(qualified),new_condition_benefit=bool(qualified and new_passes),pressure_new_passes=new_passes,new_physical_runs=len(records),training_samples=0,PPO_updates=0,supervised_updates=0,new_normal_runs=0,new_normal_checks=0,old_pass_losses=losses,added_terminations=terms,qualification_scope='Eight prospectively frozen new evaluation combinations only;not full matrix,independent training replication or normal certificate.',formal_selection_changed=False,candidate130_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-started);assert len(records)==32;dump(out/'summary.json',summary);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',summary);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});print('COMPLETE131 '+json.dumps(dict(new_physical_runs=len(records),passes=passes,pressure_new_passes=new_passes,old_pass_losses=losses,added_terminations=terms,qualified=qualified)),flush=True)


if __name__=='__main__':main()
