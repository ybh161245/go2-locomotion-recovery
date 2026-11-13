"""124 frozen123 student versus frozen118 on eight new30N combinations; no training."""
import argparse,copy,json,pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from run_go2_scoped_new_conditions import verify_frozen,semantic_signature
from go2_bilateral_pilot import Stack
from go2_scoped_expert_blend import attach_scoped_expert_blend
from go2_successful_supply_student import attach_successful_supply_student
from go2_release_regression import attach_release_capture
from go2_traction_protocol import attach_traction_schedule,save_traction_probe
from validate_go2_context_states import attach_initial_state,run_transient
from screen_go2_mass_course import factory,exact,norm_state
from go2_action_transfer import norm_load
from evaluate_go2_release_regression import raw_equal
import evaluate_go2_robust as ev
CONFIG='configs/go2-supply-new-conditions.json'
FROZEN='docs/data/go2-successful-supply-student/frozen-candidate.json'
FROZEN_SHA='bf881280b63182b296b487ceea6c6dfee72c2a22942329175d71b828b9dc1333'

def verify_supply_frozen():
    verify_frozen();assert sha(ROOT/FROZEN)==FROZEN_SHA;f=read(ROOT/FROZEN);assert f['selected_checkpoint']==250
    for n,h in {**f['models'],**f['runtime_code_sha256']}.items():assert sha(ROOT/n)==h,n
    old=read(ROOT/f['config']);saved=read(ROOT/old['run']/'input-sha256.json');assert sha(ROOT/f['config'])==saved[f['config']];assert read(ROOT/old['run']/'summary.json')['qualified'];return f

def prepare():
    f=verify_supply_frozen();cfg=read(ROOT/f['config']);cfg.pop('successful_supply_student');sources=read(ROOT/'docs/data/go2-scoped-expert-blend/summary.json')['records'];conditions=[]
    specs=[('mu08-phase-622','mu08-B-core30',6.22,None),('mu08-phase-634','mu08-B-core30',6.34,None),('mu02-phase-622','mu02-B-positive30',6.22,None),('mu02-phase-634','mu02-B-positive30',6.34,None),('revisit-new-E','revisit-B-positive30',6.20,dict(id='124-E',seed=224101,rpy_deg=[1.25,-.75,5],body_velocity=[0,.035,0])),('mu04-new-F','mu04-B-core30',6.36,dict(id='124-F',seed=224102,rpy_deg=[-1.25,.75,-5],body_velocity=[0,-.035,0])),('negative-new-G','mu02-B-negative30',6.20,dict(id='124-G',seed=224103,rpy_deg=[.75,1,3.5],body_velocity=[0,.025,0])),('negative-revisit-H','revisit-B-negative30',6.36,dict(id='124-H',seed=224104,rpy_deg=[-.75,-1,-3.5],body_velocity=[0,-.025,0]))]
    for name,key,onset,profile in specs:
        r=next(r for r in sources if r['case']==key);p=copy.deepcopy(r['protocol']);q=copy.deepcopy(profile or r['profile']);p.update(id=name,tier='frozen124',profile=q['id']);p['events'][0]['start']=onset;assert len(p['events'])==1 and p['seconds']==12 and abs(p['events'][0]['force'][1])==30;ref={k:r[k] for k in ['folder','case','kind','seed','profile','protocol','steps','passed','terminated','peak_tilt','recovery_seconds']};conditions.append(dict(id=name,protocol=p,profile=q,source118=ref))
    candidates={semantic_signature(c['protocol'],c['profile']):c['id'] for c in conditions};assert len(candidates)==8;historical=[];matches=[]
    def scan(obj,path):
        if isinstance(obj,dict):
            if isinstance(obj.get('protocol'),dict) and isinstance(obj.get('profile'),dict):
                key=semantic_signature(obj['protocol'],obj['profile'])
                if key in candidates:matches.append(dict(condition=candidates[key],summary=path,folder=obj.get('folder')))
            for v in obj.values():scan(v,path)
        elif isinstance(obj,list):
            for v in obj:scan(v,path)
    for path in sorted((ROOT/'artifacts/runs').glob('*/summary.json')):
        name=path.relative_to(ROOT).as_posix();historical.append(dict(path=name,sha256=sha(path)));scan(read(path),name)
    assert not matches,matches;cfg.update(experiment_id='GO2-B-261028-124',run='artifacts/runs/20261028-supply-new-conditions',public='docs/data/go2-supply-new-conditions',predeclared_plan='docs/experiments/2026-10-28-supply-new-conditions.md');cfg['supply_new_conditions']=dict(frozen_candidate=FROZEN,frozen_candidate_sha256=FROZEN_SHA,student_model='artifacts/runs/20261028-successful-supply-student/checkpoint-250.zip',student_norm='artifacts/runs/20261028-successful-supply-student/checkpoint-250-vecnormalize.pkl',conditions=conditions,controllers=['baseline118','student123'],kinds=['zero','pressure'],max_physical_runs=32,training_samples=0,historical_signature_check=dict(summaries=historical,matches=matches,scope='Direct existing artifacts/runs/*/summary.json nested protocol/profile semantic combinations; not every conceivable historical file.'),qualification='Student all8 pressure and8 zero pass full12s/8deg/2s/.5s stable;no baseline passing-case loss or added termination. New30N evaluation combinations only,not broad robustness or normal certification.',stop='No new model/weights/gate/norm/bank changes after opening;no switch to1000 checkpoint or threshold fit. On failure preserve and diagnose actual supply/state before any next finite experiment. No180 or automatic formal adoption.',counting='32actual runs,8newpressure combinations,6distinctzero physical settings per controller;two phase pairs have repeated zero dynamics;0reuse,0training,0new46s normal.')
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg);p=ROOT/cfg['predeclared_plan'];assert not p.exists();lines=['# 124：冻结成功供给学生的新条件配对验证','', '2026-10-28用户要求下一步，go2保持暂停。单一问题：固定123选中250学生能否保持高μ捕获与低μ恢复整合，适应未运行的新初态/起推相位；本轮无训练/机制修改。','', '固定123清单SHA '+FROZEN_SHA+'、250模型与归一化、118原作用域/20银行/.1/fast/原生物理。原118使用parent+.90学习混合float64；123仅作用域内学生直接float32供给，其余parent/low不变。','', '|条件|旧协议|起推秒|seed|姿态度|初始body速度|','|---|---|---:|---:|---|---|']
    for c in conditions:lines.append(f"|{c['id']}|{c['source118']['case']}|{c['protocol']['events'][0]['start']}|{c['profile']['seed']}|{c['profile']['rpy_deg']}|{c['profile']['body_velocity']}|")
    lines+=['', '8新组合×118/123×零力/受推=32新实际，上限固定。六正向捕获/恢复与两负向保护；幅值30N、时长/指令/材料时程沿原协议。新seed/姿态/速度同时改，不单独归因初态某一分量。历史语义扫描先存，参数曾见不等于组合曾跑；8受推组合每版只有6类不同零力设置，仍全部真实运行，不能称32独立实验或独立训练种子。','', '资格事前固定：学生8受推与8零力全部完整12秒、不终止、全程max绝对roll/pitch<=8°、撤力后<=2秒进入平面速度误差<=.08/yaw<=.15且连续25步；相对同初态/同协议118不得丢通过/新增终止。允许原预算内连续取舍，逐条报告，不用平均抵消失败。0新正常检查，123三46秒324预算仅属123，不能计本轮新认证。','', '全部自身零力首推前8原数组精确，两版初态和首次改供给前原数组/完整native查询状态精确，作用域状态单列；保存原生integration/warmstart/ctrl/当前世界力/材料、两个专家动作及实际dtype。审计实际七个跨版本网络（每个运行六个）、原fast/物理/门槛。','', '新数据打开后不换1000/改阈值/原型或补训回填。若资格失败，先有限诊断实际选择/供给/支撑，保留所有失败，不能展开180或换正式。即使本轮全过，也只验证这8个30N组合；40N继承失败仍另列、运动自然性新视觉仍待用户，092认可不扩大。正式/107/118与123旧成绩保持。'];p.write_text('\n'.join(lines)+'\n',encoding='utf-8');print('PLAN124_FROZEN '+json.dumps(dict(new_physical_runs=32,historical_summaries=len(historical),matches=matches)),flush=True)

def make_factory(cfg,stack,expert,c,controller):
    p=c['protocol'];plain=factory(stack.D,dict(id='124-'+controller,factor='friction',level=p['friction'],scope='frozen124-new-combinations'))
    def make(*args,**kwargs):
        env=plain(*args,**kwargs);attach=attach_scoped_expert_blend if controller=='baseline118' else attach_successful_supply_student;attach(env,*stack.formal,stack.default[0],expert,stack.low[0],stack.default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],stack.bank,cfg['unified_recovery_selection'],cfg['scoped_expert_blend']['gate']);return attach_release_capture(attach_initial_state(attach_traction_schedule(env,p),c['profile']))
    return plain,make

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare();return
    cfg=read(ROOT/CONFIG);f=verify_supply_frozen();task=cfg['supply_new_conditions'];ev.CONFIG=cfg;out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir();scripts=list((ROOT/'scripts').glob('*.py'))
    for path in scripts:shutil.copy2(path,out/'source'/path.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');names={CONFIG,cfg['predeclared_plan'],FROZEN,f['config'],*f['models'],'artifacts/runs/20261028-successful-supply-student/input-sha256.json','artifacts/runs/20261028-successful-supply-student/summary.json'};names.update(p.relative_to(ROOT).as_posix() for p in scripts);names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/cfg['unified_recovery_selection']['bank']).rglob('*') if p.is_file())
    for c in task['conditions']:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/c['source118']['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs);dump(out/'novelty-audit.json',task['historical_signature_check']);torch.set_num_threads(1);stack=Stack(cfg);learned=PPO.load(ROOT/cfg['scoped_expert_blend']['learned_model'],device='cpu');student=PPO.load(ROOT/task['student_model'],device='cpu');assert exact(norm_state(stack.default[1]),norm_state(norm_load(ROOT/task['student_norm'])));records=[];pairs=[];start=time.perf_counter()
    try:
        for c in task['conditions']:
            for controller in task['controllers']:
                own=[]
                for kind in task['kinds']:
                    plain,make=make_factory(cfg,stack,learned if controller=='baseline118' else student,c,controller);r=run_transient(out,controller+'-'+kind,c['protocol'],c['profile']['seed'],make,stack.adapter,kind=='zero');env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,c['protocol']);logs=env.recovery_selection_log;dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial);calls=env.scoped_calls;np.savez_compressed(folder/'expert-pair.npz',**{k:np.asarray([v[k] for v in calls]) for k in ['normalized_observation','raw_observation','parent_action','learned_action','learned_evaluated','applied','output_is_float64','scope_entry']});(folder/'snapshots.pkl').write_bytes(pickle.dumps(dict(queries=env.scoped_query_snapshots,extra_before=env.scoped_extra_before),protocol=5));r.update(controller=controller,kind=kind,condition_id=c['id'],profile=c['profile'],source118=c['source118'],scope_learned_steps=sum(l['scope_learned'] for l in logs),default_calls=len(calls),low_expert_steps=sum(l['actual_expert']=='low' for l in logs));dump(folder/'result.json',r);records.append(r);own.append(r);dump(out/'progress.json',records);print('NEW124 '+json.dumps({k:r[k] for k in ['case','controller','kind','passed','peak_tilt','recovery_seconds','terminated','scope_learned_steps']}),flush=True)
                assert raw_equal(own[0],own[1],round(c['protocol']['events'][0]['start']/.02));pairs.append(dict(zero=own[0],push=own[1],preforce_exact=True))
    except BaseException:(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:stack.close()
    comparisons=[]
    for c in task['conditions']:
        for kind in task['kinds']:
            old=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='baseline118');new=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='student123');comparisons.append(dict(case=c['id'],kind=kind,parent=old,scoped=new,full_raw_arrays_exact=raw_equal(old,new),delta_peak=new['peak_tilt']-old['peak_tilt'],delta_recovery=None if old['recovery_seconds'] is None or new['recovery_seconds'] is None else new['recovery_seconds']-old['recovery_seconds'],old_pass_lost=bool(old['passed'] and not new['passed']),added_termination=bool(not old['terminated'] and new['terminated'])))
    passes={controller:{kind:sum(r['passed'] for r in records if r['controller']==controller and r['kind']==kind) for kind in task['kinds']} for controller in task['controllers']};losses=sum(x['old_pass_lost'] for x in comparisons);terms=sum(x['added_termination'] for x in comparisons);qualified=passes['student123']==dict(zero=8,pressure=8) and not losses and not terms
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    verify_supply_frozen();s=dict(config=cfg,records=records,comparisons=comparisons,zero_pairs=pairs,passes=passes,qualified=bool(qualified),new_physical_runs=len(records),training_samples=0,PPO_updates=0,supervised_updates=0,new_normal_runs=0,new_normal_checks=0,qualification_scope='Eight predeclared frozen new30N evaluation combinations only,not independent training or broad robustness.',old_pass_losses=losses,added_terminations=terms,formal_selection_changed=False,research_candidate_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-start);assert len(records)==32;dump(out/'summary.json',s);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',s);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});print('COMPLETE124 '+json.dumps(dict(passes=passes,old_pass_losses=losses,added_terminations=terms,qualified=qualified)),flush=True)

if __name__=='__main__':main()
