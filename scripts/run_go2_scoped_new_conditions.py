"""119 frozen118 versus original098 on eight predeclared initial/onset combinations."""
import argparse,copy,json,pickle,shutil,time,traceback
import numpy as np
import torch
from stable_baselines3 import PPO
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
import evaluate_go2_robust as ev
from go2_bilateral_pilot import Stack
from go2_scoped_expert_blend import attach_scoped_expert_blend
from go2_unified_recovery_selection import attach_unified_recovery_selection
from go2_transition_env_v3 import Go2TransitionEnvV3
from go2_neighbor_entry_causal import snapshot
from go2_release_regression import attach_release_capture
from go2_traction_protocol import attach_traction_schedule,save_traction_probe
from validate_go2_context_states import attach_initial_state,run_transient
from evaluate_go2_release_regression import raw_equal
from screen_go2_mass_course import factory,exact,norm_state
from go2_action_transfer import norm_load
CONFIG='configs/go2-scoped-new-conditions.json'
FROZEN='docs/data/go2-scoped-expert-blend/frozen-candidate.json'
FROZEN_SHA='4607fde539ffedafaca1c6928c40c661807fc1b34815e6214a2014eb24783ecc'


def verify_frozen():
    assert sha(ROOT/FROZEN)==FROZEN_SHA;f=read(ROOT/FROZEN)
    assert sha(ROOT/f['config'])==f['config_sha256']
    for n,h in f['models'].items():assert sha(ROOT/n)==h,n
    for n,h in f['runtime_source_sha256'].items():assert sha(ROOT/'scripts'/n)==h,n
    assert sha(ROOT/f['selection']['bank']/'bank.npz')==f['bank_sha256']
    assert sha(ROOT/'docs/data/go2-current/selection.json')==f['formal_sha256']
    assert sha(ROOT/'docs/data/go2-release-regression/research-candidate.json')==f['parent107_sha256']
    return f


def semantic_signature(protocol,profile):
    p={k:protocol.get(k) for k in ['seconds','friction','friction_schedule','schedule','events']}
    q={k:profile.get(k) for k in ['seed','rpy_deg','body_velocity']}
    return json.dumps([p,q],sort_keys=True,separators=(',',':'))


def prepare():
    f=verify_frozen();cfg=read(ROOT/f['config']);task=cfg['scoped_expert_blend'];task.pop('sources');task.pop('inputs')
    cfg.update(experiment_id='GO2-B-261028-119',run='artifacts/runs/20261028-scoped-new-conditions',public='docs/data/go2-scoped-new-conditions',predeclared_plan='docs/experiments/2026-10-28-scoped-new-conditions.md')
    old=read(ROOT/'docs/data/go2-scoped-expert-blend/summary.json')['records'];conditions=[]
    specs=[('mu08-phase-624','mu08-B-core30',6.24,None),('mu08-phase-632','mu08-B-core30',6.32,None),('mu02-phase-624','mu02-B-positive30',6.24,None),('mu02-phase-632','mu02-B-positive30',6.32,None),('revisit-new-A','revisit-B-positive30',6.18,dict(id='119-A',seed=219101,rpy_deg=[1,-.5,4],body_velocity=[0,.03,0])),('mu04-new-B','mu04-B-core30',6.38,dict(id='119-B',seed=219102,rpy_deg=[-1,.5,-4],body_velocity=[0,-.03,0])),('negative-new-C','mu02-B-negative30',6.18,dict(id='119-C',seed=219103,rpy_deg=[.5,.75,3],body_velocity=[0,.02,0])),('nominal40-new-D','nominal-40',6.22,dict(id='119-D',seed=219104,rpy_deg=[-.5,-.75,-3],body_velocity=[0,-.02,0]))]
    for name,source,onset,profile in specs:
        r=next(r for r in old if r['case']==source);p=copy.deepcopy(r['protocol']);p['id']=name;p['events'][0]['start']=onset;p['tier']='frozen119';q=copy.deepcopy(profile or r['profile']);p['profile']=q['id'];assert len(p['events'])==1 and p['seconds']==12
        ref={k:r[k] for k in ['folder','case','kind','seed','profile','protocol','steps','passed','terminated','peak_tilt','recovery_seconds']};conditions.append(dict(id=name,protocol=p,profile=q,source118=ref))
    signatures={semantic_signature(c['protocol'],c['profile']):c['id'] for c in conditions};assert len(signatures)==8
    historical=[];matches=[]
    for path in sorted((ROOT/'artifacts/runs').glob('*/summary.json')):
        historical.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)))
        def inspect(obj):
            if isinstance(obj,dict):
                if isinstance(obj.get('protocol'),dict) and isinstance(obj.get('profile'),dict):
                    sig=semantic_signature(obj['protocol'],obj['profile'])
                    if sig in signatures:matches.append(dict(condition=signatures[sig],summary=path.relative_to(ROOT).as_posix(),folder=obj.get('folder')))
                for value in obj.values():inspect(value)
            elif isinstance(obj,list):
                for value in obj:inspect(value)
        inspect(read(path))
    assert not matches,matches
    cfg['scoped_new_conditions']=dict(frozen_candidate=FROZEN,frozen_candidate_sha256=FROZEN_SHA,conditions=conditions,max_physical_runs=32,training_samples=0,controllers=['parent098','scoped118'],kinds=['zero','pressure'],historical_signature_check=dict(summaries=historical,matches=matches,scope='Existing direct artifacts/runs/*/summary.json and nested record protocol/profile tuples, not a claim of all conceivable past artifacts.'),qualification='scoped118 all8 pressure and8 zero pass full12s/8deg/2s/.5s stable; no parent passing-case losses or added terminations. No normal new certification;118 normal budgets unchanged.',stop='No weight/gate/bank/model change after opening. If fail retain failures and diagnose observed state/action/support before another finite branch; no sweep/PPO/full180/formal adoption.',counting='32 actual runs,8 push protocols,6 distinct physical zero settings per controller; the two same-initial phase pairs have repeated zero dynamics. No condition reuse or independent training replication.',scope='Frozen new evaluation combinations, not independent training seeds or broad robustness.')
    assert not (ROOT/CONFIG).exists();dump(ROOT/CONFIG,cfg)
    lines=['# 119：冻结正向作用域后的新初态与起推相位验证','', '2026-10-28，用户返回并要求下一步；go2定时续作暂停。先冻结本方案，再打开全部结果。','', '单一问题：118在已见开发12/12通过，能否在相邻起推相位及新初态组合下保持收益与旧通过。固定118冻结清单、6网络/归一化、0.90、原观测判据、20银行/0.1半径/原fast/原生PD；不训练或修改任何控制参数。','', '|条件|参考旧场景|起推秒|seed|姿态度|初始body速度|','|---|---|---:|---:|---|---|']
    for c in conditions:lines.append(f"|{c['id']}|{c['source118']['case']}|{c['protocol']['events'][0]['start']}|{c['profile']['seed']}|{c['profile']['rpy_deg']}|{c['profile']['body_velocity']}|")
    lines+=['','前三种材料/正负方向的幅值、持续时间、指令和材料时程沿对应旧协议；前四条件只改起推时刻，后四同时改显式机身初态/种子/起推时刻，不能单因素归因。所有协议完整12秒。','', '有限预算：8条件×原107/098栈与冻结118×零力/受推=32新实际。首四相位条件中两对零力动力学重复，8逻辑零力条件实际只有6类独特初态/材料/指令；仍全部真实运行并分开列计，不冒充32独立场景。0训练/更新、0旧运行条件复用、0新正常运动协议。','', '资格先存：118八受推及八零力均完整12秒、全程max(|roll|,|pitch|)≤8度、撤力后≤2秒进入平面速度误差≤.08和yaw误差≤.15并连续25步稳定；相较配对parent不得丢失通过或新增终止。必须全部满足，不能以平均提升抵消失败。原正常质量预算保持，但118的324检查不重复计成本轮新认证。','', '完整保留两版同初态、各自零力首推前8原数组、完整查询integration/warmstart/控制/当前力/材料与新增作用域状态，实测入口相对旧118的raw50/规范化33维及前.2秒历史差；新初态若收敛为相近入口如实报告。','', '历史新颖性查既有run summary嵌套protocol/profile语义组合（忽略标签，不忽略种子/姿态/速度/物理时程），不等于全谱系未见任何相同参数。新条件打开后不回调阈值/权重/模型/原型，不回填118或旧分数；失败另立有限机制诊断，禁止本轮扫参/追加PPO/180/自动正式替换。所有退步和失败保留，新视觉待用户。']
    (ROOT/cfg['predeclared_plan']).write_text('\n'.join(lines)+'\n',encoding='utf-8');print('PREDECLARED_119 '+json.dumps(dict(conditions=8,max_physical=32,historical_summaries=len(historical),matches=0)),flush=True);return cfg


def attach_parent_capture(env,stack,cfg):
    env.scoped_calls=[];env.scoped_query_snapshots={};env.scoped_extra_before={}
    class Default:
        def predict(self,observation,deterministic=True):
            a,other=stack.default[0].predict(observation,deterministic=deterministic)
            env.scoped_calls.append(dict(normalized_observation=np.asarray(observation).copy(),raw_observation=Go2TransitionEnvV3.observe(env).copy(),parent_action=np.asarray(a).copy(),learned_action=np.zeros_like(a),learned_evaluated=False,applied=np.asarray(a).copy(),output_is_float64=False,scope_entry=False));return a,other
    attach_unified_recovery_selection(env,*stack.formal,Default(),stack.low[0],stack.default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],stack.bank,cfg['unified_recovery_selection']);native=env.modify_joint_action;native_step=env.step;native_reset=env.reset
    def step(action):
        i=env.step_count;prior=env.recovery_selection_log[-1]['weight'] if env.recovery_selection_log else 0.;candidate=snapshot(env,native) if prior==0 else None;result=native_step(action);l=env.recovery_selection_log[-1];l.update(scope_enabled=None,scope_entry=False,scope_learned=False,expert_output_dtype='float32' if l['actual_expert'] in ['default','low'] else 'base-float64')
        if l['selection_entry']:assert candidate is not None;env.scoped_query_snapshots[i]=candidate
        return result
    def reset(*a,**kw):env.scoped_calls=[];env.scoped_query_snapshots={};env.scoped_extra_before={};return native_reset(*a,**kw)
    env.step=step;env.reset=reset;return env


def make_factory(cfg,stack,learned,condition,controller):
    p=condition['protocol'];plain=factory(stack.D,dict(id='119-'+controller,factor='friction',level=p['friction'],scope='frozen-new-conditions'))
    def make(*a,**kw):
        env=plain(*a,**kw)
        if controller=='scoped118':attach_scoped_expert_blend(env,*stack.formal,stack.default[0],learned,stack.low[0],stack.default[1],cfg['confirmed_selector']['selector'],cfg['fast_capture'],stack.bank,cfg['unified_recovery_selection'],cfg['scoped_expert_blend']['gate'])
        else:assert controller=='parent098';attach_parent_capture(env,stack,cfg)
        return attach_release_capture(attach_initial_state(attach_traction_schedule(env,p),condition['profile']))
    return plain,make


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.prepare_only:prepare();return
    cfg=read(ROOT/CONFIG);f=verify_frozen();ev.CONFIG=cfg;out=ROOT/cfg['run'];out.mkdir(exist_ok=False);(out/'source').mkdir()
    for src in (ROOT/'scripts').glob('*.py'):shutil.copy2(src,out/'source'/src.name)
    dump(out/'acceptance.json',cfg);shutil.copy2(ROOT/cfg['predeclared_plan'],out/'predeclared-plan.md');names={CONFIG,cfg['predeclared_plan'],FROZEN,f['config'],'docs/data/go2-current/selection.json','docs/data/go2-release-regression/research-candidate.json',*f['models']};names.update('scripts/'+n for n in f['runtime_source_sha256']);names.add('scripts/run_go2_scoped_new_conditions.py');names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/f['selection']['bank']).rglob('*') if p.is_file())
    for c in cfg['scoped_new_conditions']['conditions']:names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/c['source118']['folder']).iterdir() if p.is_file())
    inputs={n:sha(ROOT/n) for n in sorted(names)};dump(out/'input-sha256.json',inputs);dump(out/'novelty-audit.json',cfg['scoped_new_conditions']['historical_signature_check']);torch.set_num_threads(1);stack=Stack(cfg);learned=PPO.load(ROOT/cfg['scoped_expert_blend']['learned_model'],device='cpu');other=norm_load(ROOT/cfg['scoped_expert_blend']['learned_norm']);assert exact(norm_state(stack.default[1]),norm_state(other));records=[];pairs=[];start=time.perf_counter()
    try:
        for c in cfg['scoped_new_conditions']['conditions']:
            for controller in cfg['scoped_new_conditions']['controllers']:
                own=[]
                for kind in cfg['scoped_new_conditions']['kinds']:
                    plain,make=make_factory(cfg,stack,learned,c,controller);r=run_transient(out,controller+'-'+kind,c['protocol'],c['profile']['seed'],make,stack.adapter,kind=='zero');env=plain.instances[-1];folder=ROOT/r['folder'];save_traction_probe(plain,folder,c['protocol']);logs=env.recovery_selection_log;dump(folder/'selector.json',logs);np.savez_compressed(folder/'initial-state.npz',**env.validation_initial);calls=env.scoped_calls
                    np.savez_compressed(folder/'expert-pair.npz',**{k:np.asarray([call[k] for call in calls]) for k in ['normalized_observation','raw_observation','parent_action','learned_action','learned_evaluated','applied','output_is_float64','scope_entry']});(folder/'snapshots.pkl').write_bytes(pickle.dumps(dict(queries=env.scoped_query_snapshots,extra_before=env.scoped_extra_before),protocol=5));r.update(controller=controller,kind=kind,condition_id=c['id'],profile=c['profile'],source118=c['source118'],scope_learned_steps=sum(l['scope_learned'] for l in logs),default_calls=len(calls),low_expert_steps=sum(l['actual_expert']=='low' for l in logs));dump(folder/'result.json',r);records.append(r);own.append(r);dump(out/'progress.json',records);print('NEW119 '+json.dumps({k:r[k] for k in ['case','controller','kind','passed','peak_tilt','recovery_seconds','terminated','scope_learned_steps']}),flush=True)
                cut=round(c['protocol']['events'][0]['start']/.02);assert raw_equal(own[0],own[1],cut);pairs.append(dict(zero=own[0],push=own[1],preforce_exact=True))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    finally:stack.close()
    comparisons=[]
    for c in cfg['scoped_new_conditions']['conditions']:
        for kind in cfg['scoped_new_conditions']['kinds']:
            a=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='parent098');b=next(r for r in records if r['case']==c['id'] and r['kind']==kind and r['controller']=='scoped118');comparisons.append(dict(case=c['id'],kind=kind,parent=a,scoped=b,full_raw_arrays_exact=raw_equal(a,b),delta_peak=b['peak_tilt']-a['peak_tilt'],delta_recovery=None if a['recovery_seconds'] is None or b['recovery_seconds'] is None else b['recovery_seconds']-a['recovery_seconds'],old_pass_lost=bool(a['passed'] and not b['passed']),added_termination=bool(not a['terminated'] and b['terminated'])))
    passes={controller:{kind:sum(r['passed'] for r in records if r['controller']==controller and r['kind']==kind) for kind in ['zero','pressure']} for controller in ['parent098','scoped118']};qualified=passes['scoped118']==dict(zero=8,pressure=8) and not any(x['old_pass_lost'] or x['added_termination'] for x in comparisons)
    for n,h in inputs.items():assert sha(ROOT/n)==h,n
    verify_frozen();s=dict(config=cfg,records=records,comparisons=comparisons,zero_pairs=pairs,passes=passes,qualified=bool(qualified),new_physical_runs=len(records),training_samples=0,PPO_updates=0,supervised_updates=0,new_normal_runs=0,new_normal_checks=0,qualification_scope='Eight predeclared frozen new combinations only; no broad or training-seed replication claim.',old_pass_losses=sum(x['old_pass_lost'] for x in comparisons),added_terminations=sum(x['added_termination'] for x in comparisons),formal_selection_changed=False,research_candidate_changed=False,visual_acceptance='pending',elapsed_seconds=time.perf_counter()-start);assert len(records)==32;dump(out/'summary.json',s);public=ROOT/cfg['public'];public.mkdir(exist_ok=False);dump(public/'summary.json',s);dump(out/'manifest.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'});print('COMPLETE119 '+json.dumps(dict(passes=passes,qualified=qualified,old_pass_losses=s['old_pass_losses'],added_terminations=s['added_terminations'])),flush=True)


if __name__=='__main__':main()
