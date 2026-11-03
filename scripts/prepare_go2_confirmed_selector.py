"""Predeclare061 a single-entry-rule change with all learned tensors frozen."""
import ast,copy
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump


def verify_single_change():
    old=ast.parse((ROOT/'scripts/go2_recovery_selector.py').read_text(encoding='utf-8'))
    new=ast.parse((ROOT/'scripts/go2_confirmed_selector.py').read_text(encoding='utf-8'))
    oldfn=next(n for n in old.body if isinstance(n,ast.FunctionDef));newfn=next(n for n in new.body if isinstance(n,ast.FunctionDef))
    find=lambda f:next(n for n in ast.walk(f) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='risk' for t in n.targets))
    a=find(oldfn);b=find(newfn);assert isinstance(b.value,ast.BoolOp) and isinstance(b.value.op,ast.And) and len(b.value.values)==2
    assert ast.dump(a.value,include_attributes=False)==ast.dump(b.value.values[0],include_attributes=False)
    expected=ast.parse("abs(lin[1])>=spec['vy_confirmation']",mode='eval').body
    assert ast.dump(expected,include_attributes=False)==ast.dump(b.value.values[1],include_attributes=False)
    b.value=copy.deepcopy(a.value);newfn.name=oldfn.name
    assert ast.dump(oldfn,include_attributes=False)==ast.dump(newfn,include_attributes=False)
    return dict(single_change_proved=True,change='old risk AND abs(body_vy)>=0.08; all state machine/blending/return logic AST identical',
        old_source_sha256=sha(ROOT/'scripts/go2_recovery_selector.py'),new_source_sha256=sha(ROOT/'scripts/go2_confirmed_selector.py'))


def main():
    cfg=read(ROOT/'configs/go2-recovery-specialist-r2.json');specialist=cfg['run'];prior_screen=cfg['screen_run'];spec=cfg['recovery_specialist']
    proof=verify_single_change()
    cfg.update(experiment_id='GO2-B-261026-061',run='artifacts/runs/20261026-confirmed-selector',
        screen_run='artifacts/runs/20261026-confirmed-selector',public='docs/data/go2-confirmed-selector',
        predeclared_plan='docs/experiments/2026-10-26-confirmed-selector.md')
    selector=copy.deepcopy(spec['selector']);selector['vy_confirmation']=.08
    cfg['confirmed_selector']=dict(specialist_train=specialist,specialist_checkpoint=131072,prior_screen=prior_screen,
        selector=selector,change_proof=proof,core_runs=12,max_followup_runs=6,
        baseline='artifacts/runs/20261025-mass-course-reference',followup='artifacts/runs/20261025-exposure-pressure-reference',
        candidate_count=1,model_hashes={})
    for n in [f'{specialist}/checkpoint-131072.zip',f'{specialist}/checkpoint-131072-vecnormalize.pkl',*spec['formal_models']]:
        cfg['confirmed_selector']['model_hashes'][n]=sha(ROOT/n)
    p=ROOT/'configs/go2-confirmed-selector.json';assert not p.exists();dump(p,cfg)
    plan=ROOT/cfg['predeclared_plan'];assert not plan.exists()
    plan.write_text('''# GO2-B-261026-061：冻结专家的侧向速度确认选择器（实验前声明）

2026-10-26，用户授权逐轮执行。060已证明已见快照直接恢复5/6，但组合启动提前介入，正常新越界9项。本轮0训练、仅1个候选，先保存规则、模型SHA和预算。

## 纠正旧解释和设计依据

060记录把启动倾角误触发称为俯仰，逐轴归因不充分。本轮在任何新物理运行前读取D三质量正常原始CSV：低速直行段横滚绝对峰值约3.55/3.83/3.43°，侧向速度绝对峰值约.0498/.0485/.0509m/s，故正常启动也可因横滚越3°触发。旧“俯仰”说法更正为“正常启动倾角”，此纠正追加在本轮，原060封存不改写。

单改动为原risk条件 AND abs(body_vy)>=.08m/s。旧tilt3°/vy.12/angular_xy.6及2步确认保持。侧速确认值在已见正常数据上有约.029m/s间隔，尚不代表泛化或噪声鲁棒。入场后安全退出、最短/最长持续、4步平滑和冷却全部保持。源码AST证明除了此AND比较，其余函数实现与060完全相同。新代码另存，旧选择器和所有旧结果保留。

## 不变量和固定测试

冻结专家131072模型/归一化、D及其归一化、固定转向适配器和原物理，哈希在配置先保存。相同seed71001/106101、质量1/.95/1.05，先完整测3质量×（46秒正常+30N/.6秒正侧6/6.2/6.4秒起推）12运行；不根据中途结果调阈值或追加候选。

记录全部选择器输入/状态权重和实际关节动作。正常要求专家权重全0，完整原始轨迹与同质量D逐位相同，原相对D正常/支撑/RL97%保护通过；D已有的名义RL29.570mm缺口和增重10项原正常失败照常列出，不把精确保留D称为阶段绝对达标。压力要求首次施力之前专家权重全0、原始qpos/qvel/目标/观测/动作/世界力前缀逐位等于D，避免利用启动改步态而通过。

主要晋级条件为上述正常/前缀保护 AND保持060九压力9/9无终止（原8°/2秒定义完全保持）。同时报告相对D5/9的收益、相对0609/9的丢项和所有退步。若只满足部分则保留研究进展，不自动采用。交还是否在安全25步/150上限退出、是否反复触发、权重变化和后续表现均公开，本轮不改交还。

## 条件扩展、退出和可视

只有12核心运行全部满足主要条件，才测试原6条旧开发保留（3质量×正6.1/负6.3秒30N），要求保留D全部通过且无新终止，并核对受推前D前缀。最多18新物理评估，不开原独立时刻/种子或180矩阵，不重训。任何原正常绝对缺口仍需后续解决；旧开发保留不是独立泛化。

固定展示名义46秒正常、名义和减重6.2秒难点各12秒，三列D/060原型/本轮候选；从保存状态渲染，显示专家权重。原始数据/模型SHA/运行时源码/失败/视频全部封存并校验。结果另存2026-10-26-confirmed-selector-results.md，视觉待用户。若不合格本轮到此结束，不后验搜索更多阈值。
''',encoding='utf-8')
    print('Frozen061 single AND gate, learned hashes,12+conditional6 runs,0training.',flush=True)


if __name__=='__main__':main()
