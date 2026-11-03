"""Freeze062 known-condition regression, explicit duplicate handling and no-loss gates."""
import copy
import numpy as np
import evaluate_go2_robust as ev
from archive_project import ROOT
from diagnose_go2_update import read,sha,dump
from evaluate_go2_probe_value import extension_cases


def case_key(c):
    return (c['phase'],float(c['force']),tuple(map(float,c['direction'])),float(c['duration']),tuple(map(float,c['pulses'])),float(c['torque']),c['friction'])


def main():
    cfg=read(ROOT/'configs/go2-confirmed-selector.json');ev.CONFIG=cfg;cases=extension_cases()
    assert len(cases)==31 and len({case_key(c) for c in cases})==30
    oldrun=cfg['run'];cfg.update(experiment_id='GO2-B-261026-062',run='artifacts/runs/20261026-recovery-regression',
        screen_run='artifacts/runs/20261026-recovery-regression',public='docs/data/go2-recovery-regression',
        predeclared_plan='docs/experiments/2026-10-26-recovery-regression.md')
    original=read(ROOT/'docs/data/go2-targeted-d/screen.json');d={r['case']:r for r in original['results'] if r['variant']=='joint-65536'}
    assert set(d)=={c['id'] for c in cases}
    for c in cases:assert case_key(c)==case_key(d[c['id']]['condition']) and d[c['id']]['seed']==cfg['development_seed']
    with np.load(ROOT/d['case-048']['folder']/'states.npz') as a,np.load(ROOT/d['repeat-1-offset-0']['folder']/'states.npz') as b:
        assert a.files==b.files
        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    cfg['recovery_regression']=dict(candidate_run=oldrun,known_cases=cases,aliases={'repeat-1-offset-0':'case-048'},
        reuse_cases=['F30-D0.6-S1-O0','F30-D0.6-S1-O0.2','F30-D0.6-S1-O0.4'],new_pressure_runs=27,new_normal_runs=2,
        reference_screen='docs/data/go2-targeted-d/screen.json',reference_variant='joint-65536',
        original_screen='docs/data/go2-sustained-A/screen.json',original_variants=['working','previous-stack'],
        normal_reference=cfg['quality_guard']['reference_folder'],models=copy.deepcopy(cfg['confirmed_selector']['model_hashes']),
        selector_source_sha256=sha(ROOT/'scripts/go2_confirmed_selector.py'),candidate_config_sha256=sha(ROOT/'configs/go2-confirmed-selector.json'),
        visual_cases=[dict(case='case-044',stem='positive-sustained',title='40N正侧持续推'),
            dict(case='case-045',stem='negative-sustained',title='40N反侧持续推'),
            dict(case='repeat-1-offset-0.25',stem='positive-repeat',title='40N重复侧推 · 相位后移0.25秒')])
    p=ROOT/'configs/go2-recovery-regression.json';assert not p.exists();dump(p,cfg)
    plan=ROOT/cfg['predeclared_plan'];assert not plan.exists()
    plan.write_text('''# GO2-B-261026-062：冻结恢复候选的已知工况回归（实验前声明）

2026-10-26，用户逐轮授权下一步。061核心9/9及旧开发6/6，但仍需检查40N持续/重复、20/25N和不同动作协议。只评估已研究条件，不训练或改变足端映射/选择器，不把回归称独立泛化。

## 冻结与预算

保持061专家131k、D、转向适配器、全部归一化/入口/交还/物理/动作完全不变，模型与选择器SHA预存。沿用extension_cases31记录：核心13（含0力、40N短/持续双侧、重复冲击各相位），以及20/25/30N×正反侧×6/6.2/6.4秒单次0.6秒共18记录，均为既有开发。全部seed106101、名义质量，其他规则原样。

case-048与repeat-1-offset-0条件相同，旧D所有原始数组也逐位相同；复用为别名，按31记录和30独立条件同时报告，不重复仿真、不用重复项放大通过率。061已测名义30N正侧三条直接复用，实际新增压力27次。时刻沿用原50Hz round规则（0.25秒相位名义值有离散取整），不改历史协议。

另新增两条047质量保护内部正常协议（42/41秒、seed147001/147002），运行完整命令切换，检查原362项质量限制、专家介入及与D参考完整原始轨迹相等。其既有D首转误差等缺口仍按原规则列明，不冒称原绝对3°全部达标。总预算29次新物理运行、0训练。既有三质量46秒正常与15压力保持证据复用，不重跑、不打开原独立保留集或180矩阵。

## 预声明判定

主要保持条件：新候选不丢D在31记录中任何原通过、不相对D新增安全终止；不丢原正式工作版working或最初研究previous-stack在共同核心13中的任何原通过。任何一条新增失败必须单列，即使总通过增加。D已有失败在更强扰动下仍失败不自动算退步，也不要求所有40N都通过。

所有压力首次施力前专家权重0、原始观测/动作/状态/目标/世界力前缀与同协议D逐位相同；0力控制整12秒也必须如此。两条新增内部正常应362保护通过、专家权重全0且状态/目标/足端/朝向数组逐位同D。正常数值和肉眼验收区分。

同时报告相对D已测压力收益（新增通过、终止减少、均峰）与所有角度/恢复变化。通过保持门只获得本回归范围研究资格，原名义RL26.614<29.570mm及增重10项旧缺口不因此解决，正式模型不自动替换，不宣布阶段4完成。

## 停止、可视与留存

无论中途成败都完成固定29次，失败保存；不得根据结果改阈值/重训/追加候选或足端补丁。有新增丢项则暂停足端改动，先明确失效情形；若保持通过，下一轮再单独研究原足端旧缺口。三个固定视频案例为40N正侧持续、反侧持续、正侧重复后移0.25秒，三列正式工作版/D/061；全表包含失败，不挑视频冒称全局最优。

保存运行时源码、输入SHA、模型/归一化、原新状态、外力、电机遥测、选择器记录、重复别名/复用索引和全部视频；逐文件SHA/ZIP CRC封存。结果另存2026-10-26-recovery-regression-results.md，用户视觉验收待确认。
''',encoding='utf-8')
    print('Frozen062:31records30conditions,27newpressure+2normal,0training.',flush=True)


if __name__=='__main__':main()
