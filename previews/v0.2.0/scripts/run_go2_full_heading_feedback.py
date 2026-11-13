"""197: full signed cumulative-heading feedback versus196 excess feedback."""
import argparse
import csv
import pickle
import shutil
import traceback
from unittest.mock import patch

import numpy as np

import go2_task
from go2_157 import ROOT, read, sha, write_new
from run_go2_catchup_new_conditions import measure, verify

CONFIG = ROOT/"configs/go2-full-heading-feedback.json"
PLAN = ROOT/"docs/experiments/2026-11-06-full-heading-feedback.md"
RUN = ROOT/"artifacts/runs/20261106-full-heading-feedback"
PUBLIC = ROOT/"docs/data/go2-full-heading-feedback"


def attach(env, stack):
    native = env.modify_joint_action
    env.full_heading_calls = []

    def modify(action):
        shadow = np.asarray(native(action)).copy()
        log = env.recovery_selection_log[-1]
        error = float(env.heading_error)
        permitted = bool(log["formal_weight"] == 1 and log["weight"] == 0
                         and env.command[2] != 0 and not env.is_reversal)
        ramp = float(np.clip((env.turn_age-1.)/.5, 0, 1))
        extra = ramp*float(np.clip(.4*error, -.06, .06)) if permitted else 0.
        actual = shadow.copy()
        if extra:
            observation = np.asarray(log["teacher_observation"], np.float32).copy()
            observation[11] += np.float32(.25*extra)
            policy, norm = stack.formal
            actual = np.clip(policy.predict(norm.normalize_obs(observation), deterministic=True)[0], -1, 1).astype(shadow.dtype)
        env.full_heading_calls.append(dict(step=env.step_count,heading_error=error,
            permitted=permitted,ramp=ramp,extra=extra,shadow=shadow,actual=actual.copy()))
        return actual

    env.modify_joint_action = modify
    return env


def prepare():
    verify()
    for path in (CONFIG, PLAN, RUN, PUBLIC):
        assert not path.exists(),path
    parent = read(ROOT/"configs/go2-catchup-new-conditions.json")
    names = ["scripts/run_go2_full_heading_feedback.py","scripts/go2_task.py",
             "scripts/run_go2_catchup_new_conditions.py",
             "configs/go2-catchup-new-conditions.json","docs/data/go2-catchup-new-conditions/summary.json",
             "docs/data/go2-current/selection.json"]
    write_new(CONFIG,dict(experiment=197,definitions=parent["definitions"],budget=4,training=0,
        inputs={name:sha(ROOT/name) for name in names},
        mechanism="Full signed heading_error feedback:extra=ramp*clip(.4*error,+-.06);retain194scope/nonreversal/ramp. Zero heading target,including overshoot correction.",
        gate="All4complete/no termination/whole8deg/recovery2s/25steps/34qualitychecks/6sturnerror3deg;repair196right pressure and retain3oldpasses.",
        stop="4candidate runs once,4exact196candidate references reused.No tuning if failure."))
    PLAN.write_text("""# 197　完整有符号累计朝向误差反馈

2026-11-06。196候选在新右转受推条件把8.154度误差降至3.101度，尾段追加量随超出3度的部分减小。当前试验将反馈目标从仅处理同向超差改为累计误差趋零，同时允许对已超转的状态减速。公式extra=ramp*clip(.4*heading_error,±.06)，沿194完整formal/恢复权重0/首次非反转向作用域，以及一秒后.5秒引入斜坡。其余网络、物理、实际请求和原评分不改。

沿196两条已打开新历史，各zero/pressure，预算4新候选实际、0训练。196对应4条原候选只读复用，原1574条亦只作已保存参考。成功要求4条完整原步数无终止、全程8度、原撤力2秒内平面.08/yaw.15连续25步、原34动作质量项及6秒转角3度，修复目标并保3个旧成功。所有连续yaw/目标跳变/力矩和恢复代价保留。

固定一个完整有符号反馈机制，不增加增益/死区/时长组合。输出和下层影子分开保存，实际native目标核对。新路径与196在首个实际动作差异前的8数组要求精确，该差异在两种反馈输出首次不同时定义。4条一次完成。失败即拒绝本轮资格，继续原停止规则，不在本轮改参。通过后仍须保护193旧范围和原46秒动作及冻结新条件。
""",encoding="utf-8")
    print("PREPARED197 fixed full signed error feedback4physics",flush=True)


def main():
    task=read(CONFIG);verify()
    RUN.mkdir(exist_ok=False)
    for source,target in [(CONFIG,"prospective.json"),(PLAN,"predeclared-plan.md"),(ROOT/"scripts/run_go2_full_heading_feedback.py","runtime.py")]:
        shutil.copy2(source,RUN/target)
    parent=read(ROOT/"docs/data/go2-catchup-new-conditions/summary.json")
    original_load=go2_task.load_stack;instances=[]
    def load(spec):
        cfg,stack,factory,ev=original_load(spec)
        def wrapped(reference):
            plain,native=factory(reference)
            def make(*a,**kw):
                env=attach(native(*a,**kw),stack);instances.append(env);return env
            return plain,make
        return cfg,stack,wrapped,ev
    records=[]
    try:
        for name,digest in task["inputs"].items():assert sha(ROOT/name)==digest,name
        for side,kind in [("right","pressure"),("left","pressure"),("right","zero"),("left","zero")]:
            definition=next(d for d in task["definitions"] if d["side"]==side)
            with patch.object(go2_task,"load_stack",load):
                result=go2_task.run_task(definition["task"],RUN/(side+"-"+kind),kind)
            env=instances[-1];calls=env.full_heading_calls;folder=ROOT/result["folder"]
            (folder/"full-heading-calls.pkl").write_bytes(pickle.dumps(calls,protocol=5))
            old=next(r for r in parent["records"] if r["side"]==side and r["kind"]==kind and r["arm"]=="catchup194")
            old_folder=ROOT/old["result"]["folder"]
            old_calls=pickle.loads((old_folder/"catchup-calls.pkl").read_bytes())
            first=next((i for i,c in enumerate(calls) if not np.array_equal(c["actual"],old_calls[i]["actual"])),len(calls))
            with np.load(folder/"states.npz") as a,np.load(old_folder/"states.npz") as b:
                for key in a.files:np.testing.assert_array_equal(a[key][:first],b[key][:first])
                for i,c in enumerate(calls):np.testing.assert_array_equal(np.clip(env.nominal+.35*c["actual"],*env.model.actuator_ctrlrange.T),a["targets"][i])
            metrics=measure(result,definition["task"])
            records.append(dict(side=side,kind=kind,result=result,metrics=metrics,
                first_divergence_vs196=first,maximum_extra=max(abs(c["extra"]) for c in calls),
                active_frames=sum(c["extra"]!=0 for c in calls)))
            print("RESULT197",side,kind,result["passed"],result["turn_angles"],result["recovery_seconds"],
                [c for c in result["checks"] if not c["passed"]],flush=True)
        verify()
        for name,digest in task["inputs"].items():assert sha(ROOT/name)==digest,name
        summary=dict(experiment=197,records=records,new_physical=4,reused_candidate196=4,
            training=0,fits=0,qualified=all(r["result"]["passed"] for r in records),visual="pending")
        write_new(RUN/"summary.json",summary);PUBLIC.mkdir(exist_ok=False);write_new(PUBLIC/"summary.json",summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8");raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true")
    prepare() if p.parse_args().prepare else main()
