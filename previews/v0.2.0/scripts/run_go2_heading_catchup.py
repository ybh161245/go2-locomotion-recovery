"""194: fixed excess-heading catchup on non-reversal formal turns."""
import argparse
import csv
import pickle
import shutil
import traceback
from unittest.mock import patch

import numpy as np

import go2_task
from go2_157 import ROOT, read, sha, write_new

CONFIG = ROOT / "configs/go2-heading-catchup.json"
PLAN = ROOT / "docs/experiments/2026-11-06-heading-catchup.md"
RUN = ROOT / "artifacts/runs/20261106-heading-catchup"
PUBLIC = ROOT / "docs/data/go2-heading-catchup"


def attach(env, stack):
    native = env.modify_joint_action
    env.catchup_calls = []

    def modify(action):
        shadow = np.asarray(native(action)).copy()
        log = env.recovery_selection_log[-1]
        requested = float(env.command[2])
        error = float(env.heading_error)
        permitted = bool(log["formal_weight"] == 1 and log["weight"] == 0
                         and requested != 0 and not env.is_reversal)
        excess = np.sign(error) * max(abs(error)-np.radians(3), 0) if error*requested > 0 else 0.
        ramp = float(np.clip((env.turn_age-1.)/.5, 0, 1))
        extra = ramp * float(np.clip(.4*excess, -.06, .06)) if permitted else 0.
        actual = shadow.copy()
        if extra:
            obs = np.asarray(log["teacher_observation"], np.float32).copy()
            obs[11] += np.float32(.25*extra)
            policy, norm = stack.formal
            actual = np.clip(policy.predict(norm.normalize_obs(obs), deterministic=True)[0], -1, 1).astype(shadow.dtype)
        env.catchup_calls.append(dict(step=env.step_count,heading_error=error,permitted=permitted,
            extra=extra,ramp=ramp,shadow=shadow,actual=actual.copy()))
        return actual

    env.modify_joint_action = modify
    return env


def prepare():
    for path in (CONFIG, PLAN, RUN, PUBLIC):
        assert not path.exists(), path
    parent = read(ROOT / "configs/go2-fast-turn-push.json")
    names = ["scripts/run_go2_heading_catchup.py", "scripts/go2_task.py",
             "configs/go2-fast-turn-push.json", "docs/data/go2-fast-turn-push/summary.json",
             "artifacts/runs/20261023-turn-recovery-0.4/config.json",
             "docs/data/go2-current/selection.json"]
    write_new(CONFIG, dict(experiment=194,definitions=parent["definitions"],budget=4,training=0,
        gain=.4,limit=.06,deadband_degrees=3,ramp_start_s=1,ramp_duration_s=.5,
        inputs={name:sha(ROOT/name) for name in names},
        gate="All4pressure/zero tasks complete1000/no termination/whole8deg/recovery2s/25steps/all34absolutechecks/turnerror3deg;repair right pressure and retain3previous passes.",
        stop="One mechanism,4runs;no parameter search. Any oldpassloss or uncorrected target rejects candidate."))
    PLAN.write_text("""# 194　首次转向累计欠角的有限补偿

2026-11-06。193右转受推完整恢复但欠转3.659度，首秒比零力少3.704度，后5秒仅追回.367度。原环境记录heading_error并输入适配器，额外recovery_gain只在is_reversal启用，本次直行切入右转为首次转向。原模型输出在有较大欠角时也可能给出很小的追加转速，故仅保留误差观测不足以保证累计角达标。

唯一改动在原完整formal转弯且恢复混合权重0、首次非反转向分支。以自身累计heading_error和请求yaw的符号识别同向欠转。取excess=sign(error)*max(abs(error)-3deg,0)，一秒后用原反转向追回的.5秒ramp，追加clip(.4*excess,±.06)到formal输入的yaw通道，保持原修正和输出dtype。增益.4/上限.06及一秒/.5秒ramp来自现有反转向参数，3度死区引用原任务角度预算。本机制新增首次转向补偿作用域，不读取力/步号/真摩擦/历史轨迹。该数值选择在本轮固定。

预算4新实际、0训练拟合。沿193两历史/seed/命令/材料各pressure和zero，原193四条只读基线。原1000步/无终止/全程8度/撤力2秒平面.08yaw.15连续25步及全部34绝对质量项/转角3度门槛不变。要求修复右受推且保住其余3旧通过，连续yaw/倾角/恢复/动作代价均报告。有失败不追加本轮调参。

实际输出到native目标逐步匹配、第一次实际输出改变前8数组精确，记录自身误差/追加量/实际和下层影子动作。通过仅表示已打开开发范围，正常46秒及冻结新条件另立资格。
""", encoding="utf-8")
    print("PREPARED194 fixed gain.4/deadband3deg/extra limit.06", flush=True)


def main():
    task = read(CONFIG)
    RUN.mkdir(exist_ok=False)
    shutil.copy2(__file__, RUN/"runtime.py")
    shutil.copy2(CONFIG, RUN/"prospective.json")
    shutil.copy2(PLAN, RUN/"predeclared-plan.md")
    old = read(ROOT/"docs/data/go2-fast-turn-push/summary.json")
    original_load = go2_task.load_stack
    instances = []

    def load(spec):
        cfg, stack, factory, ev = original_load(spec)
        def wrapped_factory(reference):
            plain, original = factory(reference)
            def make(*args, **kwargs):
                env = attach(original(*args, **kwargs), stack)
                instances.append(env)
                return env
            return plain, make
        return cfg, stack, wrapped_factory, ev

    records = []
    try:
        for name,digest in task["inputs"].items():
            assert sha(ROOT/name)==digest,name
        # Open the target first, but preserve all four predeclared results.
        for side,kind in [("right","pressure"),("left","pressure"),("right","zero"),("left","zero")]:
            definition = next(d for d in task["definitions"] if d["side"]==side)
            with patch.object(go2_task,"load_stack",load):
                result = go2_task.run_task(definition["task"], RUN/(side+"-"+kind),kind)
            env = instances[-1]
            folder = ROOT/result["folder"]
            calls = env.catchup_calls
            (folder/"catchup-calls.pkl").write_bytes(pickle.dumps(calls,protocol=5))
            baseline = next(r["result"] for r in old["records"] if r["side"]==side and r["kind"]==kind)
            first = next((i for i,c in enumerate(calls) if not np.array_equal(c["actual"],c["shadow"])),len(calls))
            with np.load(folder/"states.npz") as a,np.load(ROOT/baseline["folder"]/"states.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key][:first],b[key][:first])
                for i,c in enumerate(calls):
                    target=np.clip(env.nominal+.35*c["actual"],*env.model.actuator_ctrlrange.T)
                    np.testing.assert_array_equal(target,a["targets"][i])
                d=np.abs(np.diff(a["targets"],axis=0))
                quality=dict(max_target_step=float(d.max()),p95_target_step=float(np.percentile(d,95)))
            with (folder/"steps.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            yaw_rmse=float(np.sqrt(np.mean([(float(row["yaw_rate"])-float(row["command_yaw"]))**2 for row in rows])))
            record=dict(side=side,kind=kind,result=result,first_actual_change=first,
                maximum_extra=max(abs(c["extra"]) for c in calls),active_frames=sum(c["extra"]!=0 for c in calls),
                yaw_rmse=yaw_rmse,**quality)
            records.append(record)
            print("RESULT194",side,kind,result["passed"],result["turn_angles"],result["recovery_seconds"],
                  [c for c in result["checks"] if not c["passed"]],flush=True)
        for name,digest in task["inputs"].items():
            assert sha(ROOT/name)==digest,name
        qualified=all(r["result"]["passed"] for r in records)
        summary=dict(experiment=194,records=records,new_physical=4,training=0,qualified=qualified,
                     visual="pending",scope="Opened193 histories only")
        write_new(RUN/"summary.json",summary)
        PUBLIC.mkdir(exist_ok=False)
        write_new(PUBLIC/"summary.json",summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare",action="store_true")
    prepare() if parser.parse_args().prepare else main()
