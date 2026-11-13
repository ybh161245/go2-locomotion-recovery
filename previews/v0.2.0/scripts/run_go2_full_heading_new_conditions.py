"""199: frozen197 versus157 on two prospective fast-turn histories."""
import argparse
import csv
import pickle
import shutil
import time
import traceback
from unittest.mock import patch

import mujoco
import numpy as np

import go2_task
from go2_157 import ROOT, read, sha, write_new
from run_go2_full_heading_feedback import attach

CONFIG = ROOT / "configs/go2-full-heading-new-conditions.json"
PLAN = ROOT / "docs/experiments/2026-11-06-full-heading-new-conditions.md"
RUN = ROOT / "artifacts/runs/20261106-full-heading-new-conditions"
PUBLIC = ROOT / "docs/data/go2-full-heading-new-conditions"
FREEZE = ROOT / "docs/data/go2-full-heading-retention/protected-candidate-freeze.json"


def verify():
    for name, digest in read(FREEZE)["sha256"].items():
        assert sha(ROOT/name) == digest, name


def prepare():
    verify()
    for path in (CONFIG, PLAN, RUN, PUBLIC):
        assert not path.exists(), path
    definitions = []
    for side, seed, fast, onset, force in [("left",499101,5.8,9.86,30),("right",499102,6.2,10.10,-30)]:
        task = dict(sequence=dict(seed=seed,segments=[
            dict(name="stand",motion="stand",seconds=2),
            dict(name="walk",motion="walk",seconds=2),
            dict(name="fast straight",motion="trot",seconds=fast),
            dict(name=side,motion=side,seconds=6),
            dict(name="final stop",motion="stand",seconds=4)]),
            push=dict(start_s=onset,duration_s=.2,body_y_n=force))
        go2_task.validate_task(task)
        definitions.append(dict(side=side,task=task))
    names = ["scripts/run_go2_full_heading_new_conditions.py","scripts/go2_task.py",
             "docs/data/go2-full-heading-retention/protected-candidate-freeze.json",
             "docs/data/go2-current/selection.json"]
    write_new(CONFIG,dict(experiment=199,definitions=definitions,arms=["frozen157","fullheading197"],
        kinds=["zero","pressure"],new_physical_budget=8,training=0,fits=0,
        inputs={name:sha(ROOT/name) for name in names},
        gate="Candidate all4passes under original8deg/2s/25steps/wholeprotocol/34quality/6sturnerror3deg;zero oldpassloss/newtermination among4pairs.At least1gain to call improvement,otherwise coverage.",
        stop="All8once;no gain/deadband/ramp or network changes after opening. Failure blocks scope qualification."))
    PLAN.write_text("""# 199　冻结197的相反侧推与新快进历史验证

2026-11-06。198正常324项和193旧任务保护通过后，冻结197完整累计反馈全部代码与参数。新左转seed499101，站2秒/慢进2秒/快进5.8秒/转向6秒/停止4秒，总19.8秒，9.86至10.06秒正身体y30 N短推，在9.80秒转向后开始。新右转seed499102，快进6.2秒，总20.2秒，10.10至10.30秒负身体y30 N短推，跨越10.20秒转向。相对196的同方向转弯，本轮侧推反向，并改变快进时长和相位。各原157/197配zero/pressure，总8新实际，0训练拟合。

评估seed沿原生初态微扰，恢复状态独立性按实际输入描述。原生材料/reset与0.2/0.4速度、±.5yaw相同。资格为候选四条完整990/1010步、无终止、全程8度、末撤力2秒平面.08/yaw.15连续25步，全部34原动作项含六秒角误差3度。原通过零损失、零新增终止。至少一失败转通过才能称新收益，否则仅覆盖。

配置/代码/原冻结SHA在运行前保存。每格一次，即使失败也完成全部8条，不调参回填。自己zero/pressure推前8数组精确，版本初态/物理及首次输出差异前8数组精确，实际输出到native目标逐位核对，按真实姿态重建世界力。逐条转角/恢复/峰值/yawRMSE/目标跳变与50Hz力矩差值全部保留，新视频视觉待用户。失败阻止新资格，通过再考虑适用压力矩阵。
""",encoding="utf-8")
    print("PREPARED199 8physical,197frozen,opposite sidepush",flush=True)


def measure(result, task):
    folder=ROOT/result["folder"]
    with (folder/"steps.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    for start,end,command,phase in go2_task.validate_task(task):
        actual=rows[start:end]
        assert all(row["phase"]==phase for row in actual)
        np.testing.assert_array_equal([[float(row[key]) for key in ("command_vx","command_vy","command_yaw")]
                                      for row in actual],np.tile(command,(len(actual),1)))
    peak=max(max(abs(float(row["roll_deg"])),abs(float(row["pitch_deg"]))) for row in rows)
    push=task["push"];onset=round(push["start_s"]/.02);withdrawal=round((push["start_s"]+.2)/.02)
    band=[np.hypot(float(row["vx"])-float(row["command_vx"]),float(row["vy"]))<=.08
          and abs(float(row["yaw_rate"])-float(row["command_yaw"]))<=.15 for row in rows[withdrawal:]]
    recovery=next((i*.02 for i in range(len(band)-24) if all(band[i:i+25])),None)
    assert recovery==result["recovery_seconds"] and peak==result["peak_tilt"]
    with np.load(folder/"states.npz") as z,np.load(folder/"initial-state.npz") as ini,np.load(folder/"telemetry.npz") as tele,np.load(folder/"physics.npz") as physics:
        pre=np.concatenate([ini["qpos"][None,:],z["qpos"][:-1]])
        force_error=0.
        for i,qpos in enumerate(pre):
            rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,qpos[3:7])
            active=result["kind"]=="pressure" and onset<=i<withdrawal
            expected=np.r_[rotation.reshape(3,3)@np.array([0,push["body_y_n"] if active else 0,0]),np.zeros(3)]
            force_error=max(force_error,float(np.max(np.abs(expected-z["world_wrench"][i]))))
        assert force_error<1e-11
        changes=np.abs(np.diff(z["targets"],axis=0));limits=np.max(np.abs(physics["after_actuator_forcerange"]),axis=1)
        return dict(max_target_step=float(changes.max()),p95_target_step=float(np.percentile(changes,95)),
            max_50Hz_torque_fraction=float(np.max(np.abs(tele["actuator_force"])/limits)),force_error=force_error,
            yaw_rmse=float(np.sqrt(np.mean([(float(r["yaw_rate"])-float(r["command_yaw"]))**2 for r in rows]))))


def main():
    task=read(CONFIG);verify()
    RUN.mkdir(exist_ok=False)
    shutil.copy2(__file__,RUN/"runtime.py");shutil.copy2(CONFIG,RUN/"prospective.json");shutil.copy2(PLAN,RUN/"predeclared-plan.md")
    original_load=go2_task.load_stack;instances=[]
    def load(spec):
        cfg,stack,factory,ev=original_load(spec)
        def wrapped(reference):
            plain,original=factory(reference)
            def make(*a,**kw):
                env=attach(original(*a,**kw),stack);instances.append(env);return env
            return plain,make
        return cfg,stack,wrapped,ev
    records=[];started=time.perf_counter()
    try:
        for name,digest in task["inputs"].items():assert sha(ROOT/name)==digest,name
        for definition in task["definitions"]:
            for arm in task["arms"]:
                zero=None
                for kind in task["kinds"]:
                    output=RUN/(definition["side"]+"-"+arm+"-"+kind)
                    if arm=="fullheading197":
                        with patch.object(go2_task,"load_stack",load):
                            result=go2_task.run_task(definition["task"],output,kind)
                    else:result=go2_task.run_task(definition["task"],output,kind)
                    folder=ROOT/result["folder"]
                    metrics=measure(result,definition["task"])
                    if kind=="zero":zero=folder
                    else:
                        end=round(definition["task"]["push"]["start_s"]/.02)
                        with np.load(folder/"states.npz") as a,np.load(zero/"states.npz") as b:
                            for key in a.files:np.testing.assert_array_equal(a[key][:end],b[key][:end])
                    if arm=="fullheading197":
                        env=instances[-1];calls=env.full_heading_calls
                        (folder/"full-heading-calls.pkl").write_bytes(pickle.dumps(calls,protocol=5))
                        baseline=next(r["result"] for r in records if r["side"]==definition["side"] and r["arm"]=="frozen157" and r["kind"]==kind)
                        first=next((i for i,c in enumerate(calls) if not np.array_equal(c["actual"],c["shadow"])),len(calls))
                        with np.load(folder/"states.npz") as a,np.load(ROOT/baseline["folder"]/"states.npz") as b:
                            for key in a.files:np.testing.assert_array_equal(a[key][:first],b[key][:first])
                            for i,c in enumerate(calls):np.testing.assert_array_equal(np.clip(env.nominal+.35*c["actual"],*env.model.actuator_ctrlrange.T),a["targets"][i])
                        for filename in ["initial-state.npz","physics.npz"]:
                            with np.load(folder/filename) as a,np.load(ROOT/baseline["folder"]/filename) as b:
                                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                    records.append(dict(side=definition["side"],arm=arm,kind=kind,result=result,metrics=metrics))
                    print("RESULT199",definition["side"],arm,kind,result["passed"],result["peak_tilt"],result["recovery_seconds"],result["turn_angles"],
                          [c for c in result["checks"] if not c["passed"]],flush=True)
        pairs=[]
        for definition in task["definitions"]:
            for kind in task["kinds"]:
                a=next(r for r in records if r["side"]==definition["side"] and r["kind"]==kind and r["arm"]=="frozen157")
                b=next(r for r in records if r["side"]==definition["side"] and r["kind"]==kind and r["arm"]=="fullheading197")
                pairs.append(dict(side=definition["side"],kind=kind,old_pass=a["result"]["passed"],new_pass=b["result"]["passed"],
                    added_termination=not a["result"]["terminated"] and b["result"]["terminated"],
                    old_angle=a["result"]["turn_angles"][0],new_angle=b["result"]["turn_angles"][0],
                    peak_delta=b["result"]["peak_tilt"]-a["result"]["peak_tilt"],
                    old_recovery=a["result"]["recovery_seconds"],new_recovery=b["result"]["recovery_seconds"],
                    **{key+"_delta":b["metrics"][key]-a["metrics"][key] for key in b["metrics"]}))
        gains=sum(not p["old_pass"] and p["new_pass"] for p in pairs)
        losses=sum(p["old_pass"] and not p["new_pass"] for p in pairs)
        qualified=all(p["new_pass"] for p in pairs) and not losses and not any(p["added_termination"] for p in pairs)
        verify()
        for name,digest in task["inputs"].items():assert sha(ROOT/name)==digest,name
        summary=dict(experiment=199,records=records,pairs=pairs,gains=gains,old_pass_losses=losses,qualified=qualified,
            new_physical=8,training=0,fits=0,wall_seconds=time.perf_counter()-started,visual="pending")
        write_new(RUN/"summary.json",summary);PUBLIC.mkdir(exist_ok=False);write_new(PUBLIC/"summary.json",summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8");raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true")
    prepare() if p.parse_args().prepare else main()
