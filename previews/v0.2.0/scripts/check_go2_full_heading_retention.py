"""198: early-stop normal and193 retention gates for frozen197."""
import argparse
import pickle
import shutil
import time
import traceback
from unittest.mock import patch

import numpy as np
import go2_task
from go2_157 import ROOT, read, sha, write_new
from run_go2_full_heading_feedback import attach
from run_go2_frozen157_exit_matrix import load, make_factory
from go2_parameter_probe import save_probe
from go2_contact_residual_learning import save_calls
from run_go2_fast_turn_push import measure
import evaluate_go2_robust as ev

CONFIG=ROOT/"configs/go2-full-heading-retention.json"
PLAN=ROOT/"docs/experiments/2026-11-06-full-heading-retention.md"
RUN=ROOT/"artifacts/runs/20261106-full-heading-retention"
PUBLIC=ROOT/"docs/data/go2-full-heading-retention"
FREEZE=ROOT/"docs/data/go2-full-heading-feedback/frozen-development-candidate.json"


def verify():
    for name,digest in read(FREEZE)["sha256"].items():
        assert sha(ROOT/name)==digest,name


def prepare():
    verify()
    for path in (CONFIG,PLAN,RUN,PUBLIC):assert not path.exists(),path
    refs=read(ROOT/"configs/go2-turn-guard-normal.json")["normal_references"]
    definitions=read(ROOT/"configs/go2-fast-turn-push.json")["definitions"]
    names=["scripts/check_go2_full_heading_retention.py","scripts/go2_task.py",
           "docs/data/go2-full-heading-feedback/frozen-development-candidate.json",
           "configs/go2-frozen157-exit-matrix.json","docs/data/go2-157-runtime/normal-budget-reference.json",
           "configs/go2-fast-turn-push.json","docs/data/go2-current/selection.json"]
    write_new(CONFIG,dict(experiment=198,normal_refs=refs,definitions=definitions,max_physical=7,training=0,
        inputs={n:sha(ROOT/n) for n in names},
        order=["normal171701","right-pressure193","left-pressure193","right-zero193","left-zero193","normal171702","normal171703"],
        gate="Original324normal budget and RL29.5702276mm;193all4original34quality/full8deg/2s/25step/turn3deg.",
        stop="Stop after first failing run;remaining tasks not run.No adjustment or extra physics."))
    PLAN.write_text("""# 198　197补偿的旧收益保护，失败早停

2026-11-06。197在196四历史通过开发，正常与193旧收益尚待检验。固定197机制和参数，最多7新实际、0训练。

顺序固定为原46秒正常seed171701，接着193右受推、左受推、右零力、左零力，最后补正常seed171702/3。正常沿原108×3相对/绝对预算及反向右转RL高度P95≥29.5702276 mm。193完整1000步、无终止、全程8度/撤力2秒/连续25步、全部34动作预算及转角3度。

首条未达就停止剩余运行，未打开格标为未测，保留失败原始数据和视频。成功则完整原324正常与193四条任务都通过，再另立冻结新条件验证。原预算不随结果改写。

每条保存实际和下层影子、输入源码SHA、状态物理和全部检查。正常首次实际改变前8数组对原157精确、实际输出到native目标逐步匹配。任务恢复/力/命令依原193针对性工具核查，避免全网络和梯度重演。必要视频只从已保存轨迹导出。
""",encoding="utf-8")
    print("PREPARED198 atmost7,stopfirstqualityfailure",flush=True)


def evidence(env,result,baseline):
    folder=ROOT/result["folder"];calls=env.full_heading_calls
    (folder/"full-heading-calls.pkl").write_bytes(pickle.dumps(calls,protocol=5))
    first=next((i for i,c in enumerate(calls) if not np.array_equal(c["actual"],c["shadow"])),len(calls))
    with np.load(folder/"states.npz") as a,np.load(ROOT/baseline["folder"]/"states.npz") as b:
        for key in a.files:np.testing.assert_array_equal(a[key][:first],b[key][:first])
        for i,c in enumerate(calls):np.testing.assert_array_equal(np.clip(env.nominal+.35*c["actual"],*env.model.actuator_ctrlrange.T),a["targets"][i])
    result.update(active_frames=sum(c["extra"]!=0 for c in calls),maximum_extra=max(abs(c["extra"]) for c in calls),
                  first_actual_change=first)


def normal(reference):
    loaded=load(read(ROOT/"configs/go2-frozen157-exit-matrix.json"));stack=loaded[0]
    try:
        plain,original=make_factory(ev.CONFIG,loaded)
        def make(*a,**kw):return attach(original(*a,**kw),stack)
        r=ev.normal_run(RUN,"fullheading197",reference["seed"],make,stack.adapter)
        env=plain.instances[-1];p=ROOT/r["folder"]
        save_probe(plain,p);save_calls(env,p)
        write_new(p/"selector.json",env.recovery_selection_log)
        np.savez_compressed(p/"initial-state.npz",**env.validation_initial)
        checks=ev.normal_checks(r,read(ROOT/"docs/data/go2-157-runtime/normal-budget-reference.json"))
        assert len(checks)==108
        rl=r["segments"]["reverse right"]["RL"]["clearance_p95_m"]*1000
        r.update(checks=checks,RL_mm=rl,kind="normal",
                 passed=bool(r["steps"]==2300 and not r["terminated"] and all(c["passed"] for c in checks) and rl>=29.5702276))
        evidence(env,r,reference)
        write_new(p/"normal-protection.json",r)
        return r
    finally:stack.close()


def main():
    task=read(CONFIG);verify()
    RUN.mkdir(exist_ok=False)
    for src,name in [(CONFIG,"prospective.json"),(PLAN,"predeclared-plan.md"),(ROOT/"scripts/check_go2_full_heading_retention.py","runtime.py")]:
        shutil.copy2(src,RUN/name)
    original_load=go2_task.load_stack;instances=[]
    def wrapped_load(spec):
        cfg,stack,factory,ev=original_load(spec)
        def wrapped_factory(ref):
            plain,original=factory(ref)
            def make(*a,**kw):
                env=attach(original(*a,**kw),stack);instances.append(env);return env
            return plain,make
        return cfg,stack,wrapped_factory,ev
    previous=read(ROOT/"docs/data/go2-fast-turn-push/summary.json")
    records=[];started=time.perf_counter()
    try:
        for n,h in task["inputs"].items():assert sha(ROOT/n)==h,n
        for label in task["order"]:
            if label.startswith("normal"):
                seed=int(label[6:]);r=normal(next(ref for ref in task["normal_refs"] if ref["seed"]==seed))
            else:
                side,kind=label[:-3].split("-")
                definition=next(d for d in task["definitions"] if d["side"]==side)
                with patch.object(go2_task,"load_stack",wrapped_load):
                    r=go2_task.run_task(definition["task"],RUN/label,kind)
                baseline=next(x["result"] for x in previous["records"] if x["side"]==side and x["kind"]==kind)
                evidence(instances[-1],r,baseline)
                write_new(ROOT/r["folder"]/"retention.json",dict(result=r,metrics=measure(r,definition)))
            records.append(dict(label=label,result=r))
            print("RETENTION198",label,r["passed"],r["turn_angles"],
                  [c for c in r["checks"] if not c["passed"]],flush=True)
            if not r["passed"]:break
        verify()
        for n,h in task["inputs"].items():assert sha(ROOT/n)==h,n
        summary=dict(experiment=198,records=records,new_physical=len(records),training=0,
            qualified=len(records)==7 and all(r["result"]["passed"] for r in records),
            not_run=task["order"][len(records):],wall_seconds=time.perf_counter()-started,visual="pending")
        write_new(RUN/"summary.json",summary);PUBLIC.mkdir(exist_ok=False);write_new(PUBLIC/"summary.json",summary)
    except Exception:
        (RUN/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8");raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepare",action="store_true")
    prepare() if p.parse_args().prepare else main()
