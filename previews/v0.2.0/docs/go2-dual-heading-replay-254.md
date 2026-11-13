# Go2 变速直行实验

249实验在平地零外力仿真中完成了22秒的站立、加速、减速和停步。254提供该固定任务的本地运行入口。已有对照视频左侧为248，右侧为249，待用户视觉验收。[22秒对照视频](../artifacts/dual-heading-handoff-video-249/comparison.mp4)

在项目根目录运行以下命令，使用已有虚拟环境和模型文件。`show`读取历史评分。`run`以seed 648101重新运行固定任务，并把轨迹、复现核查和评分写入新的输出目录。每次运行需使用新的目录名。

```powershell
.venv/Scripts/python.exe scripts/go2_dual_heading_replay.py show
.venv/Scripts/python.exe scripts/go2_dual_heading_replay.py run --output artifacts/runs/my-dual-heading-replay-01
```

|时间|速度指令|
|---|---:|
|0–2秒|站立|
|2–6秒|0.4 m/s|
|6–12秒|0.8 m/s|
|12–16秒|0.4 m/s|
|16–22秒|停止|

已测稳态均速为0.759 m/s，最大航向偏差3.926°，最大横移14.56厘米，停步后0.48秒进入稳定区间。四脚在高速稳态窗口的离地P95均超过20毫米。加速后最长连续稳定为11步，即0.22秒，原门槛为25步。`result.json`的`replay_exact`记录复现是否一致，`task_passed`记录运动任务是否通过。完整评分在`score.json`中。[249实验原记录](experiments/2026-11-08-dual-heading-handoff-results.md)

该预设依赖项目内归档的配置、模型和原轨迹，首次核验使用当前工作区及其虚拟环境。新的加速历史、整合模式下的转向和高速受推仍需独立实测。0.8 m/s为请求值，已测动作形态为小跑。
