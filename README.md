# Go2 Locomotion & Recovery

## 最新实验预览 v0.2.0

在原冻结 157 运行版之外，项目新增 212 低速运动候选、249 变速原型及 254 固定历史重放入口，并整理了截至 2026-11-12 的 253、255 研究记录。[查看版本说明和测试范围](previews/v0.2.0/README.md) · [下载运行包](https://github.com/ybh161245/go2-locomotion-recovery/releases/tag/v0.2.0)

- 212 在 9 条既有指令历史的零力/30 N 短推对照中 18/18 逻辑条件通过（含结果复用），并完成单机独立目录复现。
- 244 分档直行的 1.0 m/s 指令下实际均速约 0.930 m/s；249 在固定零力条件下完成 22 秒变速与停步，254 对该历史的 1100 步精确重放。
- 以上均为限定场景的仿真结果；1.2 档及完整变速资格的边界、后续 253/255 未晋级结果保留在版本说明和数据中。

## 历史首版 v0.1.0

MuJoCo Go2 四足机器人仿真：速度跟踪、正常小跑与有限侧推恢复。

作者：**杨博涵** · [ORCID](https://orcid.org/0009-0006-8308-5477) · [GitHub](https://github.com/ybh161245/go2-locomotion-recovery)

这是冻结157的工程源码预览版，提供推理与三个精确重放入口；尚不是从零训练的一键复现或真机控制软件。实现包含冻结PPO运动/专家组合、50维随机特征恢复头和70维接触残差头，不是单一端到端PPO。

## 已测结果与边界

|范围|实测结果|
|---|---|
|指定正常动作|3条46秒、324项原质量预算通过；指定正常视频已获用户视觉接受|
|冻结后30N新组合|8/8受推、4/4唯一零力通过；只支持已测离散组合|
|原60定义×3评估种子压力矩阵|147/180；27提前终止、6完整但倾角失败|
|独立运行包|正常、成功恢复、已知失败三条重放，全部8组原始数组精确匹配|

三个评估种子不是三条独立训练链。复杂地形、广泛方向/相位、高速猎犬步态和真机均未验证。已知失败公开保留；更多实验结论及动作代价见 [RESULTS.md](RESULTS.md)。

## 安装与重放

已验证环境为Windows、CPython 3.13.14与锁定依赖。其它系统尚未验证。

1. 克隆仓库；安装 `requirements-lock.txt` 声明的依赖。
2. 从 [Releases](https://github.com/ybh161245/go2-locomotion-recovery/releases) 下载 `go2-v0.1.0-runtime.zip`。**仅下载GitHub自动生成的Source code.zip不含模型。若尚无运行附件，则当前只能阅读源码，待附件发布后再重放。**
3. 在仓库根目录校验并恢复附件，然后执行下面的命令。安装器不会覆盖内容不同的已有文件。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/install_runtime_assets.py --archive <运行附件ZIP的路径>
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --check
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --load-only
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --replay --case recovery --output artifacts/runs/my-recovery-01
```

`--check`、`--load-only`不推进物理。`--replay`每次新增一次仿真，输出目录必须不存在；case可选`normal`（46秒）、`recovery`（12秒）和`failure`（原7.56秒终止）。failure命令成功只代表失败边界精确复现，不表示机器人通过。

## 视频与数据

可选 `go2-v0.1.0-visuals.zip` 解压到仓库根目录后，打开 `artifacts/第一版集中视觉验收-157-rc1.html`。正常/成功恢复页左145旧版、右157；失败页两侧均失败。用户只接受了展示的右侧正常与phase626成功恢复，未扩大到全部压力测试。

目录与SHA见 [DATA_INDEX.md](DATA_INDEX.md)、[发布清单](release/assets.json)。这里公开代表运行输入、原180格结果表和组件比较；全部训练历史、失败轨迹及完整研究训练入口仍在母项目留存，尚未全部发布，不能据此宣称完整训练研究已复现。

## 许可与贡献

原创源码采用 [MIT](LICENSE)；本项目生成的模型/数据授权见 [MODEL_DATA_LICENSE.md](MODEL_DATA_LICENSE.md)。第三方Go2模型保留其BSD-3-Clause许可和来源，依赖各自许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。不分发Python解释器、虚拟环境、wheel缓存或FFmpeg二进制。

问题或改进建议请通过Issues提交，并附场景、运行版本和原始结果。欢迎复现实测边界，不将有限通过解读为任意推力下不倒。
