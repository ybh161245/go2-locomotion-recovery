# Go2 运动控制与变速实验预览

本包包含经过固定历史验证的 212 低速运动候选、249 变速实验和 254 精确重放入口；另附截至 2026-11-12 的 253、255 研究记录、关键数据和检查点。157 仍为此前公开的冻结基线。212、249 为不同功能候选，254 是 249 的重放入口；253、255 不替换已选基线。

## 运行环境

已验证 Windows、CPython 3.13.14 和 `requirements-lock.txt` 中的版本。解压后在本包根目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe scripts/go2_speed_hold.py --controller 212 --side right --kind zero --output artifacts/runs/my-212-right-zero
.venv/Scripts/python.exe scripts/go2_dual_heading_replay.py show
.venv/Scripts/python.exe scripts/go2_dual_heading_replay.py run --output artifacts/runs/my-254-replay
```

每次运行使用不存在的新输出目录；不要在不可信的包中加载 pickle 或模型文件。`README.md` 为原始 212 说明；`manifest-212.json` 与 `configs/go2-dual-heading-local-entry.json` 保留原有输入哈希。`manifest-v020.json` 核对整个本包，不包含自身。

## 记录范围

- 212：9 条既有指令历史各设零力和 30 N 短推，统一比较 18/18 逻辑条件通过，包含历史结果复用；独立目录复现限定于已测本机环境。
- 244：0.4、0.6、0.8、1.0 m/s 四个指令档在分档直行开发评估中通过；1.0 档实际均速约 0.930 m/s。1.2 档速度及右前足离地未通过。
- 249/254：固定零外力 22 秒序列跑完，包含 0.4→0.8→0.4 m/s 和停步，254 同历史重放与归档数据一致；加速连续稳定门槛未通过，不能当作完整资格通过。
- 253：一次续训在固定混合任务上改善速度与横移，但连续稳定和部分负载预算未通过；检查点仅作为研究记录，不代表已采用候选。
- 255：错相足序为静态几何预检，未进行动态训练或验证，几何误差未达原门槛。

`docs/experiments/`、`docs/data/` 中保留原始结果与部分研究记录；记录中的少量历史链接指向完整本地研究工程，未全部包含在此预览。复杂地形和真实机器人不属于已验证范围；不将指定场景结果扩展为通用性能。源码采用 MIT，项目生成的模型与数据见 `MODEL_DATA_LICENSE.md`，第三方素材保留各自许可。
