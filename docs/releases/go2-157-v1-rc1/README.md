# Go2 157 · 工程第一版候选 rc1

纯仿真速度跟踪与有限侧推恢复；冻结157，PPO运动/专家组合加观测驱动随机特征残差。当前为本地候选，正式选择未更换，未公开发布。

- 先看`artifacts/第一版集中视觉验收-157-rc1.html`：正常46秒、恢复成功与已知失败；附167的四组30N并排视频。
- `docs/releases/go2-157-v1-rc1/SCOPE.md`列已测范围、连续动作代价与未完成项；新恢复视觉尚待用户。
- `docs/releases/go2-157-v1-rc1/attachments.json`提供必需附件与可选视频的路径、SHA和当前来源。源码Git树不装入大模型、原始轨迹、网格或依赖二进制。
- `docs/releases/go2-157-v1-rc1/THIRD_PARTY_NOTICES.md`记录实际锁定依赖及原生模型许可。

## 运行

先按附件清单恢复所需模型/资产/参考raw到对应相对路径。可用`python docs/releases/go2-157-v1-rc1/restore_attachments.py --from-directory <原157有效包目录>`校验并恢复38个必需附件；已有不同内容会停止而不会覆盖。可加`--verify-only`只检查。可选视频按attachments.json同路径另行复制。

在Windows CPython3.13.14环境安装requirements-lock.txt，再使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --check
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --load-only
.\.venv\Scripts\python.exe -I -B scripts/go2_157.py --replay --case recovery --output artifacts/runs/my-recovery-01
```

`--check`和`--load-only`0物理；每次`--replay`新增1次仿真，输出目录必须不存在。case还可选normal和failure，分别为完整46秒正常和7.56秒终止的原失败边界。失败case命令验收成功只代表边界复现。

当前已实测的独立包在母项目`artifacts/delivery/go2-157-v1-validated`，37个同版本wheel离线安装和三个代表重放精确通过。该包是原始P2版本，不能用它的旧manifest校验本rc1的新说明文件；两个版本分别留存。rc1仅准备源文件与附件清单，未另建环境或重跑物理。

## 复现范围

本候选提供推理/重放与有限已测结果。完整训练代码与失败证据仍在母项目；最终统一对照、多训练种子与训练复现不包含在本候选的三个演示资格内。源码MIT与第三方各自许可分别保留，模型/数据附件的发布许可在最终发布前确认。
