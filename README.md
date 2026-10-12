# EGDM-HGPPO：S/M 训练与 L 跨规模测试

当前协议为 `paper-sm100-1`，检查点格式同名。唯一入口是 `scripts/run_31_paper.py`；不带子命令只显示帮助。本文方法保留动态异质图、事件触发与学习门控，派工固定为 β=1 ATC。所有学习方法从规则先验或各自登记初始化开始训练；无图消融与 MLP-PPO 共用一份结果。

## 实验口径

一个 epoch：选择一个固定算例，8 个环境从相同初态各执行一个完整回合，采样期间冻结策略，然后执行一次 PPO 更新周期（内部优化 3 遍）。曲线的 TWT 是 8 个完整随机回合的均值，标准差带不是置信区间。唯一目标为 TWT。

| 集合 | 算例数 | 用途 |
|---|---:|---|
| pilot | S/M 各 1，合计 2 | 联合预实验训练，每例 100 epoch |
| validation | S/M 各 5，合计 10 | 预实验结束后开发评估，不是正式测试 |
| train | S/M 各 18，合计 36 | 正式训练，每例 100 epoch |
| main | S/M 各 12，合计 24 | 正式主测试 |
| generalization | L 共 12 | 跨规模测试，原 main/L 文件内容不变 |
| tiny | 8 | 工程与参考解检查 |

92 个活动实例由 manifest 登记。24 个旧 L 训练/开发实例及 12 个 XL 实例已删除，原参数、种子、哈希与删除原因保留在清单中，`prepare` 不重建它们。每参数组合一个固定实例，活动实例缺失时只允许原参数、种子重建后通过原 SHA256，不能重抽样替换。

预实验每方法/种子为 200 epoch、1,600 完整回合；正式为 3,600 epoch、28,800 完整回合。种子 0/1/2 各自独立训练。每例按稳定循环被访问 100 次，不逐例训练完再换例。当前默认 CPU、8 个环境、8 个工作进程、4 个更新线程。

取消训练中验证和最佳权重选择；最终只评估完整预算后的末次模型。训练统计表另报每例第 91–100 次访问中 80 条回合的均值与波动，不能将其作为固定策略或独立测试结果。三次确定性回放只检查可复现性，不充当额外种子。

## 环境与首次检查

在 PowerShell 中执行：

```powershell
Set-Location 'D:\Python project\code_Chongyu_EGDM_HGPPO'
$py = 'E:\anaconda3\envs\python3.13\python.exe'
& $py -m pip install -r requirements.txt
& $py scripts/run_31_paper.py prepare
& $py scripts/run_31_paper.py smoke
& $py scripts/run_31_paper.py matrix --suite pilot-joint --phase all --dry-run
& $py scripts/run_31_paper.py matrix --suite formal --phase all --dry-run
```

`smoke` 只运行短验收：S/M 各 1 个八回合 epoch、7 个唯一学习实现的微型训练和末次回放、规则/优化入口、数据与合并测试、Visdom 及离线导出。它不运行正式主测试或 L 目标值评测。

## 单机联合预实验

```powershell
& $py scripts/run_31_paper.py pilot-joint
& $py scripts/run_31_paper.py evaluate-final --suite pilot-joint
& $py scripts/run_31_paper.py report --suite pilot-joint
```

第一条只训练，不自动评测。第二条在汇总电脑顺序执行全部末次模型的 3 次独立进程回放及规则/启发式对照，仅使用 10 个开发评估实例。`report` 保留失败、缺失、强对照和冲突，缺失实验不会被标记完成。

`pilot-single` 是可选诊断：只训练主方法的 S/M 两个独立模型，每例 100 epoch，不是必跑流程。运行它之后可用 `evaluate-final --suite pilot-single` 评测同规模开发实例。

## 多台电脑并行训练

各机器使用相同源码、依赖版本、固定数据和科学配置。复制源码时不要复制活动进程的锁文件。每台电脑只启动一个训练监督任务，内部使用本机并行环境。

例如三台电脑分配如下（每条命令的种子均为 0/1/2）：

```powershell
# 电脑 A
& $py scripts/run_31_paper.py pilot-joint --methods full shared_relation no_rule_prior --seeds 0 1 2
# 电脑 B
& $py scripts/run_31_paper.py pilot-joint --methods mlp periodic_gate --seeds 0 1 2
# 电脑 C
& $py scripts/run_31_paper.py pilot-joint --methods gat dan --seeds 0 1 2
```

也可按种子拆分，例如 `--methods full --seeds 2`。`--jobs` 接受 matrix 中的精确作业 ID；`--workers 2` 可减少工作进程，仍保持 8 环境。`--background` 隐藏启动并写独立标准流；重复进程受检查和锁保护。不同硬件不保证训练逐位一致，硬件与软件配置均落盘。

默认运行编号 `run01`。若需另一次独立运行，在所有相关命令使用预先约定的 `--run-id run02`；它形成独立结果，不能根据好坏替换原编号。训练过滤器不改变协议、数据或种子预算。

每个训练任务保存于：

```text
result/paper_sm100/<protocol_hash>/<suite>/<method>/seed_<n>/<run_id>/
```

包含 `latest.pt`、`config_snapshot.json`、`run.json`、`artifact.json`、原始回合、epoch 汇总、失败记录及任务状态。`artifact.json` 在作业结束时封存内容哈希。不要同步正在写入的任务目录。

为适配 Windows 路径长度，目录中的协议哈希采用 SHA256 前16位；注册、检查点及导入仍核对完整 SHA256，前缀冲突会明确拒绝。建议各机使用较短的仓库与结果根目录。

在汇总电脑导入其他电脑结束后的目录：

```powershell
& $py scripts/run_31_paper.py collect --from 'E:\transfers\computer_B\paper_sm100'
& $py scripts/run_31_paper.py collect --from 'E:\transfers\computer_C\paper_sm100'
& $py scripts/run_31_paper.py evaluate-final --suite pilot-joint
& $py scripts/run_31_paper.py report --suite pilot-joint
```

导入支持结果根目录或单个运行目录。相同内容重复导入不重复计数；异内容保留在 `_conflicts` 并阻止完整裁决；不兼容或未封存内容明确拒绝。不要手工覆盖冲突文件，不按 TWT 选取重复任务。总表在汇总电脑重新生成，PID、锁、服务状态位于 `_local/<host>`，不作为可移植任务状态。

最终评测首次登记汇总电脑的硬件/软件配置；后续所有方法沿用该电脑顺序评测。RH-MILP 的限时预算和模型推理延迟不跨硬件比较。

## 正式实验：先审阅，再授权

预实验报告完整且通过数值门槛后，由用户审阅。只有用户明确授权才执行：

```powershell
& $py scripts/run_31_paper.py authorize-formal --human-reviewed-pilot
```

该命令打印授权 JSON 的路径，绑定协议与预实验报告。将该文件复制到训练电脑，再运行：

```powershell
& $py scripts/run_31_paper.py formal --methods full shared_relation no_rule_prior --approval-file 'E:\transfers\formal-approval.json'
```

其余机器分配方式同预实验。所有训练结束后，将正式任务导入汇总电脑：

```powershell
& $py scripts/run_31_paper.py collect --from 'E:\transfers\formal_node_B\paper_sm100'
& $py scripts/run_31_paper.py collect --from 'E:\transfers\formal_node_C\paper_sm100'
& $py scripts/run_31_paper.py evaluate-final --suite formal
& $py scripts/run_31_paper.py report --suite formal
```

正式结束评测覆盖 24 个 S/M 主测试、12 个 L 泛化实例、全部对比和消融及微型参考解。L 不参与预实验或正式训练调参。规则包含 Fixed-EDD、Fixed-ATC、Periodic-ATC、RH-MILP、RH-ALNS，另保留 Opportunity-ATC 强参考；ALNS 使用三个随机种子。MILP 保持每调用 0.20 秒，ALNS 每事件 40 次迭代。

## 续训、状态、Visdom 与离线导出

```powershell
& $py scripts/run_31_paper.py status
& $py scripts/run_31_paper.py resume --suite pilot-joint --methods full --seeds 0
& $py scripts/run_31_paper.py evaluate-final --suite pilot-joint --resume
& $py scripts/run_31_paper.py monitor --start-server
& $py scripts/run_31_paper.py monitor --offline
```

失败后先查看任务 `failures/` 与本机 `_local/<host>/<suite>/` 日志。只有完整采样和 PPO 更新成功才提交 epoch；恢复从最近提交点重跑整个未提交批次，保留中断证据。旧检查点、科学配置或源码不匹配明确拒绝。迁移续训前必须导入封存任务；运行中损坏包不可跳过校验。

Visdom 绑定 `127.0.0.1`，从 8097 起查找端口并输出实际 URL。主图横轴为累计 epoch，纵轴为每例 8 回合 TWT 均值，保留标准差和 Periodic-ATC β=1/95% 基准线；epoch 0 不虚构训练均值。不同实例分图；失败标记不作为零 TWT。独立消费者断线自动补绘，离线导出 CSV/JSON/PNG/PDF。

报告位于 `<protocol_hash>/<suite>/_reports/<run_id>/`：`training-tail.csv`、`main-results.csv`、`ablation-results.csv`、`generalization-L.csv`、附录强参考、负荷/迁移效应、门控行为、参考值类别、失败/缺失明细和论断裁决。最终表只使用末次模型；质量和速度分别裁决。

当前交付只验证工程流程，不宣称长训练性能达标。详见 [实验规格](docs/experiment-spec.md)、[实现说明](docs/paper-implementation.md) 与 [清理验收记录](docs/cleanup-report.md)。
