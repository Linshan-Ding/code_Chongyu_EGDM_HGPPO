# EGDM-HGPPO：S/M 训练与 L 跨规模测试（协议 paper-sm100-2）

当前协议为 `paper-sm100-2`，检查点格式同名。唯一入口是 `scripts/run_31_paper.py`；不带子命令只显示帮助。本文方法保留动态异质图、事件触发与学习门控，候选重构方案由三准则生成，派工以 β=1 ATC 为先验、由学习残差调整。所有学习方法从规则先验或各自登记初始化开始训练；初始贪心策略与 Periodic-ATC 逐位一致。无图消融与 MLP-PPO 共用一份结果。旧协议 `paper-sm100-1` 的预实验记录保留在 `docs/pilot_results/`，其诊断与本轮改动见 [docs/refactor-sm100-2.md](docs/refactor-sm100-2.md)。

## 实验口径

一次更新：从训练集按种子置换取 8 个算例，每个算例 8 个环境从相同初态各执行一个完整回合（共 64 回合），采样期间冻结策略，然后执行一次 PPO 更新周期（内部最多 3 遍，超过 target-KL 提前停止）。优势是回合级自评判值（相对该算例 Periodic-ATC 的 TWT，同算例组内中心化），不用评论家自举；正则是对候选先验的 KL，系数随更新退火。每 10 次更新（含第 0 次与末次）在 10 个开发算例上做确定性贪心评估，曲线以 Periodic-ATC = 1.0 为基准线，几何均值改善时保存 `best.pt`。唯一目标为 TWT。

| 集合 | 算例数 | 用途 |
|---|---:|---|
| train（pilot v2 子集） | 12（S/M × D1/D3/D4 × 两档负荷） | 联合预实验训练 |
| validation | 10 | 训练中贪心验证、best.pt 选择、预实验结束评测 |
| train | 36 | 正式训练 |
| main | 24 | 正式主测试 |
| generalization | 12（L） | 跨规模测试 |
| tiny | 8 | 工程与参考解检查 |

92 个活动实例由 manifest 登记，参数、种子与算例内容沿用 `paper-sm100-1`；活动实例缺失时只允许原参数、种子重建后通过原 SHA256。算例 ID 由设计参数推导、可直接写入论文：`<规模>_<场景>_rho<负荷×100>_tau<迁移倍数×100>`，小数点写作 `p`（如 `S_D3_rho72p5_tau100`、`M_D1_rho90_tau125`、`L_D4_rho95_tau200`）；工程算例为 `tiny_S<工位数>_J<订单数>_rho<负荷×100>`。manifest 登记的 ID 必须与参数推导一致（`agent/paper/data.py: case_id`），否则拒绝加载；两个 pilot 算例的 ID 与哈希另行固定。`train_instance_perturbation: arrivals` 时训练回合使用算例的到达流扰动（文件不变，锚点 TWT 在 worker 内重新测量）。

预实验每方法/种子 150 次更新、9,600 完整回合；正式 600 次更新（按实测吞吐修订后登记）。种子 0/1/2 各自独立训练。当前默认 CPU、8 个工作进程、4 个更新线程；`device: cuda` 时学习器更新在 GPU 上进行，采样仍在 CPU。

结束评测对每个学习任务的 `latest.pt`（末次模型）与 `best.pt`（开发集选出）各做三次确定性回放。**准入与论断只看 `latest`**；`best` 在开发集上的分数带选择偏差，仅在正式主测试/L 上作为无偏结果报告。训练统计表另报每例最后 10 次访问的回合均值与波动，不能作为固定策略或独立测试结果。

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

`smoke` 只运行短验收：S/M 各 1 个八回合更新并核对零残差锚点轨迹、8 个唯一学习实现的微型训练（含一次贪心验证与 best.pt）、`latest`/`best` 末次回放、规则/优化入口、数据与合并测试、离线导出。它不运行正式主测试或 L 目标值评测。

## 单机联合预实验

```powershell
& $py scripts/run_31_paper.py pilot-joint
& $py scripts/run_31_paper.py evaluate-final --suite pilot-joint
& $py scripts/run_31_paper.py report --suite pilot-joint
```

第一条只训练，不自动评测。第二条在汇总电脑顺序执行全部末次与开发集最优模型的 3 次独立进程回放及规则/启发式对照，仅使用 10 个开发评估实例。`report` 保留失败、缺失、强对照和冲突，缺失实验不会被标记完成。

对照运行用独立配置（协议哈希与结果树各自独立）：`--config configs/paper_RA.yaml`（仅训练层改动：v1 候选、固定 ATC 派工）、`configs/paper_RAB3.yaml`（加触发事件门控与重构预算）、`configs/paper_GAE.yaml`（GAE + 评论家的估计器消融）。

`pilot-single` 是可选诊断：只训练主方法的 S/M 两个独立模型（各 6 个训练算例），不是必跑流程。

## 单算例收敛诊断

在正式预实验之前，可先在一个固定的小算例上检验各学习方法能否收敛、最终目标值能到哪里：

```powershell
& $py scripts/run_31_paper.py single --instance S_D3_rho72p5_tau100 --methods full gat dan --seeds 0 1 2 --updates 100 --episodes 32
```

`single` 只用 `--instance` 这一个登记算例：先跑全部规则/优化对照（RH-ALNS 三个算法种子），以 Periodic-ATC 的 TWT 为参照 1.0；再对 `--methods` 中的每个学习方法、每个种子训练 `--updates` 次更新（每次更新为该算例 `--episodes` 条完整回合，默认 32），每 `--validation-every`（默认 5）次更新在同一算例上做一次贪心验证并保存 `best.pt`；结束后对 `latest.pt` 与 `best.pt` 各做一次贪心评测并导出曲线。结果在 `result/paper_sm100/<protocol_hash>/single/<算例ID>/`，汇总在其 `_reports/<run_id>/`（默认 `single01`）：`baselines.csv`、`learners.csv`、`validation-curves.csv`、`convergence.csv`（最后 30% 验证点的相对变化 < 1% 记为收敛）、`ranking.csv` 与 `summary.md`。训练与评测算例相同，这只是可学习性与收敛性诊断，不是泛化证据，也不进入预实验准入或任何论断；`--run-id`、`--workers`、`--config`、`--resume` 与其他子命令同义。

## 多台电脑并行训练

各机器使用相同源码、依赖版本、固定数据和科学配置。复制源码时不要复制活动进程的锁文件。每台电脑只启动一个训练监督任务，内部使用本机并行环境。

例如三台电脑分配如下（每条命令的种子均为 0/1/2）：

```powershell
# 电脑 A
& $py scripts/run_31_paper.py pilot-joint --methods full shared_relation no_rule_prior --seeds 0 1 2
# 电脑 B
& $py scripts/run_31_paper.py pilot-joint --methods mlp periodic_gate fixed_dispatch --seeds 0 1 2
# 电脑 C
& $py scripts/run_31_paper.py pilot-joint --methods gat dan --seeds 0 1 2
```

也可按种子拆分，例如 `--methods full --seeds 2`。`--jobs` 接受 matrix 中的精确作业 ID；`--workers N` 可改工作进程数（上限为每次更新的回合数 64），每次更新的回合数不变。`--background` 隐藏启动并写独立标准流；重复进程受检查和锁保护。不同硬件不保证训练逐位一致，硬件与软件配置均落盘。

默认运行编号 `run01`。若需另一次独立运行，在所有相关命令使用预先约定的 `--run-id run02`；它形成独立结果，不能根据好坏替换原编号。训练过滤器不改变协议、数据或种子预算。

每个训练任务保存于：

```text
result/paper_sm100/<protocol_hash>/<suite>/<method>/seed_<n>/<run_id>/
```

包含 `latest.pt`、`best.pt`（若开发集改善）、`config_snapshot.json`、`run.json`、`artifact.json`、原始回合（`episodes.csv`）、更新汇总（`log.csv`）、验证明细（`evaluations.csv`、`validation_summary.csv`、`validation/epoch_*/`）、失败记录及任务状态。`artifact.json` 在作业结束时封存内容哈希。不要同步正在写入的任务目录。

目录中的协议哈希采用 SHA256 前 16 位；注册、检查点及导入仍核对完整 SHA256。导入其他电脑结束后的目录：

```powershell
& $py scripts/run_31_paper.py collect --from 'E:\transfers\computer_B\paper_sm100'
& $py scripts/run_31_paper.py collect --from 'E:\transfers\computer_C\paper_sm100'
& $py scripts/run_31_paper.py evaluate-final --suite pilot-joint
& $py scripts/run_31_paper.py report --suite pilot-joint
```

导入支持结果根目录或单个运行目录。相同内容重复导入不重复计数；异内容保留在 `_conflicts` 并阻止完整裁决；不兼容或未封存内容明确拒绝。最终评测首次登记汇总电脑的硬件/软件配置；后续所有方法沿用该电脑顺序评测。

## 正式实验：先审阅，再授权

预实验报告完整且通过数值门槛后（`latest` 在 S、M 各自开发集平均 gap ≤ −5%，对 mlp 不劣于 −1%，三种子、全部可行、回放一致），由用户审阅。只有用户明确授权才执行：

```powershell
& $py scripts/run_31_paper.py authorize-formal --human-reviewed-pilot
& $py scripts/run_31_paper.py formal --methods full shared_relation no_rule_prior --approval-file 'E:\transfers\formal-approval.json'
& $py scripts/run_31_paper.py collect --from 'E:\transfers\formal_node_B\paper_sm100'
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

只有完整采样和 PPO 更新成功才提交一次更新；恢复从最近提交点重跑整个未提交批次，保留中断证据。旧检查点、科学配置或源码不匹配明确拒绝。

曲线：每个训练算例一张图（该算例各次更新复制的 TWT 均值与标准差、Periodic-ATC 基准线）、一张贪心开发验证图（几何均值 TWT/Periodic-ATC，基准线 1.0 与 0.95）、一张相对训练批次图（采样策略相对 TWT、门控率、锚点率）与诊断图（策略损失、KL、先验散度、裁剪比例、学习率等）。失败标记不作为零 TWT。离线导出 CSV/JSON/PNG/PDF。

报告位于 `<protocol_hash>/<suite>/_reports/<run_id>/`：`validation-curves.csv`、`training-tail.csv`、`main-results.csv`、`ablation-results.csv`、`facets.csv`、`comparisons.csv`（含 Wilcoxon p 值）、`comparisons-best.csv`、`generalization-L.csv`、附录强参考、负荷/迁移效应、门控行为、参考值类别、失败/缺失明细和论断裁决。

当前交付只验证工程流程，不宣称长训练性能达标。详见 [实验规格](docs/experiment-spec.md)、[实现说明](docs/paper-implementation.md)、[重构说明](docs/refactor-sm100-2.md) 与 [清理验收记录](docs/cleanup-report.md)。
