# SM100-2 实验规格

协议 `paper-sm100-2`；唯一目标 TWT。固定数据注册表（92 个活动算例、36 条已删除记录）与 `paper-sm100-1` 共用，文件与哈希不变。

## 训练更新的定义

一次更新 = `instances_per_update`（默认 8）个训练算例 × `episodes_per_instance`（默认 8）条完整随机回合，共 `environments` = 64 条回合，全部用同一冻结策略采样，然后执行一次 PPO 周期（最多 `ppo_update_epochs` = 3 遍，`target_kl` = 0.02 时提前停止）。算例按种子置换轮转（`rollout.update_plan`），每个周期内每个算例恰被访问一次；同一算例的 N 条复制共享初态哈希。`train_instance_perturbation: arrivals` 时，每次更新对每个算例抽一个到达流扰动种子（复制共享），算例文件不变，锚点 TWT 在 worker 内对扰动后算例重新测量。

## 信用分配与正则

- `advantage_mode: self_critical`：回合优势 = (B − TWT)/B，B 为该算例（或其扰动）上 Periodic-ATC 的 TWT；同算例复制组内中心化（`group_center`）；回合内所有决策共享该优势（γ = 1，无自举）。`critic: none` 时不训练评论家；`auxiliary` 时评论家以蒙特卡洛回报为目标但不参与优势。`advantage_mode: gae` 保留为估计器消融（`configs/paper_GAE.yaml`）。
- 正则：KL(π‖候选先验) 系数从 `kl_prior_coef_start` = 1.0 线性退火到 `kl_prior_coef_end` = 0.01（`kl_anneal_updates` = 100）；熵奖励默认 0（DAN 无先验，保留 `entropy_dan` = 0.01）。优势标准化与各项均值只统计有真实选择的事件。
- 学习率：`lr_warmup_updates` = 10 后线性衰减到 `lr_final_ratio` = 0.1；梯度裁剪 0.5。

## 决策模型

- 候选方案 v2（`candidate_version: v2`）：`periodic_anchor`（Periodic-ATC 的贪心重构，仅到期时非空）、`keep`、`lcr_now`（LCR 准则贪心，不受周期限制）、LCR/压力/交期加权三准则正收益单步迁移并集的前 8 个、前 4 个单步的最优后续（双步）、最优双步的第三步、`opportunity`；上限 `max_candidates` = 24；候选特征含 3 个准则标志。`candidate_version: v1` 保留原生成器。
- 学习派工（`learned_dispatch: true`）：候选的派工在方案选定后逐步决策，每步选项为当前前缀下合法的（就绪工序, 空闲单元）对，先验把 `dispatch_prior` = 0.9 压在 ATC(β=1) 最优对上，残差头零初始化，初始行为与规则基线逐位一致；事件联合对数概率 = 门控/方案 + 各派工步之和。`fixed_dispatch` 为该机制的消融变体。
- 门控开关：`gate_trigger_mode: trigger_events` 只在订单到达、迁移完成、"工序完成且某阶段空转或压力 ≥ 1.5" 时保留学习型重构候选；`reconfiguration_budget` 达到后只剩锚点与 keep。默认均关（`configs/paper_RAB3.yaml` 打开）。
- 势函数 `potential_mode: future_only`：Φ = −η Σ w·max(0, Rmin − max(0, d − t))，恒等式 Σshaping = −Φ(s0) 保持。`transit_aware_features`：候选打分中在途资源的目标能力按 (1 − travel/period) 折减；门控特征新增在途资源数、剩余在途时间、预留单元数。

## 数据集合与套件

| 集合 | 算例数 | 用途 |
|---|---:|---|
| train（pilot v2 子集） | 12（S/M × D1/D3/D4 × 负荷 0.7/0.9，迁移倍数 1.25） | `pilot-joint` 训练 |
| train | 36 | `formal` 训练 |
| validation | 10 | 训练中贪心验证、best.pt 选择、预实验结束评测 |
| main | 24 | 正式主测试（不参与任何选择） |
| generalization | 12（L） | 正式跨规模测试 |
| tiny | 8 | 工程与参考解检查 |

预算：`pilot_updates` = 150，`formal_updates` = 600（按实测吞吐修订后登记）。每 `validation_every_updates` = 10 次更新（含更新 0 与末次）在 validation 集合上做确定性贪心评估，记录 `evaluations.csv`、`validation_summary.csv`，几何均值改善时写 `best.pt`。`latest.pt` 为末次模型。两者都进入结束评测与报表；**准入与论断只看 `latest`**，`best` 在开发集上的分数带选择偏差，只在正式主测试/L 上作为无偏结果报告。

## 结束评测与报表

结束评测在登记的汇总主机顺序执行：每个学习任务的 `latest` 与 `best` 各 3 次新进程确定性回放，规则 Fixed-EDD / Fixed-ATC / Periodic-ATC / RH-MILP / RH-ALNS（3 个算法种子）与附录 Opportunity-ATC。报表新增 `validation-curves.csv`、`facets.csv`（规模 × 场景）、`comparisons-best.csv`，配对比较给出按算例配对（种子取均值）的 Wilcoxon 双侧 p 值。消融：mlp（兼无图）、shared_relation、periodic_gate、no_rule_prior、fixed_dispatch；外部学习对照：gat、dan。

正式准入：预实验报告完整；`latest` 在 S、M 各自的开发集平均 gap ≤ −5%；对 mlp 的平均改善 ≥ −1%；三种子同预算、全部可行、回放一致。p95 延迟 ≤ 100 ms 单独裁决。数值通过后仍须用户审阅并显式授权。

## 对照运行

| 配置 | 含义 |
|---|---|
| `configs/paper.yaml` | 层 A + B1 + B2（主配置） |
| `configs/paper_RA.yaml` | 仅层 A：v1 候选、固定 ATC 派工 |
| `configs/paper_RAB3.yaml` | 层 A + B，开启触发事件门控与重构预算 |
| `configs/paper_GAE.yaml` | 估计器消融：GAE + 评论家 |

每个配置的科学参数不同，协议哈希与结果树各自独立，互不覆盖。
