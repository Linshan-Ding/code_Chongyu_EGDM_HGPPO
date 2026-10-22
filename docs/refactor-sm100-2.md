# sm100-1 预实验不收敛的诊断与 sm100-2 重构说明

## 1. 诊断（依据 `docs/pilot_results/_reports/run01/` 与 sm100-1 源码）

sm100-1 的预实验：7 个学习方法 × 3 种子各 200 epoch（每 epoch 一个固定算例、8 条同初态回合、一次 PPO 周期），训练曲线从头到尾几乎平坦；`full` 在开发集上落后 Periodic-ATC（S +8.7%，M +24.2%）。按证据强度排序的原因：

1. **学习信号信噪比小于 1。** 每次更新只有同一算例的 8 条同初态轨迹，8 回合均值的标准误是 TWT 的 2% 到 6%，而候选集合内可达的改进只有几个百分点。训练尾部相邻 epoch 均值标准差与 8 回合标准误之比的中位数为 0.91，曲线的抖动完全是采样噪声。代码里实际没有折扣，λ = 0.95 按事件截断（有效视野约 20 个事件），而重构的代价与收益在几十到几百个事件后才体现，优势几乎全靠一个只见过同算例 3,300 个转移的评论家自举。
2. **熵奖励是唯一方向稳定的推力，而且推向更多重构。** 熵项作用在 ≤15 路候选分布上，其中 14 个候选都是重构；没有任何 KL 锚定把策略拉回先验。实测 `full` 的贪心重构频率为 0.19，Periodic-ATC 为 0.08 到 0.12，学习策略的重构频率与相对 TWT 单调正相关（M 规模三分位：0.118 → 1.04×，0.165 → 1.32×，0.252 → 1.56×）。
3. **候选集合里没有比锚点更好的动作，派工不可学。** 非锚点候选来自单独使用时表现最差的压力打分器，不过滤负收益；学习上限等于"永远选锚点"。dan 用固定周期重构加学习派工反而优于 `full`，说明派工侧才有可学收益，但方案承诺的双匹配已被移除。
4. **评估协议放大问题。** 没有贪心验证曲线，只看含 10% 随机重构的随机策略均值；只保留末次权重；预实验用 2 个算例训练、10 个不同场景算例评估；训练算例与到达流固定。
5. **表征让"刚重构完"的状态显得更好。** 候选打分把迁移中的资源立即计入目标阶段能力，门控特征在重构后立即改善。
6. **算力结构。** 仿真器与候选构造是纯 Python、逐环境串行，GPU 无用武之地。

## 2. 改动清单（协议 `paper-sm100-2`，科学配置全部进协议哈希）

| 层 | 改动 | 文件 |
|---|---|---|
| A1 | 回合级自评判优势：(B − TWT)/B，B 为该算例 Periodic-ATC 的 TWT，同算例复制组内中心化，回合内共享；评论家可关（`critic: none`）或辅助 | `agent/paper/ppo.py` |
| A2 | KL(π‖候选先验) 正则，系数 1.0 → 0.01 线性退火；熵奖励默认 0（DAN 保留）；优势标准化与各项均值只统计有真实选择的事件 | `ppo.py`、`network.py` |
| A3 | 每次更新 8 个算例 × 8 条回合（种子置换轮转），`log.csv` 一行一次更新并记录相对 TWT、门控率、锚点率、先验散度、裁剪比例、学习率 | `rollout.py`、`train.py`、`matrix.py`、`report.py`、`monitor.py` |
| A4 | 每 10 次更新在 10 个开发算例上做确定性贪心验证，写 `evaluations.csv`/`validation_summary.csv`，几何均值改善时保存 `best.pt`；结束评测对 `latest` 与 `best` 各回放三次，准入只看 `latest` | `train.py`、`rollout.py`、`runtime.py`、`matrix.py`、`report.py` |
| A5 | 学习率 warmup 10 次更新后线性衰减到 10%，`target_kl` = 0.02 提前停止，价值裁剪改为开关，调度器状态进检查点 | `ppo.py` |
| A6 | 势函数 `future_only`（只计未来拖期，终态为零）；候选打分中在途资源能力按 (1 − travel/period) 折减；门控特征新增在途资源数、剩余在途时间、预留单元数 | `environment/reward.py`、`env.py`、`graph_builder.py`、`resource_features.py` |
| B1 | 候选生成器 v2：LCR / 压力 / 交期加权三准则、正收益过滤、`lcr_now`、双步与三步、上限 24、准则标志特征 | `candidates.py`、`resource_features.py` |
| B2 | 学习派工残差头：ATC(β=1) 先验 0.9，零初始化残差，逐步选择，联合对数概率；`fixed_dispatch` 为其消融 | `dispatch.py`、`network.py`、`policies.py`、`trajectory.py`、`config.py` |
| B3 | 触发事件门控与重构预算开关（默认关，`configs/paper_RAB3.yaml` 打开） | `candidates.py` |
| C1 | 预实验训练集改为 12 个训练算例，预算 150 次更新 | `matrix.py` |
| C2 | 训练时到达流扰动（`train_instance_perturbation: arrivals`），worker 内重测锚点 | `data/generator.py`、`rollout.py`、`evaluate.py` |
| C3 | 每回合记录决策/仿真耗时；学习器可在 GPU 更新（采样、回放与 DAN 仍在 CPU）；`workers` 上限放开到每次更新的回合数 | `rollout.py`、`graph.py`、`network.py`、`ppo.py`、`train.py`、`scripts/run_31_paper.py` |
| C4 | `FORMAT = paper-sm100-2`，旧检查点自动拒绝；数据注册表共用 | `config.py`、`data.py` |
| C5 | 报表：验证曲线、规模 × 场景分面、`best` 与 `latest` 分列、按算例配对的 Wilcoxon p 值 | `report.py`、`monitor.py` |

未改动：网络宽度（hidden 64、2 层、单头）、基线与消融集合（新增 `fixed_dispatch`）、数据文件。

## 3. 验收与对照运行

1. `python -m pytest tests -q`（26 项）与 `python scripts/run_31_paper.py smoke`。
2. 预实验 v2 对照（单卡 GPU 电脑，各 2 种子先行）：

| 运行 | 配置 | 预期 |
|---|---|---|
| RA | `configs/paper_RA.yaml`（仅层 A） | 贪心开发曲线从 1.0 起步、前 50 次更新内可见下降；`best` 不劣于锚点（由构造保证） |
| RAB | `configs/paper.yaml`（层 A + B1 + B2） | 开发集几何均值相对 Periodic-ATC ≤ 0.95（S、M 各自），`gate_fraction` ≤ Periodic-ATC 的 1.5 倍 |
| RAB3 | `configs/paper_RAB3.yaml` | 与 RAB 对比门控收紧的效果 |
| GAE | `configs/paper_GAE.yaml` | 估计器消融 |

   判据：收敛（最后 30% 更新内开发曲线几何均值相对变化 < 1%）；种子间最终策略变异系数 < 5%。
3. 通过后再按实测吞吐修订 `formal_updates`，走 `authorize-formal` → `formal` → `evaluate-final` → `report`。

## 4. 曲线应该怎么读

- 每个训练算例一张图：各次更新该算例 8 条复制的 TWT 均值与标准差（采样策略，含探索）。
- 贪心开发验证图是唯一的收敛证据：纵轴为 TWT/Periodic-ATC 的几何均值，起点为 1.0（零残差锚点），虚线 0.95 为准入目标；`best_rel_twt_geomean` 单调不增。
- 诊断图：`kl_prior` 应随系数退火而放开，`clip_fraction` 维持在 0.05 到 0.3 之间，`gate_rate` 不应长期高于 Periodic-ATC 的 1.5 倍。
