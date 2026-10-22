# SM100-2 实现说明

`run_31_paper.py` 是唯一入口，协议/检查点为 `paper-sm100-2`。科学身份由源码、科学配置与活动数据 ID/参数/哈希构成，排除磁盘路径、工作进程数、更新线程、设备、Visdom 端口和运行编号；运行编号仍是任务命名空间的一部分。

`matrix` 提前定义训练/结束评测作业；`pilot-joint`/`formal` 只训练，`evaluate-final` 只做结束评测；支持方法、种子、作业 ID 筛选。每个学习任务生成 `latest` 与 `best` 两种检查点的回放作业。

固定数据（`agent/paper/data.py`）：活动算例 ID 由 `case_id(parameters, split)` 从设计参数推导（`<规模>_<场景>_rho<负荷×100>_tau<迁移倍数×100>`、`tiny_S<工位数>_J<订单数>_rho<负荷×100>`），`registered_manifest` 拒绝任何与参数不一致的 ID；`rename_registry` 是已执行过的一次性迁移（按原参数与种子重生成、逐字段比对、仅 ID 不同、重新登记哈希），不属于日常流程。

单算例诊断（`agent/paper/single.py`，子命令 `single`）：对一个登记算例依次运行全部规则/优化对照、各学习方法的训练（`train` 以该算例同时作训练与验证集，签名另含 `suite=single` 与算例身份）、`latest`/`best` 贪心评测与曲线导出，结果写入 `<protocol_hash>/single/<算例ID>/`，汇总在 `_reports/<run_id>/`。它与 `pilot-*`/`formal` 的结果树、报表和准入完全隔离。

训练循环（`agent/paper/train.py`）：`rollout.update_plan` 给出本次更新的 (算例, 复制) 槽位；`rollout.Pool.collect` 由持久 worker 采样，每条回合记录锚点 TWT、相对 TWT、门控比例、迁移数、决策与仿真耗时；`ppo.PPO.update` 按 `advantage_mode` 计算优势，损失 = PPO 裁剪项 + β_t·KL(π‖先验) − 熵系数·H，`target_kl` 提前停止，学习率按更新数调度；到期时 `Pool.validate` 在 worker 内对开发算例做确定性贪心评估，写 `evaluations.csv`/`validation_summary.csv`，改善则原子保存 `best.pt`。只有完整回合与 PPO 更新成功后才原子保存 `latest.pt` 并发布 `epochs/NNNNNN.json`。

决策模型（`agent/paper/candidates.py`、`dispatch.py`、`network.py`）：候选方案 v2 由 LCR / 压力 / 交期加权三准则生成并过滤负收益；`Network.select` 选方案，`Network.dispatch` 逐步选派工（ATC 先验 + 零初始化残差），`policies.learned_action` 合成物理动作；`Transition.action` 为 (方案索引, 派工选择元组)，`evaluate_actions` 据存储的派工步重算联合对数概率、熵与先验散度。

设备：`device: cuda` 时学习器网络与 PPO 更新在 GPU 上进行（图批在 `graph.move_graph` 中搬运），采样 worker、回放与 DAN 始终在 CPU；检查点与下发给 worker 的权重为 CPU 张量。`scripts/run_31_paper.py` 在 `device: cpu` 时隐藏加速器。本轮改动在 CPU 上验证，GPU 路径按同一代码路径实现但尚未在本机实测。

结果每方法/种子/运行隔离，封存、导入、冲突隔离与此前相同。报告新增 `validation-curves.csv`、`facets.csv`、`comparisons-best.csv`；配对比较含按算例配对的 Wilcoxon p 值；准入只看 `latest`。
