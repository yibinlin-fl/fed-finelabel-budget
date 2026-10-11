# 数据协议与标签可见性边界（P0）

## P2 实现补充

当前用户授权进入 P2，历史 P1 停止约束已完成。`cifar_io.prepare_experiment` 按公共 ID 先从官方训练集留出独立验证集，再选择联邦训练子集；所有客户端分区只接受所选池的 coarse。`SimulationData` 持有隐藏 train 真值，只在可信控制器 `p2.py` 内创建各方法独立 Oracle；训练/查询不接收该对象。

验证/测试图像与真值仅进入 `OfflineEvaluator`，训练和查询模块禁止依赖 evaluator 或 cifar_io。每轮验证仅作日志诊断，不反馈查询、超参数或早停；测试仅在每个方法全部固定预算阶段训练结束后计算一次指标。ID 命名空间明确记录：训练内部 ID 映射到训练原 ID；官方测试 ID 具有独立 split，不与训练内部 ID 混用。

五种方法初始 2% 共用同一公共分区随机抽样，新增 ID 先通过控制器当前阶段配额检查，再向 Oracle.query 解锁；重复候选、已购候选或数量不匹配整批拒绝。checkpoint 只保存已购标签快照和公开 ID，不保存未购真值、评估标签或特征。训练轮边界恢复在可信控制器回放已购 ID，恢复费用等于累计已购数量，不产生重复额外计费。

查询只收到当前客户端公开 pool、ID、自己已购快照与模型。条件熵只访问已知 coarse 对应五个预测 logit；group balance 只用公开 coarse 的已购样本覆盖数；diversity 特征由本地模型计算并在本地选样时使用，不进入控制器日志或服务器接口。

所有方法使用相同绝对轮次种子、固定步数/批大小、有放回采样和公共客户端规模聚合权重，阶段间延续权重。45k 训练池预算是900/1800/3600，1000 样本 Pilot 是20/40/80。预算统计不得以包含验证样本的50k作为实际训练池分母。

## P1 历史实现补充

`p1.py` 是可信控制器，唯一持有 bootstrap 返回的 oracle；`training/` 无 `cifar_io` 或 oracle 导入。控制器只为客户端传递公共样本和已购标签快照，`ClientDataset` 将快照缩减为自身 ID 并复制，未知 fine 为 -1。训练 API 不接收评估数据或 oracle。损失检查已购 fine 与公开 coarse 层级的一致性，每个样本仅计一个监督项。

P1 smoke 固定轮数，不读取 val/test，日志不含隐藏标签；checkpoint 仅包含模型、公开层级、配置与训练成本。合成层级只用于工程测试，真实 smoke 的层级仍来自官方训练配对。预算、分区与种子记录在 run.json，自动测试检查依赖方向、零 fine 客户端、损失与可复现训练。原 P0 表中的“P1 待实现”现已由此补充实现；独立离线性能 evaluator 留待后续明确授权阶段。

## 数据类型与信任边界

| 组件 | 可以访问 | 不可以访问 |
|---|---|---|
| `PublicPool` | 样本图像、ID、coarse label | 未购买 fine label |
| `partition` | 样本 ID、coarse label、seed | 任意 fine label |
| 客户端训练器（P1 待实现） | 自己客户端的样本、coarse、已购买 fine；模型权重 | Oracle 的 hidden_truth、别人的原始数据、test fine |
| 本地查询策略（P2 待实现） | 客户端 public pool、已购买标签、当前模型预测/本地特征 | 隐藏的 fine、离线评估标签 |
| 可信控制器 | 由完整 train fine 构建的 `FineLabelOracle` | 不得将 oracle 传给客户端或查询模块 |
| 可信离线评估器（P1/P2 待实现） | 验证/测试真值，只输出指标 | 不能把验证/测试真值反馈给策略做样本选择 |

## 具体实现约束

- `cifar_io.py` 是可信边界。原始 CIFAR 官方 pickle 内带 `fine_labels`，因此训练/查询代码禁止直接加载该 pickle 或使用 `torchvision.datasets.CIFAR100.targets`。
- `Cifar100Hierarchy` 在可信启动阶段由完整标签配对推导。taxonomy / fine-to-coarse 映射是公开任务定义，**可以**进入粗粒度损失；单个样本 hidden fine label **不可以**。
- `PublicPool.sample(i)` 返回影像副本、coarse 和 ID，不返回完整训练样本的 true fine label。
- `FineLabelOracle.query` 原子校验整批 ID/预算；已购买 ID 重复查询不会重复计费；`purchased_labels()` 只返回一份不可变、已付费标签快照。
- Python `__private` 不是强安全边界。能读取磁盘 pickle 或反射 `FineLabelOracle` 的恶意代码仍能拿到隐藏标签。因此本项目承诺**实验级防意外泄漏**，不能承诺密码学隔离或差分隐私。
- 目前 P0 没实现物理隔离的分进程 oracle / client。P1 开发时必须把 API 依赖方向控制住，继续加负向测试。
- 数据划分、客户端采样、seed、损失权重、query budget 要在配置与报告中记录。IID/Dirichlet 基于 coarse 而不是 fine。
- P1 中训练集/验证集分割也必须无泄漏：线上可访问部分为 coarse/public，细标签只由预购授权；验证/测试 fine 只由隔离 evaluator 用于离线性能汇报，不可作查询引导或 early stopping。初期建议固定轮数。

## 协议统计

预算计数：50,000 training examples 的细标签申请数依次为 1,000 → 2,000 → 4,000（精细标注累计 2% → 4% → 8%）。P0 preflight 的 1000 个合成样本分别是 20 → 40 → 80。合成样本的数字**不是模型精度**。
