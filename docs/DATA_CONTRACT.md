# 数据协议与标签可见性边界（P0）

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
