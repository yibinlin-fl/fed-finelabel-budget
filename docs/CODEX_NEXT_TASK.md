# 发送给 Codex 的第一条指令（P1 — 仅基础训练闭环）

你现在接手一个已经存在的联邦学习科研项目 `fed-finelabel-budget`。**不要重新搭建、覆盖或重构项目骨架。**

先阅读并遵守：
- `AGENTS.md`
- `README.md`
- `docs/research_plan.md`
- `docs/DATA_CONTRACT.md`
- `docs/P0_REPORT.md`

## 你的本轮目标：P1 基础训练闭环

开始前：若当前目录还不是 Git 仓库，请先执行 `git init` 并将当前 P0 文件创建为初始提交（如果 Git 身份未配置，记录原因，允许跳过提交）；保证不把数据或模型文件提交进去。

1. **先检查已有工程并真实执行** `python -m pytest -q` 和 `python -m fed_finelabel_budget.cli preflight --config configs/p0.yaml`。排除环境错误，核验 P0 数据协议，不要直接重写已有测试。
2. 在不破坏 P0 数据隔离的前提下，新增 `models/resnet_cifar.py`（CIFAR stem 的 ResNet18，100 类输出）、`training/losses.py`（fine CE + coarse marginal likelihood）和 `training/fedavg.py`（单进程串行客户端模拟，按样本数正确加权聚合），以及必要的 PyTorch 数据适配。
3. 严格区分**已购买 fine 的样本**与**只有 coarse 的样本**：前者仅计 fine CE，后者使用 `-(logsumexp(z_{S(c)})-logsumexp(z))`。不要默认对同一样本重复施加两种损失。提供数值稳定、边界和梯度测试。
4. 先用**合成小图像 + 2～3 个客户端**完成 CPU smoke training（至少验证一次真实 forward/backward、FedAvg 权重聚合、参数变化与 checkpoint）。单独记录它是 smoke，不是 CIFAR-100 性能结果。
5. 如果官方 CIFAR-100 已下载，则增加真实数据 1～2 轮的小规模 smoke 命令；如果还没下载，只提供明确下载命令并记录没跑，不要因此阻塞其他 P1 交付。
6. 随机种子、数据 partition、模型初始化、客户端抽样、fine query 必须可复现；不要将训练真值/测试真值传给查询选择逻辑。
7. 增加 `configs/p1_smoke.yaml`、可执行 `scripts/run_p1_smoke.py`、测试，以及 `reports/P1_REPORT.md`。报告写明真实执行命令、环境版本、通过的测试、模型权重是否发生更新、剩余问题。
8. 只做 **P1**，不实现 B0–B4 查询算法、不运行多随机种子 V100 pilot、不设计论文创新算法，不加入 CLIP / MoE / teacher / pseudo-label / 蒸馏。

## 质量与停止条件

- 如发现既有 P0 bug：给出复现用例、最小修复及回归测试。
- 必须真跑测试；结果不得编造。对新依赖使用当前运行环境可支持的版本。
- 不允许因训练过长占用 Codex 会话：smoke 设置很小的 step/round 限额；完整训练以后交给 V100 脚本后台管理（由用户发起）。
- 完成后**停止，不要擅自进入 P2**。输出修改文件、测试结果、运行命令、下一步 P2 建议。
