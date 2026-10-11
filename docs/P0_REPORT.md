# P0-Setup 实测报告

生成日期：2026-10-11  
课题：Budgeted Federated Fine-Label Acquisition with Coarse Supervision  
本轮阶段：**P0-Setup COMPLETE / P1 NOT STARTED**。

## 执行环境

- Python: 3.13.5
- NumPy: 2.3.5
- PyYAML: 6.0.3
- pytest: 9.0.2
- 操作环境：隔离的 Linux CPU Python 环境；**没有执行 GPU / V100 训练**。

## 实际执行命令与结果

```bash
PYTHONPATH=src python -m pytest -q
# 18 passed in 0.13s

PYTHONPATH=src python -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
# status: PASS
# synthetic_samples: 1000
# clients: 10
# fine_query_targets: [20, 40, 80]
# fine_queries_observed: [20, 40, 80]
# remaining_queries: 0

python -m pip install -e '.[dev]' --no-deps --no-build-isolation -q
python -m pytest -q
# 18 passed in 0.07s

python -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
# status: PASS
```

## 已实现及验证的项目模块

- Python 打包、独立工程结构、可复现实验配置。
- CIFAR-100 官方 pickle 格式读取器（本轮使用测试夹具模拟官方字段，真实官方数据**尚未下载或核验**）。
- 官方标签配对推导 20×5 层次映射，拒绝矛盾映射和不完整层级；包含非连续类 ID 单测。
- `PublicPool` 只提供图片与粗标签；`FineLabelOracle` 只返回已购买 fine 标签，重复查询不计费，失败批次回滚。
- IID / coarse-Dirichlet，随机种子确定，所有样本 exactly once。
- 全局预算初始 2%、后续额外 2%、4%，保证累计 2%/4%/8% 数量。
- 纯合成离线数据流完整自检，不使用 GPU 或真实训练数据。

## 尚未做 / 不可声称

- 没有 ResNet18、FedAvg 或真实训练循环。无真实精度、训练耗时或通信成本结果。
- 没有下载 CIFAR-100 官方 tar.gz，本轮没有真实文件完整性校验。
- 没有实现 B0–B4 查询基线、P1 烟测或 V100 实验。
- 标签 Oracle 仅具实验协议上的*逻辑隔离*，不能防范 Python 恶意反射/直接读取磁盘文件，不能宣称密码学隐私或差分隐私。
- 不声明论文方法的新颖性或实验增益。

## 下一步

将整个工程文件夹在 Codex 打开，执行 `docs/CODEX_NEXT_TASK.md` 的 P1 任务，并将 Codex 的 `reports/P1_REPORT.md` 发回继续评审。**严格先 P1，再决定是否跑 V100 Pilot。**
