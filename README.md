# Fed Fine-Label Budget — 第二篇论文初始工程

> **当前阶段 P1：基础训练闭环已实现。**
> 包含数据协议、ResNet18、混合监督损失和串行 FedAvg；CPU 合成 smoke 仅验证工程，不构成 CIFAR-100 性能结果。B0–B4 与 P2 尚未实现。真实执行记录见 `reports/P1_REPORT.md`。

## 课题

**Budgeted Federated Fine-Label Acquisition with Coarse Supervision**  
具有粗粒度监督的预算受限联邦细标签获取。

研究问题：每个客户端已有粗标签，且只有有限预算能向标注 oracle 申请 fine label。在不分享样本的情况下，如何协调客户端 × coarse-group 的 fine 标签申请预算，提高全局 100 类分类的标注效率？

**Status**: CONDITIONAL_PILOT — 尚未证明创新性/性能增益。见 `docs/research_plan.md`。

## 30 秒本地自检（不需要 GPU / 不下载数据）

要求：Python **3.10+**，建议使用虚拟环境。终端要位于项目根目录。

**Windows PowerShell**：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
```

如果 PowerShell 禁止执行激活脚本，可以不用激活，改用 `\.venv\Scripts\python.exe -m pip ...` 等命令直接执行。

**Linux / macOS（含 V100 服务器）**：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
```

预期：pytest 全通过，preflight JSON 显示 `"status": "PASS"` 和 `"fine_queries_observed": [20, 40, 80]`。preflight 用的是 **1000 个合成样本**，只是数据流/预算验证，不是模型训练，更不是 CIFAR-100 准确率。

## 下载与核验公开数据（可等 P1 再执行）

```bash
python scripts/download_cifar100.py --root data/cifar100
python -m fed_finelabel_budget.cli data-check --root data/cifar100
```

仅从官方 `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` 下载，脚本会验证 torchvision 使用的官方 MD5。实际训练集 50,000 张、测试集 10,000 张。数据、模型权重、训练日志和运行结果被 `.gitignore` 排除；ZIP 内没有数据集。**仅对可信任来源的官方 pickle 文件执行读取**。

## 项目结构

```text
fed-finelabel-budget/
├── AGENTS.md                         # Codex 约束：科研边界、完成准则
├── README.md                         # 当前说明与运行命令
├── configs/p0.yaml                   # 统一 seeds/客户端/预算配置
├── docs/
│   ├── research_plan.md              # 本项目既有完整选题审计与任务书
│   ├── DATA_CONTRACT.md              # 不泄漏协议及其边界
│   ├── CODEX_NEXT_TASK.md            # 可直接复制的下一步任务
│   └── P0_REPORT.md                  # 打包前的真实测试记录
├── src/fed_finelabel_budget/
│   ├── cli.py                        # preflight / data-check
│   └── data/
│       ├── cifar_io.py               # 可信引导进程读取真实 fine 标签
│       ├── hierarchy.py              # 正确构建官方 20×5 层次映射
│       ├── public_pool.py            # 不包含 fine 标签的公开数据对象
│       ├── label_oracle.py           # 按预算原子查询细标签
│       ├── partition.py              # IID / coarse-only Dirichlet
│       └── budget.py                 # 2% → 4% → 8% 计数与初始随机选择
├── scripts/download_cifar100.py      # 用户显式下载官方数据
├── tests/                             # pytest 单元测试
├── results/                           # 不提交结果/模型文件
└── reports/                           # 不提交未经实际训练的数字
```

## 与原研究任务书保持一致的关键事项

- CIFAR-100 官方 20 个 coarse × 每组 5 个 fine 类别；**不能假定 fine 编号每 5 个连续归一组**，必须从官方配对推导。
- 训练/查询只知道 public images + coarse labels + 已购买 fine labels。
- 非 IID 仅根据 coarse labels 划分，绝不能参考还没购买的 fine labels。
- 初始预算 2%（已计费），后续追加 2% 与 4%，所以累计 2% → 4% → 8%。
- 不允许训练算法接触 oracle 本体、原始完整标签字典或测试/验证集真值。
- 在 P1 设计训练时：为已购买 fine 的样本使用 fine loss；未购买样本使用 coarse loss，不能默认重复监督。
- 原论文基线与本设置可能不完全一致；先核查定义再做对比。

## Codex 如何开始

1. 下载并解压本 ZIP，在 Codex 里 **打开解压后的 `fed-finelabel-budget` 文件夹**（不是 ZIP 文件）。
2. 不要让 Codex 再初始化一次新项目，不要复制到 GDAP / CLE / RAHFL 的仓库。
3. 发送 `docs/CODEX_NEXT_TASK.md` 的内容给 Codex，或让它直接读取该文件并执行。
4. Codex 完成后，把它写出的 `reports/P1_REPORT.md` 或终端执行记录发回 ChatGPT 继续审查。

## P0 / P1 范围

| 阶段 | 实际交付状态 |
|---|---|
| P0-Setup：工程、层级读取、分区、Oracle、预算、单测、preflight | **已包含（本地经过测试）** |
| P1：ResNet18、层级损失、FedAvg、数据集适配、CPU smoke train | **已实现，执行证据见 P1_REPORT** |
| P2：B0–B4 公平对照策略与 V100 pilot | 未实现 |
| P3：方法可行性判断与创新机制 | 未开始 |

注意：Python 层的隐藏标签不是强安全隔离；信任边界的具体描述见 `docs/DATA_CONTRACT.md`。

## P1 安装与最小训练

使用 Python 3.10+，安装 `python -m pip install -e ".[dev,train]"`。本机默认 Python 3.8 不可用，已使用 `D:/anaconda3/python.exe` (3.12.7) 创建 `.venv-p1`。在项目根目录 PowerShell 执行：

```powershell
# 此项目已安装依赖的本机环境
$p1TestTemp = Join-Path (Get-Location) ('results/pytest-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p1TestTemp
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --config configs/p1_smoke.yaml
```

使用新的 `--basetemp` 路径避开本机沙箱临时目录权限错误，避免复用或清理已有目录。环境、缓存、测试临时文件保留且不入 Git。

其他兼容环境中直接用 `python` 运行以上命令即可。配置分别记录 data / partition / init / client_sampling / query / training 六个种子。合成默认 150 样本、3 客户端、1 轮，每客户端至多 1 步；初购 2% = 3 个 fine，后续预算仅记录，P1 不追加查询。`--device cpu|cuda` 覆盖设备；本次仅验证 CPU。

输出 `results/p1_smoke/run.json`（配置、版本、Git commit 与 dirty 标记、实际预算/计算成本）、`rounds.jsonl`（本地损失与聚合权重）、`checkpoint.pt`（模型状态及配置；保存后回读逐张量验证）。checkpoint 不保存 oracle、完整 fine 真值或图像；目前供权重恢复检查使用，没有实现精确断点续训。

FedAvg 对所选客户端按完整本地训练集样本数加权，浮点参数与 BatchNorm 浮点缓冲均加权；`num_batches_tracked` 取各客户端最大值。每次本地训练重新初始化 SGD，固定轮数，没有早停。已购样本只用 fine CE，未知 fine=-1 的样本只用 logsumexp coarse loss，损失按批样本均值归一化。图像固定映射到 [-1,1]；P1 无增强和验证集调参。

如需在官方数据上做受限 smoke，先运行（本次未下载）：

```powershell
& ./.venv-p1/Scripts/python.exe scripts/download_cifar100.py --root data/cifar100
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli data-check --root data/cifar100
& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --mode cifar100 --device cpu --output results/p1_cifar_smoke
```

真实数据 smoke 由可信控制器从训练集按公共 ID 抽 150 样本，预算分母为该子集；层级由完整官方训练集配对推导，购买请求保留原始 ID，再映射为子集 ID。它不读取测试集，也不报告测试准确率。正式 50k 的 P2 预算仍是 1000 → 2000 → 4000，不能引用 smoke 子集预算作为正式实验。
