# Fed Fine-Label Budget — 第二篇论文初始工程

> **当前阶段 P2：真实数据验证与 B0–B4 基线已实现。**
> CPU Pilot 只验证训练与查询流程，不构成论文性能结论。执行证据见 `reports/P2_REPORT.md`；P1 历史证据仍见 `reports/P1_REPORT.md`。

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
| P2：真实数据验证、B0–B4、独立评估、预算递增与恢复 | **已实现；V100 尚未执行** |
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

真实数据 smoke 由可信控制器从训练集按公共 ID 抽 150 样本，预算分母为该子集；层级由完整官方训练集配对推导，购买请求保留原始 ID，再映射为子集 ID。它不读取测试集，也不报告测试准确率。若联邦训练池是全部 50k，预算为 1000 → 2000 → 4000；当前 P2 独立留出验证集后的分母见下文，不能混用 smoke 子集预算。

## P2 运行和恢复

```powershell
& ./.venv-p1/Scripts/python.exe scripts/download_cifar100.py --root data/cifar100
& ./.venv-p1/Scripts/python.exe scripts/validate_p2_data.py
& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --mode cifar100 --output results/p1_real_smoke
& ./.venv-p1/Scripts/python.exe scripts/run_pilot.py --config configs/p2_cpu_pilot.yaml
# 只恢复指定基线；checkpoint 在每轮结束时原子保存
& ./.venv-p1/Scripts/python.exe scripts/run_pilot.py --config configs/p2_cpu_pilot.yaml --baseline B2 --resume results/p2_cpu_pilot/B2/checkpoint.pt --output results/p2_resumed
```

CPU 默认 1000 个真实训练样本、10 客户端、coarse Dirichlet alpha=.3、单组种子、三个预算阶段各 1 轮、每客户端 1 步、batch=4。独立验证集从官方训练集按公共 ID 划 5000 张，Pilot 固定使用其中 100 张评估；测试集固定抽 100 张，仅每个基线最终 8% 阶段离线评估。CPU 入口拒绝完整数据/大训练预算，CUDA 配置不会静默回退到 CPU。

初购集合完全共享，之后所有策略按各客户端公开剩余样本数分配相同规则的配额，具体 ID 留在客户端选择。B0/B1 配对随机；B2 coarse 组内条件熵；B3 基于已购样本公开 coarse 覆盖数进行水位填充并随机选组内样本；B4 先取熵 top-5k 候选，再对归一化本地 ResNet 特征做 farthest-first。B4 无 NxN 距离矩阵，特征不上传；均是简单基线，不声称新的协调机制。

各阶段延续模型，无早停和学习率搜索。每轮固定 `max_steps`，每客户端用有放回采样生成 `batch_size * max_steps` 个样本，B0 仅从已购 fine 集采样，其余从完整公共本地池采样。所有方法聚合权重统一按本地公共样本数；若 B0 某客户端完全没有已购 fine，入口报错以防默默改变参与或计算预算，建议可行的 `min_client_samples>=50`。日志同时记录训练数据规模、重复使用计数、fine/coarse 监督样本使用数、时间与查询分配。

`metrics.csv` 是逐轮**验证指标**，macro-F1 固定对 100 类平均，缺失类记 0。各 coarse 群的 fine accuracy、预测类别数/最大预测占比、最终 test 指标见基线 summary.json。每轮 checkpoint 含模型、已购快照、查询账本、轮次、配置与协议哈希；恢复重新构建同一分区/初购和 oracle，仅回放已购 ID。随机过程按绝对轮次/阶段重建，optimizer 每轮重置，可在轮边界一致恢复；不支持本地 batch 中途恢复。相同环境之外不保证位级一致。

V100 使用兼容 CUDA 的 PyTorch/torchvision 后，由用户显式执行：

```bash
python -m pip install -e '.[dev,train]'
bash scripts/run_p2_v100.sh
# 或指定基线及已保存的轮边界 checkpoint
bash scripts/run_p2_v100.sh --baseline B2 --resume results/p2_v100_seed101/B2/checkpoint.pt
```

V100 配置是待执行的单组种子模板，脚本不会循环多种子。配置固定 20 轮/阶段、20 步/客户端、batch=64，没有测试集调参。每次单独输出目录以保留结果；种子变更应明确记入配置。正式比较前先在 V100 跑小规模 CUDA smoke。

**预算分母补充：** 使用独立 5k 验证集后，完整联邦训练池为 45k，2%/4%/8% 对应 **900/1800/3600**；1000 样本 CPU Pilot 为 **20/40/80**。上文 P0 的 50k→1000/2000/4000 仅对应无 held-out 验证集的分母，不能混用。所有 P2 方法共享同一分母。

## P2.5 学习能力诊断与单种子准备

真实执行结果见 `reports/P2_5_REPORT.md`。诊断 Oracle 明确授权独立训练子集的全细标签；验证来自同一固定 5k held-out，官方测试集不读取。全细标签诊断不属于预算基线，权重标记 `diagnostic_full_fine`，P2 恢复入口拒绝加载。未标注样本训练和查询隔离协议不变。

```powershell
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_diagnostic.yaml --only audit
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_diagnostic.yaml --only overfit
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_diagnostic.yaml --only central
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_diagnostic.yaml --only fedavg
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_diagnostic.yaml --only bn
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p25 --config configs/p25_cpu_stable_diagnostic.yaml --only central
# GPU 诊断需要单独安装兼容 CUDA 的 PyTorch，再使用 p25_local_diagnostic.yaml。
```

CPU 配置仅允许 128 张过拟合、256 张中央监督、128 张双客户端诊断，分别最多 40 epochs / 10 epochs / 10 rounds。过拟合只根据已授权训练准确率 >=95% 提前结束；其余按固定轮数运行，无验证早停，无强增强。训练模块仅接收已授权 ClientDataset，不接收 Oracle 或离线 evaluator。

原始 `sample_mean` 损失和 SGD momentum=0 仍是默认值。可配置 `loss.aggregation` 为 `sample_mean` 或 `group_mean`，以及非负 `fine_weight/coarse_weight`。前者是加权单样本损失总和除以 batch 样本数；后者是两个组各自均值的加权和，无隐式权重归一化，缺失组记零。`reduction=none` 始终返回未加权逐样本项供审计。本阶段只比较梯度，没有用 group_mean 替换已完成的基线。

V100 上显式执行以下命令（本机未执行 GPU 实验）：

```bash
bash scripts/setup_p25_v100.sh
bash scripts/run_p25_v100.sh
# 断点恢复示例：激活相同环境后，只恢复指定方法
python -m fed_finelabel_budget.p2 --config configs/p25_v100_b0_b1.yaml --baseline B0 --output results/p25_v100_B0 --resume results/p25_v100_B0/B0/checkpoint.pt
```

新 V100 模板只含 B0、B1，固定单组种子，公共 coarse Dirichlet alpha=.3，45k 训练 / 5k 验证，官方 test_samples=0。预算仍为900/1800/3600；每阶段30轮、每轮10客户端、每客户端50步、batch64；两种方法共同 SGD lr=.02 / momentum=.9（每轮重置），原始 sample_mean、两项权重1。lr=.02 来自本阶段独立全细标签诊断的早期数值稳定性控制，并已完整保留 lr=.05 的原始诊断；不是根据官方测试集选择。参数仍是待验证的预实验配置，与 P2 极短 CPU 配置有明确差异。B2–B4 和多种子没有启动。逐轮验证 CE、预测熵、直方图随日志保存；样本与 fine/coarse 使用数见 rounds.jsonl。BN 重校准只在独立反事实诊断中执行，未加入基线训练。
