# P2 真实数据验证与基础标注策略实测报告

日期：2026-10-11。P2 已完成，停止于 P2。没有启动多种子或 V100 实验，没有设计 P3 算法。

## 环境和数据证据

本机 Python 3.12.7，torch 2.14.1+cpu，NumPy 2.5.4，Windows CPU；CUDA 不可用。沿用 P1 的 .venv-p1，无新增依赖。真实 Pilot 源码提交 `3ca840b642dd1aabe140a4d19f91110b640d4c61`，运行时 git_dirty=False。P1 基线提交为 a9b9cf3；最终报告另作提交，源码运行版本仍由 run.json 明确记录。

官方 URL https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 实际重定向到 https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz。下载文件 169001437 bytes，官方 MD5 `eb9058c3a382ffc7106e4002c42a8d85` 匹配，已解包并真实读取 50000 train / 10000 test。层级由官方 fine/coarse 配对推导，20×5，测试配对一致，确认不是每5个连续fine ID一组。完整映射保存在 results/p2_data_validation.json 和 Pilot manifest.json。

首次 urllib 下载无进展，中止后产生不完整压缩包；curl 单连接较慢，改为官方主机 Range 分段下载，部分连接遇到 SSL/超时后补齐剩余字节。16 段长度和合并 MD5 已校验。操作脚本 results/download_ranges.py 和分段文件均保留在 Git 忽略目录，没有批量删除。修复原下载器直接写正式包的缺陷：新版本使用 .download 暂存，30秒网络超时，校验成功后原子替换；两项失败复现测试确保中断/损坏不会发布正式包。

实际全训练集分区诊断：

| 设置 | 10 客户端规模 | exactly-once / 重放 |
|---|---|---|
| iid | [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000] | PASS / PASS |
| dirichlet_0.3 | [4904, 6210, 6849, 3415, 6696, 5988, 2321, 6930, 1955, 4732] | PASS / PASS |
| dirichlet_1.0 | [4955, 5715, 3746, 4493, 5763, 4108, 6401, 5887, 3629, 5303] | PASS / PASS |

真实图像诊断：通过Oracle购买ID 0..7，ID 8..15保持fine=-1；fine CE均值=4.73886490，coarse marginal均值=2.93429375，loss和全部梯度有限。真值没有直接进入训练器。

真实数据 P1 smoke：150个公共ID随机子集、3客户端、1轮、3梯度步，购买3 fine，其余用coarse。更新 62 个参数张量，checkpoint回读一致，训练函数实测 5.4926 秒。没有读验证/测试集。

## 五种基线和接口

| 基线 | 训练监督 | 新增查询 |
|---|---|---|
| B0 | 只从已购fine数据集采样，只计算fine CE | 本地随机 |
| B1 | 已购fine仅CE，未知fine仅coarse marginal | 本地随机，与B0配对 |
| B2 | 同B1 | 当前公开coarse五个logit的条件熵，float64 log-softmax |
| B3 | 同B1 | 按已购样本的公开coarse覆盖数水位填充，组内随机；不统计隐藏fine分布 |
| B4 | 同B1 | 条件熵 top-5k shortlist，再在归一化avgpool特征上farthest-first |

新增配额统一按客户端公开剩余样本数做largest remainder分配；这些是简单本地基线，没有引入新的跨客户端协调算法。B4保留一维最小距离数组，内存O(ND)，不创建NxN距离矩阵，计算O(kMD)；特征只存在本地调用。查询只返回样本ID。

ClientPublicView只物化自己客户端公共样本；训练只接收ClientDataset已购快照。SimulationData/Oracle只在可信控制器，训练/查询无cifar_io、oracle或评估模块导入。条件熵不访问hidden truth；eval返回指标而不返回标签/预测数组。Python逻辑隔离仍不防恶意反射或原始文件读取。

## 实际 Pilot 配置与监督/预算协议

配置 configs/p2_cpu_pilot.yaml，10客户端，coarse Dirichlet alpha=.3、min_client_samples=50；split=20261011，partition=101，init=102，client_sampling=103，query=104，training=105。先按公共ID从官方train留出5000验证集，从剩余45000中固定选1000联邦训练样本。验证评估固定100张，test固定100张仅各基线最后计算；没有根据这些指标调参/早停。

B0–B4初始2%使用完全相同20个标注ID；累计20→40→80。各基线独立Oracle，每次新增请求必须匹配当前阶段差额且不得包含已购/重复ID，成功才解锁。每阶段账本记录新增ID、数量、客户端累计预算和公开coarse覆盖数。

阶段间统一延续权重；每阶段1通信轮，10客户端全参与，每客户端1梯度步、batch=4、SGD lr=.01、无增强。全部方法有放回固定步数采样，保证每方法每阶段10步/40 examples_seen，总计30步/120 examples_seen，五方法共150步/600 examples_seen。B0只重采样已购fine，其他基线从公共池采样。所有方法按公共客户端规模聚合，BN浮点状态加权、整数计数取max。

完整45k联邦池的2%/4%/8%应为900/1800/3600，不能引用包含验证集的50k分母1000/2000/4000。当前1000样本Pilot只对应20/40/80。所有方法使用同一预算分母、分区、初始化、参与序列和计算配置。

## 验证指标与训练成本（真实执行）

以下是100张验证子集的工程诊断，macro-F1始终对全部100类取平均、缺失类为0。指标不是论文最终性能，不能用来比较算法增益。

| 方法 | 累计预算 | 通信轮 | train loss | Val Acc % | Val Macro-F1 % | 步数 / examples_seen | 训练秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 20 (2%) | 1 | 4.777404 | 0.0000 | 0.0000 | 10 / 40 | 2.5384 |
| B0 | 40 (4%) | 2 | 4.866598 | 0.0000 | 0.0000 | 10 / 40 | 2.4767 |
| B0 | 80 (8%) | 3 | 4.729615 | 0.0000 | 0.0000 | 10 / 40 | 2.4752 |
| B1 | 20 (2%) | 1 | 3.004403 | 0.0000 | 0.0000 | 10 / 40 | 2.5005 |
| B1 | 40 (4%) | 2 | 3.074141 | 0.0000 | 0.0000 | 10 / 40 | 2.4519 |
| B1 | 80 (8%) | 3 | 3.106482 | 0.0000 | 0.0000 | 10 / 40 | 2.5086 |
| B2 | 20 (2%) | 1 | 3.004403 | 0.0000 | 0.0000 | 10 / 40 | 2.4759 |
| B2 | 40 (4%) | 2 | 3.114306 | 0.0000 | 0.0000 | 10 / 40 | 2.5329 |
| B2 | 80 (8%) | 3 | 3.068377 | 0.0000 | 0.0000 | 10 / 40 | 2.4816 |
| B3 | 20 (2%) | 1 | 3.004403 | 0.0000 | 0.0000 | 10 / 40 | 2.4859 |
| B3 | 40 (4%) | 2 | 3.074141 | 0.0000 | 0.0000 | 10 / 40 | 2.5088 |
| B3 | 80 (8%) | 3 | 3.142257 | 0.0000 | 0.0000 | 10 / 40 | 2.5152 |
| B4 | 20 (2%) | 1 | 3.004403 | 0.0000 | 0.0000 | 10 / 40 | 2.4921 |
| B4 | 40 (4%) | 2 | 3.114306 | 0.0000 | 0.0000 | 10 / 40 | 2.5395 |
| B4 | 80 (8%) | 3 | 3.068377 | 0.0000 | 0.0000 | 10 / 40 | 2.5242 |

客户端累计已购数量（c0..c9；每行求和等于累计预算）：

| 方法 | 预算 | 客户端 fine 标签数量 |
|---|---:|---|
| B0 | 20 | [1, 2, 3, 2, 2, 4, 1, 1, 2, 2] |
| B0 | 40 | [2, 4, 6, 4, 4, 8, 3, 2, 3, 4] |
| B0 | 80 | [4, 8, 12, 8, 8, 15, 6, 5, 6, 8] |
| B1 | 20 | [1, 2, 3, 2, 2, 4, 1, 1, 2, 2] |
| B1 | 40 | [2, 4, 6, 4, 4, 8, 3, 2, 3, 4] |
| B1 | 80 | [4, 8, 12, 8, 8, 15, 6, 5, 6, 8] |
| B2 | 20 | [1, 2, 3, 2, 2, 4, 1, 1, 2, 2] |
| B2 | 40 | [2, 4, 6, 4, 4, 8, 3, 2, 3, 4] |
| B2 | 80 | [4, 8, 12, 8, 8, 15, 6, 5, 6, 8] |
| B3 | 20 | [1, 2, 3, 2, 2, 4, 1, 1, 2, 2] |
| B3 | 40 | [2, 4, 6, 4, 4, 8, 3, 2, 3, 4] |
| B3 | 80 | [4, 8, 12, 8, 8, 15, 6, 5, 6, 8] |
| B4 | 20 | [1, 2, 3, 2, 2, 4, 1, 1, 2, 2] |
| B4 | 40 | [2, 4, 6, 4, 4, 8, 3, 2, 3, 4] |
| B4 | 80 | [4, 8, 12, 8, 8, 15, 6, 5, 6, 8] |

最终8%后一次性测试指标（100张官方test ID子集；不用于任何选择或调参）：

| 方法 | Test Acc % | Test Macro-F1 % | 预测类别数 | 最大预测类别占比 |
|---|---:|---:|---:|---:|
| B0 | 1.0000 | 0.0345 | 5 | 57% |
| B1 | 1.0000 | 0.0230 | 3 | 86% |
| B2 | 1.0000 | 0.0222 | 3 | 89% |
| B3 | 1.0000 | 0.0220 | 3 | 90% |
| B4 | 1.0000 | 0.0222 | 3 | 89% |

最终8%验证集各coarse群的fine Top-1 Accuracy（百分数；—表示该100张验证子集中无此群）：

| coarse | B0 | B1 | B2 | B3 | B4 |
|---|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 14 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 15 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 18 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

## 训练异常与公平性边界

所有执行轮均有参数更新，loss、梯度、模型state和评估logit有限；全部基线checkpoint逐张量回读一致，没有NaN/Inf或未更新故障。预测分布诊断如下：

| 方法 | 最终Val预测类别数 | 最大类别占比 | 实际fine/coarse监督使用次数（所有轮，允许重复） | 总查询秒 |
|---|---:|---:|---|---:|
| B0 | 3 | 67% | 120 / 0 | 0.0088 |
| B1 | 3 | 96% | 3 / 117 | 0.0090 |
| B2 | 3 | 96% | 4 / 116 | 18.7845 |
| B3 | 3 | 96% | 4 / 116 | 0.0123 |
| B4 | 3 | 96% | 4 / 116 | 18.5569 |

本次B1–B4最终验证预测的最大类占比96%，存在明显的短程预测集中现象，不能声称模型已正常收敛。每客户端只有3个更新、验证/测试子集很小，缺失类别会压低macro-F1；本轮不做收敛/优势判断，也未根据test指标修改配置。未发现训练/评估转换或标签映射错误；正式多种子前需在V100单种子较长诊断中确认预测集中是否消退。

B2/B4尽管本轮loss和指标相同，新增查询集合不同：从2%到4%的集合对称差为14个ID，从4%到8%为48个ID。不能把这个极小训练预算下的指标相同解释为两种策略等价。

可比性因素：固定梯度步和batch控制计算，但B0重复使用少量fine样本，其他基线多数更新来自coarse，独立样本覆盖/有效监督本身不同。公共规模加权是显式共同协议，与按fine-only子集规模加权的另一版本不同，正式论文应说明。group-balanced只平衡本地已购coarse样本覆盖，不保证fine平衡。B2/B4多出模型推理/特征距离成本，查询耗时已记录，不能声称所有方法总墙钟成本相同。Non-IID最小规模约束会条件化分区；未读fine来满足该约束。BN聚合、无增强、固定lr、采样有放回等设置需保持配对。

## 测试、恢复与真实命令

P2开始前原P0/P1测试28 passed in 12.60s；新增最小query/eval测试12 passed in 6.91s；控制器测试4 passed in 8.78s；下载错误复现2 passed in 0.07s。完成新目录恢复修复后，最小恢复回归1 passed in 7.40s；最终完整回归48 passed in 14.15s，退出码0，原测试断言未重写；P0 preflight继续PASS，[20,40,80]，剩余0。

测试覆盖五策略接口、严格相同累计数量、无重复/已购查询、隐藏fine变更不影响选择、共享2%初购、条件熵极端数值、B0/B1区别、public客户端隔离、val/test排除、100类macro-F1和空coarse群、固定步数、阶段/轮边界checkpoint恢复及协议不匹配拒绝。恢复测试在轻量模型上对比完整运行与实际保存/加载后运行，最终权重逐张量完全相同，查询账本和指标相同；ResNet真实checkpoint恢复/回读另在smoke与实际Pilot验证。

checkpoint在每轮结束时原子替换，保存模型、已购快照、initial IDs、阶段查询账本、next_round、配置和协议hash；恢复仅在可信控制器重建Oracle并回放已购ID，预算等于已购数量。每轮optimizer重建；随机数由绝对轮次/阶段种子重建。支持轮边界，不支持本地batch中途恢复，不保证跨PyTorch/设备版本位级一致。

以下命令从工程根目录实际执行成功，退出码均为0：

```powershell
# 官方分段下载操作脚本；之前失败/中止命令见环境段
& ./.venv-p1/Scripts/python.exe -u results/download_ranges.py
& ./.venv-p1/Scripts/python.exe scripts/download_cifar100.py --root data/cifar100
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli data-check --root data/cifar100
& ./.venv-p1/Scripts/python.exe scripts/validate_p2_data.py
& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --mode cifar100 --device cpu --output results/p1_real_smoke
& ./.venv-p1/Scripts/python.exe scripts/run_pilot.py --config configs/p2_cpu_pilot.yaml
& ./.venv-p1/Scripts/python.exe scripts/run_pilot.py --config configs/p2_cpu_pilot.yaml --baseline B2 --resume results/p2_cpu_pilot/B2/checkpoint.pt --output results/p2_resumed
$p2TestTemp = Join-Path (Get-Location) ('results/pytest-p2-final-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p2TestTemp
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli preflight --config configs/p0.yaml
git diff --check
```

运行文件：results/p2_cpu_pilot/run.json、manifest.json、metrics.csv、各B0..B4/rounds.jsonl、summary.json、checkpoint.pt；数据诊断results/p2_data_validation.json；真实smoke results/p1_real_smoke/；完整测试日志results/p2_pytest.txt。数据/模型/运行日志不入Git，报告表格由这些实测JSON生成。

## 文件变更和V100准备

新增 src/fed_finelabel_budget/p2.py、query/{__init__,strategies}.py、eval/{__init__,metrics,offline}.py、training/baselines.py；configs/p2_cpu_pilot.yaml、configs/p2_v100.yaml；scripts/run_pilot.py、run_p2_v100.sh、validate_p2_data.py；tests/test_p2_query.py、test_p2_eval_training.py、test_p2_controller.py、test_download.py；reports/P2_REPORT.md；.gitattributes确保sh脚本检出为LF。修改 data/cifar_io.py（可信split与模拟器）、data/public_pool.py（客户端视图）、training/fedavg.py（固定步数/绝对轮次/公共权重/梯度有限检查）、training/seeds.py（CUDA确定性env）、scripts/download_cifar100.py（原子下载）、README、DATA_CONTRACT、.gitignore。P0/P1入口继续可用。

具备开展V100单种子验证和后续正式配对多种子实验的代码基础；尚未在V100实测CUDA/Volta兼容性或完整45k训练收敛，因此正式多种子前应先做小规模CUDA smoke。configs/p2_v100.yaml是未执行、未基于test调参的模板：45k/5k、10客户端、alpha=.3、每阶段20轮/每客户端20步/batch64、全部5方法，同预算900/1800/3600。脚本只跑配置的一组种子，不自动发起多种子。

后续由用户显式启动：

```bash
python -m pip install -e '.[dev,train]'
bash scripts/run_p2_v100.sh
# 指定单个基线恢复
bash scripts/run_p2_v100.sh --baseline B2 --resume results/p2_v100_seed101/B2/checkpoint.pt
```

部署环境需选择支持V100的CUDA/PyTorch构建并先验证实际kernel；当前CPU轮子不能直接当GPU环境。单组Pilot和验证子集不足以判断研究收益、创新性或选题可行性，不进行P3研发。

恢复修复记录：代码审查发现已完成checkpoint迁移到新输出目录时缺少目标checkpoint；补上原子保存和输出日志，并在原断点恢复测试中增加完成状态的回归。真实B2完成状态也执行了跨目录恢复；保存模型和评估指标一致，没有新增梯度步或查询预算。

实测账本审计结果见results/p2_audit.json：五方法共享初购，分区exactly-once、验证ID隔离、累计预算一致、没有重复查询，每方法30步/120 examples_seen；真实ResNet完成状态跨目录恢复的权重与指标完全一致。审计脚本results/audit_p2.py真实执行退出码0。
