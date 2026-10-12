# P2.6：V100 基线验证与实验协议校准

日期：2026-10-12。起点commit：`aca7570`。本阶段不设计新方法，不进入P3。

## 1. 执行状态：必须区分的四类证据

| 类型 | 真实状态 |
|---|---|
| 本阶段实际GPU训练/真实V100 baseline | **无**；CUDA preflight和前5轮GPU Pilot没有执行 |
| 本阶段已执行 | 真实CIFAR-100公共ID采样审计；带真实forward/backward的合成小模型单元测试；全82项回归；脚本语法/CLI检查 |
| 已有真实训练有效性证据 | P2.5独立CPU诊断：128张过拟合95.31%；中央控制80.86%；全fine双客户端FedAvg47.66%，均为训练准确率。详见P2_5_REPORT，不是GPU baseline对比 |
| 已准备但未执行 | V100 sample_mean原对比与paired控制、全5k验证监控、2%前5轮和后续分段恢复、独立group_mean诊断配置 |

本机实际GPU为RTX3050 Laptop / 4GB，不是V100；PyTorch2.14.1+cpu，CUDA unavailable。任务没有提供可用V100服务器连接。SSH配置存在性检查被沙箱拒绝，不能据此断言用户没有远程服务器；本阶段没有猜测连接或请求读取私密配置。采用用户明确允许的交付分阶段Linux指令方案，**没有以CPU指标冒充GPU结果**。

本次回归环境：`{"python": "3.12.7", "torch": "2.14.1+cpu", "torchvision": "0.29.1+cpu", "numpy": "2.5.4", "pytest": "9.1.1", "platform": "Windows-11-10.0.26200-SP0", "hardware": "Intel64 Family 6 Model 141 Stepping 1, GenuineIntel", "cuda_available": false}`。运行元数据记录起点commit和dirty状态，报告不会把后续提交伪装为已经执行训练时的commit。

## 2. B0/B1原对比公平性审计

原`configs/p25_v100_b0_b1.yaml`完全保留，Git blob检查一致：`b7bc2a8252f0d1e8c4c854238466ab0a2ea02dbd`。原`p2.py`、`training/fedavg.py`、查询策略和P2/P2.5结果均未覆盖。P2.6通过独立入口新增监控和控制，原入口仍可复现原实现。

### 2.1 输入与fine监督不同

- B0：本地训练dataset只含已购fine样本；有放回RandomSampler只在这个小集合采样。全部batch样本计算Fine CE。
- B1：dataset包含完整本地公共池；同样有放回采样，但已购fine仅占2%/4%/8%。已购用fine CE，其余用coarse marginal；sample_mean除以完整batch样本数。
- 共享client种子和随机数并不意味着同一输入序列：两个dataset长度与索引到公共ID的映射不同。
- 同预算、同scheduled batches和同输入暴露次数，不等于同fine暴露、同非零梯度、同优化效果或同墙钟成本。原B0/B1是“实际预算方法”对比，不能仅用其差值归因于coarse loss。

### 2.2 真实数据ID-only采样审计（没有训练模型）

实际读取官方train，通过seed20261011按公共ID分出45k训练/完整5k验证；coarse Dirichlet alpha=.3、10客户端、min500、seed101。Oracle按900→1800→3600真实解锁快照；初购seed104，追加查询沿用B0/B1配对Random。审计只利用public ID及是否已购买，不使用隐藏fine类别分布。

下表是用当前Torch的**实际RandomSampler复放规定序列**得到的数量，覆盖三个各30轮阶段；没有模型forward、没有执行90轮训练、没有GPU Accuracy/耗时结果。

| 阶段 / 累计已购 | 方法 | 输入暴露次数 | fine暴露次数 | 不同输入数 | fine重复暴露次数 | 含fine的batch / scheduled batch |
|---|---|---:|---:|---:|---:|---:|
| 1 / 900 | B0 | 960,000 | 960,000 | 900 | 959,100 | 15,000 / 15,000 |
| 1 / 900 | B1 | 960,000 | 19,159 | 45,000 | 18,259 | 10,883 / 15,000 |
| 2 / 1800 | B0 | 960,000 | 960,000 | 1,800 | 958,200 | 15,000 / 15,000 |
| 2 / 1800 | B1 | 960,000 | 38,423 | 45,000 | 36,623 | 13,922 / 15,000 |
| 3 / 3600 | B0 | 960,000 | 960,000 | 3,600 | 956,400 | 15,000 / 15,000 |
| 3 / 3600 | B1 | 960,000 | 76,652 | 45,000 | 73,052 | 14,915 / 15,000 |

2%前5轮：两种方法各160,000次输入暴露，B0 fine=`160000`次，B1 fine=`3215`次。原P2极短真实CPU Pilot的120次fine（B0）与3次fine（B1）也与这一采样差异相符，但不重新当作P2.6 GPU性能证据。

2%阶段B0平均每张已购图像重复使用960000/900≈1066.67次；B1约19159/900≈21.29次。B0的反复过采样是原协议的真实组成，不是预算重复计费；Oracle仍只购买900个不同标签。B1既有fine出现少的问题，又按N归一化，fine项总体约占2%，不能把更多输入图像等同更多fine训练信号。

`fine_positive_batches`只表示ID序列中有fine样本，**不是实际测量的非零梯度次数**。模型可能产生零梯度、动量仍可更新参数；真实GPU运行才会产生相应统计。这里的数值来自CPU版Torch2.14.1，Linux模板锁定Torch2.5.1，跨版本具体RandomSampler序列可能不同；服务器应以实际日志为准，并依靠paired流哈希而非盲用本表作为期望值。

### 2.3 BN、优化器和成本混杂

B0的BN仅见重复的已购小池，B1的BN见完整池；即使running statistics聚合规则相同，其输入分布不同。两者SGD都在每客户端每轮重置momentum=.9状态，局部动量因各自损失和数据而不同。原P2.5发现早期BN推理统计不匹配风险，没有在本阶段给任何方法偷偷重校准BN。

原对比每轮10客户端×50批×64输入，500个本地scheduled batch/32000输入；每阶段15000批/960000输入。原SGD在有监督的batch每次step，包括零梯度时可能发生的动量更新。额外coarse计算、数据构造、监控/显存开销都可能影响实际成本；本阶段**没有真实V100成本测量，没有声称同FLOPs或同耗时**。

## 3. 新paired-stream控制：PS_FINE / PS_COARSE

这是一组独立工程因果控制，不替代B0/B1及B2–B4的预算策略排名。

| 项目 | PS_FINE | PS_COARSE |
|---|---|---|
| 初始模型、client选择、已购标签 | 同一init102、相同seed和购买账本 | 同左 |
| 输入流 | 完整公共本地池的有放回序列 | 完全相同序列 |
| 已购fine项 | Fine CE求和/N | 同一Fine CE求和/N |
| 未购项 | 不提供监督loss | coarse marginal求和/N |
| BN前向规则 | 全部采样图像进入train-mode BN | 同左 |
| 没有任何监督的batch | 前向更新BN；不backward、不optimizer.step | 同一规则（一般存在coarse监督） |
| 有监督但梯度为零 | 通常SGD step保留，包括动量行为 | 同左 |

未知fine仍为−1；训练只接收ClientDataset的已购快照与public taxonomy，不能接收Oracle、原始reader、验证/测试真值。损失矩阵可以统一校验公开coarse映射，但PS_FINE没有把未知样本coarse项加入loss或梯度。

BN规则相同不意味着最终BN缓冲逐数相同：模型权重在添加coarse梯度后会不同，这是处理效应的一部分。train-mode BN还会使fine CE通过batch统计依赖同batch未知图像；两组采用相同输入和BN规则来控制这一点，没有冻结BN或私下采用不同BN处理。

输入流由training_seed + absolute_round×1000003 + cid确定，与模型预测无关。每客户端每轮保存完整输入序列SHA-256、fine ID暴露Counter哈希、fine使用数及不同fine数；两组共享同一初购和后续Random账本。跨组核查还比较初始化hash、已购快照hash、配置/协议/源码hash、Torch、硬件、lr/momentum和BN/optimizer规则。

全未购batch的控制组必须跳过step，否则此前fine梯度形成的动量会在没有监督的batch继续推动参数。这一规则按“有没有监督”判断，不按loss或梯度大小私下选择样本。原B0/B1均一直有监督，P2.6监控入口保留其正常SGD行为。测试分别检查optimizer.step、非零梯度和参数真的变化，三个计数不会混称一个数字。

**尚未实际测量coarse贡献**。paired fine暴露相同，但处理组存在更多可监督batch，backward次数和成本通常较大。例如本地ID审计2%阶段仅10883/15000个batch含fine，PS_FINE预计只在这些batch有监督，PS_COARSE通常15000个batch都有监督；真实非零梯度次数需实际运行测量。不能声称它同时匹配了有效更新次数或计算成本。

## 4. P2.6专属分阶段入口

`configs/p26_v100.yaml`沿用原模型、45k/5k split、coarse Dirichlet、预算、种子、batch64、50步、lr=.02/momentum=.9、30轮/阶段；sample_mean、fine/coarse权重均显式固定为1。独立输出`results/p26_sample_mean/{B0,B1,PS_FINE,PS_COARSE}`。

### 4.1 运行前检

环境准备复用P2.5的官方固定CUDA轮子脚本；不调用其自动90轮训练脚本。P2.6 preflight执行官方archive MD5、raw train与官方archive字节hash一致性、pytest、真实16张训练图像独立Oracle授权的CUDA fine/mixed反向传播与checkpoint校验。所有证据写入P2.6目录。

CUDA检查实际设备必须V100，没有CPU fallback。预检证书记录commit、config/source/data hash、种子、Python/Torch、CUDA runtime、GPU名称/总显存和nvidia-smi驱动信息。训练入口要求证书仍匹配当前配置、源码、Git commit、Torch、硬件、训练文件hash。预检若失败停止；本机没有生成虚构的GPU预检证书。

P2.5 CUDA preflight只新增可选`--output`参数，原默认路径和行为保留；P2.6指定新路径，避免覆盖P2.5结果。

### 4.2 明确的恢复链

| 显式动作 | 本次全局轮次 | 累计预算 | 必须已有 |
|---|---|---:|---|
| start2 | 1–5 | 900 | 空方法目录；有效preflight |
| finish2 | 6–30 | 900 | 第5轮checkpoint；健康门槛 |
| start4 | 31–35 | 1800 | 第30轮checkpoint；健康门槛 |
| finish4 | 36–60 | 1800 | 第35轮checkpoint；健康门槛 |
| start8 | 61–65 | 3600 | 第60轮checkpoint；健康门槛 |
| finish8 | 66–90 | 3600 | 第65轮checkpoint；健康门槛 |

首次不能直接30/90轮，已有非空方法目录不能重新初始化。恢复加载同一模型状态，重建Oracle时只回放已付费ID并核对标签快照；spent和账本继续保持累计数量，阶段内不会再次购买。变更配置/模型初始化/源码/Torch或不匹配的账本会拒绝恢复。账本需逐阶段增量正确、无重复，初购列表必须匹配，预算与轮次一致。

每轮原子保存`checkpoint.pt`；段末保留`checkpoint-round005.pt`等不可混淆快照。公开manifest保存原始train/validation ID及客户端分区，不保存隐藏fine或图像。每次独立invocation文件记录来源commit、配置hash、源码hash、硬件、种子、恢复文件和轮区间。

Linux包装脚本通过flock对同方法写入加锁，每次只执行一个明确方法/片段，没有任何多种子或全方法循环。paired后续片段额外验证两组此前共同前缀：例如PS_FINE先到30后，PS_COARSE仍可只核验共同1–5轮再到30；下一阶段需要两组都完成30轮。核验读取原子checkpoint，避免读取尚在写入的JSONL。

## 5. 每轮监控与健康门槛

保存字段：

- sample加权训练loss；fine/coarse原始loss sum、各组监督均值、对总loss贡献分量。
- 完整5000张独立验证集Fine Top-1、固定100类Macro-F1、Fine CE、预测熵、100类预测直方图、各coarse组下的fine Accuracy。
- 累计预算/比例、客户端分配账本、fine/coarse/无监督次数、总输入、重复暴露、不同fine数量、流hash。
- scheduled batches、backward calls、optimizer steps、nonzero-gradient steps、实际参数更新次数、零梯度batch数。
- 本地/聚合BN模块数、running mean最大绝对值、running var范围/负值数/有限性、counter范围。
- lr、momentum、optimizer/BN策略、训练与评估时间、CUDA allocated/reserved峰值、种子/commit/config/source/protocol hash。

工程门槛检查有限loss/梯度与模型state、完整5k验证/直方图数量和指标范围、预算/轮次/计数、BN非负有限；验证CE>100阻止后续续训并保留检查材料。极端BN范围、零准确率、预测集中和无fine暴露作为警告。阈值固定、所有方法相同，不按Accuracy挑最佳checkpoint或为方法调不同权重；它不是官方测试集早停。

训练模块不访问验证对象。可信控制器每轮只调用离线evaluator输出指标，不把验证真值或指标送回查询策略。官方test_samples严格为0；本阶段没有重新评估官方测试集。

## 6. 损失敏感性协议

sample_mean原定义保持；PS_FINE/PS_COARSE已购fine项都除以完整N，fine梯度系数一致。`group_mean`仅存在于明确命名的`p26_v100_group_mean_diagnostic.yaml`，角色必须为group_mean_diagnostic，输出必须含此名称；两项权重仍都是1。

group_mean下fine/coarse分别求均值后加和，缺失组为零。paired两组fine项仍使用相同分母，但这与sample_mean是**另一个协议**，必须单独预检、单独记录、单独输出。恢复协议hash禁止在中途切换损失；两组配置核对禁止混用协议。没有执行group_mean性能训练，也没有依据验证指标为各baseline分别设置权重。

## 7. 真实测试与执行结果

最终**82 passed in 22.82s**（原65 + 新17参数化测试实例）。新增覆盖paired梯度系数、流/fine暴露/BN规则一致、全未知只更新BN不改参数、有监督零梯度仍保留原SGD、B0/B1与旧更新兼容、完整预算恢复链、续训与人工连续训练逐张量相同、防重复启动/跨协议/坏账本、BN/CE/预算健康门槛及CUDA/配置限制。

checkpoint和健康逻辑测试使用明确标记的`FakeValidation`测试替身来模拟5k计数，**其虚拟指标不构成训练/验证结果**。GPU配置schema测试中的V100名称也是单元测试mock，不是实际V100可用证据。真实ResNet前向/梯度和P0–P2.5标签隔离测试仍在全回归中通过。

| 执行记录 | 真实结果 | 本机保留文件 |
|---|---|---|
| 第一轮专项 | 1 failed, 9 passed in 14.19s | results/p26_pytest_targeted.txt |
| 修正篡改fixture后的专项 | 13 passed in 13.91s | results/p26_pytest_targeted_fixed.txt |
| 回归首轮 | 78 passed in 22.23s | results/p26_pytest_all.txt |
| 零梯度语义复查后 | 79 passed in 23.29s | results/p26_pytest_final.txt |
| 配置门槛新增后 | 82 passed in 25.04s | results/p26_pytest_final_82.txt |
| 最终源码回归 | 82 passed in 22.82s | results/p26_pytest_release.txt |

首个失败是fixture直接篡改共享序列化列表，账本变化同时改了initial_ids，因此提前被初始化校验拒绝，未命中测试期望的ledger报错。修正fixture复制该列表再篡改，确认独立账本校验拒绝坏checkpoint，未放宽生产校验。

真实执行命令（exit0）：

```powershell
& ./.venv-p1/Scripts/python.exe scripts/audit_p26_sampling.py
$p26Temp=Join-Path (Get-Location) ('results/pytest-p26-'+[guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p26Temp
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.p26 --help
```

新Python脚本编译检查、`bash -n scripts/run_p26_v100.sh`和`git diff --check`均通过。Git Bash语法检查因Windows sandbox对象限制使用提升权限，但只执行-n，不运行安装/训练。实际CPU ID审计日志/JSON、全部测试输出保留在ignored results；没有删除文件或覆盖P2/P2.5结果。

## 8. Linux实际运行指令（准备完成，尚未执行）

```bash
# 从项目根目录；不要使用原run_p25_v100.sh自动串行完整90轮
bash scripts/setup_p25_v100.sh
bash scripts/run_p26_v100.sh preflight B0
bash scripts/run_p26_v100.sh start2 B0
bash scripts/run_p26_v100.sh start2 B1
# 审阅各目录health.json和rounds.jsonl后，显式继续
bash scripts/run_p26_v100.sh finish2 B0
bash scripts/run_p26_v100.sh finish2 B1
```

paired独立控制（同样一次只运行一组）：

```bash
bash scripts/run_p26_v100.sh start2 PS_FINE
bash scripts/run_p26_v100.sh start2 PS_COARSE
source .venv-p25/bin/activate
python scripts/verify_p26_pair.py --root results/p26_sample_mean
bash scripts/run_p26_v100.sh finish2 PS_FINE
bash scripts/run_p26_v100.sh finish2 PS_COARSE
```

后续4%和8%支持恢复，但每段须显式启动；下面不是本阶段已执行结果：

```bash
bash scripts/run_p26_v100.sh start4 B0
bash scripts/run_p26_v100.sh finish4 B0
bash scripts/run_p26_v100.sh start8 B0
bash scripts/run_p26_v100.sh finish8 B0
# B1或paired方法各自指定method，以同一阶段协议执行
# 独立group_mean诊断需重新preflight和单独目录
bash scripts/run_p26_v100.sh preflight PS_FINE configs/p26_v100_group_mean_diagnostic.yaml
bash scripts/run_p26_v100.sh start2 PS_FINE configs/p26_v100_group_mean_diagnostic.yaml
```

Linux需bash/flock（util-linux）、P2.5固定环境（Python3.10–3.12、Torch2.5.1+CUDA12.1/torchvision0.20.1）、实际可运行V100驱动和足够磁盘。安装及CUDA运行未在本机验证；GPU preflight将在服务器验证设备、真实反向传播和checkpoint后签发证书，不假定“脚本准备好=GPU实验完成”。

## 9. 文件交付与证据hash

- 新增`training/paired.py`：监控FedAvg、masked fine/control监督、BN/梯度/暴露计数。
- 新增`p26.py`：可信分阶段/预算/模型恢复、健康门槛、元数据、公开manifest、paired核验。
- 新增两份P2.6专属sample_mean及group_mean_diagnostic配置。
- 新增`run_p26_v100.sh`、`certify_p26_preflight.py`、`verify_p26_pair.py`、`audit_p26_sampling.py`。
- `p25_cuda_preflight.py`仅增加可选output路由，保持原默认行为。
- 新增`tests/test_p26_protocol.py`；更新README/.gitignore与本报告。原P2/P2.5训练、采样算法、V100配置和历史数字保持可复现。

本机证据SHA-256：

- `results/p26_sampling_audit.json`: `abb67a55fcdba2dc4fa93814a90708ebc0482368a08006745902ae644b5191a1`
- `results/p26_pytest_release.txt`: `5e86ab6264334f46f10c28b21ab724a851a7c600aeba42cf182c29a0b279bca3`
- `results/p26_environment.json`: `720a5d28e895340cff7f2ca912c3836097f9d820501c8ad7c9967b6dbcf77b6c`
- `configs/p26_v100.yaml`: `95a82573a5f0a86016c38b1a10b5f702fe2bd44f0a73c657a78a9f8c3bd7fd3e`
- `configs/p26_v100_group_mean_diagnostic.yaml`: `875e3fa6a645a194d604f765f18ed06142933a2cbf5053f1a1d0ce2b980e37c9`

## 10. 尚未解决的公平性风险与停止条件

1. 原B0/B1的fine过采样与BN分布差异仍保留，必须同时报告输入/fine暴露和成本，不能将其差值直接归为coarse监督贡献。
2. paired匹配fine暴露和输入流，**不匹配额外coarse计算与有效更新次数**；额外监督带来的优化步数是处理的一部分。需实际记录GPU成本，不能宣称相同计算成本。
3. BN在Non-IID、小已购集合和本地统计平均下的推理偏差仍存在。规则统一，不对个别方法调校缓存；paired模型权重不同也会导致BN数值不同。
4. 局部optimizer每轮重置；无监督batch跳过step，有监督零梯度保留正常SGD。动量与coarse梯度相互作用属于控制协议，需要论文说明，不隐藏这一因素。
5. sample_mean对低比例fine仍可能稀释；group_mean只诊断，不用于挑选“更好数字”或混为同一比较。
6. 没有V100全45k学习曲线，没有B0/B1或paired真实GPU差异，没有真实峰值显存/墙钟/FLOPs结果；服务器预检和前5轮是下一步明确门槛。
7. 脚本对一个方法加锁；直接绕过包装器并发运行同目录不受该shell锁保护，使用提供的分阶段脚本。跨环境位级复现不保证，恢复锁定源码/Torch并记录硬件。

本阶段已完成**协议实现、审计、恢复与回归验证，以及允许的无V100交付方案**。只建议先在V100显式执行preflight和B0/B1前5轮，审阅健康信息，再决定后续段与paired对照；没有启动完整90轮或B0–B4多种子，没有进入P3。
