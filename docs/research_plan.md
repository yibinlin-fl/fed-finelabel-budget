# 第二篇论文｜窄方向选题审计与第一轮实验任务书

日期：2026-10-09
状态：A = CONDITIONAL_PILOT；B = DEPRIORITIZE；不是已验证的新颖性声明。

## 0. 研究约束
- 第二篇与 CLE-HFL / RAHFL / GDAP 独立。
- 不再走“先挖已有算法失效机制、再提出修复”的路线；优先任务驱动、正向方法设计。
- 切口窄、公开数据不受限，初步实验可在 RTX 3050 4GB / 服务器 V100 运行。
- 先查新、复现基础任务、做止损实验；不要立即实现复杂算法。

## 1. 候选 B 审计：模态缺失传感器联邦学习
结论：DEPRIORITIZE（不作为这轮主选题）。

近邻论文：
1. FLISM, 2024: Federated Learning with Incomplete Sensing Modalities，研究缺失传感器、鲁棒表示与跨客户端知识迁移。https://arxiv.org/html/2405.11828v1
2. MFCPL, 2025: Cross-modal prototype based multimodal federated learning under severely missing modality. https://www.sciencedirect.com/science/article/pii/S1566253525002921
3. MMO-FL, 2025/2026: Multimodal Online Federated Learning with Modality Missing in Internet of Things. https://arxiv.org/abs/2505.16138
4. Flux, 2026-08: Multimodal Federated Learning under Dual-Axis Modality Missingness. 同时考虑客户端级、样本级缺失；在感知数据模拟中已有 Markov 时间相关的模态故障。https://arxiv.org/html/2608.09240v1
5. FedMultimodal (KDD 2023): UCI-HAR 公共基准、下载与缺失模态脚本都齐全。https://github.com/usc-sail/fed-multimodal

判断：数据易获取≠题目新颖；随机/间歇/持续缺失、模态感知聚合、原型补齐已经具有密集近邻工作，不宜再做微小改动。

## 2. 候选 A 审计：粗标签向细标签的预算受限升级
临时任务名称：Budgeted Federated Fine-Label Acquisition with Coarse Supervision。

**研究问题**：所有客户端已有可用粗类别标签，但只能为有限样本申请细粒度人工标注。在不共享原始图像、并不使用服务器公共精细标注集的情况下，如何将有限的细标签获取预算在客户端和粗类群之间分配，从而提升全局细粒度分类器的预算效率？

最近邻成果（必须在投稿前进一步核查）：
- FedMT (arXiv 2210.02042, 更新至 2024)：粗细/mixed-type label 空间学习；CIFAR100 实验常使用服务器已拥有的少量细类别标注。与我们的“所有细标签只在客户端由预算逐步解锁”设定不同，但问题高度相关。https://arxiv.org/html/2210.02042v4
- FedBDFT (2026-08)：客户端标签粒度不一致的层次分类；已有端到端 FedBDFT 算法。其 CIFAR100 采用自建 WordNet 层级（论文描述 22 个粗节点、最大深度 5），不能直接把数值和 CIFAR100 官方两层 20 超类方案当作同一 benchmark。https://arxiv.org/html/2608.09236
- FairFAL (CVPR 2026)：不均衡与极端 Non-IID 条件下的联邦主动标注。https://openaccess.thecvf.com/content/CVPR2026/html/Zong_Federated_Active_Learning_Under_Extreme_Non-IID_and_Global_Class_Imbalance_CVPR_2026_paper.html
- Coordination on a Budget (2026-08)：少标签预算的跨客户端主动选择协调。https://arxiv.org/html/2608.18634v1
- FAST (IEEE IoT Journal, 2025)：基础模型弱标注 + 低预算人工细化，公开实现。虽非严格 taxonomy-consistent coarse labels，仍须检查它的 query/refinement 机制。https://github.com/haoyuan-l/FAST-official
- Cost-Effective Active Learning for Hierarchical Multi-Label Classification (IJCAI 2018)：非联邦环境下，层次标签/查询成本的主动学习已有先例。https://www.ijcai.org/proceedings/2018/411

审计结论：
- **不能宣称新颖**：“粗细标签联合训练”、“置信度高/低选择标注”、“主动学习”、“类别平衡”单独都已有工作。
- 尚值得检验的组合约束：已有可靠粗标签 + 部分客户端精细标签可按预算解锁 + 客户端与粗类之间的联合预算决策 + 不传输样本及 embeddings。
- 这只是可能的区别，不自动构成方法创新。假如简单的层次条件熵/平衡采样已经足够，应该停止研发，而不是加模块硬凑。

## 3. 数据集与严格标注协议
### 3.1 Pilot 数据
- CIFAR-100 官方原始层级：20 个 coarse、100 个 fine，每个 coarse 对应 5 个 fine，训练 50,000，测试 10,000。下载约 161MB。https://cave.cs.toronto.edu/kriz/cifar.html
- 第一轮采用原始 coarse/fine 映射；不要混用 FedBDFT 的自建 WordNet 多级层次。
- 所有训练样本先只暴露 coarse label；真实 fine label 被封装为一个 oracle，只在 sample index 被 query 后才能访问并加入训练集。
- 不能使用 fine label 做客户端采样、初始带标注样本的按细类分层、采样策略调参或构造先验；可使用 coarse 进行控制。
- 隐藏的真实 fine 标签仅用于模拟人工标注回复及最终独立测试评估。
- 不用测试集做采样算法选择、超参搜索或早停。
- 训练集再划分 train/val（例如 45k/5k，划分细节记录）；为了避免在训练中使用未购买 fine labels，**验证集的细类别标签只能用于最终离线评估/研究者调参；不可进入客户端查询策略或训练流程**。如果把 val 视为服务器可访问的 fine 公共集，就违反研究设定；最好将验证作为离线 evaluator，决策算法无访问权。

### 3.2 客户端划分
- 第一版 10 clients，IID 和 coarse-label Dirichlet(alpha 0.3, 1.0)；同 seed 共享完全相同划分。
- 划分依据 coarse 仅；若将来加入基于 fine 的人为 Non-IID 划分，明确它是离线模拟器操作，不得被在线算法看到。
- 客户端数据规模下限、空 coarse 群、单客户端没有任何细标签等都要有测试覆盖。

### 3.3 标注预算
- 初始 fine 标签：从每客户端样本中随机抽取 2%（初始已购买，计入总预算）。
- 额外 fine query 预算：全局 +2%、+4%（在所有客户端样本上统计）；每个策略具有完全相同的累计 fine query 数。
- 查询时，未知 fine 的 coarse 标签仍然保留用于训练；已查询 fine 的数据应转为 fine loss，避免把同一监督重复加权（除非作独立消融）。
- 细化预算在 pilot 中用“人工 fine-query 次数”衡量；**没有真实人力工时证据时，不要将其宣称为真实货币节省**。

## 4. 合理的基础训练模型
模型输出对 100 个 fine 类的 logits z，并有官方 20-to-100 映射 S(c)。
- 已有 fine 标注样本：`L_fine(x,y) = -log softmax(z(x))[y]`。
- 只有 coarse 标注样本：`L_coarse(x,c) = -log sum_{j in S(c)} softmax(z(x))[j]`。
- 数值实现时使用 `logsumexp(z[S(c)]) - logsumexp(z)`；不要先 softmax 然后手工 sum+log 导致下溢。
- 联邦训练：相同 ResNet18（CIFAR stem）+ FedAvg；探索期固定参与比例、本地 epoch、通信轮数及总梯度步数。
- 优先不用伪标签、教师模型、CLIP、原型库或复杂个性化模型，以便知道结果由哪一部分产生。

## 5. Pilot Baselines（先实现，禁止一开始写复杂候选模型）
B0. Same-budget fine-only FedAvg：只使用已购买的 fine 标签；其余样本不训练。衡量 coarse 标签本身价值。
B1. Coarse-aware + Random：同样的 coarse/fine 混合损失，随机选择 fine 查询样本。
B2. Coarse-aware + Local Conditional Entropy：只按对应 coarse 组内 fine posterior 的熵局部选样。
B3. Coarse-aware + Group-balanced Random：根据 coarse 组现有 fine 标注覆盖调整预算，再组内随机选样。
B4. Coarse-aware + Local Entropy-Diversity（例如已选 embedding 距离，embedding 留在本地）：检验简单多样性修正的价值。
Optional UB. Full fine-label FedAvg：100% fine 标签的上界，不参与同预算策略排名。

**不应把 FedMT/FedBDFT/FairFAL 直接塞入 pilot 的主排名**，因为标签持有位置、层级定义及标注预算协议可能不同。正式论文需要明确复现或合理适配最相关方法，并单列原设置与受控适配设置，不得把不同协议下的结果直接比较。

## 6. Pilot 的核心判断
衡量：
- Acc@100、macro-F1@100（独立 test）；每 20 个 coarse 群的子类准确率。
- 同一累计精细标注预算下的性能对比；标注预算-性能曲线下面积（AULC）。
- 各策略的累计查询次数和累计训练成本；同训练轮数、客户端抽样 seed、初始化 seed 配对对照。
- IID / alpha 0.3 / alpha 1.0；至少 3 个 seeds。

强制判据：
1. 若 `Coarse-aware + Random` 相对 `Fine-only` 也没有稳定收益，则当前粗标签利用方式可能有问题，暂不开发预算分配策略。
2. 若组内 conditional entropy 已经和其他复杂策略接近，且不同 non-IID/预算下无稳定可重复改进，暂停选题。
3. 只有若客户端 × coarse 群的预算协调显示相对强简单基线的**可复现、实质性的增益**，才进入新方法设计和更大数据集。
4. 不允许用测试集寻优、不允许只汇报一个最好 seed、不允许对不同算法使用不同查询次数或训练预算。

## 7. 可能的方法层：这里只列待验证机制，不提前实现
分层预算协调：各客户端在本地对每个 coarse 类计算可导出的非原始数据统计（候选池数量、组内细分类条件熵汇总、当前已购买 fine 数量），服务器只根据统计信息分配 `(client, coarse group)` 额度，然后客户端独立选取具体样本请求 fine 标注。

隐私注意：聚合统计并不天然具备差分隐私保证，少样本客户端的组计数也可能泄露信息。只能宣称原始图像不上传。正式论文需要报告统计量通信成本和潜在信息泄露风险。

## 8. 首批必须产出的文件和测试
- `configs/pilot.yaml`：随机种子、client、alpha、预算、查询轮次、训练轮次等全部显式化。
- `data/cifar100_hierarchy.py`：读官方粗/细标签和 20-to-100 映射，检查每个细类仅属于一个粗类。
- `data/label_oracle.py`：隐藏标签访问与解锁接口，禁止训练/选择模块直接索引未解锁标签。
- `data/partition.py`：固定 client partition；无 fine 标签泄漏。
- `models/resnet_cifar.py`
- `training/fedavg.py`、`training/losses.py`
- `query/random.py`、`query/conditional_entropy.py`、`query/group_balanced.py`
- `eval/metrics.py`、`scripts/run_pilot.py`
- `tests/test_no_label_leakage.py`、`tests/test_budget_accounting.py`、`tests/test_hierarchy.py`、`tests/test_seed_replay.py`。
- `reports/pilot_results.csv`、`reports/pilot_summary.md`，不得填入模拟训练结果。

## 9. 可直接交给 Codex 的任务说明
你是负责机器学习实验平台实现的工程师。为上述第 3–6 节实现最小实验项目。**这不是新算法开发阶段**：仅实现可重现的 CIFAR-100 粗/细标签预算实验协议、严格标签 oracle、基础 FedAvg、混合监督损失、4 种查询基线、同预算评估及自动化测试。尽量少用依赖：Python、PyTorch、torchvision、numpy、pandas、pytest、PyYAML。要支持 `--device cuda` 与 `cpu`，无 GPU 时允许 smoke-test，V100 上支持单卡串行客户端仿真。

执行要求：
1. 先写数据规范与不泄漏约束；再实现代码；先跑单元测试/烟测，不要求大规模训练立刻完毕。
2. 所有随机种子（partition / init / client sampling / query）显式控制；不同 baseline 可配对复现。
3. `query` 模块不得读取隐藏 fine label；oracle 只给已授权 index 返回 label；不能用 test fine 标签选策略。
4. 训练期间记录每轮准确率、精细标签使用数、训练成本、实验配置和 git commit hash。
5. 严格比较同预算+同联邦计算轮数；加上不查询 fine 的纯粗监督 sanity check。
6. 先给出 README：从空环境到下载 CIFAR-100、运行测试、运行 smoke 和 pilot 的完整命令。
7. 若任务需要自行作决定，优先实现简单且可审计的设计；不要引入新模型、深度蒸馏、CLIP 或个性化 FL。
8. 不得伪造执行成功、训练精度、运行耗时或 GPU 验证结论；无法在当前环境跑就说明具体原因。

## 10. 阶段结论
- 不推荐以“间歇性缺失传感器”作为本轮主线。
- 暂定唯一优先试验任务：**有可靠粗标签的预算受限联邦细标签获取**（conditional pilot，不是已经完成的选题）。
- 真正决定是否投下一阶段算力的是：在公平的标注与计算预算下，是否存在足够的跨客户端预算协调增益，而非简单方法叠加。
