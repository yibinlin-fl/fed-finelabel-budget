# P1 基础训练闭环实测报告

日期：2026-10-11。项目：fed-finelabel-budget。
结论：P1 已完成并停止。所有数字来自本机实际测试；没有 CIFAR-100 性能结论、GPU 结果或方法增益结论。

## 工程与环境

实际工程根目录是 `D:/AVibCodingProjects/fed-finelabel-budget-P0-starter/fed-finelabel-budget`，未创建第二个工程，未改写 P0 架构。
已完整阅读 AGENTS.md、README.md、research_plan、DATA_CONTRACT、P0_REPORT、CODEX_NEXT_TASK。

原目录没有 Git，已初始化并提交 P0：`b6eb5b805824ce57189a710197a593007c42eebf`（28 文件，1074 行）。数据、结果、环境和模型不入 Git。P1_REPORT 特别解除 reports 的忽略规则，其余运行结果仍忽略。

本机 Windows 11 build 26200；默认 Python 3.8.1 不满足 >=3.10。使用 `D:/anaconda3/python.exe` 的 Python 3.12.7 创建 `.venv-p1`。实际安装：torch 2.14.1+cpu、torchvision 0.29.1+cpu、NumPy 2.5.4、PyYAML 6.0.3、pytest 9.1.1（完整安装清单见 `results/p1_smoke/environment.txt`）。CUDA availability = False，本次全部 CPU，1 个线程。

环境问题及处理：

- `python -m venv .venv` 的 ensurepip 失败，未获得可用 pip；改用 uv 和 Python 3.12.7 创建 `.venv-p1`。保留原目录，没有批量删除。
- uv 默认缓存 `D:/DevCache/uv` 无访问权限；改为项目内 `.uv-cache`。
- 沙箱内安装依赖两次遇到 Windows PE/构建权限错误，获自动审批后重试安装成功；无需用户进一步操作。
- 首次 P0 `pytest -q` 为 15 passed、3 errors，全部 errors 是 pytest 无法创建沙箱系统临时目录，未出现 P0 断言失败。改用项目内每次全新 `--basetemp` 路径，原 18 项全部通过。未修改已有测试断言，也未清理已有目录。

未发现需要修复的 P0 代码错误。新增了 P1 的数值、梯度、隔离、聚合、重放和适配器回归测试。

## 文件变更

新增：

- `models/__init__.py`、`models/resnet_cifar.py`（位于 `src/fed_finelabel_budget/`）：CIFAR ResNet18。
- `training/__init__.py`、`training/dataset.py`、`training/losses.py`、`training/seeds.py`、`training/fedavg.py`：安全数据适配、损失、随机数控制与联邦闭环。
- `src/fed_finelabel_budget/p1.py`：可信控制器、配置加载、预算预购、受限训练、日志与 checkpoint。
- `configs/p1_smoke.yaml`、`scripts/run_p1_smoke.py`：可直接执行的最小 CPU 配置与入口。
- `tests/test_p1_losses.py`、`tests/test_p1_training.py`：9 个新单元测试用例（含参数化）。
- `reports/P1_REPORT.md`：本报告。

修改：

- `tests/test_loader.py`：增加官方格式夹具的 P1 子集 ID/Oracle 映射集成测试，保留原 3 项测试。
- `README.md`：更新阶段状态，补安装、测试、smoke 与真实数据命令。
- `docs/DATA_CONTRACT.md`：补 P1 实际信任边界与监督规则。
- `.gitignore`：忽略新环境、缓存、临时目录，允许提交本报告。
- `pyproject.toml`：更新项目描述；沿用已有 train 依赖组。

## 已实现的功能与约定

1. 不加载预训练权重的 ResNet18：3×3 stride=1 stem、无 maxpool、100 类输出。
2. 已购买 fine 样本仅 fine CE；未知 fine=-1 样本仅 coarse marginal likelihood。使用两个 logsumexp 的差，批样本均值归一化；不重复监督已购样本。
3. 客户端数据适配只物化自身公共图像/ID/coarse，并复制自身已购标签；标签与公开层级不匹配时拒绝。固定 uint8→[-1,1] 转换，无增强。
4. 单进程串行 FedAvg；各客户端从同一全局模型出发，用独立 SGD 本地训练，按完整本地训练数据集样本数加权浮点参数和 BN 浮点状态。BN 整数计数取最大值，避免平均整数计数产生截断。客户端每次重建 optimizer。
5. IID/coarse-Dirichlet 沿用 P0；partition / init / client sampling / query / training / synthetic data 种子显式分开。训练重放测试逐张量完全一致，适用本次 CPU 软件环境；不承诺跨版本/设备位级一致。
6. 每轮记录实际参与客户端、聚合权重、梯度步数、fine/coarse examples_seen 与平均训练损失；run.json 记录配置、环境、Git commit、dirty 状态和预算。固定轮数，无 early stopping，不读 test/val。
7. checkpoint 保存 CPU 模型状态、公开层级、配置、训练历史与 Git commit，回读逐张量检查。不含原始数据或隐藏标签。已验证权重保存恢复，没有实现精确中断续训/optimizer RNG 恢复。
8. smoke 入口强制 100..300 样本、2..3 客户端、1..2 轮、每客户端每轮最多 1..2 步，防止误启动正式实验。实际默认 150 样本、3 客户端、1 轮、各 1 步。

## 实际成功执行的命令

以下均从工程根目录执行。安装失败与环境纠正见上节；以下列的是成功命令。

```powershell
git init
git add .
git commit -m "Save tested P0 scaffold before P1"
uv --cache-dir .uv-cache venv --python D:/anaconda3/python.exe .venv-p1
uv --cache-dir .uv-cache pip install --link-mode copy --python .venv-p1/Scripts/python.exe -e '.[dev,train]'

# 每轮 pytest 都生成此前不存在的路径，不复用已有目录
$p1TestTemp = Join-Path (Get-Location) ('results/pytest-p0-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p1TestTemp
# 原 P0: 18 passed in 0.16s, exit 0

$p1TestTemp = Join-Path (Get-Location) ('results/pytest-loss-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q tests/test_p1_losses.py --basetemp $p1TestTemp
# 4 passed in 5.36s, exit 0

$p1TestTemp = Join-Path (Get-Location) ('results/pytest-all-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p1TestTemp
# 中间回归: 27 passed in 7.45s, exit 0

& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --config configs/p1_smoke.yaml
# PASS, exit 0

$p1TestTemp = Join-Path (Get-Location) ('results/pytest-adapter-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q tests/test_loader.py::test_p1_real_data_controller_subset_ids --basetemp $p1TestTemp
# 1 passed in 10.97s, exit 0

$p1TestTemp = Join-Path (Get-Location) ('results/pytest-final-' + [guid]::NewGuid().ToString())
& ./.venv-p1/Scripts/python.exe -m pytest -q --basetemp $p1TestTemp 2>&1 | Tee-Object -FilePath results/p1_smoke/pytest.txt
# 最终回归: 28 passed in 11.37s, exit 0

& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli preflight --config configs/p0.yaml | Tee-Object -FilePath results/p1_smoke/preflight.json
# PASS, fine_queries_observed=[20,40,80], remaining_queries=0, exit 0

uv --cache-dir .uv-cache pip freeze --python .venv-p1/Scripts/python.exe | Set-Content results/p1_smoke/environment.txt
git diff --check
# 无 whitespace errors
```

日志记录 Git 为 P0 commit 且 dirty=true，表示 smoke 是基于该 P0 提交上的本次 P1 工作区执行，未把 P0 当成 P1 训练代码版本。

## 单元测试与训练结果

最终 28 项全部通过，涵盖原 P0 的 18 项及 P1 新增 10 项。P1 覆盖：非连续层级损失、数值参考对照、double gradcheck、全 fine/全 coarse、极端 logits、空批次/越界/冲突拒绝、unequal-size FedAvg、BN 整数计数、客户端快照、禁止训练导入 trusted reader/oracle、零 fine 客户端、确定性重放、ResNet18 真正 forward/backward、官方格式夹具的子集适配与 checkpoint。

独立合成 smoke 的真实输出：

| 项目 | 观测 |
|---|---|
| 图像/客户端/轮次 | 150 个随机 uint8 32×32 图像；3 客户端；1 轮 |
| 梯度步数/训练样本使用数 | 合计 3 步，150 examples_seen |
| 已购 fine / coarse loss 样本使用数 | 3 / 147 |
| 累计预算目标/实际消费/剩余 | [3,6,12] / 3 / 9；只执行初始购买 |
| 各客户端样本数与权重 | 各 50，权重各 1/3 |
| 客户端 0/1/2 平均训练 loss | 3.046830177307129 / 3.08659291267395 / 3.040795087814331 |
| 更新参数张量数 | 62 |
| 训练函数实测耗时 | 5.454497200000333 秒；不含安装、初始化和 checkpoint IO |
| checkpoint 回读 | 所有模型 state 张量一致，PASS |
| val/test 读取 | False |

这些损失和耗时仅是合成工程 smoke 证据，不能解读为 CIFAR-100 准确率或收敛性。未产出 test/train accuracy CSV，未伪造 Acc@100、macro-F1 或 AULC。

运行文件在 `results/p1_smoke/`：`run.json`、`rounds.jsonl`、`checkpoint.pt`、`pytest.txt`、`preflight.json`、`environment.txt`，均不提交 Git。

## 标签泄漏与预算风险检查

没有发现当前实现中的未购买 fine 泄漏路径：training 不导入 cifar_io/oracle；输入只有公开样本与已购快照，未知 fine 只能以 -1 进入损失；训练器和选择函数不获得测试/验证真值。公开 fine-to-coarse taxonomy 可用，不等于样本真值。

Oracle 在可信控制器内执行初购，初购计入预算。P0 的原子回滚/重复不收费测试继续通过。P1 实际只消费 2%，没有提前执行追加预算或 P2 选择策略。分区依然 exactly-once 且不访问 fine。真实数据适配用公共 ID 抽子集，原 ID 查询 oracle，子集 ID 配训练，集成测试已核对 100 个夹具样本中 2 个 fine 的一致使用。

剩余边界：Python 封装不是强安全隔离，恶意反射/直接读取原始 pickle 仍不在防护承诺内；已购 Mapping 的真实性依赖可信控制器，类型系统无法证明付费。smoke 限制步数可能使部分标签尚未被梯度使用，日志区分已购标签数与实际 examples_seen；正式 P2 必须统一步数、抽样与预算，不能只比较轮数。BN 状态聚合规则已明确，后续对照必须一致使用。

## 官方数据与 P2 条件

本次 `data/cifar100/cifar-100-python/train` 不存在，没有下载官方数据，未执行真实 CIFAR-100 smoke 或 GPU/V100 实验。官方格式夹具测试不是官方数据验证。

待用户后续启动真实 smoke 时可运行：

```powershell
& ./.venv-p1/Scripts/python.exe scripts/download_cifar100.py --root data/cifar100
& ./.venv-p1/Scripts/python.exe -m fed_finelabel_budget.cli data-check --root data/cifar100
& ./.venv-p1/Scripts/python.exe scripts/run_p1_smoke.py --mode cifar100 --device cpu --output results/p1_cifar_smoke
```

工程已具备进入 P2 开发的基础训练条件；尚不具备直接启动 B0–B4 正式比较的完整功能。后续授权 P2 时仍需基线实现、独立离线 evaluator/指标、相同查询及计算预算控制、官方数据实测和服务器 CUDA 环境核验。真实 smoke 预算分母只是子集，正式 50k 预算必须仍为 1000→2000→4000。P1 不验证研究创新性和性能增益。

本轮停止在 P1；未实现 P2 策略、教师/伪标签/蒸馏或额外复杂算法，未启动长时间实验。
