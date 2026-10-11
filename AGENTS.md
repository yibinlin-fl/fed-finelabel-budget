# Codex instructions — 第二篇论文 / fed-finelabel-budget

You are maintaining a **research-grade, reproducible** federated-learning experiment scaffold.

## Non-negotiable rules

1. Read `README.md`, `docs/research_plan.md`, `docs/DATA_CONTRACT.md`, and `docs/CODEX_NEXT_TASK.md` before editing. This project is **independent** of CLE-HFL / RAHFL / GDAP.
2. P0 currently includes data protocol and tests, **NOT model training**. Implement ONLY the phase explicitly requested by the user. Do not claim an algorithm has been validated.
3. **No fine-label leakage.** Client partitioning, training, and selection cannot see unpurchased fine labels. `cifar_io.py` is trusted bootstrap and must not be imported in client/query modules. In particular do not expose raw `torchvision.datasets.CIFAR100.targets` to query logic.
4. Only `FineLabelOracle.query(ids)` may unlock ground truth under a configured budget. All simulation truth is controller-side; query code receives *only* public data, already purchased labels, and model outputs. Python encapsulation prevents accidental leaks, not deliberate raw-file/introspection attacks.
5. Use coarse-only Dirichlet and IID partitions with reproducible seeds. Always verify exactly-once partitioning and identical budgets/training budgets across compared strategies.
6. The official CIFAR-100 hierarchy must be derived from real label pairs, not naïve 5-consecutive-ID grouping. Do not substitute FedBDFT WordNet taxonomy.
7. Never use test labels to choose queries/thresholds/hyperparameters, or pretend offline private evaluation labels are server-side public fine labels.
8. No fabricated metrics, logs, time estimates, GPU results, or literature novelty claims. Separate executed evidence from proposed future steps.
9. Do not build additional teachers, pseudo-label models, CLIP, distillation, personalization, or novel aggregation at P1. Keep dependencies reasonable. Use `torch`/`torchvision` only for P1 training; adjust CUDA packages to server environment.
10. Run the smallest relevant test first, then all tests; report commands, exit codes, errors, and changed files. Avoid reading huge binaries/training logs into model context.

## Immediate phase

See `docs/CODEX_NEXT_TASK.md`. After completing it, STOP before longer V100 runs. Leave test/train CSVs empty unless actually produced by real runs. Keep raw data/models out of Git.
