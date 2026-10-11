#!/usr/bin/env bash
# Explicit user launch; B0 then B1 only, single seed, no official test access.
set -euo pipefail
source .venv-p25/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; assert torch.cuda.is_available(); assert 'V100' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
python scripts/download_cifar100.py --root data/cifar100
python -m pytest -q
python scripts/p25_cuda_preflight.py
python -m fed_finelabel_budget.p2 --config configs/p25_v100_b0_b1.yaml --baseline B0 --output results/p25_v100_B0
python -m fed_finelabel_budget.p2 --config configs/p25_v100_b0_b1.yaml --baseline B1 --output results/p25_v100_B1
python scripts/plot_p25.py --baselines results/p25_v100_B0/B0 results/p25_v100_B1/B1 --output results/p25_v100_curves.png
