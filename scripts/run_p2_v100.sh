#!/usr/bin/env bash
# User launches explicitly. One configured seed; no automatic multi-seed loop.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
python scripts/download_cifar100.py --root data/cifar100
python -m fed_finelabel_budget.cli data-check --root data/cifar100
python scripts/run_pilot.py --config configs/p2_v100.yaml "$@"
