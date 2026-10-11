#!/usr/bin/env bash
# Run from project root. Pinned official CUDA wheels retain Volta support;
# check actual runtime architectures/device before any experiment.
set -euo pipefail
python3 -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), 'Use Python 3.10-3.12 for the pinned Torch wheels (3.12 recommended)'"
python3 -m venv .venv-p25
source .venv-p25/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.1.3 PyYAML==6.0.3 pytest==9.0.2 matplotlib==3.10.8
python -m pip install -e . --no-deps
python -c "import torch; assert torch.cuda.is_available(); assert 'sm_70' in torch.cuda.get_arch_list(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); x=torch.ones(8,8,device='cuda'); print((x@x).sum().item())"
