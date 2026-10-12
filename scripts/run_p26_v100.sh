#!/usr/bin/env bash
# Exactly one method/segment; never loop all methods or all 90 rounds.
set -euo pipefail
action=${1:?Usage: run_p26_v100.sh preflight|start2|finish2|start4|finish4|start8|finish8 METHOD [CONFIG]}
method=${2:-B0}
config=${3:-configs/p26_v100.yaml}
case "$method" in B0|B1|PS_FINE|PS_COARSE) ;; *) echo "Invalid method" >&2; exit 2;; esac
source .venv-p25/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; assert torch.cuda.is_available() and 'V100' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
root=$(python - "$config" <<'PY'
import sys,yaml
from fed_finelabel_budget.p26 import validate_config
c=yaml.safe_load(open(sys.argv[1]))
validate_config(c)
print(c['output'])
PY
)
if [[ "$action" == preflight ]]; then
  mkdir -p "$root"
  python scripts/download_cifar100.py --root data/cifar100
  testlog="$root/preflight-pytest-$(date +%s)-$$.txt"
  python -m pytest -q --basetemp "results/p26-pytest-$(date +%s)-$$" | tee "$testlog"
  python scripts/p25_cuda_preflight.py --output "$root/cuda_preflight.json"
  python scripts/certify_p26_preflight.py --config "$config" --test-log "$testlog"
  exit 0
fi
mkdir -p "$root/_locks"
exec 9>"$root/_locks/$method.lock"
flock -n 9 || { echo "Method already running" >&2; exit 2; }
case "$action" in
  start2) until=5; resume=() ;;
  finish2) from=5; until=30; resume=(--resume "$root/$method/checkpoint.pt") ;;
  start4) from=30; until=35; resume=(--resume "$root/$method/checkpoint.pt") ;;
  finish4) from=35; until=60; resume=(--resume "$root/$method/checkpoint.pt") ;;
  start8) from=60; until=65; resume=(--resume "$root/$method/checkpoint.pt") ;;
  finish8) from=65; until=90; resume=(--resume "$root/$method/checkpoint.pt") ;;
  *) echo "Unknown segment" >&2; exit 2 ;;
esac
if [[ "$method" == PS_* && "$action" != start2 ]]; then
  python scripts/verify_p26_pair.py --root "$root" --until-round "$from"
fi
python -m fed_finelabel_budget.p26 --config "$config" --method "$method" --until-round "$until" "${resume[@]}"
