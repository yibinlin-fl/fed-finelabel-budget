"""Preserve the executed CUDA/test/data gate under the P2.6 namespace only."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import yaml
from fed_finelabel_budget.p26 import runtime_metadata, validate_config

p=argparse.ArgumentParser()
p.add_argument("--config",required=True)
p.add_argument("--test-log",required=True)
args=p.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text())
validate_config(cfg)
output=Path(cfg["output"])
cuda=json.loads((output/"cuda_preflight.json").read_text())
assert cuda["checkpoint_exact"] and all(x["gradients_finite"] for x in cuda["losses"].values())
test_log=Path(args.test_log).read_text()
assert "passed" in test_log and "failed" not in test_log and "ERROR" not in test_log
archive=Path(cfg["dataset"]["root"])/"cifar-100-python.tar.gz"
h=hashlib.md5()
with archive.open("rb") as f:
    for block in iter(lambda:f.read(1024*1024),b""):
        h.update(block)
assert h.hexdigest()=="eb9058c3a382ffc7106e4002c42a8d85"
metadata=runtime_metadata(cfg)
with tarfile.open(archive,"r:gz") as tar:
    source=tar.extractfile("cifar-100-python/train")
    extracted_hash=hashlib.sha256()
    for block in iter(lambda:source.read(1024*1024),b""):
        extracted_hash.update(block)
assert extracted_hash.hexdigest()==metadata["training_file_hash"], "Raw training file differs from official archive"
metadata.update(preflight_pass=True,test_log=args.test_log,data_md5=h.hexdigest(),cuda_preflight=cuda)
(output/"preflight.json").write_text(json.dumps(metadata,indent=2))
print("P2.6 V100 preflight certified")
