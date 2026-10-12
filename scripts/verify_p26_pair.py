"""Cross-arm pairing gate; does not choose models or inspect hidden labels."""
import argparse
import json
import torch
from pathlib import Path
from fed_finelabel_budget.p26 import verify_pair

p=argparse.ArgumentParser()
p.add_argument("--root",default="results/p26_sample_mean")
p.add_argument("--until-round",type=int)
args=p.parse_args()
def rows(method):
    rows=torch.load(Path(args.root)/method/"checkpoint.pt",map_location="cpu",weights_only=True)["rounds"]
    if args.until_round is not None:
        if len(rows)<args.until_round:
            raise ValueError("Both paired arms must finish the required prefix first")
        rows=rows[:args.until_round]
    return rows
result=verify_pair(rows("PS_FINE"),rows("PS_COARSE"))
Path(args.root,f"paired_verification_round{result['rounds']:03}.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
