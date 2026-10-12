"""Trusted real-data ID-only exposure audit. NO model/GPU training or metrics."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import RandomSampler
import yaml
from fed_finelabel_budget.data.cifar_io import prepare_experiment
from fed_finelabel_budget.data.partition import coarse_dirichlet_partition
from fed_finelabel_budget.data.budget import initial_random_ids,cumulative_targets
from fed_finelabel_budget.p2 import acquire_to_target,query_increment


def main():
    cfg=yaml.safe_load(Path("configs/p26_v100.yaml").read_text())
    d,s,t=cfg["dataset"],cfg["seeds"],cfg["training"]
    sim=prepare_experiment(d["root"],split_seed=s["split"],validation_size=5000,test_samples=0)
    p=cfg["partition"]
    parts=coarse_dirichlet_partition(sim.pool.coarse_labels(),10,p["alpha"],s["partition"],p["min_client_samples"],p["max_attempts"])
    targets=cumulative_targets(len(sim.pool),.02,[.02,.04])
    oracle=sim.new_oracle(targets[-1])
    rng=np.random.default_rng(s["client_sampling"])
    rows=[]
    stage_summaries=[]
    for stage,target in enumerate(targets):
        selected=(initial_random_ids(parts,.02,s["query"]) if stage==0 else
            query_increment("B0",sim.pool,parts,oracle.purchased_labels(),target-oracle.spent,s["query"]+stage*100003,
                            None,sim.hierarchy,"cpu",64))
        acquire_to_target(oracle,selected,target)
        bought=oracle.purchased_labels()
        exposure={k:Counter() for k in ("B0","B1")}
        fine_exposure={k:Counter() for k in ("B0","B1")}
        fine_batches={k:0 for k in ("B0","B1")}
        for ri in range(stage*30,(stage+1)*30):
            selected=sorted(map(int,rng.choice(sorted(parts),t["clients_per_round"],replace=False)))
            row={"round":ri+1,"budget":target,"methods":{}}
            for method in ("B0","B1"):
                nf=0
                for cid in selected:
                    ids=[int(i) for i in parts[cid] if int(i) in bought] if method=="B0" else list(map(int,parts[cid]))
                    sampler=RandomSampler(range(len(ids)),replacement=True,num_samples=t["batch_size"]*t["max_steps"],
                        generator=torch.Generator().manual_seed(s["training"]+ri*1000003+cid))
                    drawn=[ids[i] for i in sampler]
                    fine=[i for i in drawn if i in bought]
                    exposure[method].update(drawn)
                    fine_exposure[method].update(fine)
                    nf+=len(fine)
                    fine_batches[method]+=sum(any(i in bought for i in drawn[j:j+t["batch_size"]]) for j in range(0,len(drawn),t["batch_size"]))
                row["methods"][method]={"fine_exposures":nf,"input_exposures":len(selected)*t["batch_size"]*t["max_steps"]}
            rows.append(row)
        summary={"stage":stage,"budget":target,"methods":{}}
        for method in ("B0","B1"):
            summary["methods"][method]={"input_exposures":sum(exposure[method].values()),
                "fine_exposures":sum(fine_exposure[method].values()),"distinct_inputs":len(exposure[method]),
                "distinct_fine":len(fine_exposure[method]),"repeated_fine_exposures":sum(fine_exposure[method].values())-len(fine_exposure[method]),
                "fine_positive_batches":fine_batches[method],"scheduled_batches":30*t["clients_per_round"]*t["max_steps"]}
        stage_summaries.append(summary)
    result={"purpose":"ID-only sampling audit; NOT GPU or CPU model training","config":cfg,
            "budgets":targets,"client_sizes":{k:len(v) for k,v in parts.items()},
            "validation_distribution":sim.validation.label_distribution(),"stages":stage_summaries,"rounds":rows,
            "paired_stream_rule":"PS_FINE and PS_COARSE both reuse B1 public-pool stream; exact hashes asserted by training/tests"}
    Path("results/p26_sampling_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(stage_summaries,indent=2))


if __name__=="__main__":
    main()
