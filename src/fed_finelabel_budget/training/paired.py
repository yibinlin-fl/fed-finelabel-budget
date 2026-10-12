"""Auditable P2.6 serial FedAvg. Receives only client purchased snapshots."""
from copy import deepcopy
from collections import Counter
import hashlib
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from .fedavg import aggregate_states
from .losses import MixedSupervisionLoss
from .seeds import seed_everything

METHODS = ("B0", "B1", "PS_FINE", "PS_COARSE")


def supervision_terms(logits, coarse, fine, hierarchy, method, *, aggregation="sample_mean", raw_criterion=None):
    """Both paired arms have identical fine coefficients for identical batches.

    PS_FINE forward includes ALL inputs (BN policy), loss excludes unknowns.
    Empty-supervision batches return connected zero but optimizer must skip.
    """
    if method not in METHODS or aggregation not in ("sample_mean", "group_mean"):
        raise ValueError("invalid supervision protocol")
    if method == "B0" and (fine < 0).any():
        raise ValueError("B0 cannot sample unpurchased fine")
    known = fine >= 0
    # Use log-space validated hierarchy loss; this never indexes hidden truth.
    criterion = raw_criterion or MixedSupervisionLoss(hierarchy).to(logits.device)
    raw = criterion(logits, coarse, fine, reduction="none")
    fine_sum = raw[known].sum()
    use_coarse = method in ("B1", "PS_COARSE")
    coarse_sum = raw[~known].sum() if use_coarse else logits.new_zeros(())
    nf, nc = int(known.sum()), int((~known).sum()) if use_coarse else 0
    if aggregation == "sample_mean":
        fine_term, coarse_term = fine_sum / len(fine), coarse_sum / len(fine)
    else:
        fine_term, coarse_term = fine_sum / max(nf, 1), coarse_sum / max(nc, 1)
    return fine_term + coarse_term, {"fine_sum": float(fine_sum.detach()),
        "coarse_sum": float(coarse_sum.detach()), "fine_component": float(fine_term.detach()),
        "coarse_component": float(coarse_term.detach()), "fine_count": nf, "coarse_count": nc}


def bn_statistics(model):
    means, variances, counters = [], [], []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
            means.append(module.running_mean.detach().float().cpu())
            variances.append(module.running_var.detach().float().cpu())
            counters.append(int(module.num_batches_tracked))
    if not variances:
        return {"modules": 0, "finite": True, "negative_variances": 0, "counter_min": None, "counter_max": None,
                "mean_max_abs": None, "variance_min": None, "variance_max": None}
    m,v = torch.cat(means),torch.cat(variances)
    return {"modules":len(variances),"finite":bool(torch.isfinite(m).all() and torch.isfinite(v).all()),
            "negative_variances":int((v<0).sum()),"mean_max_abs":float(m.abs().max()),
            "variance_min":float(v.min()),"variance_max":float(v.max()),
            "counter_min":min(counters),"counter_max":max(counters)}


def train_round(model, clients, hierarchy, method, *, round_index, clients_per_round,
                batch_size, max_steps, learning_rate, momentum, sampling_seed, training_seed,
                device="cpu", aggregation="sample_mean", aggregation_counts=None):
    if method not in METHODS or not clients or any(not len(ds) for ds in clients.values()):
        raise ValueError("invalid/nonempty clients required")
    if not 1 <= clients_per_round <= len(clients) or min(batch_size,max_steps)<1:
        raise ValueError("invalid training limits")
    if not np.isfinite(learning_rate) or learning_rate<=0 or not np.isfinite(momentum) or not 0<=momentum<1:
        raise ValueError("invalid optimizer")
    rng=np.random.default_rng(sampling_seed)
    for _ in range(round_index+1):
        selected=sorted(map(int,rng.choice(sorted(clients),clients_per_round,replace=False)))
    model.to(device)
    criterion=MixedSupervisionLoss(hierarchy).to(device)
    if device=="cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start=time.perf_counter()
    states,weights,records=[],[],[]
    for cid in selected:
        seed=training_seed+round_index*1000003+cid
        seed_everything(seed)
        local=deepcopy(model).to(device).train()
        optimizer=torch.optim.SGD(local.parameters(),lr=learning_rate,momentum=momentum)
        sampler=RandomSampler(clients[cid],replacement=True,num_samples=batch_size*max_steps,
                              generator=torch.Generator().manual_seed(seed))
        loader=DataLoader(clients[cid],batch_size=batch_size,sampler=sampler,num_workers=0)
        stream=hashlib.sha256()
        all_ids, fine_ids = Counter(), Counter()
        stats={"client":cid,"dataset_samples":len(clients[cid]),"batches":0,"examples_seen":0,
               "fine_examples_seen":0,"coarse_examples_seen":0,"unsupervised_examples_seen":0,
               "backward_calls":0,"optimizer_steps":0,"nonzero_gradient_steps":0,
               "parameter_update_steps":0,"zero_gradient_batches":0,"fine_loss_sum":0.,
               "coarse_loss_sum":0.,"loss_sum":0.,"fine_component_sum":0.,"coarse_component_sum":0.}
        for x,c,f,ids in loader:
            stream.update(ids.numpy().astype("<i8").tobytes())
            all_ids.update(map(int,ids))
            fine_ids.update(map(int,ids[f>=0]))
            x,c,f=x.to(device),c.to(device),f.to(device)
            optimizer.zero_grad(set_to_none=True)
            # Both paired arms run every image through train-mode BN, including
            # unpurchased images. BN state evolves differently only via weights.
            z=local(x)
            loss,terms=supervision_terms(z,c,f,hierarchy,method,aggregation=aggregation,raw_criterion=criterion)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            nonzero=False
            if terms["fine_count"]+terms["coarse_count"]:
                loss.backward()
                stats["backward_calls"]+=1
                for p in local.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise RuntimeError("nonfinite gradient")
                        nonzero=nonzero or bool(torch.count_nonzero(p.grad))
            if nonzero:
                stats["nonzero_gradient_steps"]+=1
            else:
                stats["zero_gradient_batches"]+=1
            if terms["fine_count"]+terms["coarse_count"]:
                # Preserve normal SGD (including momentum on zero gradients)
                # whenever supervision exists, exactly as original B0/B1.
                before=[p.detach().clone() for p in local.parameters()]
                optimizer.step()
                stats["optimizer_steps"]+=1
                stats["parameter_update_steps"]+=int(any(not torch.equal(a,p.detach()) for a,p in zip(before,local.parameters())))
                del before
            # Do NOT let old momentum move a batch with no supervision.
            stats["batches"]+=1
            stats["examples_seen"]+=len(f)
            stats["fine_examples_seen"]+=terms["fine_count"]
            stats["coarse_examples_seen"]+=terms["coarse_count"]
            stats["unsupervised_examples_seen"]+=len(f)-terms["fine_count"]-terms["coarse_count"]
            stats["fine_loss_sum"]+=terms["fine_sum"]
            stats["coarse_loss_sum"]+=terms["coarse_sum"]
            stats["loss_sum"]+=float(loss.detach())*len(f)
            stats["fine_component_sum"]+=terms["fine_component"]*len(f)
            stats["coarse_component_sum"]+=terms["coarse_component"]*len(f)
        stats.update({"stream_hash":stream.hexdigest(),"unique_examples_seen":len(all_ids),
            "repeated_exposures":stats["examples_seen"]-len(all_ids),
            "fine_exposure_hash":hashlib.sha256(str(sorted(fine_ids.items())).encode()).hexdigest(),
            "unique_fine_seen":len(fine_ids),"bn":bn_statistics(local)})
        states.append({k:v.detach().cpu().clone() for k,v in local.state_dict().items()})
        weights.append(len(clients[cid]) if aggregation_counts is None else aggregation_counts[cid])
        records.append(stats)
        del local,optimizer
    model.load_state_dict(aggregate_states(states,weights))
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise RuntimeError("nonfinite global state")
    if device=="cuda":
        torch.cuda.synchronize()
    examples=sum(c["examples_seen"] for c in records)
    result={"round":round_index+1,"method":method,"clients":records,"aggregation_weights":[n/sum(weights) for n in weights],
        "learning_rate":learning_rate,"momentum":momentum,"optimizer_policy":"reset_each_client_round_skip_no_supervision",
        "bn_policy":"train_forward_all_sampled_images","loss_protocol":aggregation,
        "train_loss":sum(c["loss_sum"] for c in records)/examples,
        "fine_loss_component":sum(c["fine_component_sum"] for c in records)/examples,
        "coarse_loss_component":sum(c["coarse_component_sum"] for c in records)/examples,
        "fine_ce_mean":sum(c["fine_loss_sum"] for c in records)/max(1,sum(c["fine_examples_seen"] for c in records)),
        "coarse_loss_mean":sum(c["coarse_loss_sum"] for c in records)/max(1,sum(c["coarse_examples_seen"] for c in records)),
        "bn":bn_statistics(model),"training_seconds":time.perf_counter()-start,
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None,
        "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved() if device=="cuda" else None}
    for key in ("examples_seen","fine_examples_seen","coarse_examples_seen","unsupervised_examples_seen",
                "batches","backward_calls","optimizer_steps","nonzero_gradient_steps","parameter_update_steps","zero_gradient_batches"):
        result[key]=sum(c[key] for c in records)
    return result
