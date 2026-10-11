"""B0/B1 random; B2 conditional entropy; B3 coarse balance; B4 diversity.

Features and logits stay inside each serial client invocation. The controller
receives selected IDs only. Memory is O(ND), never a pairwise NxN matrix.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..training.dataset import ClientDataset

BASELINES = ("B0", "B1", "B2", "B3", "B4")


def conditional_entropy(logits, coarse, hierarchy):
    z = torch.as_tensor(logits, dtype=torch.float64)
    c = torch.as_tensor(coarse, dtype=torch.long)
    if z.shape != (len(c), 100) or ((c < 0) | (c >= 20)).any() or not torch.isfinite(z).all():
        raise ValueError("invalid public predictions/coarse labels")
    groups = torch.tensor(hierarchy.coarse_to_fine, dtype=torch.long, device=z.device)
    logp = z.gather(1, groups[c.to(z.device)]).log_softmax(1)
    return -(logp.exp() * logp).sum(1).cpu().numpy()


def allocate_quotas(capacities, count):
    """Proportional public eligible counts, largest remainder; exact global sum."""
    if count < 0 or count > sum(capacities.values()) or any(v < 0 for v in capacities.values()):
        raise ValueError("invalid query count")
    total = sum(capacities.values())
    if total == 0:
        return {k: 0 for k in capacities}
    quotas = {k: count * v // total for k, v in capacities.items()}
    for k in sorted(capacities, key=lambda k: (-(count * capacities[k] % total), k))[:count - sum(quotas.values())]:
        quotas[k] += 1
    return quotas


def choose_ids(strategy, ids, coarse, count, seed, hierarchy, *, logits=None, features=None,
               purchased_coarse=()):
    """Pure public-array interface; ids must already exclude purchased samples."""
    ids = np.asarray(ids, dtype=np.int64)
    coarse = np.asarray(coarse, dtype=np.int64)
    if strategy not in BASELINES or len(ids) != len(set(ids)) or coarse.shape != ids.shape:
        raise ValueError("invalid strategy/candidate IDs")
    if ((coarse < 0) | (coarse >= 20)).any() or not 0 <= count <= len(ids):
        raise ValueError("invalid coarse/count")
    if count == 0:
        return []
    rng = np.random.default_rng(seed)
    if strategy in ("B0", "B1"):
        selected = rng.choice(len(ids), count, replace=False)
    elif strategy == "B3":
        # Water-fill coarse coverage using bought samples' PUBLIC coarse labels.
        buckets = {int(c): list(rng.permutation(np.flatnonzero(coarse == c))) for c in np.unique(coarse)}
        coverage = np.bincount(np.asarray(purchased_coarse, dtype=int), minlength=20)
        selected = []
        for _ in range(count):
            available = [c for c, bucket in buckets.items() if bucket]
            lowest = min(coverage[c] for c in available)
            group = int(rng.choice([c for c in available if coverage[c] == lowest]))
            selected.append(buckets[group].pop())
            coverage[group] += 1
    else:
        entropy = conditional_entropy(logits, coarse, hierarchy)
        rank = np.lexsort((ids, -entropy))
        if strategy == "B2":
            selected = rank[:count]
        else:
            f = np.asarray(features, dtype=np.float64)
            if f.ndim != 2 or len(f) != len(ids) or not np.isfinite(f).all():
                raise ValueError("B4 requires finite local features")
            # Entropy shortlist, then farthest-first; one min-distance vector.
            shortlist = rank[:min(len(ids), max(count, 5 * count))]
            f = f[shortlist]
            f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
            distances = np.full(len(f), np.inf)
            chosen = []
            current = 0
            for _ in range(count):
                chosen.append(current)
                distances = np.minimum(distances, np.sum((f - f[current]) ** 2, axis=1))
                distances[chosen] = -np.inf
                current = int(np.argmax(distances))
            selected = shortlist[chosen]
    return [int(ids[i]) for i in selected]


def local_query(strategy, pool, client_ids, purchased, count, seed, model, hierarchy,
                device="cpu", batch_size=64):
    candidates = sorted(set(map(int, client_ids)) - set(purchased))
    if count == 0:
        return []
    public_coarse = pool.coarse_labels()
    coarse = np.array([public_coarse[i] for i in candidates], dtype=np.int64)
    logits = features = None
    if strategy in ("B2", "B4"):
        dataset = ClientDataset(pool, candidates, {}, hierarchy)
        logits_batches, feature_batches = [], []
        captured = []
        hook = model.avgpool.register_forward_hook(lambda _m, _i, out: captured.append(out.detach().flatten(1).cpu())) if strategy == "B4" else None
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                for images, *_ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
                    captured.clear()
                    logits_batches.append(model(images.to(device)).cpu())
                    if hook is not None:
                        feature_batches.append(captured[0])
            logits = torch.cat(logits_batches)
            features = torch.cat(feature_batches).numpy() if hook is not None else None
        finally:
            if hook is not None:
                hook.remove()
            model.train(was_training)
    bought_coarse = [int(public_coarse[i]) for i in client_ids if int(i) in purchased]
    return choose_ids(strategy, candidates, coarse, count, seed, hierarchy, logits=logits,
                      features=features, purchased_coarse=bought_coarse)
