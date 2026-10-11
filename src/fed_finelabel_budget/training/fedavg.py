"""Serial FedAvg; no oracle, raw CIFAR reader, or evaluator dependency."""
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from .seeds import seed_everything


def aggregate_states(states, counts):
    if not states or len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("require nonempty states with positive sample counts")
    total = sum(counts)
    result = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            result[key] = sum(s[key] * (n / total) for s, n in zip(states, counts))
        else:
            # BatchNorm num_batches_tracked is a counter, not a model parameter.
            result[key] = torch.stack([s[key] for s in states]).amax(0)
    return result


def train_fedavg(model, clients, criterion, *, rounds, clients_per_round, local_epochs,
                 batch_size, learning_rate, max_steps, sampling_seed, training_seed,
                 device="cpu", start_round=0, fixed_steps=False, aggregation_counts=None, momentum=0.0):
    if not clients or not 1 <= clients_per_round <= len(clients):
        raise ValueError("invalid participating client count")
    if min(rounds, local_epochs, batch_size, max_steps) < 1 or learning_rate <= 0:
        raise ValueError("training limits and learning rate must be positive")
    if any(len(ds) == 0 for ds in clients.values()):
        raise ValueError("empty clients must be excluded explicitly")
    if not np.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite in [0,1)")
    model.to(device)
    criterion.to(device)
    rng = np.random.default_rng(sampling_seed)
    for _ in range(start_round):
        rng.choice(sorted(clients), clients_per_round, replace=False)
    history = []
    for round_index in range(start_round, start_round + rounds):
        selected = sorted(map(int, rng.choice(sorted(clients), clients_per_round, replace=False)))
        states, counts, records = [], [], []
        for cid in selected:
            seed = training_seed + round_index * 1000003 + cid
            seed_everything(seed)
            local = deepcopy(model).to(device).train()
            optimizer = torch.optim.SGD(local.parameters(), lr=learning_rate, momentum=momentum)
            generator = torch.Generator().manual_seed(seed)
            if fixed_steps:
                # Equal gradient/batch budgets including small fine-only datasets.
                sampler = RandomSampler(clients[cid], replacement=True,
                                        num_samples=batch_size * max_steps, generator=generator)
                loader = DataLoader(clients[cid], batch_size=batch_size, sampler=sampler, num_workers=0)
            else:
                loader = DataLoader(clients[cid], batch_size=batch_size, shuffle=True,
                                    generator=generator, num_workers=0)
            steps = examples = fine_examples = 0
            total_loss = 0.0
            for _ in range(local_epochs):
                for images, coarse, fine, _ids in loader:
                    images, coarse, fine = images.to(device), coarse.to(device), fine.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(local(images), coarse, fine)
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite local loss")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in local.parameters()):
                        raise RuntimeError("nonfinite gradient")
                    optimizer.step()
                    steps += 1
                    examples += len(images)
                    fine_examples += int((fine >= 0).sum())
                    total_loss += float(loss.detach()) * len(images)
                    if steps >= max_steps:
                        break
                if steps >= max_steps:
                    break
            states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            counts.append(len(clients[cid]) if aggregation_counts is None else aggregation_counts[cid])
            records.append({"client": cid, "dataset_samples": len(clients[cid]), "steps": steps,
                            "examples_seen": examples, "fine_examples_seen": fine_examples,
                            "coarse_examples_seen": examples - fine_examples,
                            "mean_loss": total_loss / examples})
            del local, optimizer
        model.load_state_dict(aggregate_states(states, counts))
        history.append({"round": round_index + 1, "clients": records,
                        "aggregation_weights": [n / sum(counts) for n in counts]})
    return history
