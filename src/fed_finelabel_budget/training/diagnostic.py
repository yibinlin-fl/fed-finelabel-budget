"""Bounded central diagnostics on explicitly purchased training labels only."""
import torch
from torch.utils.data import DataLoader
from .seeds import seed_everything


def supervised_epoch(model, dataset, optimizer, *, seed, batch_size, device):
    if not len(dataset) or any(f < 0 for f in dataset.fine):
        raise ValueError("diagnostics require explicitly authorized fine labels")
    seed_everything(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model.train()
    total = examples = steps = 0
    for images, _coarse, fine, _ids in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images.to(device)), fine.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite diagnostic loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite diagnostic gradient")
        optimizer.step()
        total += float(loss.detach()) * len(images)
        examples += len(images)
        steps += 1
    return {"train_loss": total / examples, "examples": examples, "steps": steps}
