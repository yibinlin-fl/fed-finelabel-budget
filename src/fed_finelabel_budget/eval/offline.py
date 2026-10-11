import numpy as np
import torch
from .metrics import classification_metrics


class OfflineEvaluator:
    """Truth encapsulated here. Returns only metrics, never labels or predictions."""
    def __init__(self, images, fine, coarse):
        self.__images = np.asarray(images).copy()
        self.__fine = np.asarray(fine).copy()
        self.__coarse = np.asarray(coarse).copy()
        if not len(images) or len(images) != len(fine) or len(images) != len(coarse):
            raise ValueError("invalid offline evaluation split")

    def label_distribution(self):
        """Offline audit only; never supplied to partition/training/query APIs."""
        return {"samples": len(self.__fine),
                "fine_histogram": np.bincount(self.__fine, minlength=100).tolist(),
                "coarse_histogram": np.bincount(self.__coarse, minlength=20).tolist()}

    def evaluate(self, model, device="cpu", batch_size=64, *, diagnostics=False):
        predictions = []
        total_ce = total_entropy = 0.0
        was_training = model.training
        try:
            model.eval()
            with torch.inference_mode():
                for start in range(0, len(self.__images), batch_size):
                    images = torch.from_numpy(self.__images[start:start + batch_size].transpose(0, 3, 1, 2).copy()).float()
                    z = model(images.div(255).sub(.5).div(.5).to(device))
                    if not torch.isfinite(z).all():
                        raise RuntimeError("nonfinite evaluation logits")
                    predictions.extend(z.argmax(1).cpu().tolist())
                    if diagnostics:
                        y = torch.tensor(self.__fine[start:start + batch_size], device=device)
                        total_ce += float(torch.nn.functional.cross_entropy(z, y, reduction="sum"))
                        lp = z.log_softmax(1)
                        total_entropy += float(-(lp.exp() * lp).sum())
        finally:
            model.train(was_training)
        metrics = classification_metrics(self.__fine, predictions, self.__coarse)
        if diagnostics:
            metrics.update({"fine_ce": total_ce / len(predictions), "prediction_entropy": total_entropy / len(predictions),
                            "prediction_histogram": np.bincount(predictions, minlength=100).tolist()})
        return metrics
