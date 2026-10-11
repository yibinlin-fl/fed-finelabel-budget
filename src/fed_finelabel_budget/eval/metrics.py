import numpy as np


def classification_metrics(truth, predictions, coarse):
    truth, predictions, coarse = [np.asarray(x, dtype=np.int64) for x in (truth, predictions, coarse)]
    if truth.ndim != 1 or not len(truth) or truth.shape != predictions.shape or truth.shape != coarse.shape:
        raise ValueError("invalid evaluation arrays")
    if ((truth < 0) | (truth >= 100) | (predictions < 0) | (predictions >= 100)).any():
        raise ValueError("fine labels outside 0..99")
    if ((coarse < 0) | (coarse >= 20)).any():
        raise ValueError("coarse labels outside 0..19")
    confusion = np.bincount(truth * 100 + predictions, minlength=10000).reshape(100, 100)
    tp = confusion.diagonal()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(100, dtype=float), where=denominator != 0)
    predicted_counts = np.bincount(predictions, minlength=100)
    return {"accuracy": float(np.mean(truth == predictions)), "macro_f1": float(f1.mean()),
            "samples": len(truth), "fine_classes_present": int(np.count_nonzero(confusion.sum(1))),
            "predicted_classes": int(np.count_nonzero(predicted_counts)),
            "max_prediction_fraction": float(predicted_counts.max() / len(truth)),
            "coarse_accuracy": {str(c): float(np.mean(truth[coarse == c] == predictions[coarse == c]))
                                if np.any(coarse == c) else None for c in range(20)}}


def client_budget_statistics(partitions, purchased, coarse_labels=None):
    result = {}
    for cid, ids in sorted(partitions.items()):
        bought = [int(i) for i in ids if int(i) in purchased]
        result[str(cid)] = {"samples": len(ids), "purchased": len(bought)}
        if coarse_labels is not None:
            result[str(cid)]["purchased_by_coarse"] = np.bincount(np.asarray(coarse_labels)[bought], minlength=20).tolist()
    return result
