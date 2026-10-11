"""Scientific diagnostic curves from actually produced artifacts only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", default="results/p25_diagnostic")
    parser.add_argument("--baselines", nargs="*")
    parser.add_argument("--central-control")
    parser.add_argument("--output", default="reports/P2_5_CURVES.png")
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    if args.baselines:
        data = [(Path(p).name, json.loads((Path(p)/"summary.json").read_text())["rounds"]) for p in args.baselines]
    else:
        data = [(n, json.loads((Path(args.diagnostics)/n/"history.json").read_text())) for n in ("overfit","central","fedavg")]
        if args.central_control:
            data.append(("central lr=.02",json.loads((Path(args.central_control)/"central/history.json").read_text())))
    for name, rows in data:
        x = [r.get("iteration",r.get("communication_rounds")) for r in rows]
        axes[0,0].plot(x, [r["train"]["accuracy"] if "train" in r else float("nan") for r in rows], label=name)
        axes[0,1].plot(x, [r["validation"]["accuracy"] for r in rows], label=name)
        axes[1,0].plot(x, [r["validation"]["macro_f1"] for r in rows], label=name)
        axes[1,1].plot(x, [r["validation"]["fine_ce"] for r in rows], label=name)
        axes[2,0].plot(x, [r.get("train_loss",float("nan")) for r in rows], label=name)
        axes[2,1].plot(x, [r["validation"]["prediction_entropy"] for r in rows], label=name)
    axes[1,1].set_yscale("log")
    for ax,title in zip(axes.flat,["Authorized train accuracy","Validation accuracy","Validation Macro-F1 (100 classes)","Validation fine CE (log scale)","Sample-weighted training loss","Validation prediction entropy (nats)"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch (central) / communication round (FedAvg)")
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("P2.5 engineering diagnostics — independent full-fine authorization" if not args.baselines else "B0/B1 single-seed budget pilot — validation only")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output,dpi=160)


if __name__ == "__main__":
    main()
