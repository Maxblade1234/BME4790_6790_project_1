"""
compare_results.py

Combine the 2-, 3- and 4-layer CNN results into report-ready tables and figures.

Looks for each model's outputs in results/cnn_{N}_layer/ (falls back to
results/ for the 2-layer run, which was committed there). Models whose
results are missing are skipped.

Outputs (results/comparison/):
    comparison_table.csv   accuracy, macro P/R/F1, params, best epoch, train time
    per_class_f1.csv       per-class F1 for every model
    confusion_matrices.png row-normalized confusion matrices side by side
    training_curves.png    loss and accuracy vs epoch for every model

Figures have no in-plot titles (put titles in the report captions).

Run:
    python compare_results.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results")
OUT = RESULTS / "comparison"
DEPTHS = [2, 3, 4]
CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast", "Immature Granulocyte",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet",
]
SHORT = ["BA", "EO", "ERB", "IG", "LY", "MO", "NE", "PLT"]


def find_dir(depth):
    candidates = [RESULTS / f"cnn_{depth}_layer"]
    if depth == 2:
        candidates.append(RESULTS)
    for d in candidates:
        if (d / "metrics.json").exists():
            return d
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = {d: find_dir(d) for d in DEPTHS}
    models = {d: p for d, p in models.items() if p is not None}
    if not models:
        raise SystemExit("No results found.")
    print("Found:", {d: str(p) for d, p in models.items()})

    rows, f1_cols = [], {}
    for depth, d in models.items():
        m = json.loads((d / "metrics.json").read_text())
        s_path = d / "training_summary.json"
        s = json.loads(s_path.read_text()) if s_path.exists() else {}
        rows.append({
            "Model": f"{depth}-layer CNN",
            "Test accuracy (%)": round(100 * m["accuracy"], 2),
            "Macro precision": round(m["macro_precision"], 4),
            "Macro recall": round(m["macro_recall"], 4),
            "Macro F1": round(m["macro_f1"], 4),
            "Test loss": round(m["loss"], 4),
            "Trainable parameters": s.get("trainable_parameters"),
            "Best epoch": s.get("best_epoch"),
            "Epochs run": s.get("epochs_completed"),
            "Training time (h)": round(s["training_time_seconds"] / 3600, 2)
            if "training_time_seconds" in s else None,
        })
        cr = pd.read_csv(d / "classification_report.csv")
        cr = cr[cr["class"].isin(CLASS_NAMES)].set_index("class")
        f1_cols[f"{depth}-layer"] = cr["f1_score"].round(4)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison_table.csv", index=False)
    f1 = pd.DataFrame(f1_cols).reindex(CLASS_NAMES)
    f1.to_csv(OUT / "per_class_f1.csv")
    print("\n", table.to_string(index=False))
    print("\nPer-class F1\n", f1.to_string())

    # Confusion matrices (row-normalized = per-class recall)
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), squeeze=False)
    for ax, (depth, d) in zip(axes[0], models.items()):
        cm = np.loadtxt(d / "confusion_matrix.csv", delimiter=",")
        cmn = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(8), SHORT)
        ax.set_yticks(range(8), SHORT)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.text(0.5, 1.03, f"({chr(96 + list(models).index(depth) + 1)}) {depth}-layer",
                transform=ax.transAxes, ha="center")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8,
                 label="Fraction of true class")
    fig.savefig(OUT / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Training curves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for depth, d in models.items():
        h = pd.read_csv(d / "history.csv")
        line, = a1.plot(h["epoch"], h["train_loss"], label=f"{depth}-layer train")
        a1.plot(h["epoch"], h["val_loss"], "--", color=line.get_color(),
                label=f"{depth}-layer val")
        a2.plot(h["epoch"], h["train_accuracy"], color=line.get_color(),
                label=f"{depth}-layer train")
        a2.plot(h["epoch"], h["val_accuracy"], "--", color=line.get_color(),
                label=f"{depth}-layer val")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Cross-entropy loss (unitless)")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Accuracy (%)")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {OUT}/")


if __name__ == "__main__":
    main()
