from __future__ import annotations
import os
import matplotlib.pyplot as plt
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Terminal table — printed after every run using the `rich` library.
# Shows per-prompt results and a summary row.
# ---------------------------------------------------------------------------

def print_rich_table(outputs: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        print("[report] `rich` not installed — skipping terminal table. Run: pip install rich")
        return

    console = Console()

    # ------------------------------------------------------------------
    # Per-prompt results table
    # ------------------------------------------------------------------
    table = Table(
        title="Prompt Results",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("ID",         style="cyan",  no_wrap=True)
    table.add_column("Bucket",     style="magenta")
    table.add_column("LNS Score",  justify="right")
    table.add_column("Correct?",   justify="center")
    table.add_column("Label",      style="bold")

    # Colour map for bucket labels
    label_styles = {
        "correct_certain":    "green",
        "correct_uncertain":  "yellow",
        "incorrect_certain":  "red bold",   # overconfident hallucination — most dangerous
        "incorrect_uncertain": "red",
        "unscored":           "dim",
    }

    for entry in outputs:
        label     = entry.get("bucket_label", "unscored")
        is_correct = entry.get("is_correct")
        lns       = entry.get("lns_score", "n/a")

        correct_str = "✓" if is_correct else ("✗" if is_correct is False else "—")
        correct_style = "green" if is_correct else ("red" if is_correct is False else "dim")

        table.add_row(
            entry.get("id", ""),
            entry.get("bucket", ""),
            f"{lns:.4f}" if isinstance(lns, float) else str(lns),
            f"[{correct_style}]{correct_str}[/{correct_style}]",
            f"[{label_styles.get(label, '')}]{label}[/{label_styles.get(label, '')}]",
        )

    console.print(table)

    # ------------------------------------------------------------------
    # Summary metrics panel
    # ------------------------------------------------------------------
    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Prompts scored       : {metrics.get('num_scored')} / {metrics.get('num_prompts')}")
    console.print(f"  Accuracy             : {metrics.get('accuracy', 'n/a')}")
    console.print(f"  Hallucination rate   : {metrics.get('hallucination_rate', 'n/a')}")
    console.print(
        f"  [red bold]Overconfident halluc : {metrics.get('overconfident_hallucination_rate', 'n/a')}[/red bold]"
    )
    console.print(f"  Avg LNS score        : {metrics.get('avg_lns_score', 'n/a')}")
    console.print(f"  correct_certain      : {metrics.get('correct_certain', 0)}")
    console.print(f"  correct_uncertain    : {metrics.get('correct_uncertain', 0)}")
    console.print(f"  incorrect_certain    : {metrics.get('incorrect_certain', 0)}")
    console.print(f"  incorrect_uncertain  : {metrics.get('incorrect_uncertain', 0)}")


# ---------------------------------------------------------------------------
# Charts — saved as PNG to outputs/<run_id>/charts/
# Requires matplotlib.
# ---------------------------------------------------------------------------

def extract_binary_labels_scores(outputs: List[Dict[str, Any]]):
    y_true = []
    y_score = []

    for entry in outputs:
        is_correct = entry.get("is_correct")
        lns_score = entry.get("lns_score")

        if is_correct is None or lns_score is None:
            continue

        if not isinstance(lns_score, (int,float)):
            continue

        y_true.append(0 if is_correct else 1)
        y_score.append(-float(lns_score))

    return y_true, y_score

def binary_clf_curve(y_true: List[int], y_score: List[float]):
    pairs = sorted(zip(y_true,y_score),key=lambda x : x[0], reverse=True)
    fps, tps, thresholds = [], [], []
    tp, fp = 0.0, 0.0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            fps.append(fp)
            tps.append(tp)
            thresholds.append(prev_score)

        if label == 1:
            tp += 1.0
        else:
            fp += 1.0

        prev_score = score

    if prev_score is not None:
        fps.append(fp)
        tps.append(tp)
        thresholds.append(prev_score)
    
    return fps, tps, thresholds

def roc_points(y_true: List[int], y_score: List[float]):
    if not y_true or len(set(y_true)) < 2:
        return [], []

    fps, tps, _ = binary_clf_curve(y_true,y_score)
    pos = sum(y_true)
    neg = len(y_true) - pos

    if pos == 0 or neg == 0:
        return [], []
    
    fpr =  [0.0] + [fp/neg for fp in fps]
    tpr = [0.0] + [tp / pos for tp in tps]

    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr.append(1.0)
        tpr.append(1.0)
    
    return fpr, tpr

def pr_points(y_true: List[int], y_score: List[float]):
    if not y_true or len(set(y_true)) < 2:
        return [], []
    
    pairs = sorted(zip(y_score, y_true), key=lambda x : x[0], reverse=True)
    total_pos = sum(y_true)

    if total_pos == 0:
        return [], []
    
    tp = 0.0
    fp = 0.0
    recalls = [0.0]
    precisions = [1.0]

    for _, label in pairs:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_pos if total_pos > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
    
    return recalls, precisions

def precision_fraction(sorted_correct: List[int], keep_fraction: float):
    n = len(sorted_correct)
    if n == 0:
        return 0.0
    
    k = max(1, int(round(n* keep_fraction)))
    kept = sorted_correct[:k]
    return sum(kept) / len(kept)

def rejection_curve_points(y_true: List[int], y_score: List[float], num_points = 51):
    if not y_true:
        return [], []
    
    correctness = [1 - y for y in y_true]
    pairs_unc = sorted(zip(y_score, correctness), key=lambda x: x[0])
    sorted_correct = [c for _, c in pairs_unc]
    rejection_rates = []
    retained_accuracies = []
    for i in range(num_points):
        reject_rate = i / (num_points - 1)
        keep_fraction = 1.0 - reject_rate
        acc = precision_fraction(sorted_correct,keep_fraction)
        rejection_rates.append(reject_rate)
        retained_accuracies.append(acc)

    return rejection_rates, retained_accuracies

def save_roc_curve(outputs: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: str) -> None:
    y_true, y_score = extract_binary_labels_scores(outputs)
    fpr , tpr = roc_points(y_true,y_score)

    auroc = metrics.get("auroc",None)
    fig, ax = plt.subplots(figsize=(7,5))
    label = f"ROC Curve = {auroc}"
    ax.plot(fpr,tpr,linewidth=2, label=label)
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="random")
    ax.set_title("ROC Curve: Hallucination Detection")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir,"roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[report] Saved ROC Curve: {path}")


def save_pr_curve(outputs: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: str) -> None:
    y_true, y_score = extract_binary_labels_scores(outputs)
    recalls , precisions = pr_points(y_true,y_score)

    auprc = metrics.get("auprc",None)
    positive_rate = sum(y_true) / len(y_true)
    fig, ax = plt.subplots(figsize=(7,5))
    label = f"PR Curve = {auprc}"
    ax.plot(recalls,precisions,linewidth=2, label=label)
    # ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="random")
    ax.axhline(y=positive_rate, linestyle="--", linewidth=1, label=f"random baseline {positive_rate}")
    ax.set_title("Precision-Recall Curve: Hallucination Detection")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir,"pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[report] Saved PR Curve: {path}")


def save_rejection_curve(outputs: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: str) -> None:
    y_true, y_score = extract_binary_labels_scores(outputs)
    rejection_rates , retained_accuracies= rejection_curve_points(y_true,y_score)

    base_accuracy = sum(1 - y for y in y_true) / len(y_true)
    prr = metrics.get("prr", None)

    fig, ax = plt.subplots(figsize=(7,5))
    label = f"Rejection Curve = {prr}"
    ax.plot(rejection_rates,retained_accuracies,linewidth=2, label=label)
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="random")
    ax.axhline(y=base_accuracy, linestyle="--", linewidth=1, label=f"no rejection baseline {base_accuracy}")
    ax.set_title("Precision-Rejection Curve: Hallucination Detection")
    ax.set_xlabel("Rejection Rate")
    ax.set_ylabel("Accuracy of retained predictions")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out_dir,"rejection_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[report] Saved Rejection Curve: {path}")



def save_charts(outputs: List[Dict[str, Any]], metrics: Dict[str, Any], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe on headless machines
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[report] `matplotlib` not installed — skipping charts. Run: pip install matplotlib")
        return

    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Chart 1: 4-bucket bar chart
    # Shows how many prompts fell into each correctness×certainty category.
    # The `incorrect_certain` bar is the key signal: it represents
    # overconfident hallucinations — wrong answers the model was confident about.
    # This count is expected to increase when the model is heavily pruned.
    # ------------------------------------------------------------------
    labels  = ["correct\ncertain", "correct\nuncertain", "incorrect\ncertain", "incorrect\nuncertain"]
    keys    = ["correct_certain",  "correct_uncertain",  "incorrect_certain",  "incorrect_uncertain"]
    colors  = ["#2ecc71",          "#f1c40f",            "#e74c3c",            "#e67e22"]
    counts  = [metrics.get(k, 0) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)

    # Annotate bar heights
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(count),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_title("4-Bucket Outcome Counts", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of prompts")
    ax.set_ylim(0, max(counts) + 1.5 if counts else 2)
    ax.axvline(x=1.5, color="grey", linestyle="--", linewidth=0.8)   # divider: correct | incorrect

    # Legend note for the dangerous bucket
    danger_patch = mpatches.Patch(color="#e74c3c", label="⚠ incorrect_certain = overconfident hallucination")
    ax.legend(handles=[danger_patch], loc="upper right", fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(charts_dir, "bucket_counts.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"[report] Saved chart → {path1}")

    # ------------------------------------------------------------------
    # Chart 2: LNS score per prompt (horizontal bar)
    # Each bar represents one prompt's avg_logprob (LNS score).
    # Bars are coloured by bucket_label so you can see at a glance
    # which prompts the model was confident about and whether it was right.
    # The vertical dashed line shows the lns_threshold (certainty boundary).
    # ------------------------------------------------------------------
    label_colors = {
        "correct_certain":    "#2ecc71",
        "correct_uncertain":  "#f1c40f",
        "incorrect_certain":  "#e74c3c",
        "incorrect_uncertain": "#e67e22",
        "unscored":           "#95a5a6",
    }

    ids       = [e.get("id", f"p{i}") for i, e in enumerate(outputs)]
    lns_vals  = [e.get("lns_score", 0.0) for e in outputs]
    bar_colors = [label_colors.get(e.get("bucket_label", "unscored"), "#95a5a6") for e in outputs]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ids) * 0.45)))
    y_pos = range(len(ids))
    ax.barh(list(y_pos), lns_vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(ids, fontsize=9)
    ax.invert_yaxis()   # top = first prompt

    # Threshold line
    lns_threshold = -2.0   # matches default in metrics.py / yaml
    ax.axvline(x=lns_threshold, color="black", linestyle="--", linewidth=1.0, label=f"lns_threshold ({lns_threshold})")

    ax.set_xlabel("avg_logprob (LNS score)  ← uncertain | confident →")
    ax.set_title("LNS Score per Prompt (coloured by outcome)", fontsize=13, fontweight="bold")

    legend_patches = [mpatches.Patch(color=c, label=l) for l, c in label_colors.items()]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    path2 = os.path.join(charts_dir, "lns_per_prompt.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"[report] Saved chart → {path2}")


    save_roc_curve(outputs=outputs, metrics=metrics, out_dir=charts_dir)
    save_pr_curve(outputs=outputs, metrics=metrics, out_dir=charts_dir)
    save_rejection_curve(outputs=outputs, metrics=metrics, out_dir=charts_dir)
