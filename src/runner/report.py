from __future__ import annotations
import os
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
