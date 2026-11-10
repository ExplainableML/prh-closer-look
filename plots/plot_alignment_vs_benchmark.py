import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from scipy.stats import linregress

from plots import common
from src.utils import CAPTION_TYPES, to_modality_pair_name
from src import paths

# metric reported per benchmark task (key in results[task] of the lm-eval json)
TASK_METRICS = {
    "mmlu": "acc,none",
    "arc_challenge": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
}

# x-axis label per benchmark task (falls back to the upper-cased task name)
TASK_LABELS = {
    "arc_challenge": "ARC",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
}

# substrings shortened or removed in the model names written on the plot
DISPLAY_NAME_REPLACEMENTS = [
    ("DeepSeek", "DS"),
    ("-pretrain-lvd1689m", ""),
    ("-patch14-224", ""),
]


def display_name(model):
    """Shortened model name for plot labels."""
    for old, new in DISPLAY_NAME_REPLACEMENTS:
        model = model.replace(old, new)
    return model


def normalize_model_name(name: str) -> str:
    """
    Convert model names between benchmark/alignment naming schemes.

    Example:
        bigscience_bloomz-1b1 -> bloomz-1b1
        meta-llama_Llama-2-7b-hf -> llama-2-7b-hf
    """

    # remove org prefixes
    if "_" in name:
        name = name.split("_", 1)[1]

    # optional cleanup
    name = name.replace("/", "_")

    return name


# alignment CSV written by measure_alignment.py, e.g.
#   model_x=<lang>_model_y=<vision>.csv           layer-wise scores
#   model_x=<lang>_model_y=<vision>_summary.csv   calibrated score
ALIGNMENT_FILE_RE = re.compile(
    r"^model_x=(?P<model_x>.+?)_model_y=(?P<model_y>.+?)"
    r"(?P<seed>_seed=\d+)?(?P<summary>_summary)?\.csv$"
)

LAYER_MODES = ("final", "max", "max calibrated")

# look of each layer mode in the plot (drawn in the order of LAYER_MODES)
LAYER_MODE_STYLES = {
    "final": {"color": "#ff7f0f", "linestyle": "-", "filled": True},
    "max": {"color": "#1f78b4", "linestyle": "--", "filled": False},
    "max calibrated": {"color": "#1f78b4", "linestyle": "-", "filled": True},
}


def load_benchmark_results(benchmark_dir, task):
    """
    Load benchmark scores from benchmark_results directory, using the
    metric of the task given in TASK_METRICS.

    Expected filename format:
        bloomz-1b1_mmlu_*.json
    """

    metric_key = TASK_METRICS[task]

    benchmark_dir = Path(benchmark_dir)

    scores = {}

    for file in benchmark_dir.glob(f"*_{task}_*.json"):

        model_name = file.name.split(f"_{task}_")[0]
        if "qwen3.5" in model_name.lower() or "gemma-3" in model_name.lower():
            continue

        with open(file, "r") as f:
            data = json.load(f)

        try:
            score = data["results"][task][metric_key]
        except KeyError:
            print(f"Skipping {file} (missing metric)")
            continue

        scores[model_name] = score

    return scores


def load_alignment_results(
    alignment_root,
    dataset,
    fixed_model,
    layer_mode,
    random_init=False,
    pca_mode=None,
    caption_type=None,
    metric="mutual_knn",
    topk=10,
):
    """
    Alignment scores of every language model against the fixed vision model.

    layer_mode:
        "final"          -> score of the last layer pair
        "max"            -> max score over all layer pairs
        "max calibrated" -> calibrated score from the *_summary.csv files

    Returns {language_model: [score per seed directory]}.
    """

    if layer_mode not in LAYER_MODES:
        raise ValueError(f"layer_mode must be one of {LAYER_MODES}, got {layer_mode!r}")

    mode = "random_init_both" if random_init else "pretrained"
    language_dir = to_modality_pair_name("language", "vision", captions_x=caption_type)
    use_summary = layer_mode == "max calibrated"

    scores = {}

    for _, run_dir in common.find_runs(alignment_root, dataset):
        alignment_dir = run_dir / language_dir / f"pca={pca_mode}" / metric / f"topk={topk}" / mode

        for file in alignment_dir.glob("*.csv"):
            match = ALIGNMENT_FILE_RE.match(file.name)
            if match is None or match["model_y"] != fixed_model:
                continue

            # calibrated scores live in *_summary.csv, layer-wise scores in the others
            if bool(match["summary"]) != use_summary:
                continue

            df = pd.read_csv(file)

            if layer_mode == "final":
                score = common.select_layer_row(df, "last")["score"]
            elif layer_mode == "max":
                score = common.select_layer_row(df, "best")["score"]
            else:
                score = df["calibrated_score"].iloc[0]

            model_name = normalize_model_name(match["model_x"])
            scores.setdefault(model_name, []).append(score)

    return scores


def spread_positions(x, min_gap):
    """
    Label positions close to x with neighbouring labels at least min_gap apart.

    Overlapping labels are merged into clusters; each cluster is laid out with
    spacing min_gap and centred on the mean x of its members. Returns the
    positions in the original order of x.
    """
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]

    # each cluster: [indices into xs]
    clusters = [[i] for i in range(len(xs))]

    def layout(cluster):
        center = xs[cluster].mean()
        offsets = (np.arange(len(cluster)) - (len(cluster) - 1) / 2) * min_gap
        return center + offsets

    merged = True
    while merged:
        merged = False
        for c in range(len(clusters) - 1):
            if layout(clusters[c + 1])[0] - layout(clusters[c])[-1] < min_gap:
                clusters[c] = clusters[c] + clusters.pop(c + 1)
                merged = True
                break

    positions = np.empty(len(xs))
    for cluster in clusters:
        positions[cluster] = layout(cluster)

    out = np.empty(len(xs))
    out[order] = positions
    return out


def annotate_model_names(ax, models, x, fontsize=14):
    """
    Write model names (rotated) above the plot, spread out so they do not
    overlap, with a dotted line at each model's actual x position.
    """
    # minimum label distance: one rotated line of text, converted to data units
    fig = ax.get_figure()
    gap_px = 1.2 * fontsize * fig.dpi / 72
    x_min, x_max = ax.get_xlim()
    min_gap = gap_px * (x_max - x_min) / ax.get_window_extent().width

    label_x = spread_positions(x, min_gap)

    for model, xi, li in zip(models, x, label_x):
        ax.axvline(
            x=xi,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.4,
            zorder=1,
        )

        ax.annotate(
            display_name(model),
            xy=(li, 1.02),
            xycoords=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            alpha=0.9,
            rotation=90,
        )


def create_scatter_plot(
    ax,
    benchmark_scores,
    alignment_scores,
    label=None,
    color=None,
    linestyle="-",
    filled=True,
    annotate_models=False,
):
    """
    Draw one group of models on ax:
        x-axis = benchmark score
        y-axis = mean alignment over seeds (+/- 1 std)
    with a linear regression line. Axis styling is done by style_axes.

    Returns False (and draws nothing) if fewer than 2 models have both scores.
    """

    common_models = sorted(
        set(benchmark_scores.keys()) & set(alignment_scores.keys())
    )

    print(f"Found {len(common_models)} matching models")

    if len(common_models) < 2:
        print(f"Warning: skipping {label or 'group'} (needs at least 2 matching models)")
        return False

    x = np.array([benchmark_scores[m] for m in common_models])
    y, yerr = np.array([
        common.mean_std(alignment_scores[m])
        for m in common_models
    ]).T

    # Linear regression; r and p are the Pearson correlation and its p-value
    fit = linregress(x, y)
    y_pred = fit.intercept + fit.slope * x

    print(
        f"{label or ''} Pearson r = {fit.rvalue:.4f}, "
        f"p = {fit.pvalue:.4g}"
    )

    sort_idx = np.argsort(x)

    ax.plot(
        x[sort_idx],
        y_pred[sort_idx],
        linestyle=linestyle,
        color=color,
        label="_nolegend_",
    )

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        markersize=8,
        elinewidth=2,
        capsize=3,
        label=f"{label or 'data'}",
        color=color,
        ecolor=to_rgba(color, alpha=0.5),
        mfc=(to_rgba(color, alpha=0.7) if filled else "none"),
    )

    # after plotting, so the x-axis limits used for label spacing are final
    if annotate_models:
        annotate_model_names(ax, common_models, x)

    print(f"{label or ''} R^2 = {fit.rvalue ** 2:.4f}")

    return True


def style_axes(ax, task):
    """Labels, grid, legend and tick sizes; call once after all groups are drawn."""
    ax.set_xlabel(f"{TASK_LABELS.get(task, task.upper())} accuracy", fontsize=16)
    ax.set_ylabel("Alignment (Mutual kNN)", fontsize=16)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper left",
        frameon=True,
        fontsize=16,
    )
    ax.tick_params(axis="both", labelsize=16)


def plot_filename(task, dataset, caption_type=None, metric="mutual_knn", file_format="pdf"):
    """
    e.g. arc_challenge_mutual_knn_wit_1024.pdf,
         gsm8k_mutual_knn_docci_with_summaries_caption_10.pdf
    """
    parts = [task, metric, dataset]
    if caption_type is not None:
        parts.append(caption_type)
    return "_".join(parts) + f".{file_format}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Alignment of language models with a fixed vision model against their benchmark accuracy (final, max and max calibrated layer).",
    )

    # benchmark args
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=str(paths.BENCHMARK_DIR),
        help="Directory containing benchmark json files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
        choices=list(TASK_METRICS),
        help="Benchmark task; its metric is taken from TASK_METRICS",
    )

    # alignment args
    parser.add_argument(
        "--alignment_root",
        type=str,
        default=str(paths.ALIGNMENT_DIR),
        help="Root alignment results directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wit_1024",
        help="Alignment dataset name",
    )
    parser.add_argument(
        "--fixed_model",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Fixed vision model (model_y) all language models are compared against",
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        default=None,
        choices=CAPTION_TYPES,
        help="Caption type of the language features (cc3m_recaptioned, docci_with_summaries); also added to the output file name",
    )

    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Fixed y-axis range, e.g. to give subplots of different datasets the same span; default: matplotlib's automatic range",
    )

    # output args
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(paths.PLOTS_DIR / "benchmark_alignment_plots"),
        help="Output dir for saving figure",
    )
    parser.add_argument(
        "--file_format",
        type=str,
        choices=["pdf", "png"],
        default="pdf",
        help="File format of the saved figure",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    fixed_model = args.fixed_model.replace("/", "_")

    benchmark_scores = load_benchmark_results(
        benchmark_dir=args.benchmark_dir,
        task=args.task,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    annotated = False

    for layer_mode in LAYER_MODES:
        alignment_scores = load_alignment_results(
            alignment_root=args.alignment_root,
            dataset=args.dataset,
            fixed_model=fixed_model,
            layer_mode=layer_mode,
            caption_type=args.caption_type,
        )

        # model names are written once, by the first group that is drawn
        drawn = create_scatter_plot(
            ax,
            benchmark_scores,
            alignment_scores,
            label=f"{layer_mode} layer",
            annotate_models=not annotated,
            **LAYER_MODE_STYLES[layer_mode],
        )
        annotated = annotated or drawn

    if not annotated:
        raise SystemExit("Error: no layer mode had enough matching models to plot.")

    style_axes(ax, args.task)
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)

    save_dir = Path(args.save_dir) / fixed_model
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / plot_filename(
        task=args.task,
        dataset=args.dataset,
        caption_type=args.caption_type,
        file_format=args.file_format,
    )

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved to: {save_path}")


if __name__ == "__main__":
    main()
