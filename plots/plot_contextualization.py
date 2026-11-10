from itertools import combinations, product
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
import numpy as np

from src import utils
from plots import common
from src import paths


MODEL_NAMES = {
    "nvidia/NV-Embed-v2": "NV",
    "Qwen/Qwen3-Embedding-8B": "Q3",
    "facebook/dinov2-large": "D2",
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "D3",
    "openai/clip-vit-large-patch14": "CLIP",
    "google/siglip2-so400m-patch14-224": "SigLIP 2",
    "Qwen/Qwen3-VL-Embedding-8B": "Q3-VL",
}

STYLES = {
    "Random": {
        "color": "#777777",
        "marker": "o",
        "filled": True,
    },
    "Cross-modal": {
        "color": "#1f77b4",
        "marker": "o",
        "filled": True,
    },
    "VLM": {
        "color": "#ff7f0e",
        "marker": "o",
        "filled": True,
    },
    "Unimodal": {
        "color": "#1f77b4",
        "marker": "D",
        "filled": False,
    },
}

GROUP_LABEL = "D2/D3\nx\nNV/Q3"

XTICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

FULL_N = 1024  # samples per dataset run; a subsample of this size is the full dataset


def aggregate_runs(run_files, which):
    """
    Mean and standard deviation of the scores over runs.

    run_files: list of (run name, csv file)
    which: layer pair of each run, "best" or "last" (see common.select_layer_row)
    """
    rows = [common.select_layer_row(pd.read_csv(f), which) for _, f in run_files]
    score, score_std = common.mean_std([row["score"] for row in rows])

    return {
        "score": score,
        "score_std": score_std,
        "n_runs": len(rows),
        "runs": [name for name, _ in run_files],
        "layers": [(int(row["model_x_layer"]), int(row["model_y_layer"])) for row in rows],
    }


def model_pairs(category, language_models, vision_models, vlms):
    """
    (model_x, model_y, modality_x, modality_y) of all pairs of a category.
    """
    if category in ["Random", "Cross-modal"]:
        return [(l, v, "language", "vision") for l, v in product(language_models, vision_models)]
    if category == "Unimodal":
        return (
            [(a, b, "language", "language") for a, b in combinations(language_models, 2)]
            + [(a, b, "vision", "vision") for a, b in combinations(vision_models, 2)]
        )
    if category == "VLM":
        return [(m, m, "language", "vision") for m in vlms]


def load_alignment_pairs(
    language_models,
    vision_models,
    vlms,
    base_dir,
    dataset,
    caption_type=None,
    seeds=None,
    topk=10,
    subsample_n=None,
    vlm_score="last",
):
    """
    Load the alignment scores of all model pairs of a dataset, averaged over all runs.

    Runs are the seed directories of the dataset or, with subsample_n (subsample ablation),
    the subsampling seeds (topk=<k>_N=<n>/..._seed=*.csv) within those directories.

    vlm_score:
        "last": score of the last layer pair (projection) for VLMs
        "best": best layer pair for VLMs, as for all other categories
    """
    base_dir = Path(base_dir)
    subsampled = subsample_n is not None
    runs = common.find_runs(base_dir, dataset, seeds)
    runs = [("-" if seed is None else str(seed), d) for seed, d in runs]

    records = []

    for category, mode in [
        ("Random", "random_init_both"),
        ("Cross-modal", "pretrained"),
        ("Unimodal", "pretrained"),
        ("VLM", "pretrained"),
    ]:
        which = vlm_score if category == "VLM" else "best"

        for model_x, model_y, modality_x, modality_y in model_pairs(category, language_models, vision_models, vlms):
            modality_pair = utils.to_modality_pair_name(
                modality_x,
                modality_y,
                caption_type if modality_x == "language" else None,
                caption_type if modality_y == "language" else None,
            )
            mutual_knn_dir = Path(modality_pair) / "pca=None" / "mutual_knn"
            model_file = f"model_x={model_x.replace('/', '_')}_model_y={model_y.replace('/', '_')}"

            if subsampled:
                # run name: subsampling seed, prefixed by the dataset seed if there is one
                run_files = [
                    (f.stem.split("_seed=")[-1] if name == "-" else f"{name}:{f.stem.split('_seed=')[-1]}", f)
                    for name, d in runs
                    for f in sorted((d / mutual_knn_dir / f"topk={topk}_N={subsample_n}" / mode).glob(f"{model_file}_seed=*.csv"))
                ]
            else:
                run_files = [
                    (name, d / mutual_knn_dir / f"topk={topk}" / mode / f"{model_file}.csv")
                    for name, d in runs
                ]
                run_files = [(name, f) for name, f in run_files if f.exists()]

            if not run_files:
                if category == "Random":
                    print(f"No random init runs for {model_file} (topk={topk}) in {dataset}, skipping")
                    continue
                raise FileNotFoundError(
                    f"No files found for {model_file} ({mode}, topk={topk}, N={subsample_n}) in any run of {dataset}"
                )

            print(f"{model_file} {mode} topk={topk}: {len(run_files)} runs ({', '.join(name for name, _ in run_files)})")

            records.append({
                "category": category,
                "pair": MODEL_NAMES[model_x] if category == "VLM" else f"{MODEL_NAMES[model_x]} - {MODEL_NAMES[model_y]}",
                **aggregate_runs(run_files, which),
            })

    results = pd.DataFrame(records)

    if results["n_runs"].nunique() > 1:
        print(f"WARNING: {dataset} {caption_type}: pairs have different numbers of runs:")
        print(results[["category", "pair", "n_runs"]].to_string(index=False))

    return results


def place_labels(ax, labels, key_y, base_offset=0.1, tier_gap=0.03, pad=0.004, max_shift=0.015, fontsize=9):
    """
    Draw the labels of one row and stack overlapping ones in tiers away from the line.
    Expects the final axis limits and an inverted y axis (smaller y is higher up).

    labels: list of dicts with
        x:    x position (data units)
        text: label text
        side: "above" (stacked in tiers above the line) or "right" (right of x, on the line)

    A label overlapping its neighbour by at most max_shift is nudged to the right
    instead of being lifted to the next tier.
    """
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()

    def data_extent(text):
        bbox = text.get_window_extent(renderer)
        (x0, y0), (x1, y1) = to_data.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        return x0, x1, abs(y1 - y0)

    tier_ends = []  # right edge of the last label in each tier
    tier_heights = []

    # labels right of a group sit on the line, their upper half blocks the start of the first tier
    for label in labels:
        if label["side"] == "right":
            text = ax.text(label["x"], key_y, label["text"], fontsize=fontsize, ha="left", va="center")
            _, x1, height = data_extent(text)

            if not tier_ends:
                tier_ends.append(x1)
                tier_heights.append(0.0)
            tier_ends[0] = max(tier_ends[0], x1)
            tier_heights[0] = max(tier_heights[0], height / 2 - base_offset)

    above = sorted((l for l in labels if l["side"] == "above"), key=lambda l: l["x"])

    # assign tiers from left to right: lowest tier where the label fits, possibly after a small nudge
    placed = []

    for label in above:
        text = ax.text(label["x"], key_y, label["text"], fontsize=fontsize, ha="center", va="bottom")
        x0, x1, height = data_extent(text)

        tier, shift = len(tier_ends), 0.0
        for t, end in enumerate(tier_ends):
            overlap = end + pad - x0
            if overlap <= max_shift:
                tier, shift = t, max(overlap, 0.0)
                break

        if tier == len(tier_ends):
            tier_ends.append(x1)
            tier_heights.append(height)
        else:
            tier_ends[tier] = x1 + shift
            tier_heights[tier] = max(tier_heights[tier], height)

        text.set_x(label["x"] + shift)
        placed.append((label, text, tier))

    tier_offsets = np.cumsum([base_offset] + [h + tier_gap for h in tier_heights[:-1]])

    for label, text, tier in placed:
        offset = tier_offsets[tier]
        text.set_y(key_y - offset)

        # leader line for labels lifted away from their marker
        if tier > 0:
            ax.plot(
                [label["x"], label["x"]],
                [key_y - base_offset, key_y - offset],
                color="0.6",
                lw=0.5,
                zorder=1,
            )


def draw_contextualization(ax, results, key_y):

    jitter = [-0.075, -0.025, 0.025, 0.075]
    labels = []

    for category, group in results.groupby("category", sort=False):
        style = STYLES[category]

        for k, row in enumerate(group.itertuples()):
            y = key_y + (jitter[k] if category in ["Random", "Cross-modal"] else 0)
            has_std = not pd.isna(row.score_std)

            ax.errorbar(
                row.score,
                y,
                xerr=row.score_std if has_std else None,
                fmt=style["marker"],
                markersize=8,
                color=style["color"],
                ecolor=(*plt.matplotlib.colors.to_rgb(style["color"]), 0.5),
                markerfacecolor=style["color"] if style["filled"] else "none",
                markeredgecolor=style["color"],
                markeredgewidth=2,
                capsize=3 if has_std else 0,
                elinewidth=1.5,
                zorder=3,
            )

            if category in ["Unimodal", "VLM"]:
                labels.append({"x": row.score, "text": row.pair, "side": "above"})

        # one label for the whole group of D2/D3 x NV/Q3 pairs
        if category == "Random":
            labels.append({"x": group.score.max() + 0.01, "text": GROUP_LABEL, "side": "right"})
        elif category == "Cross-modal":
            labels.append({"x": group.score.mean(), "text": GROUP_LABEL, "side": "above"})

    place_labels(ax, labels, key_y)


def legend_handles():
    """
    Colour legend (training) and shape legend (alignment).
    """
    color_handles = [
        Line2D(
            [0], [0],
            marker="_",
            linestyle="None",
            markerfacecolor=STYLES[category]["color"],
            markeredgecolor=STYLES[category]["color"],
            markeredgewidth=3,
            markersize=10,
            label=label,
        )
        for category, label in [
            ("Random", "Random"),
            ("Cross-modal", "Unimodal training"),
            ("VLM", "Cross-modal training"),
        ]
    ]

    shape_handles = [
        Line2D(
            [0], [0],
            marker=STYLES["Cross-modal"]["marker"],
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
            label="Cross-modal alignment",
        ),
        Line2D(
            [0], [0],
            marker=STYLES["Unimodal"]["marker"],
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=2,
            markersize=8,
            label="Unimodal alignment",
        ),
    ]

    return color_handles, shape_handles


def plot_contextualizations(rows, output_file):
    """
    rows: list of (y tick label, results), drawn from top to bottom.
    """
    fig, ax = plt.subplots(
        figsize=(10, 1.0 * len(rows)),
    )

    # Light guide lines
    for y in range(len(rows)):
        ax.axhline(y, color="0.93", lw=1, zorder=0)

    # Vertical guide lines (none on the right edge)
    for x in XTICKS[:-1]:
        ax.axvline(x, color="0.93", lw=1, zorder=0)

    # Axes (set before drawing, label placement measures text in data units)
    ax.set_xticks(XTICKS)
    ax.set_xlim(-0.02, XTICKS[-1])
    ax.set_xlabel("Mutual kNN alignment", fontsize=13)

    ax.set_ylim(-0.5, len(rows) - 0.75)

    # Put first row at the top
    ax.invert_yaxis()

    for y, (_, results) in enumerate(rows):
        draw_contextualization(ax, results, key_y=y)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [label for label, _ in rows],
        fontsize=11,
        fontfamily="monospace",
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0, pad=12)

    color_handles, shape_handles = legend_handles()

    legend1 = fig.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.25, 1.1),
        frameon=True,
        ncol=3,
        columnspacing=0.8,
    )

    fig.legend(
        handles=shape_handles,
        loc="upper center",
        bbox_to_anchor=(0.78, 1.1),
        frameon=True,
        ncol=2,
        columnspacing=0.8,
    )

    ax.add_artist(legend1)

    plt.tight_layout()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved to:", output_file)


def dataset_label(dataset, caption_type):
    """
    y tick label of a dataset, e.g. "WIT", "COCO", "DOCCI 100%", "DOCCI 25%".
    """
    name = dataset.split("_")[0].upper()

    if "docci" not in dataset:
        return name
    if caption_type in [None, "description"]:
        return f"{name} 100%"
    return f"{name} {caption_type.split('_')[-1]}%"


def parse_dataset(value):
    """<dataset> or <dataset>-<caption_type> -> (dataset, caption type or None)."""
    dataset, _, caption_type = value.partition("-")
    return dataset, caption_type or None


def parse_pair(value):
    """k:N -> (k, N), for the subsample ablation."""
    try:
        k, n = (int(v) for v in value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"pair must be k:N with integers, e.g. 5:512, got {value!r}")
    if not (0 < k < n):
        raise argparse.ArgumentTypeError(f"pair must have 0 < k < N, got {value!r}")
    return k, n


def output_name(mode, vlm_score, dataset=None, caption_type=None):
    """e.g. contextualization_last_vlms.pdf, contextualization_k_ablation_docci_with_summaries_description_last_vlms.pdf"""
    if mode == "datasets":
        return f"contextualization_{vlm_score}_vlms.pdf"
    parts = ["contextualization", mode, dataset] + ([caption_type] if caption_type else []) + [vlm_score, "vlms"]
    return "_".join(parts) + ".pdf"


def print_ratios(pairs, tolerance=0.05):
    """Print k/N of each pair and warn if they differ by more than tolerance (relative)."""
    ratios = [k / n for k, n in pairs]
    for (k, n), r in zip(pairs, ratios):
        print(f"k = {k:4d}, N = {n:5d}: k/N = {r:.5f} (N/k = {n / k:.1f})")
    if max(ratios) > (1 + tolerance) * min(ratios):
        print(f"WARNING: k/N differs by more than {tolerance:.0%} between the pairs")


if __name__ == "__main__":
    # options shared by all plot types
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--base_dir",
        type=str,
        default=str(paths.ALIGNMENT_DIR),
        help="Alignment results directory (output_dir of measure_alignment.py)",
    )
    shared.add_argument("--output_dir", type=str, default=str(paths.PLOTS_DIR / "contextualization"), help="Where the plot is saved")
    shared.add_argument(
        "--language_models",
        nargs="+",
        default=[
            "nvidia/NV-Embed-v2",
            "Qwen/Qwen3-Embedding-8B",
        ],
        help="Unimodal language models",
    )
    shared.add_argument(
        "--vision_models",
        nargs="+",
        default=[
            "facebook/dinov2-large",
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        ],
        help="Unimodal vision models",
    )
    shared.add_argument(
        "--vlms",
        nargs="+",
        default=[
            "openai/clip-vit-large-patch14",
            "google/siglip2-so400m-patch14-224",
            "Qwen/Qwen3-VL-Embedding-8B"
        ],
        help="Vision-language models, compared between their own language and vision features",
    )
    shared.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Only use runs with these dataset seeds (default: all runs found)",
    )
    shared.add_argument(
        "--vlm_score",
        type=str,
        default="last",
        choices=["last", "best"],
        help="VLM score: last layer pair (projection) or best layer pair",
    )

    parser = argparse.ArgumentParser(
        description="Contextualization plots: alignment of unimodal model pairs next to random init and VLM baselines.",
    )
    modes = parser.add_subparsers(dest="mode", required=True, metavar="{datasets,k_ablation,subsample_ablation}")

    datasets_parser = modes.add_parser(
        "datasets",
        parents=[shared],
        help="One row per dataset (k=10, full dataset)",
        description="One row per dataset, at k=10 on the full dataset.",
    )
    datasets_parser.add_argument(
        "--datasets",
        nargs="+",
        type=parse_dataset,
        default=[("wit_1024", None)],
        help="<dataset> or <dataset>-<caption_type>, e.g. wit_1024 coco docci_with_summaries-caption_10",
    )

    k_parser = modes.add_parser(
        "k_ablation",
        parents=[shared],
        help="One row per k, on the full dataset",
        description="Ablation over the number of neighbours k at fixed N (the full dataset): one row per k.",
    )
    k_parser.add_argument(
        "--dataset",
        type=parse_dataset,
        required=True,
        help="<dataset> or <dataset>-<caption_type>, e.g. docci_with_summaries-description",
    )
    k_parser.add_argument("--topk", type=int, nargs="+", required=True, help="Values of k, one row each, e.g. 1 5 10 100")

    subsample_parser = modes.add_parser(
        "subsample_ablation",
        parents=[shared],
        help="One row per (k, N) pair, on subsamples of size N",
        description=(
            "Ablation over the subsample (gallery) size N, with k chosen per N: one row per k:N pair, "
            "averaged over the subsampling seeds of topk=<k>_N=<N>. "
            f"A pair with N={FULL_N} uses the full dataset (topk=<k>). "
            "The k/N ratio of the pairs is printed, not enforced."
        ),
    )
    subsample_parser.add_argument(
        "--dataset",
        type=parse_dataset,
        required=True,
        help="<dataset> or <dataset>-<caption_type>, e.g. docci_with_summaries-description",
    )
    subsample_parser.add_argument(
        "--pairs",
        type=parse_pair,
        nargs="+",
        required=True,
        help="k:N pairs, one row each, e.g. 1:102 5:512 10:1024",
    )

    args = parser.parse_args()

    datasets = args.datasets if args.mode == "datasets" else [args.dataset]
    for dataset, caption_type in datasets:
        utils.check_caption_type(parser, dataset, caption_type)

    load_args = {
        "language_models": args.language_models,
        "vision_models": args.vision_models,
        "vlms": args.vlms,
        "base_dir": args.base_dir,
        "seeds": args.seeds,
        "vlm_score": args.vlm_score,
    }

    rows = []

    if args.mode == "datasets":
        for dataset, caption_type in args.datasets:
            rows.append((dataset_label(dataset, caption_type), load_alignment_pairs(
                **load_args,
                dataset=dataset,
                caption_type=caption_type,
            )))
        output_file = output_name(args.mode, args.vlm_score)

    else:
        dataset, caption_type = args.dataset

        if args.mode == "k_ablation":
            pairs = [(k, None) for k in args.topk]
        else:
            pairs = args.pairs
            print_ratios(pairs)

        for k, n in pairs:
            label = f"k = {k}" if n is None else f"k = {k},\nN = {n}"
            rows.append((label, load_alignment_pairs(
                **load_args,
                dataset=dataset,
                caption_type=caption_type,
                topk=k,
                subsample_n=None if n == FULL_N else n,
            )))
        output_file = output_name(args.mode, args.vlm_score, dataset, caption_type)

    plot_contextualizations(rows, Path(args.output_dir) / output_file)
