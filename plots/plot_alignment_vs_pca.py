import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import utils
from plots import common
from src import paths

# Raw feature dimensions, only fallback
ORIGINAL_DIMS = {
    "bigscience_bloomz-1b1": 1536,
    "huggyllama_llama-7b": 4096,
    "google_gemma-2b": 2048,
    "mistralai_Mistral-7B-v0.3": 4096,
    "Qwen_Qwen3-8B": 4096,
    "deepseek-ai_DeepSeek-R1-Distill-Qwen-32B": 5120,
    "google_siglip2-so400m-patch14-224": 1152,
    "nvidia_NV-Embed-v2": 4096,
    "Qwen_Qwen3-Embedding-8B": 4096,
    "Qwen_Qwen3-VL-Embedding-8B": 4096,
    "openai_clip-vit-large-patch14": 768
}

# Alignment setting of the PCA experiments (scripts/measure_alignment_pca*.sh)
ALIGNMENT_SETTING = dict(
    modality_x="language",
    modality_y="vision",
    metric="mutual_knn",
    topk=10,
    kernel=None,
    rbf_sigma=None,
    subsample_n=None,
    random_init_x=False,
    random_init_y=False,
    captions_y=None,
)


def parse_pca_mode(value):
    """PCA mode, or "None" for the features without PCA."""
    return None if value == "None" else utils.parse_pca_mode(value)


def alignment_dir(alignment_root, dataset, pca_mode, captions_x=None, seed=None):
    """Folder measure_alignment.py writes the CSVs of one run to."""
    return Path(utils.to_alignment_filename(
        alignment_root,
        dataset,
        captions_x=captions_x,
        pca_mode=utils.to_pca_tag(pca_mode),
        seed=seed,
        **ALIGNMENT_SETTING,
    ))


def find_seeds(alignment_root, dataset):
    """Seeds of the runs of a dataset, [None] for an unseeded dataset (e.g. wit_1024)."""
    return [seed for seed, _ in common.find_runs(alignment_root, dataset)]


def load_alignment_results(
    alignment_root,
    dataset,
    model_x,
    model_y,
    pca_modes,
    captions_x=None,
    use_final_layer=False,
):
    """
    Returns {pca_mode: [{"seed", "score", "x_dim", "x_var"}, ...]} with one
    entry per seed, for the pca modes that have results.
    """
    seeds = find_seeds(alignment_root, dataset)
    scores = {}

    for pca_mode in pca_modes:
        results = []

        for seed in seeds:
            file = (
                alignment_dir(alignment_root, dataset, pca_mode, captions_x, seed)
                / f"model_x={model_x}_model_y={model_y}.csv"
            )
            if not file.exists():
                continue

            df = pd.read_csv(file)

            # with 1 PCA dimension all L2-normalized features are +-1, so kNN
            # reduces to tie-breaking and the score is meaningless
            if pca_mode is not None:
                trivial = (df["x_pca_dim"] == 1) | (df["y_pca_dim"] == 1)
            else:
                trivial = pd.Series(False, index=df.index)

            # row (layer combination) the score comes from; with per-layer PCA
            # the dims differ between rows, so they must be read from this row
            if use_final_layer:
                row = common.select_layer_row(df, "last")
                is_trivial = trivial[row.name]
            elif (~trivial).any():
                row = common.select_layer_row(df, "best", mask=~trivial)
                is_trivial = False
            else:
                # every layer combination is 1-dimensional
                row = common.select_layer_row(df, "last")
                is_trivial = True

            score = 0.0 if is_trivial else row["score"]

            if "x_dim" in row:
                x_dim = row["x_dim"]
            elif pca_mode is not None:
                x_dim = row["x_pca_dim"]
            else:
                x_dim = ORIGINAL_DIMS.get(model_x, np.nan)
                if np.isnan(x_dim):
                    print(f"Warning: unknown dimension of {model_x} for pca=None, skipping it in the dims plot")

            # variance this layer actually retains (>= pca_mode)
            x_var = 1.0 if pca_mode is None else row["x_pca_var_layer"]

            results.append({
                "seed": seed,
                "score": score,
                "x_dim": x_dim,
                "x_var": x_var,
            })

        if results:
            scores[pca_mode] = results

    return scores


def extract_models_x(alignment_root, dataset, model_y, captions_x=None):
    """All model_x with a pca=None result against model_y (in the first seed)."""
    seeds = find_seeds(alignment_root, dataset)
    if not seeds:
        return []

    base_dir = alignment_dir(alignment_root, dataset, None, captions_x, seeds[0])
    models_x = set()
    for file in base_dir.glob("*.csv"):
        match = re.fullmatch(r"model_x=(.+?)_model_y=(.+?)\.csv", file.name)
        if match and match.group(2) == model_y:
            models_x.add(match.group(1))

    return sorted(models_x)


def plot_filename(dataset, model_y, layer_mode, x_axis, file_format="pdf"):
    """
    e.g. wit_1024__y=facebook_dinov2-large__max__x=dims.pdf
    """
    return (
        f"{dataset}__y={model_y}__{layer_mode}"
        f"__x={x_axis}.{file_format}"
    )


def summarize_runs(scores, x_of_run):
    """
    Summarize the runs (seeds) of every PCA mode of one model_x.

    scores:   {pca_mode: [{"score": ..., "x_dim": ..., "x_var": ...}, ...]}
    x_of_run: function run -> x value of that run

    Returns arrays (x_mean, x_std, y_mean, y_std) sorted by x_mean,
    or None if there is no valid run.
    """
    processed = []

    for results in scores.values():
        run_scores = np.array([r["score"] for r in results], dtype=float)
        run_x = np.array([x_of_run(r) for r in results], dtype=float)

        # Ignore invalid / missing values
        valid = np.isfinite(run_scores) & np.isfinite(run_x)
        run_scores = run_scores[valid]
        run_x = run_x[valid]

        if len(run_scores) == 0:
            continue

        processed.append((*common.mean_std(run_x), *common.mean_std(run_scores)))

    if not processed:
        return None

    processed.sort(key=lambda t: t[0])
    return tuple(np.array(col) for col in zip(*processed))


def plot_alignment_curves(all_scores, x_of_run, xlabel, save_path, log_x=False):
    """
    One line per model_x: mean alignment against x across runs.

    all_scores: {model_x: {pca_mode: [run, ...]}} (see load_alignment_results)

    - Y: mean score with +/- 1 std shaded region.
    - X: mean value with horizontal +/- 1 std error bars (if it varies across runs).
    - Mean score == 0.0: use an 'x' marker.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    for model_x, scores in all_scores.items():
        summary = summarize_runs(scores, x_of_run)
        if summary is None:
            continue

        x_mean, x_std, y_mean, y_std = summary

        # Main line
        line, = ax.plot(x_mean, y_mean, linestyle="-", label=common.convert_model_name(model_x))
        color = line.get_color()

        # Nonzero scores: circles
        nonzero_mask = y_mean != 0.0
        ax.plot(
            x_mean[nonzero_mask],
            y_mean[nonzero_mask],
            linestyle="none",
            marker="o",
            color=color,
        )

        # Zero scores: crosses only
        zero_mask = y_mean == 0.0
        ax.plot(
            x_mean[zero_mask],
            y_mean[zero_mask],
            linestyle="none",
            marker="x",
            markersize=7,
            markeredgewidth=1.5,
            color=color,
        )

        # Background shading: mean +/- std for alignment
        ax.fill_between(
            x_mean,
            y_mean - y_std,
            y_mean + y_std,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        # Horizontal error bars: x varies across runs (e.g. PCA dimensions)
        if np.any(x_std > 0):
            ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_std,
                fmt="none",
                ecolor=color,
                elinewidth=1,
                capsize=3,
                alpha=0.5,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Alignment (mutual k-NN)")

    if log_x:
        ax.set_xscale("log")
    ax.invert_xaxis()

    ax.grid(True)
    ax.legend(loc="lower left")

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_alignment_vs_dims(all_scores, save_path):
    """Alignment against the number of X dimensions of the scoring layer."""
    return plot_alignment_curves(
        all_scores,
        x_of_run=lambda run: run["x_dim"],
        xlabel="Language model representation dimensions",
        save_path=save_path,
        log_x=True,
    )


def plot_alignment_vs_variance(all_scores, save_path):
    """Alignment against the variance the X scoring layer retains (1 for no PCA)."""
    return plot_alignment_curves(
        all_scores,
        x_of_run=lambda run: run["x_var"],
        xlabel="Variance retained",
        save_path=save_path,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--captions_x",
        type=str,
        default=None,
        help="Caption type of the language features (--captions_x of measure_alignment.py)",
    )

    parser.add_argument(
        "--model_y",
        type=str,
        required=True,
        help="The fixed modality model (Y) to plot everything against"
    )

    parser.add_argument(
        "--models_x",
        nargs="+",
        type=str,
        default=None,
        help="Specific list of X models to include. If omitted, all available pairs are plotted."
    )

    parser.add_argument(
        "--pca_modes",
        nargs="+",
        type=parse_pca_mode,
        required=True,
        help='Fractions of variance retained, "None" for no PCA',
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(paths.PLOTS_DIR / "alignment_vs_pca_plots"),
        help="Output dir for saving figure",
    )

    parser.add_argument(
        "--layer_mode",
        type=str,
        choices=["final", "max"],
        default="max",
        help=(
            '"final" -> use final layer representations, '
            '"max" -> use max score across layers'
        ),
    )

    parser.add_argument(
        "--file_format",
        type=str,
        choices=["png", "pdf"],
        default="pdf",
        help="File format of the saved figures",
    )

    args = parser.parse_args()

    model_y = args.model_y.replace("/", "_")

    if args.models_x:
        models_x = [mx.replace("/", "_") for mx in args.models_x]
    else:
        models_x = extract_models_x(args.alignment_root, args.dataset, model_y, args.captions_x)

    if not models_x:
        sys.exit("No models found for processing.")

    all_scores = {}

    for model_x in models_x:
        scores = load_alignment_results(
            alignment_root=args.alignment_root,
            dataset=args.dataset,
            model_x=model_x,
            model_y=model_y,
            pca_modes=args.pca_modes,
            captions_x=args.captions_x,
            use_final_layer=args.layer_mode == "final",
        )

        if scores:
            all_scores[model_x] = scores
        else:
            print(f"Warning: No data found for pair model_x={model_x}, model_y={model_y}")

    if not all_scores:
        sys.exit("Error: No data was loaded for any of the specified model pairs.")

    for x_axis, plot_fn in [
        ("dims", plot_alignment_vs_dims),
        ("variance", plot_alignment_vs_variance),
    ]:
        filename = plot_filename(
            dataset=args.dataset,
            model_y=model_y,
            layer_mode=args.layer_mode,
            x_axis=x_axis,
            file_format=args.file_format,
        )
        file = plot_fn(all_scores, save_path=Path(args.save_dir) / filename)
        print("Saved combined plot to:", file)
