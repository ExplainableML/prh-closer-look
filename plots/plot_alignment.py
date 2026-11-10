#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from src import metrics
import numpy as np
from plots.common import convert_model_name


MODALITIES = ("language", "vision")


def modalities_from_path(path):
    """
    (modality_x, modality_y) from the modality pair folder of an alignment csv,
    e.g. language_avg_caption_10_vision_cls -> ("language", "vision"); (None, None) if there is none.
    """
    for part in path.split(os.sep):
        tokens = part.split("_")
        if tokens[0] not in MODALITIES:
            continue
        rest = [t for t in tokens[1:] if t in MODALITIES]
        if rest:
            return tokens[0], rest[0]
    return None, None


def tick_positions(n, max_ticks=4):
    return np.linspace(0, n - 1, min(max_ticks, n), dtype=int)


def plot_heatmap(heatmap_data, model_x, model_y, out_file):
    """Save a heatmap PDF for the given data and highlight the maximum."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heatmap_data, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=32)

    x_ticks = tick_positions(heatmap_data.shape[1])
    y_ticks = tick_positions(heatmap_data.shape[0])
    ax.set_xticks(x_ticks, labels=heatmap_data.columns[x_ticks], rotation=0, fontsize=24)
    ax.set_yticks(y_ticks, labels=heatmap_data.index[y_ticks], fontsize=24)

    # Rows are model_x layers and columns are model_y layers (see pivot in main),
    # so model_y goes on the x-axis and model_x on the y-axis.
    ax.set_xlabel(model_y, fontsize=36, labelpad=20)
    ax.set_ylabel(model_x, fontsize=36, labelpad=10)

    max_val = heatmap_data.values.max()
    max_pos = np.unravel_index(np.argmax(heatmap_data.values), heatmap_data.shape)
    rect = plt.Rectangle((max_pos[1] - 0.5, max_pos[0] - 0.5), 1, 1,
                         linewidth=3, edgecolor="#e00000", facecolor="none")
    ax.add_patch(rect)
    ax.text(max_pos[1] - 0.5, max_pos[0] - 1.8, f"{max_val:.3f}", ha="center", va="center",
            color="#e00000", fontweight="bold", fontsize=32)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate alignment heatmaps from CSVs."
    )
    parser.add_argument(
        "--alignment_root",
        required=True,
        help="Path to alignment CSV root (input).",
    )
    parser.add_argument(
        "--heatmap_root", required=True, help="Path to output heatmaps root."
    )

    args = parser.parse_args()

    alignment_root = args.alignment_root
    heatmap_root = args.heatmap_root

    for root, _, files in os.walk(alignment_root):
        for file in files:
            if file.endswith(".csv") and "model_x=" in file and "model_y=" in file:
                if file.endswith("summary.csv"):
                    continue
                full_path = os.path.join(root, file)
                path_parts = set(full_path.split(os.sep))
                metric = next(
                    (m for m in metrics.AlignmentMetrics.SUPPORTED_METRICS if m in path_parts),
                    None,
                )
                if metric is None:
                    continue

                match = re.search(
                    r"model_x=(.+?)_model_y=(.+?)\.csv$",
                    file
                )
                if not match:
                    continue

                model_x = convert_model_name(match.group(1))
                model_y = convert_model_name(match.group(2))

                # VLM (same model on both axes, different modalities): add the modality to the labels
                label_x, label_y = model_x, model_y
                modality_x, modality_y = modalities_from_path(full_path)
                if match.group(1) == match.group(2) and modality_x != modality_y:
                    label_x = f"{model_x} {modality_x}"
                    label_y = f"{model_y} {modality_y}"

                # Load csv and prepare heatmap matrix
                df = pd.read_csv(full_path)
                df["model_x_layer_num"] = (
                    df["model_x_layer"].astype(str).str.extract(r"(\d+)").astype(int)
                )
                df["model_y_layer_num"] = (
                    df["model_y_layer"].astype(str).str.extract(r"(\d+)").astype(int)
                )

                heatmap_data = (
                    df.pivot(
                        index="model_x_layer_num",
                        columns="model_y_layer_num",
                        values="score",
                    )
                    .sort_index(axis=0)
                    .sort_index(axis=1)
                )

                relative_path = os.path.relpath(full_path, alignment_root)
                new_path = os.path.join(heatmap_root, os.path.dirname(relative_path))
                os.makedirs(new_path, exist_ok=True)

                out_file = os.path.join(new_path, f"{model_x}_{model_y}.pdf")
                plot_heatmap(
                    heatmap_data=heatmap_data,
                    model_x=label_x,
                    model_y=label_y,
                    out_file=out_file,
                )
                print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
