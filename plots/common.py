"""
Helpers shared by the plot scripts: finding the runs of a dataset, picking a score
from a layer-wise alignment csv, aggregating runs and model display names.
"""
import re
from pathlib import Path

import numpy as np


DISPLAY_NAMES = [
    ("dinov2", "DINOv2"),
    ("mae", "MAE"),
    ("siglip2", "SigLIP2"),
    ("siglip", "SigLIP"),
    ("clip", "CLIP"),
    ("bloomz-1b1", "BLOOMZ-1B1"),
    ("bloom", "Bloom"),
    ("nvidia", "NV-Embed-v2"),
    ("dinov3", "DINOv3"),
    ("qwen3-vl-embedding-8b", "Q3-VL-Emb-8B"),
    ("qwen3-embedding-8b", "Qwen3-Embedding"),
    ("qwen3.5-9b", "Qwen3.5-9B"),
    ("qwen3-8b", "Qwen3-8B"),
    ("gemma-2-9b", "Gemma-2-9B"),
    ("gemma-2b", "Gemma-2B"),
    ("mistral-7b", "Mistral-7B"),
    ("deepseek-r1-distill-qwen-32b", "DS-R1-Distill-Qwen-32B"),
    ("llama-3.1-8b", "Llama-3.1-8B"),
    ("llama-7b", "LLaMA-7B"),
]


def convert_model_name(model_name: str) -> str:
    """Display name of a model; the first matching substring in DISPLAY_NAMES wins."""
    model_name = model_name.lower()
    for key, display_name in DISPLAY_NAMES:
        if key in model_name:
            return display_name
    return model_name


def find_runs(base_dir, dataset, seeds=None):
    """
    Runs of a dataset in base_dir: the <dataset>_<seed> folders sorted by seed or,
    if there are none, the unseeded <dataset> folder (e.g. wit_1024).

    seeds: if given, only these seeds

    Returns a list of (seed, folder), seed is None for the unseeded run.
    """
    base_dir = Path(base_dir)
    pattern = re.compile(rf"{re.escape(dataset)}_(\d+)")

    runs = sorted(
        (int(m.group(1)), d)
        for d in base_dir.iterdir()
        if d.is_dir() and (m := pattern.fullmatch(d.name))
    )
    if seeds is not None:
        runs = [(seed, d) for seed, d in runs if seed in seeds]

    if runs:
        return runs
    return [(None, base_dir / dataset)] if (base_dir / dataset).is_dir() else []


def select_layer_row(df, which="best", mask=None):
    """
    Row (layer pair) of a layer-wise alignment csv.

    which:
        "best": highest score, among the rows in mask if given
        "last": last layer pair (highest layer of both models)
    """
    if which == "best":
        scores = df["score"] if mask is None else df.loc[mask, "score"]
        return df.loc[scores.idxmax()]

    return df[
        (df["model_x_layer"] == df["model_x_layer"].max())
        & (df["model_y_layer"] == df["model_y_layer"].max())
    ].iloc[0]


def mean_std(values):
    """Mean and sample standard deviation; the std is NaN for a single value."""
    values = np.asarray(values, dtype=float)
    std = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return np.mean(values), std
