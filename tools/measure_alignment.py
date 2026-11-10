import argparse
import os
from pprint import pprint

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import random

from src import paths
from src import metrics
from src import utils
from src.features import load_layers

from calibrated_similarity import calibrate_layers


def compute_score(
    x_feats,
    y_feats,
    metric="mutual_knn",
    topk=10,
    kernel="ip",
    rbf_sigma=1.0,
    normalize=True,
):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment.

    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
        metric: alignment metric to use (e.g., "mutual_knn")
        topk: number of nearest neighbors if using knn-based metrics
        normalize: whether to normalize the features before comparison

    Returns:
        best_alignment_score: the highest alignment score
        best_alignment_indices: tuple (i, j) of the best x_feat and y_feat layer indices
        all_scores: list of tuples (i, j, score) for each x-y layer pair
    """
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    kwargs = {}
    if "knn" in metric:
        kwargs["topk"] = topk
    elif "cka" in metric:
        kwargs["kernel_metric"] = kernel
        kwargs["rbf_sigma"] = rbf_sigma

    best_alignment_indices = None
    best_alignment_score = float("-inf")  # in case scores can be negative
    all_scores = []

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            score = metrics.AlignmentMetrics.measure(
                metric, x_aligned, y_aligned, width_calibration=False, **kwargs
            )

            all_scores.append((i, j, score))

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)

    return best_alignment_score, best_alignment_indices, all_scores


def make_calibration_similarity(metric, topk, kernel, rbf_sigma):
    """
    Similarity function used for the depth calibrated max score.
    """
    if metric == "mutual_knn":
        return metrics.make_similarity(
            metric,
            topk=topk,
            width_calibration=False,
        )
    if metric == "cka":
        return metrics.make_similarity(
            metric,
            kernel_metric=kernel,
            rbf_sigma=rbf_sigma,
            width_calibration=False,
        )
    raise ValueError(f"Depth calibration is not supported for metric: {metric}")


def compute_alignment(x_models, y_models, metric, topk, kernel, rbf_sigma, save_path, args, precise=True):
    """
    Computes the layer-wise alignment of all x-y model pairs and saves one csv per pair
    (and a _summary.csv with the depth calibrated max score if args.depth_calibration).

    Args:
        x_models: list of (model name, feature path) for x
        y_models: list of (model name, feature path) for y
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32
            used in measure_alignment.py
    """

    os.makedirs(save_path, exist_ok=True)

    # with identical x and y models, only compute each unordered pair once
    unique_pairs = args.unique_pairs and x_models == y_models
    pairs = [
        (i, j)
        for i in range(len(x_models))
        for j in range(len(y_models))
        if not unique_pairs or i < j
    ]

    sim_fn = make_calibration_similarity(metric, topk, kernel, rbf_sigma) if args.depth_calibration else None

    pbar = tqdm(total=len(pairs))

    loaded_x = None  # (index, features, pca info) of the currently loaded x model

    for i, j in pairs:
        x_name, x_fp = x_models[i]
        y_name, y_fp = y_models[j]
        model_file = f"model_x={x_name.replace('/', '_')}_model_y={y_name.replace('/', '_')}"

        # subsampled runs get a new random seed (and file) each time, so they are never skipped
        if not args.subsample_n and not args.force_remake:
            done = os.path.exists(f"{save_path}/{model_file}.csv")
            if args.depth_calibration:
                done = done and os.path.exists(f"{save_path}/{model_file}_summary.csv")

            if done:
                print(f"Skipping {model_file}, already exists (use --force_remake to recompute)")
                pbar.update(1)
                continue

        if loaded_x is None or loaded_x[0] != i:
            loaded_x = (i, *load_layers(x_fp, precise))
        _, x_feats, x_pca_info = loaded_x

        y_feats, y_pca_info = load_layers(y_fp, precise)

        # subsample n sample pairs
        if args.subsample_n:
            subsample_seed = random.randint(0, 10_000_000)
            torch.manual_seed(subsample_seed)
            indices = torch.randperm(x_feats[0].shape[0], device=x_feats[0].device)[:args.subsample_n]

            x_feats_sub = [feat[indices] for feat in x_feats]
            y_feats_sub = [feat[indices] for feat in y_feats]
            seed_suffix = f"_seed={subsample_seed}"
        else:
            x_feats_sub = x_feats
            y_feats_sub = y_feats
            seed_suffix = ""

        best_score, best_indices, all_scores = compute_score(
            x_feats_sub,
            y_feats_sub,
            metric=metric,
            topk=topk,
            kernel=kernel,
            rbf_sigma=rbf_sigma,
        )

        df = pd.DataFrame(
            all_scores, columns=["model_x_layer", "model_y_layer", "score"]
        )
        # dimensions of each layer as used for scoring (after PCA, if any)
        df["x_dim"] = df["model_x_layer"].map(lambda i: x_feats_sub[i].shape[1])
        df["y_dim"] = df["model_y_layer"].map(lambda j: y_feats_sub[j].shape[1])
        # PCA dimensions differ by layer, so store them per row
        df["x_pca_dim"] = None
        df["y_pca_dim"] = None
        if x_pca_info:
            df["x_pca_dim"] = df["model_x_layer"].map(
                lambda i: x_pca_info["num_dimensions_per_layer"][i]
            )
            df["x_pca_var_layer"] = df["model_x_layer"].map(
                lambda i: x_pca_info["variance_retained_per_layer"][i]
            )
        if y_pca_info:
            df["y_pca_dim"] = df["model_y_layer"].map(
                lambda j: y_pca_info["num_dimensions_per_layer"][j]
            )
            df["y_pca_var_layer"] = df["model_y_layer"].map(
                lambda j: y_pca_info["variance_retained_per_layer"][j]
            )
        df.to_csv(f"{save_path}/{model_file}{seed_suffix}.csv")
        print(f"Saved to: {save_path}/{model_file}{seed_suffix}.csv")

        # compute depth calibrated max score, on the same (sub)samples as the layer-wise scores
        if args.depth_calibration:
            # L2-normalize as in compute_score
            calibrated_agg, p_value, tau = calibrate_layers(
                [F.normalize(x, p=2, dim=-1) for x in x_feats_sub],
                [F.normalize(y, p=2, dim=-1) for y in y_feats_sub],
                sim_fn,
                agg="max",
            )

            # save calibrated aggregate score to extra csv
            summary_df = pd.DataFrame(
                [{
                    "calibrated_score": calibrated_agg.item(),
                    "p_value": p_value,
                    "tau": tau,
                    "best_x_layer": best_indices[0],
                    "best_y_layer": best_indices[1],
                }]
            )

            summary_df.to_csv(f"{save_path}/{model_file}{seed_suffix}_summary.csv", index=False)
            print(f"Saved to: {save_path}/{model_file}{seed_suffix}_summary.csv")

        pbar.update(1)

        torch.cuda.empty_cache()

    pbar.close()


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """

    parser = argparse.ArgumentParser(
        description="Layer-wise alignment of all x-y model pairs; one csv per pair with the score of every layer pair.",
    )
    parser.add_argument("--dataset", type=str, default="wit_1024", help="Dataset name as used in extract_features.py")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Dataset seeds (runs) to measure, one after another (default: the unseeded run)",
    )
    parser.add_argument("--modality_x", type=str, required=True, choices=["vision", "language"], help="Modality of the x features")
    parser.add_argument("--modality_y", type=str, required=True, choices=["vision", "language"], help="Modality of the y features")
    parser.add_argument(
        "--metric",
        type=str,
        default="mutual_knn",
        choices=metrics.AlignmentMetrics.SUPPORTED_METRICS,
        help="Alignment metric",
    )
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbours k for the kNN metrics")
    parser.add_argument(
        "--subsample_n",
        type=int,
        default=None,
        help=(
            "Measure on a random subsample of N samples; every run draws a new subsample, "
            "its seed is saved in the file name (_seed=...). Not with --pca_mode"
        ),
    )
    parser.add_argument("--kernel", type=str, default="ip", choices=["ip", "rbf"], help="CKA only: kernel")
    parser.add_argument("--rbf_sigma", type=float, default=1.0, help="CKA with the rbf kernel only: kernel width")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(paths.FEATURES_DIR),
        help="Root directory of the features (output_dir of extract_features.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(paths.ALIGNMENT_DIR),
        help="Root directory for the alignment results",
    )
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Exact computation; without it, TF32 matmuls are enabled for speed",
    )
    parser.add_argument(
        "--force_remake",
        action="store_true",
        help="Recompute pairs whose csv already exists (otherwise they are skipped)",
    )
    parser.add_argument(
        "--depth_calibration",
        action="store_true",
        help="Also compute the depth calibrated max score and save it to the _summary.csv",
    )
    parser.add_argument(
        "--unique_pairs",
        action="store_true",
        help="If x and y models are identical, only compute each pair once and skip self-pairs",
    )
    parser.add_argument(
        "--language_models",
        nargs="+",
        default=None,
        metavar="LLM",
        help="Space-separated list of LLM model names.",
    )
    parser.add_argument(
        "--vision_models",
        nargs="+",
        default=None,
        metavar="LVM",
        help="Space-separated list of vision model names.",
    )
    parser.add_argument("--random_init_x", action="store_true", help="Use the randomly initialised features of the x models")
    parser.add_argument("--random_init_y", action="store_true", help="Use the randomly initialised features of the y models")
    parser.add_argument("--captions_x", type=str, default=None, choices=utils.CAPTION_TYPES, help="Caption type of the x language features")
    parser.add_argument("--captions_y", type=str, default=None, choices=utils.CAPTION_TYPES, help="Caption type of the y language features")
    parser.add_argument(
        "--pca_mode",
        type=utils.parse_pca_mode,
        default=None,
        help="Selects per-layer PCA features precomputed by reduce_features.py.",
    )
    args = parser.parse_args()

    for modality in {args.modality_x, args.modality_y}:
        if getattr(args, f"{modality}_models") is None:
            parser.error(f"--{modality}_models is required for modality {modality}")

    utils.check_caption_type(parser, args.dataset, args.captions_x, args.modality_x)
    utils.check_caption_type(parser, args.dataset, args.captions_y, args.modality_y)

    if args.unique_pairs and args.metric == "cycle_knn":
        parser.error("--unique_pairs requires a symmetric metric, cycle_knn is not")

    pca_tag = utils.to_pca_tag(args.pca_mode)
    # reduced features are only stored for PCA runs; pca=None loads the raw ones
    feature_pca_tag = pca_tag if args.pca_mode is not None else None

    if args.pca_mode is not None and args.subsample_n:
        # previously PCA was fit on the subsample; precomputed PCA is fit on all samples
        parser.error("--subsample_n is not supported with --pca_mode (PCA is precomputed on all samples)")

    if not args.precise:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    for seed in (args.seeds if args.seeds is not None else [None]):

        save_path = utils.to_alignment_filename(
            args.output_dir,
            args.dataset,
            args.modality_x,
            args.modality_y,
            args.metric,
            args.topk,
            args.kernel,
            args.rbf_sigma,
            args.subsample_n,
            args.random_init_x,
            args.random_init_y,
            args.captions_x,
            args.captions_y,
            pca_tag,
            seed
        )

        def feature_models(modality, random_init, captions):
            return [
                (
                    m,
                    utils.to_feature_filename(
                        args.input_dir,
                        args.dataset,
                        modality=modality,
                        model_name=m,
                        random_init=random_init,
                        caption_type=captions,
                        seed=seed,
                        pca_tag=feature_pca_tag,
                    ),
                )
                for m in getattr(args, f"{modality}_models")
            ]

        x_models = feature_models(args.modality_x, args.random_init_x, args.captions_x)
        y_models = feature_models(args.modality_y, args.random_init_y, args.captions_y)

        for _, fn in x_models + y_models:
            assert os.path.exists(fn), (
                f"{fn} not found"
                + (" (run reduce_features.py first)" if feature_pca_tag else "")
            )

        print(f"dataset:\t{args.dataset}")
        print(f"metric: \t{args.metric}")
        if "knn" in args.metric:
            print(f"topk:\t{args.topk}")

        print(f"models_x_paths:")
        pprint([fn for _, fn in x_models])
        print("\nmodels_y_paths:")
        pprint([fn for _, fn in y_models])

        print("\nmeasuring alignment")
        compute_alignment(
            x_models,
            y_models,
            args.metric,
            topk=args.topk,
            kernel=args.kernel,
            rbf_sigma=args.rbf_sigma,
            save_path=save_path,
            args=args,
            precise=args.precise,
        )
