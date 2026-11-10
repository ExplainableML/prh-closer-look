"""
Reduce precomputed features with a separate PCA per layer, keeping a fraction of each layer's variance.
The reduced features are saved next to the original ones (<seed>/pca=<mode>_per_layer/features.pt)
and are used by measure_alignment.py --pca_mode.
"""
import argparse
import os

import torch
from sklearn.decomposition import PCA

from src import paths
from src import utils
from src.features import load_layers


def apply_pca(feats, pca_mode):
    """
    Separate PCA for every layer (own mean, own basis), keeping the fraction
    pca_mode of each layer's variance, so the number of kept dimensions can
    differ between layers.
    """

    transformed_layers = []
    dims_per_layer = []
    variance_per_layer = []

    for f in feats:
        pca = PCA(
            n_components=pca_mode,
            svd_solver="full"
        )
        reduced = pca.fit_transform(f.detach().cpu().numpy())

        dims_per_layer.append(int(pca.n_components_))
        variance_per_layer.append(float(pca.explained_variance_ratio_.sum()))

        reduced = torch.tensor(
            reduced,
            device=f.device,
            dtype=f.dtype
        )

        transformed_layers.append(reduced)

    pca_info = {
        "num_dimensions_per_layer": dims_per_layer,
        "variance_retained_per_layer": variance_per_layer,
    }

    return transformed_layers, pca_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="wit_1024", help="Dataset name as used in extract_features.py")
    parser.add_argument("--modality", type=str, required=True, choices=["vision", "language"], help="Modality of the features to reduce")
    parser.add_argument("--models", nargs="+", required=True, help="Models whose features are reduced")
    parser.add_argument("--random_init", action="store_true", help="Reduce the randomly initialised features of the models")
    parser.add_argument(
        "--captions",
        type=str,
        default=None,
        choices=utils.CAPTION_TYPES,
        help="Caption type of the language features",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Dataset seeds whose features are reduced (default: the unseeded run)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(paths.FEATURES_DIR),
        help="Root directory of the features; the reduced features are saved next to them",
    )
    parser.add_argument(
        "--pca_modes",
        type=utils.parse_pca_mode,
        nargs="+",
        required=True,
        help="One or more fractions of variance to keep per layer, in (0, 1).",
    )
    parser.add_argument(
        "--precise",
        action="store_true",
        help="Exact outlier quantile; must match the --precise of measure_alignment.py.",
    )
    parser.add_argument("--force_remake", action="store_true", help="Recompute reduced features that already exist (otherwise skipped)")
    args = parser.parse_args()

    utils.check_caption_type(parser, args.dataset, args.captions, args.modality)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in (args.seeds if args.seeds is not None else [None]):
        for model in args.models:
            def feature_path(pca_tag=None):
                return utils.to_feature_filename(
                    args.input_dir,
                    args.dataset,
                    modality=args.modality,
                    model_name=model,
                    random_init=args.random_init,
                    caption_type=args.captions,
                    seed=seed,
                    pca_tag=pca_tag,
                )

            raw_path = feature_path()
            assert os.path.exists(raw_path), raw_path

            out_paths = {
                pca_mode: feature_path(utils.to_pca_tag(pca_mode))
                for pca_mode in args.pca_modes
            }
            todo = [
                m for m, p in out_paths.items()
                if args.force_remake or not os.path.exists(p)
            ]
            if not todo:
                print(f"Skipping {raw_path}: all reduced features exist")
                continue

            layers, pca_info = load_layers(raw_path, args.precise, device=device)
            assert pca_info is None, f"{raw_path} is already PCA-reduced"

            for pca_mode in todo:
                reduced, pca_info = apply_pca(layers, pca_mode)
                pca_info.update(
                    pca_mode=pca_mode,
                    precise=args.precise,
                    source=raw_path,
                )

                out_path = out_paths[pca_mode]
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save(
                    {"layers": [l.cpu() for l in reduced], "pca_info": pca_info},
                    out_path,
                )
                print(f"Saved to: {out_path}")

            del layers
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
