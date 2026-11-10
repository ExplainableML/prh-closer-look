import torch

from src import metrics


def prepare_features(feats, q=0.95, exact=False, device="cuda"):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats.to(device)
    elif isinstance(feats, list):
        return [
            metrics.remove_outliers(f.float(), q=q, exact=exact).to(device) for f in feats
        ]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def load_layers(feat_path, precise, device="cuda"):
    """
    Load a features file as a list of N x D tensors, one per layer, plus its
    PCA info (None for raw features).

    Raw features (written by extract_features.py):
    "feats" is N x L x D (or None for embedding-only models).
    "projected" is stored as N x 1 x D (a single projection head, e.g. CLIP)
    or N x 1 x L x D (several layers, e.g. Qwen3-VL LLM states); every
    projected layer is appended after the "feats" layers.
    Outliers are removed on load.

    Reduced features (written by reduce_features.py):
    "layers" is a list of N x D_l tensors that already had outliers removed
    before PCA, so they are returned as they are.
    """
    feat_dict = torch.load(feat_path, map_location=device)

    if "pca_info" in feat_dict:
        pca_info = feat_dict["pca_info"]
        if pca_info["precise"] != precise:
            raise ValueError(
                f"{feat_path} was reduced with precise={pca_info['precise']}, "
                f"but precise={precise} was requested. Rerun reduce_features.py "
                f"with matching --precise."
            )
        return [l.float().to(device) for l in feat_dict["layers"]], pca_info

    layers = []

    feats = feat_dict.get("feats")
    if feats is not None:
        feats = prepare_features(feats.float(), exact=precise, device=device)
        layers += [feats[:, l, :] for l in range(feats.shape[1])]

    projected = feat_dict.get("projected")
    if projected is not None:
        projected = prepare_features(projected.float(), exact=precise, device=device)
        projected = projected.reshape(projected.shape[0], -1, projected.shape[-1])
        layers += [projected[:, l, :] for l in range(projected.shape[1])]

    assert layers, f"No features found in {feat_path}"
    return layers, None
