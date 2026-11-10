import argparse
import os


DATASET_CAPTION_TYPES = {
    "cc3m_recaptioned": ["raw_caption", "shortSV_captions", "longSV_captions"],
    "docci_with_summaries": ["description", "caption_75", "caption_50", "caption_25", "caption_10"],
}

CAPTION_TYPES = [c for types in DATASET_CAPTION_TYPES.values() for c in types]

# samples per run of the datasets that are subsampled (with a seed) in extract_features.py
DEFAULT_NUM_SAMPLES = 1024
SUBSAMPLED_DATASETS = ["coco", "cc3m", "cc3m_recaptioned", "docci_with_summaries"]


def check_caption_type(parser, dataset, caption_type, modality="language"):
    """
    Error if the caption type does not belong to the dataset.
    Language features of datasets with caption types need one, everything else has none.
    """
    valid = DATASET_CAPTION_TYPES.get(dataset, []) if modality == "language" else []

    if caption_type is None and valid:
        parser.error(f"{dataset} ({modality}) needs a caption type, one of {valid}")
    if caption_type is not None and caption_type not in valid:
        parser.error(
            f"caption type {caption_type} is not valid for {dataset} ({modality}), "
            f"valid: {valid or 'none'}"
        )


def parse_pca_mode(value):
    """Fraction of variance to keep per layer, in (0, 1)."""
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"PCA mode must be a float in (0, 1), got {value!r}")

    if not (0 < value < 1):
        raise argparse.ArgumentTypeError(f"PCA mode must be a float in (0, 1), got {value}")

    return value


def to_pca_tag(pca_mode):
    if pca_mode is None:
        return "pca=None"
    return f"pca={pca_mode}_per_layer"


# pooling of the representations, fixed per modality by the encoders in models.py:
# mean pooling for language models, the CLS token for vision models
# (VLMs use their own pooling, e.g. mean over patches for SigLIP 2, and are stored like the other models)
POOLING = {"language": "avg", "vision": "cls"}


def to_pooling(modality):
    """Pooling name of a modality, used in the feature and alignment paths."""
    return POOLING[modality]


def to_modality_pair_name(modality_x, modality_y, captions_x=None, captions_y=None):
    """
    Folder name of an x-y modality pair in the alignment results,
    e.g. language_avg_vision_cls or language_avg_caption_10_vision_cls.
    """
    def name(modality, captions):
        return f"{modality}_{to_pooling(modality)}" + (f"_{captions}" if captions is not None else "")

    return f"{name(modality_x, captions_x)}_{name(modality_y, captions_y)}"


def to_feature_filename(output_dir, dataset, modality, model_name, random_init, caption_type, seed=None, pca_tag=None, num_samples=None):
    """
    Path of a features.pt; num_samples only adds a folder (samples=<num_samples>) if it is not DEFAULT_NUM_SAMPLES,
    so features of the default subsample size keep their paths.
    """
    model_name = f"{model_name.replace('/', '_')}"

    if random_init:
        mode = "random_init"
    else:
        mode = "from_pretrained"
    
    parts = [
        output_dir,
        dataset,
        modality,
        model_name,
        mode,
        to_pooling(modality),
    ]

    if caption_type is not None:
        parts.append(caption_type)
    if num_samples is not None and num_samples != DEFAULT_NUM_SAMPLES:
        parts.append(f"samples={num_samples}")
    if seed is not None:
        parts.append(str(seed))
    if pca_tag is not None:
        parts.append(pca_tag)

    parts.append("features.pt")

    save_path = os.path.join(*parts)
        
    return save_path


def to_alignment_filename(
    output_dir,
    dataset,
    modality_x,
    modality_y,
    metric,
    topk,
    kernel,
    rbf_sigma,
    subsample_n,
    random_init_x,
    random_init_y,
    captions_x,
    captions_y,
    pca_mode,
    seed,
):
    if metric == "mutual_knn":
        specification = f"topk={topk}"
    elif metric == "cka":
        specification = f"kernel={kernel}"
        if kernel == "rbf":
            specification += f"_sigma={rbf_sigma}"

    if subsample_n:
        specification += f"_N={subsample_n}"
    
    if random_init_x and random_init_y:
        mode = "random_init_both"
    elif random_init_x:
        mode = "random_init_x_pretrained_y"
    elif random_init_y:
        mode = "pretrained_x_random_init_y"
    else:
        mode = "pretrained"

    save_path = os.path.join(
        output_dir,
        dataset if not seed else f"{dataset}_{seed}",
        to_modality_pair_name(modality_x, modality_y, captions_x, captions_y),
        pca_mode,
        metric,
        specification,
        mode,
    )
    return save_path
