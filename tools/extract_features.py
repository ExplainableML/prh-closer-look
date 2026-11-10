import argparse
import gc
import os

import torch
from datasets import load_dataset, load_from_disk
from tqdm import trange
import random

from src import paths
from src import utils
from src.datautils import load_cc3m_dataset, load_coco_dataset, load_cc3m_subset
from src.models import (
    CLIPTextEncoder,
    CLIPVisionEncoder,
    SiglipTextEncoder,
    SiglipVisionEncoder,
    TextEncoder,
    VisionEncoder,
    NVEmbedEncoder,
    Qwen3VLEncoder,
)


def extract_features(model, inputs, batch_size):
    hidden_all = []
    projected_all = []

    model.eval()

    for i in trange(0, len(inputs), batch_size):
        batch = inputs[i : i + batch_size]

        with torch.no_grad():
            hidden, projected = model(batch)
            if hidden is not None:
                hidden_all.append(hidden.cpu())

            if projected is not None:
                projected = projected.unsqueeze(1)
                projected_all.append(projected.cpu())

    if len(projected_all) > 0:
        projected_all = torch.cat(projected_all)
    else:
        projected_all = None

    if len(hidden_all) > 0:
        hidden_all = torch.cat(hidden_all)
    else:
        hidden_all = None

    return hidden_all, projected_all


def extract_text_features(filenames, dataset, seed, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """
    # Extract captions
    if args.dataset == "wit_1024":
        texts = [str(x["text"][args.caption_idx]) for x in dataset]
    elif args.dataset == "cc3m":
        texts = [str(x["text"]) for x in dataset]
    elif args.dataset == "coco":
        texts = [str(example["text"]).strip().lower() for example in dataset]
    elif args.dataset == "cc3m_recaptioned":
        texts = [str(x[args.caption_type]) for x in dataset]
    elif args.dataset == "docci_with_summaries":
        texts = [str(x[args.caption_type]) for x in dataset]
    else:
        texts = [str(example["text"]) for example in dataset]
    
    for language_model in filenames:
        save_path = utils.to_feature_filename(
            output_dir=args.output_dir,
            dataset=args.dataset,
            modality="language",
            model_name=language_model,
            random_init=args.random_init,
            caption_type=args.caption_type,
            seed=seed,
            num_samples=args.num_samples,
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"processing:\t{language_model}")
        print(f"save_path: \t{save_path}")

        if "clip" in language_model.lower():
            language_model = CLIPTextEncoder(language_model, random_init=args.random_init)
        elif "siglip" in language_model.lower():
            language_model = SiglipTextEncoder(language_model, random_init=args.random_init)
        elif "nv-embed" in language_model.lower():
            language_model = NVEmbedEncoder(language_model, random_init=args.random_init)
        elif "qwen3-vl-embedding" in language_model.lower():
            language_model = Qwen3VLEncoder(language_model, random_init=args.random_init)
        else:  # no projection into bimodal embedding space
            language_model = TextEncoder(language_model, random_init=args.random_init)
        # Extract and save features for both captions
        feats, projected = extract_features(language_model, texts, args.batch_size)

        save_dict = {"feats": feats}

        if projected is not None:
            save_dict["projected"] = projected

        torch.save(save_dict, save_path)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return


def extract_image_features(filenames, dataset, seed, args):
    if args.dataset == "imagenet":
        imgs = [example["jpg"].convert("RGB") for example in dataset]
    else:
        imgs = [example["image"].convert("RGB") for example in dataset]

    for vision_model in filenames:
        save_path = utils.to_feature_filename(
            output_dir=args.output_dir,
            dataset=args.dataset,
            modality="vision",
            model_name=vision_model,
            random_init=args.random_init,
            caption_type=None,  # images do not depend on the caption type
            seed=seed,
            num_samples=args.num_samples,
        )

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"processing:\t{vision_model}")
        print(f"save_path: \t{save_path}")

        if "clip" in vision_model.lower():
            vision_model = CLIPVisionEncoder(vision_model, random_init=args.random_init)
        elif "siglip" in vision_model.lower():
            vision_model = SiglipVisionEncoder(vision_model, random_init=args.random_init)
        elif "qwen3-vl-embedding" in vision_model.lower():
            vision_model = Qwen3VLEncoder(vision_model, random_init=args.random_init)
        else:  # no projection into bimodal embedding space
            vision_model = VisionEncoder(vision_model, random_init=args.random_init)

        feats, projected = extract_features(vision_model, imgs, args.batch_size)

        save_dict = {"feats": feats}

        if projected is not None:
            save_dict["projected"] = projected

        torch.save(save_dict, save_path)

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return


def get_dataset(dataset_name, seed, num_samples=utils.DEFAULT_NUM_SAMPLES):
    # load dataset once outside
    if dataset_name == "cc3m":
        dataset = load_cc3m_dataset(num_samples=num_samples)
    elif dataset_name =="cc3m_recaptioned":
        dataset = load_cc3m_subset(num_samples=num_samples, seed=seed)
    elif dataset_name == "wit_1024":
        dataset = load_dataset("minhuh/prh", revision="wit_1024", split="train")
    elif dataset_name == "imagenet":
        dataset = load_dataset(
            "timm/imagenet-1k-wds",
            split="validation",
            streaming=True,
        )
    elif dataset_name == "stanford_cars":
        dataset = load_dataset("Donghyun99/Stanford-Cars", split="test")
    elif dataset_name == "cub_200_2011":
        dataset = load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
    elif dataset_name == "stanford_dogs":
        dataset = load_dataset("maurice-fp/stanford-dogs")
    elif dataset_name == "oxford_flowers_102":
        dataset = load_dataset("Voxel51/OxfordFlowers102")
    elif dataset_name == "coco":
        dataset = load_coco_dataset(num_samples=num_samples, seed=seed)
    elif dataset_name == "docci_with_summaries":
        # DOCCI from the original release files; the full caption is the "description" column
        dataset = load_from_disk(str(paths.DOCCI_WITH_SUMMARIES_DIR))
        if not seed:
            seed = 42
        dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract layer-wise language and/or vision features of a dataset and save one features.pt per model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=utils.DEFAULT_NUM_SAMPLES,
        help=(
            f"Size of the subsample ({', '.join(utils.SUBSAMPLED_DATASETS)}); "
            f"other sizes than {utils.DEFAULT_NUM_SAMPLES} are saved under samples=<num_samples>/"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Number of texts or images per forward pass")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wit_1024",
        choices=[
            "wit_1024",
            "coco",
            "docci_with_summaries",
            "cc3m_recaptioned",
            "cc3m",
            "imagenet",
            "stanford_cars",
            "cub_200_2011",
            "stanford_dogs",
            "oxford_flowers_102",
        ],
        help="Dataset name",
    )
    parser.add_argument("--caption_idx", type=int, default=0, help="wit_1024 only: index of the caption used for each image")
    parser.add_argument(
        "--modality",
        type=str,
        default="all",
        choices=["vision", "language", "all"],
        help="Features to extract; with all, both language and vision (the caption type only applies to language)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(paths.FEATURES_DIR),
        help="Root directory for the feature files",
    )
    parser.add_argument(
        "--language_models",
        nargs="+",
        default=[
            "bigscience/bloomz-7b1",
            "huggyllama/llama-7b",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "google/siglip2-so400m-patch14-224",
            "nvidia/NV-Embed-v2",
            "google/siglip-so400m-patch14-224",
        ],
        metavar="LLM",
        help="Space-separated list of language model names.",
    )

    parser.add_argument(
        "--vision_models",
        nargs="+",
        default=[
            "facebook/vit-mae-large",
            "facebook/dinov2-large",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "google/siglip2-so400m-patch14-224",
            "google/siglip-so400m-patch14-224",
        ],
        metavar="LVM",
        help="Space-separated list of vision model names.",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Use randomly initialised weights instead of the pretrained ones (saved under random_init/)",
    )
    parser.add_argument(
        "--caption_type",
        type=str,
        default=None,
        choices=utils.CAPTION_TYPES,
        help=(
            "Caption column of the language features; required for cc3m_recaptioned and docci_with_summaries, "
            "not allowed for other datasets, ignored for vision features"
        ),
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=0,
        help="Without --seeds: number of random seeds to draw, one run each; 0: a single run without seed (wit_1024) or default seed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds of the 1024-sample subsample, one run each (coco, cc3m_recaptioned, docci_with_summaries; "
            "docci_with_summaries uses seed 42 without one). Other datasets ignore the seed, but it is still part of the output path"
        ),
    )

    args = parser.parse_args()

    # the caption type only applies to the language features, vision features ignore it
    if args.modality in ["language", "all"]:
        utils.check_caption_type(parser, args.dataset, args.caption_type)

    if args.num_samples < 1:
        parser.error(f"--num_samples must be positive, got {args.num_samples}")
    if args.num_samples != utils.DEFAULT_NUM_SAMPLES and args.dataset not in utils.SUBSAMPLED_DATASETS:
        parser.error(f"--num_samples only applies to {', '.join(utils.SUBSAMPLED_DATASETS)}, not to {args.dataset}")

    if not args.seeds:
        if args.num_seeds > 0:
            seeds = random.sample(range(10_000_000), args.num_seeds)
        else:
            seeds = None
    else:
        seeds = args.seeds

    for seed in (seeds if seeds is not None else [None]):
        dataset = get_dataset(args.dataset, seed, args.num_samples)

        if args.modality in ["all", "language"]:
            extract_text_features(args.language_models, dataset, seed, args)

        if args.modality in ["all", "vision"]:
            extract_image_features(args.vision_models, dataset, seed, args)
