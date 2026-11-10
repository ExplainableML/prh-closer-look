import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
import json
from collections import defaultdict
import webdataset as wds
import itertools
import random
import numpy as np
from src import paths


class COCODataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        ann_path = os.path.join(data_dir, "annotations", f"captions_{split}2017.json")
        # official zips: <data_dir>/<split>2017/, COCO-Stuff layout: <data_dir>/images/<split>2017/
        img_dir = os.path.join(data_dir, f"{split}2017")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(data_dir, "images", f"{split}2017")

        self.img_dir = img_dir

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.annotations = data["annotations"]

        self.image_id_to_ann_ids = defaultdict(list)
        for i, ann in enumerate(self.annotations):
            self.image_id_to_ann_ids[ann["image_id"]].append(i)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        caption = ann["caption"]
        image_id = ann["image_id"]

        filename = f"{image_id:012d}.jpg"
        img_path = os.path.join(self.img_dir, filename)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": caption,
            "image_id": image_id,
            "annotation_id": idx
        }
    
    def get_by_image_id(self, image_id):
        ann_ids = self.image_id_to_ann_ids[image_id]
        return [self[i] for i in ann_ids]


class CC3MDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        csv_path = os.path.join(data_dir, f"{split}.csv")
        self.annotations = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.data_dir, row["path"])
        caption = row["caption"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "text": caption,
        }


def load_cc3m_dataset(num_samples, seed=42):
    if seed is not None:
        torch.manual_seed(seed)

    dataset = CC3MDataset(data_dir=str(paths.CC3M_RECAPTIONED_DIR), split="train")
    random_indices = torch.randperm(len(dataset))[:num_samples].tolist()
    subset = Subset(dataset, random_indices)

    return subset


def load_cc3m_subset(
    split="train",
    num_samples=1024,
    seed=42,
    shards_to_use=5,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # separate RNGs for clarity
    shard_rng = random.Random(seed)
    shuffle_rng = random.Random(seed + 1)

    if split == "train":
        base_path = f"{paths.CC3M_RECAPTIONED_DIR}/cc3m-train-"
        max_shard_index = 282
    elif split == "validation":
        base_path = f"{paths.CC3M_RECAPTIONED_DIR}/cc3m-validation-"
        max_shard_index = 15
    else:
        raise ValueError("Invalid split")

    shard_indices = sorted(
        shard_rng.sample(range(max_shard_index + 1), shards_to_use)
    )

    shard_paths = [
        f"{base_path}{idx:04d}.tar"
        for idx in shard_indices
    ]

    def process_sample(sample):
        captions = sample["json"]
        return {
            "image": sample["jpg"],
            "raw_caption": captions["raw_caption"],
            "shortSV_captions": captions["shortSV_captions"],
            "longSV_captions": captions["longSV_captions"],
        }

    dataset = (
        wds.WebDataset(shard_paths, shardshuffle=False)
        .shuffle(5000, rng=shuffle_rng)
        .decode("pil")
        .map(process_sample)
    )

    dataset = itertools.islice(dataset, num_samples)

    return dataset


def load_coco_dataset(num_samples, seed=42):
    if seed is not None:
        torch.manual_seed(seed)

    dataset = COCODataset(data_dir=str(paths.COCO_DIR), split="train")
    random_indices = torch.randperm(len(dataset))[:num_samples].tolist()
    subset = Subset(dataset, random_indices)

    return subset


def load_coco_validation_set():
    dataset = COCODataset(data_dir=str(paths.COCO_DIR), split="val")
    return dataset