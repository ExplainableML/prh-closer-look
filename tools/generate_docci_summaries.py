import argparse
from pathlib import Path
import json

import torch
from datasets import load_from_disk, Image, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src import paths


# Compression levels
RATIOS = {
    "caption_75": 0.75,
    "caption_50": 0.50,
    "caption_25": 0.25,
    "caption_10": 0.10,
}

# columns of the generated results, in checkpoint order
RESULT_COLUMNS = (
    ["original_words"]
    + list(RATIOS)
    + [f"{name}_words" for name in RATIOS]
)

PROMPT_TEMPLATE = """You are compressing a detailed image description to {target_ratio} of its original length.

Your goal is to reduce the LEVEL OF VISUAL DETAIL in the description.
At shorter target lengths, the output should describe substantially fewer
visual details, not merely express the same details using fewer words.
Your task is to rewrite the description as a shorter caption of {target_words} words while preserving
the important visually grounded information. The {target_words} target is a strict length requirement, not a suggestion!

IMPORTANT RULES:
- Use ONLY information contained in the original description.
- Do NOT add, infer, or invent information.
- Keep the remaining text grammatically complete and natural.
- Do NOT create sentence fragments.
- Do NOT remove function words merely to save space.
- At shorter target lengths, it is EXPECTED that many secondary details
will be completely omitted.
- Prioritize the most salient subjects and their primary actions.
- The target length is {target_words} words.
- Your caption HAS TO be LONGER than {min_words} and SHORTER than {max_words}.
- Write a natural, self-contained image caption.
- A simple caption containing only the main subject and action is preferable
to a dense caption containing many secondary details.
- Output ONLY the compressed caption. Do not explain your changes.

The original description contains approximately {original_words} words.

Original description:
{caption}

Compressed caption (~{target_words} words):
"""


def load_docci_from_files(docci_dir: str | Path, split: str = "train") -> Dataset:
    docci_dir = Path(docci_dir)

    descriptions_path = docci_dir / "docci_descriptions.jsonlines"
    images_dir = docci_dir / "images"

    examples = []

    with open(descriptions_path, "r") as f:
        for line in tqdm(f, desc="Loading DOCCI"):
            item = json.loads(line)

            if item["split"] != split:
                continue

            image_path = images_dir / item["image_file"]

            if not image_path.exists():
                print(f"Warning: missing image: {image_path}")
                continue

            examples.append(
                {
                    "example_id": item["example_id"],
                    "split": item["split"],
                    "image": str(image_path),
                    "description": item["description"],
                }
            )

    dataset = Dataset.from_list(examples)

    dataset = dataset.cast_column(
        "image",
        Image(),
    )

    print(f"Loaded {len(dataset)} {split} examples")

    return dataset


def build_prompt(caption: str, target_ratio: float) -> str:
    """Create the prompt for caption compression."""

    original_words = len(caption.split())
    target_words = max(5, round(original_words * target_ratio))

    return PROMPT_TEMPLATE.format(
        target_ratio=target_ratio,
        target_words=target_words,
        min_words=max(1, target_words - 10),
        max_words=target_words + 10,
        original_words=original_words,
        caption=caption,
    )


def generate_caption(
    model,
    tokenizer,
    caption: str,
    target_ratio: float,
    max_new_tokens: int = 256,
) -> str:

    prompt = build_prompt(caption, target_ratio)

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Only decode the newly generated tokens
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return result


def empty_results():
    return {column: [] for column in RESULT_COLUMNS}


def append_results(checkpoint: Dataset, results: dict) -> Dataset:
    """Checkpoint with the newly generated results appended."""
    return Dataset.from_dict({
        column: list(checkpoint[column]) + results[column]
        for column in RESULT_COLUMNS
    })


def main(args):

    output_path = Path(args.output)

    if output_path.exists():
        raise SystemExit(f"{output_path} already exists; choose another --output or remove it first")

    checkpoint_path = (
        output_path.parent / f"{output_path.name}_checkpoint"
    )

    # Load DOCCI
    print("Loading DOCCI...")
    dataset = load_docci_from_files(args.docci_dir, args.split)

    # Load or initialize checkpoint
    if checkpoint_path.exists():

        print(f"Found checkpoint: {checkpoint_path}")
        print("Loading checkpoint...")

        checkpoint = load_from_disk(str(checkpoint_path))
        print(f"Checkpoint contains {len(checkpoint)} examples.")

        if len(checkpoint) > len(dataset):
            raise ValueError(
                f"Checkpoint has more examples ({len(checkpoint)}) than the dataset ({len(dataset)})"
            )

    else:

        print("No checkpoint found. Starting from scratch.")
        checkpoint = Dataset.from_dict(empty_results())

    start_idx = len(checkpoint)

    if start_idx == len(dataset):
        print("Checkpoint is already complete, skipping generation.")

    else:
        # Load model
        print(f"Loading model: {args.model}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        model.eval()

        # Generate captions
        results = empty_results()

        for idx in tqdm(
            range(start_idx, len(dataset)),
            desc="Generating captions",
        ):

            original_caption = dataset[idx]["description"]
            results["original_words"].append(len(original_caption.split()))

            for name, ratio in RATIOS.items():

                summary = generate_caption(
                    model=model,
                    tokenizer=tokenizer,
                    caption=original_caption,
                    target_ratio=ratio,
                )

                results[name].append(summary)
                results[f"{name}_words"].append(len(summary.split()))

            # Save checkpoint (and any remaining examples at the end)
            if (idx + 1) % args.checkpoint_every == 0 or idx + 1 == len(dataset):

                print(f"\nSaving checkpoint at index {idx + 1}...")

                checkpoint = append_results(checkpoint, results)
                checkpoint.save_to_disk(str(checkpoint_path))

                print(f"Checkpoint saved: {len(checkpoint)} examples")

                results = empty_results()

    if len(checkpoint) != len(dataset):
        raise ValueError(
            f"Checkpoint has {len(checkpoint)} examples, but the dataset has {len(dataset)}"
        )

    # Add generated columns to original dataset
    print("\nAdding generated captions to dataset...")

    for column in RESULT_COLUMNS:

        dataset = dataset.add_column(
            column,
            checkpoint[column],
        )

    # Save final dataset
    print(f"Saving final dataset to: {output_path}")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.save_to_disk(str(output_path))

    print("Done.")

    print("\nColumns:")
    print(dataset.column_names)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Shorten the DOCCI descriptions to 75, 50, 25 and 10% of their length with a language model.",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for the Hugging Face dataset (must not exist yet).",
        default=str(paths.DOCCI_WITH_SUMMARIES_DIR)
    )

    parser.add_argument(
        "--docci_dir",
        type=str,
        help="DOCCI directory with docci_descriptions.jsonlines and images/.",
        default=str(paths.DOCCI_DIR),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "qual_dev", "qual_test"],
        help="DOCCI split whose descriptions are shortened.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Hugging Face language model that writes the shortened captions.",
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=100,
        help="Save a checkpoint every this many examples.",
    )

    args = parser.parse_args()

    main(args)
