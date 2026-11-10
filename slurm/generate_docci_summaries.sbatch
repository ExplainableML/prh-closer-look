#!/bin/bash
#SBATCH --job-name="generate docci summaries"
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH -o logs/generate_docci_summaries_%j.out
#SBATCH -e logs/generate_docci_summaries_%j.err

# submit from the repository root with the environment activated (conda activate plat_env_orig); logs/ must exist

python -m tools.generate_docci_summaries \
    --model Qwen/Qwen3-8B