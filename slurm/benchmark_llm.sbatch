#!/bin/bash
#SBATCH --job-name="lm eval"
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=45:00:00
#SBATCH -o logs/lm_eval_%j.out
#SBATCH -e logs/lm_eval_%j.err

# one task per job: sbatch slurm/benchmark_llm.sbatch <mmlu|arc_challenge|gsm8k>
# submit from the repository root with the environment activated (conda activate benchmark_env); logs/ must exist

mkdir -p logs
mkdir -p benchmark_results

# MODELS
MODELS=(
    "bigscience/bloomz-1b1"
    "huggyllama/llama-7b"
    "bigscience/bloomz-1b7"
    "bigscience/bloomz-7b1"
    "mistralai/Mistral-7B-v0.3"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "huggyllama/llama-13b"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    "google/gemma-2-9b"
    "google/gemma-2b"
    "Qwen/Qwen2.5-14B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-1B"
    "google/gemma-7b"
    "huggyllama/llama-30b"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "mistralai/Mistral-7B-v0.1"
)

# TASK and its number of few-shot examples
TASK=${1:-mmlu}
case "$TASK" in
    mmlu) NUM_FEWSHOT=5 ;;
    arc_challenge) NUM_FEWSHOT=4 ;;
    gsm8k) NUM_FEWSHOT=5 ;;
    *) echo "Unknown task: $TASK (choose mmlu, arc_challenge or gsm8k)" >&2; exit 1 ;;
esac

# SETTINGS
BATCH_SIZE=1

# RUN BENCHMARKS
for MODEL in "${MODELS[@]}"; do

    MODEL_NAME=$(basename "$MODEL")

    OUTPUT_FILE="benchmark_results/${MODEL_NAME}_${TASK}.json"

    lm_eval \
        --model hf \
        --model_args pretrained="$MODEL" \
        --tasks "$TASK" \
        --batch_size "$BATCH_SIZE" \
        --num_fewshot "$NUM_FEWSHOT" \
        --output_path "$OUTPUT_FILE"
done