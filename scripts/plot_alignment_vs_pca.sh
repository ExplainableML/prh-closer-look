python -m plots.plot_alignment_vs_pca \
    --pca_modes "None" "0.9" "0.8" "0.7" "0.6" "0.5" "0.4" "0.3" "0.2" "0.1" \
    --model_y facebook/dinov2-large \
    --models_x bigscience/bloomz-1b1 google/gemma-2b mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --dataset wit_1024 \
    --layer_mode max