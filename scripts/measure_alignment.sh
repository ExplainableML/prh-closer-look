python -m tools.measure_alignment \
    --dataset wit_1024 \
    --modality_x language \
    --language_models Qwen/Qwen3-VL-Embedding-8B \
    --modality_y vision \
    --vision_models Qwen/Qwen3-VL-Embedding-8B \
    --metric mutual_knn \
    --topk 10
    # --kernel rbf \
    # --rbf_sigma 2.03