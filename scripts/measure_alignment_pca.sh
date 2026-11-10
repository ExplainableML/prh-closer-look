#!/bin/bash
# PCA compression (Section 3.1): reduce the features of the language models and the vision model with a
# separate PCA per layer, then measure their alignment without PCA and for every PCA mode.
# The features must be extracted first (tools.extract_features).
#
# Settings can be overridden with environment variables, e.g.
#   DATASET=docci_with_summaries CAPTIONS=description SEEDS="1961180 2981585" bash scripts/measure_alignment_pca.sh
set -e

DATASET=${DATASET:-wit_1024}
CAPTIONS=${CAPTIONS:-}    # caption type of the language features (cc3m_recaptioned, docci_with_summaries)
SEEDS=${SEEDS:-}          # dataset seeds; empty for a single run (wit_1024)
LANGUAGE_MODELS=${LANGUAGE_MODELS:-"bigscience/bloomz-1b1 huggyllama/llama-7b google/gemma-2b mistralai/Mistral-7B-v0.3 Qwen/Qwen3-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"}
VISION_MODEL=${VISION_MODEL:-facebook/dinov2-large}
PCA_MODES=${PCA_MODES:-"0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"}

DATA_ARGS="--dataset $DATASET ${SEEDS:+--seeds $SEEDS}"

python -m tools.reduce_features $DATA_ARGS --modality language --models $LANGUAGE_MODELS \
    ${CAPTIONS:+--captions $CAPTIONS} --pca_modes $PCA_MODES
python -m tools.reduce_features $DATA_ARGS --modality vision --models $VISION_MODEL --pca_modes $PCA_MODES

for PCA_MODE in None $PCA_MODES; do
    PCA_ARGS=""
    if [ "$PCA_MODE" != None ]; then
        PCA_ARGS="--pca_mode $PCA_MODE"
    fi

    python -m tools.measure_alignment $DATA_ARGS \
        --modality_x language --language_models $LANGUAGE_MODELS ${CAPTIONS:+--captions_x $CAPTIONS} \
        --modality_y vision --vision_models $VISION_MODEL \
        --metric mutual_knn --topk 10 \
        $PCA_ARGS
done
