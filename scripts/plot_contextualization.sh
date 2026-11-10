for vlm_score in last best; do
    python -m plots.plot_contextualization datasets \
        --datasets wit_1024 coco docci_with_summaries-description docci_with_summaries-caption_25 docci_with_summaries-caption_10 \
        --vlm_score $vlm_score
done
