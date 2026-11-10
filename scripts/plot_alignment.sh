ALIGNMENT_DIR=$(python -m src.paths ALIGNMENT_DIR)
PLOTS_DIR=$(python -m src.paths PLOTS_DIR)

python -m plots.plot_alignment \
  --alignment_root $ALIGNMENT_DIR/docci_with_summaries_1961180/language_avg_description_vision_cls/pca=None/mutual_knn/topk=10/pretrained \
  --heatmap_root $PLOTS_DIR/heatmaps/docci_with_summaries_1961180/language_avg_description_vision_cls
