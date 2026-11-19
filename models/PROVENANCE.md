# Active detector provenance and attribution

## Active checkpoint

- **Model:** YOLOv12-small road damage detector; `D40` is the pothole class.
- **Source:** [rezzzq/yolo12s-road-damage-rdd2022](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022)
- **File:** `pothole_yolo12s_rdd2022.pt`, 18,966,618 bytes (about 18.1 MiB / 19 MB decimal).
- **SHA-256:** `ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0`
- **Checkpoint license declared by publisher:** MIT.
- **Training data identified by publisher:** RDD2022.
- **Dataset license discrepancy:** the linked Hugging Face RDD2022 mirror declares CC BY-SA 4.0, while the NLM catalog entry for the linked Figshare release declares CC BY 4.0. The model card names RDD2022 but does not resolve this disagreement. The checkpoint's MIT declaration is separate and does not determine whether redistribution of the trained weights complies with dataset terms. Resolve rights with the relevant publishers/rights holders before public redistribution or production use.
- **Dataset attribution:** Deeksha Arya, Hiroya Maeda, Sanjay Kumar Ghosh, Durga Toshniwal, and Yoshihide Sekimoto, “RDD2022: A multi-national image dataset for automatic Road Damage Detection,” [arXiv:2209.08538](https://arxiv.org/abs/2209.08538). The linked [dataset card](https://huggingface.co/datasets/dronefreak/RDD2022) describes itself as an unofficial redistribution/conversion of RDD2022 and declares CC BY-SA 4.0.
- **Model card evidence:** reports a held-out RDD2022 test split; pothole AP@0.5 72.71 and AP@0.5:0.95 45.6. These are publisher-reported benchmark numbers and are not RoadGuard results.
- **RoadGuard diagnostic:** on the local ten-image HRP4K test sample (8 labelled pothole boxes, confidence 0.25, NMS IoU 0.45, input size 1280), the configured checkpoint produced 1 true positive, 1 false positive, and 7 false negatives. Precision was 0.500, recall 0.125, and F1 0.200 for this sample only. It is not a production accuracy estimate; the per-image artifacts are intentionally excluded from the public repository because they include local evaluation output.
- **Runtime:** loaded and inferred with the installed Ultralytics 8.4.215 / PyTorch 2.15.0+cpu. The downloaded file's SHA and byte size are checked before inference.

## Comparison candidate not selected

[dronefreak/rdd2022-yolov8s](https://huggingface.co/dronefreak/rdd2022-yolov8s) reports pothole AP@0.5 73.77 and AP@0.5:0.95 45.6, with a 22.5 MB checkpoint and example predictions. Its model repository declares AGPL-3.0; the linked RDD2022 dataset separately declares CC BY-SA 4.0. It was loaded and evaluated, but made no true positive on the RoadGuard sample at 640 or 1280 pixels and had three false positives at 640. It was not selected.

## Previous checkpoint retained

`pothole_yolov8s.previous.pt` is the exact prior checkpoint backup (22,524,266 bytes; SHA-256 `af2ac6ce7bfec72e71643659ac946caf80ced84869e526a60135c457abfbb200`). Its source model card declares Apache-2.0, but its linked HF dataset declares OpenRAIL while the originating Roboflow page says CC BY 4.0. The training-data metadata discrepancy remains unresolved. The backup is retained for rollback, not treated as an independently rights-cleared distribution artifact.
