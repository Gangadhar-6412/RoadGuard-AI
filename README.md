# RoadGuard AI

Local smart-city road maintenance workflow demo: **detect → verify → prioritize → assign → notify → repair → approve**.

## Run locally

Use Python 3.12 for Streamlit Community Cloud (the app was also tested locally on Python 3.14.6). From this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

On systems without the `py` launcher, use `python` to create the virtual environment. The app uses local SQLite at `data/roadguard.sqlite3`; uploads are saved under `data/uploads/`.

Run the deterministic integration checks before launching:

```powershell
python -m unittest discover -s tests -v
```

The test-only detector is an explicitly labelled stub used solely to exercise the persisted end-to-end workflow. It is not imported by the app and is not evidence of AI detection. The product will only report a pothole candidate when a configured model returns the configured pothole class.

## Streamlit Community Cloud

Deploy `app.py` from the repository root and select Python 3.12 in the deployment's Advanced settings. The deployment uses `.streamlit/config.toml` and expects the 18,966,618-byte checkpoint at `models/pothole_yolo12s_rdd2022.pt`; it fits under GitHub's normal file-size limit. However, dataset-license metadata conflicts with the checkpoint's MIT declaration. Do not commit or distribute the checkpoint publicly until its redistribution rights have been reviewed. Without the checkpoint, the application starts but inference reports `Not evaluated`. Other model files and all `data/` runtime state are excluded by `.gitignore`. Community Cloud does not guarantee persistence for files generated while the app runs, so local SQLite and uploads are suitable only for a disposable demonstration. Use a managed database and object storage before relying on public submissions. Notifications are simulated, contractor records are seeded demo data, and the reviewer/admin name fields are not authentication.

## What works

- Citizen and municipal officer upload forms use one persisted processing pipeline.
- Upload validation, perceptual image hashing, coordinate proximity duplicate matching, advisory TrustShield assessment, transparent priority formula, ward contractor lookup, unique work order creation and recorded notification events.
- Contractor acceptance/status updates, after-repair evidence submission and administrator-controlled closure.
- Dashboard, event log, evidence review, map, CSV analytics and demo reset.
- Camera screen previews a local prerecorded clip and records monitoring events.

## Detector setup and limits

RoadGuard uses the public [YOLOv12s RDD2022 road-damage checkpoint](https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022), stored at `models/pothole_yolo12s_rdd2022.pt`. It detects five classes; `D40` is the pothole class. The model repository declares MIT for the checkpoint and identifies RDD2022 as its training data. Dataset licensing metadata conflicts: the Hugging Face RDD2022 mirror declares CC BY-SA 4.0, while the NLM catalog entry for the linked Figshare release declares CC BY 4.0. The checkpoint's MIT declaration does not settle whether redistributing trained weights complies with the dataset terms. Resolve this with the relevant rights holders before public checkpoint redistribution or production use. The model card requests an RDD2022 citation. The 18,966,618-byte weights are pinned by SHA-256 in `config.py`. Inference uses 1280 pixels, which increases CPU cost; camera work remains bounded by frame skip and the sample limit. On the ten-image HRP4K diagnostic, the configured model found 1 of 8 labelled potholes, missed 7, and produced 1 false positive; this is not a production accuracy claim. See [`models/PROVENANCE.md`](models/PROVENANCE.md) for attribution, licensing evidence, and limitations. Detailed evaluation artifacts remain local under the ignored `data/model_eval/` directory.

The previous [peterhdd YOLOv8s checkpoint](https://huggingface.co/peterhdd/pothole-detection-yolov8) is retained as `models/pothole_yolov8s.previous.pt` for rollback. Its model card says Apache-2.0; its linked Hugging Face dataset metadata says OpenRAIL while the originating Roboflow page says CC BY 4.0. That training-data discrepancy remains unresolved, so the backup is not redistributed as a validated detector. The RDD2022 YOLOv8s candidate from [dronefreak](https://huggingface.co/dronefreak/rdd2022-yolov8s) was also compared; its model repository declares AGPL-3.0, separately from the RDD2022 CC BY-SA 4.0 dataset.

Install the model if it is not present:

```powershell
.venv\Scripts\python.exe scripts\download_model.py
```

Image reports run model inference automatically, save per-box coordinates and confidence values, and show a labelled annotation next to the original photo. Confidence and NMS IoU thresholds are adjustable on the report and camera screens. Camera frame stride and maximum sampled frames are configurable; the default samples one frame every two seconds, up to 12 frames. Repeated adjacent-frame boxes are associated within the current clip and suppressed from creating another report. This is a simple visual matching heuristic, not persistent object tracking or a guarantee against duplicates in difficult footage. The camera page annotates positive frames and sends detections through the same report automation pipeline. A custom local path can override the default with `$env:ROADGUARD_YOLO_MODEL`; it must contain a model with an explicitly named pothole/road-damage class. Generic COCO weights are not treated as pothole models. If weights are missing, incompatible, or fail to load, the result remains **Not evaluated** and is held for review; predictions are never fabricated.

To verify the real checkpoint on the included attributed held-out road photo:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The integration test runs real model inference when weights are present, checks that pothole boxes and an annotated image are produced, and also tests the event/work-order pipeline separately with an explicitly labelled test-only detector stub. Run a small independent evaluation using ten held-out HRP4K test images (five positive, five clean-road negatives; CC BY 4.0; attribution is in `tests/evaluation/hrp4k/README.md`):

```powershell
.venv\Scripts\python.exe scripts\evaluate_model.py --confidence 0.25 --iou 0.45
```

The script writes per-image boxes, confidence values, box-level false positives/misses and images with HRP4K ground truth overlaid in green to `data/model_eval/`. Matching uses box IoU 0.50. Use `--image-size 1280` (the configured default) to reproduce the active detector's evaluation. This sample is deliberately small and is not a production accuracy estimate; use the recorded results as a diagnostic, not a deployment claim.

## Demo data and verification

Contractors are demo records for Wards 12, 4 and 7. Notifications are logged simulations; no external email is sent. Duplicate thresholds and priority weights are documented in the app. TrustShield uses image dimensions, visual variance, available EXIF, optional model signal and duplicate/location signals. It cannot prove authenticity. Repair evidence requires human inspection; only the administrator review page can close a work order.

No real vehicle feed, road GIS catalog, government safety standard, or official contractor directory is included. The default dataset is empty apart from clearly identified demo contractor assignments.
