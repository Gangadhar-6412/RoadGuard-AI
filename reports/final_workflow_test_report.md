# RoadGuard AI — Final Workflow Test Report

Run: 2026-11-19 04:18:36 UTC  
Database: temporary isolated SQLite database; no existing application data was used or changed.  
Checkpoint: `pothole_yolo12s_rdd2022.pt` (18,966,618 bytes), SHA-256 `ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0`.  
Test photo: `tests/evaluation/hrp4k/30811.jpg`. This is a held-out evaluation image, not a newly collected field report.

## Results

| Workflow check | Actual result |
|---|---|
| Citizen photograph inference | Pothole candidate; confidence 0.6591; 2 box(es) |
| Citizen intake gate | Pending human decision; no work order before review: True |
| Citizen review / assignment | TEST ONLY scripted reviewer recorded `Verified`; assigned `RG-WO-261119-36764` to Ward 12 Roadworks Team |
| Priority | P2 — High, 54.8/100; inputs severity=65, traffic=55, importance=60, repeats=20 |
| Citizen evidence / closure | Contractor update and after-evidence file persisted; TEST ONLY admin approval moved order to `Closed` |
| Camera video inference | 2 sampled frames, 2 candidate frames, 1 report(s), 1 repeated frame(s) suppressed |
| Camera review / assignment | No order before review: True; TEST ONLY reviewer assigned `RG-WO-261119-64913` to Ward 12 Roadworks Team |
| SQLite persistence | 2 reports, 2 work orders, 26 event records |
| External notifications | Simulated event records only; no email or contractor message was sent |

## Actual model output

Citizen image: `Pothole candidate` at confidence `0.6591`. Durable annotated outputs: `reports/annotated_citizen_30811.jpg` and `reports/annotated_camera_frame_30811.jpg`. Test-only synthetic evidence is `reports/TEST_ONLY_synthetic_after_evidence.png` and is labelled in-image as not a real repair. Box data:

```json
[
  {
    "class_id": 3,
    "label": "pothole",
    "confidence": 0.6591,
    "x1": 2081.1,
    "y1": 1551.6,
    "x2": 2243.4,
    "y2": 1621.1
  },
  {
    "class_id": 3,
    "label": "pothole",
    "confidence": 0.2682,
    "x1": 2262.2,
    "y1": 1200.7,
    "x2": 2353.8,
    "y2": 1248.4
  }
]
```

Camera frame inference:

```json
[
  {
    "frame": 0,
    "result": "Pothole candidate",
    "confidence": 0.6587,
    "boxes": [
      {
        "class_id": 3,
        "label": "pothole",
        "confidence": 0.6587,
        "x1": 2081.0,
        "y1": 1551.7,
        "x2": 2243.4,
        "y2": 1621.3
      },
      {
        "class_id": 3,
        "label": "pothole",
        "confidence": 0.2714,
        "x1": 2262.3,
        "y1": 1200.7,
        "x2": 2353.8,
        "y2": 1248.4
      }
    ]
  },
  {
    "frame": 1,
    "result": "Pothole candidate",
    "confidence": 0.6587,
    "boxes": [
      {
        "class_id": 3,
        "label": "pothole",
        "confidence": 0.6587,
        "x1": 2081.0,
        "y1": 1551.7,
        "x2": 2243.4,
        "y2": 1621.3
      },
      {
        "class_id": 3,
        "label": "pothole",
        "confidence": 0.2714,
        "x1": 2262.3,
        "y1": 1200.7,
        "x2": 2353.8,
        "y2": 1248.4
      }
    ]
  }
]
```

The video was a TEST ONLY two-frame lossless clip generated from the same still image. It validates the video inference path and duplicate-frame suppression, not live camera performance. Detections are actual configured checkpoint outputs. Reviewer, contractor update, synthetic evidence, and admin approval were scripted TEST ONLY actions in an isolated database; they are not claims of field review or completed road repairs. The synthetic evidence image explicitly says it is not repair evidence from a real site.

## Event and record evidence

Report IDs: `[1, 2]`.  
Work order IDs: `['RG-WO-261119-36764', 'RG-WO-261119-64913']`.  
Recorded event stages in order: `Received, Detection, Duplicate check, TrustShield, Priority, Human review, Human verification, Assignment, Notification, Contractor update, Repair evidence, Admin approval, Camera, Received, Detection, Duplicate check, TrustShield, Priority, Human review, Frame detection, Duplicate frame suppression, Frame detection, Camera, Human verification, Assignment, Notification`.  
All logged notification events were marked simulated: `True`.

## Limitations

The detector has documented poor recall on the evaluation sample and was not tuned in this workflow task. A model candidate is only evidence for human review; the pipeline does not dispatch based solely on confidence. Human and admin names are recorded as supplied text; the application has no authentication/authorization system, so the approval step is a workflow gate, not verified identity control. The final `Closed` state above demonstrates persistence logic only with synthetic test evidence and does not represent a completed physical repair.
