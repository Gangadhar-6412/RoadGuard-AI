"""Trace raw-class predictions and compare them to HRP4K labels without changing inference."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DETECTION_IMAGE_SIZE, DETECTION_IOU, MODEL_PATH  # noqa: E402
from scripts.evaluate_model import iou, load_ground_truth  # noqa: E402

SAMPLE = ROOT / 'tests' / 'evaluation' / 'hrp4k'
DEFAULT_OUTPUT = ROOT / 'data' / 'model_eval' / 'pipeline_debug'
THRESHOLDS = (0.01, 0.05, 0.10, 0.25, 0.50)
MATCH_IOU = 0.50
RAW_CONFIDENCE = 0.001


def match_boxes(predictions, ground_truth):
    ordered = sorted(predictions, key=lambda item: item['confidence'], reverse=True)
    used_gt = set()
    matched_pred = set()
    for pred_index, pred in enumerate(ordered):
        box = pred['xyxy']
        options = [(iou(box, gt), index) for index, gt in enumerate(ground_truth) if index not in used_gt]
        best_iou, best_index = max(options, default=(0.0, -1))
        if best_iou >= MATCH_IOU:
            used_gt.add(best_index)
            matched_pred.add(pred_index)
    return ordered, matched_pred, used_gt


def annotate_comparison(image_path, ground_truth, predictions, output_path, threshold):
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f'Could not decode {image_path}')
    height, width = original.shape[:2]
    d40 = [item for item in predictions if item['class_id'] == 3 and item['confidence'] >= threshold]
    ordered, matched_pred, matched_gt = match_boxes(d40, ground_truth)
    left = original.copy()
    right = original.copy()
    for gt_index, box in enumerate(ground_truth):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 220, 90), 5)
        cv2.putText(left, f'GT {gt_index + 1}', (x1, max(35, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 90), 3, cv2.LINE_AA)
        color = (0, 220, 90) if gt_index in matched_gt else (0, 0, 255)
        label = 'matched GT' if gt_index in matched_gt else 'MISSED GT'
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 5)
        cv2.putText(right, label, (x1, max(35, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)
    for pred_index, prediction in enumerate(ordered):
        x1, y1, x2, y2 = [int(round(value)) for value in prediction['xyxy']]
        color = (255, 210, 0) if pred_index in matched_pred else (255, 80, 220)
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 5)
        cv2.putText(right, f"D40 {prediction['confidence']:.3f}", (x1, min(height - 12, y2 + 32)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

    panel_width = 1600
    panel_height = round(height * panel_width / width)
    left = cv2.resize(left, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(right, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    header = 76
    canvas = 26 * torch.ones((panel_height + header, panel_width * 2 + 3, 3), dtype=torch.uint8).numpy()
    canvas[:header, :, :] = (30, 30, 30)
    canvas[header:, :panel_width] = left
    canvas[header:, panel_width + 3:] = right
    cv2.putText(canvas, f"Ground truth · {image_path.name}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 220, 90), 2, cv2.LINE_AA)
    cv2.putText(canvas, f'D40 predictions · confidence ≥ {threshold:.2f} · green matched / red missed',
                (panel_width + 30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 220, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--image-size', type=int, default=DETECTION_IMAGE_SIZE)
    parser.add_argument('--iou', type=float, default=DETECTION_IOU)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    names = model.names
    class_names = {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(map(str, names)))
    first_image = sorted(SAMPLE.glob('*.jpg'))[0]
    model.predict(str(first_image), conf=RAW_CONFIDENCE, iou=args.iou,
                  imgsz=args.image_size, verbose=False, device='cpu', augment=False,
                  classes=None, max_det=300)
    original_preprocess = model.predictor.preprocess
    observed_preprocessing = {}

    def capture_preprocessing(images):
        source = images[0]
        tensor = original_preprocess(images)
        observed_preprocessing.update({
            'loader_image_shape_hwc': list(source.shape), 'loader_dtype': str(source.dtype),
            'loader_channel_order': 'BGR',
            'loader_channel_means_bgr_0_255': source.mean(axis=(0, 1)).tolist(),
            'model_tensor_shape_bchw': list(tensor.shape), 'model_tensor_dtype': str(tensor.dtype),
            'model_tensor_min': float(tensor.min()), 'model_tensor_max': float(tensor.max()),
            'model_tensor_channel_means_rgb_0_1': tensor[0].mean(axis=(1, 2)).tolist(),
        })
        return tensor

    model.predictor.preprocess = capture_preprocessing
    metadata = {
        'checkpoint': str(model_path), 'sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'bytes': model_path.stat().st_size, 'ultralytics_version': __import__('ultralytics').__version__,
        'torch_version': torch.__version__, 'task': model.task, 'class_names': class_names,
        'pothole_class_id': next((class_id for class_id, name in class_names.items() if name.lower() in {'d40', 'pothole'}), None),
        'trained_image_size': model.overrides.get('imgsz'), 'inference_image_size': args.image_size,
        'stride': model.model.stride.tolist(), 'runtime_overrides': model.overrides,
        'inference': {'raw_confidence': RAW_CONFIDENCE, 'nms_iou': args.iou, 'device': 'cpu',
                      'augment': False, 'classes_filter': None, 'max_det': 300,
                      'interpretation': 'Ultralytics decoded, class-aware NMS outputs collected down to conf=0.001; not pre-NMS tensor logits.'},
        'preprocessing': {'source': 'image file path', 'loader_order': 'BGR uint8',
                          'model_order': 'RGB', 'normalization': 'float / 255 to 0..1',
                          'resize': 'Ultralytics LetterBox, stride-aligned; coordinates returned in original-image pixel space'},
        'evaluation': {'gt_format': 'single-class YOLO normalized xywh; class id 0 means pothole in HRP4K subset',
                       'matching_iou': MATCH_IOU, 'threshold_sweep': THRESHOLDS},
    }
    (output_dir / 'checkpoint_and_pipeline.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')

    per_image = []
    raw_records = []
    threshold_rows = []
    for image_path in sorted(SAMPLE.glob('*.jpg')):
        with Image.open(image_path) as image:
            width, height = image.size
        gt = load_ground_truth(image_path.with_suffix('.txt'), width, height)
        result = model.predict(str(image_path), conf=RAW_CONFIDENCE, iou=args.iou,
                               imgsz=args.image_size, verbose=False, device='cpu',
                               augment=False, classes=None, max_det=300)[0]
        predictions = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            item = {'image': image_path.name, 'class_id': class_id, 'class_name': class_names[class_id],
                    'confidence': float(box.conf[0]),
                    'xyxy': [round(float(value), 3) for value in box.xyxy[0].tolist()]}
            overlaps = [iou(item['xyxy'], truth) for truth in gt]
            item['best_pothole_gt_iou'] = round(max(overlaps, default=0.0), 5) if class_id == 3 else None
            item['best_gt_iou_any_class'] = round(max(overlaps, default=0.0), 5)
            predictions.append(item)
            raw_records.append(item)

        class_counts = Counter(item['class_id'] for item in predictions)
        per_image.append({'image_path': image_path, 'gt': gt, 'predictions': predictions})
        for threshold in THRESHOLDS:
            pothole_preds = [item for item in predictions if item['class_id'] == 3 and item['confidence'] >= threshold]
            _, matched_predictions, matched_ground_truth = match_boxes(pothole_preds, gt)
            other_class_overlaps = [item for item in predictions if item['class_id'] != 3
                                    and item['confidence'] >= threshold
                                    and item['best_gt_iou_any_class'] >= MATCH_IOU]
            threshold_rows.append({'image': image_path.name, 'threshold': threshold,
                                   'ground_truth_potholes': len(gt),
                                   'raw_post_nms_predictions_all_classes': len(predictions),
                                   'raw_class_counts_json': json.dumps({class_names[k]: v for k, v in class_counts.items()}),
                                   'pothole_predictions': len(pothole_preds),
                                   'non_pothole_predictions_overlapping_GT_iou50': len(other_class_overlaps),
                                   'true_positive': len(matched_predictions),
                                   'false_positive': len(pothole_preds) - len(matched_predictions),
                                   'false_negative': len(gt) - len(matched_ground_truth)})
        if gt:
            for threshold in (0.01, 0.25):
                annotate_comparison(image_path, gt, predictions,
                                    output_dir / 'side_by_side' / f'{image_path.stem}_conf{int(threshold*1000):03d}.jpg',
                                    threshold)

    metadata['observed_preprocessing'] = observed_preprocessing
    (output_dir / 'checkpoint_and_pipeline.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    with (output_dir / 'raw_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image', 'class_id', 'class_name', 'confidence', 'xyxy',
                                                    'best_pothole_gt_iou', 'best_gt_iou_any_class'])
        writer.writeheader()
        for row in raw_records:
            writer.writerow({**row, 'xyxy': json.dumps(row['xyxy'])})
    with (output_dir / 'thresholds_by_image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=threshold_rows[0].keys())
        writer.writeheader()
        writer.writerows(threshold_rows)

    summaries = []
    for threshold in THRESHOLDS:
        rows = [row for row in threshold_rows if row['threshold'] == threshold]
        tp, fp, fn = (sum(row[key] for row in rows) for key in ('true_positive', 'false_positive', 'false_negative'))
        all_class_counts = Counter()
        non_pothole_overlaps = 0
        for image in per_image:
            selected = [item for item in image['predictions'] if item['confidence'] >= threshold]
            all_class_counts.update(item['class_id'] for item in selected)
            non_pothole_overlaps += sum(item['class_id'] != 3 and item['best_gt_iou_any_class'] >= MATCH_IOU
                                        for item in selected)
        summaries.append({'confidence_threshold': threshold, 'nms_iou': args.iou,
                          'd40_TP': tp, 'd40_FP': fp, 'd40_FN': fn,
                          'precision': tp / (tp + fp) if tp + fp else None,
                          'recall': tp / (tp + fn) if tp + fn else None,
                          'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                          'non_pothole_class_predictions_overlapping_pothole_GT_iou50': non_pothole_overlaps,
                          'all_class_prediction_counts': {class_names[k]: all_class_counts.get(k, 0) for k in sorted(class_names)}})
    (output_dir / 'threshold_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    print(json.dumps({'metadata': metadata, 'raw_predictions': raw_records,
                      'threshold_summary': summaries}, indent=2, default=str))
    print(f'Debug artifacts: {output_dir}')


if __name__ == '__main__':
    main()
