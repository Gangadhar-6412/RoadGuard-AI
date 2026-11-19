"""Evaluate the pinned checkpoint on the small, held-out HRP4K sample."""
import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DETECTION_CONFIDENCE, DETECTION_IMAGE_SIZE, DETECTION_IOU, MODEL_PATH
from modules.detection import detect

SAMPLE = ROOT / 'tests' / 'evaluation' / 'hrp4k'
OUTPUT = ROOT / 'data' / 'model_eval'


def load_ground_truth(label_path, width, height):
    boxes = []
    for line in label_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = map(float, line.split()[:5])
        boxes.append((x * width - w * width / 2, y * height - h * height / 2,
                      x * width + w * width / 2, y * height + h * height / 2))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=MODEL_PATH,
                        help='Checkpoint to evaluate. Defaults to the configured model.')
    parser.add_argument('--confidence', type=float, default=DETECTION_CONFIDENCE)
    parser.add_argument('--iou', type=float, default=DETECTION_IOU,
                        help='Ultralytics NMS IoU threshold (matching remains fixed at 0.5).')
    parser.add_argument('--image-size', type=int, default=DETECTION_IMAGE_SIZE,
                        help='Inference image size; defaults to the configured detector size.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT,
                        help='Directory for per-image CSV, summary JSON and annotated outputs.')
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    os.environ['ROADGUARD_YOLO_MODEL'] = str(model_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in sorted(SAMPLE.glob('*.jpg')):
        label_path = image_path.with_suffix('.txt')
        with Image.open(image_path) as image:
            width, height = image.size
        truth = load_ground_truth(label_path, width, height)
        annotated_path = output_dir / f'{image_path.stem}_annotated.jpg'
        prediction = detect(str(image_path), str(annotated_path), confidence=args.confidence, iou=args.iou,
                            image_size=args.image_size)
        predicted = prediction.get('boxes', [])
        import cv2
        visual = cv2.imread(prediction.get('annotated_path') or str(image_path))
        # Green boxes show HRP4K labels; cyan boxes are model predictions.
        for x1, y1, x2, y2 in truth:
            p1, p2 = (round(x1), round(y1)), (round(x2), round(y2))
            cv2.rectangle(visual, p1, p2, (0, 220, 90), 3)
            cv2.putText(visual, 'HRP4K ground truth', (p1[0], max(22, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 90), 2, cv2.LINE_AA)
        evaluation_path = output_dir / f'{image_path.stem}_evaluation.jpg'
        cv2.imwrite(str(evaluation_path), visual)
        ranked = sorted(predicted, key=lambda b: b['confidence'], reverse=True)
        matched, used, matched_predictions = 0, set(), set()
        for pred_index, box in enumerate(ranked):
            xyxy = (box['x1'], box['y1'], box['x2'], box['y2'])
            choices = [(iou(xyxy, gt), idx) for idx, gt in enumerate(truth) if idx not in used]
            best_iou, best_idx = max(choices, default=(0.0, -1))
            if best_iou >= 0.5:
                matched += 1
                used.add(best_idx)
                matched_predictions.add(pred_index)
        false_pos = len(predicted) - matched
        missed = len(truth) - matched
        detailed_predictions = [
            {'box': [b['x1'], b['y1'], b['x2'], b['y2']], 'confidence': b['confidence'],
             'match': 'TP' if idx in matched_predictions else 'FP'}
            for idx, b in enumerate(ranked)
        ]
        records.append({
            'image': image_path.name, 'ground_truth': len(truth), 'predictions': len(predicted),
            'confidences': ';'.join(f"{b['confidence']:.4f}" for b in predicted),
            'prediction_boxes_and_match_json': json.dumps(detailed_predictions),
            'false_positive_boxes_json': json.dumps([p for p in detailed_predictions if p['match'] == 'FP']),
            'missed_ground_truth_boxes_json': json.dumps([box for idx, box in enumerate(truth) if idx not in used]),
            'true_positive_boxes_iou50': matched, 'false_positive_boxes_iou50': false_pos,
            'missed_ground_truth_boxes_iou50': missed, 'annotated_path': str(evaluation_path.resolve()),
            'result': prediction['result'], 'note': prediction['note'],
        })
        print(f"{image_path.name}: GT={len(truth)} pred={len(predicted)} TP={matched} FP={false_pos} FN={missed} conf=[{records[-1]['confidences']}] {prediction['result']}")
    csv_path = output_dir / 'hrp4k_sample_results.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    scores = [float(score) for r in records for score in r['confidences'].split(';') if score]
    summary = {
        'model': str(model_path), 'confidence_threshold': args.confidence, 'nms_iou_threshold': args.iou,
        'image_size': args.image_size,
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'model_size_bytes': model_path.stat().st_size,
        'box_match_iou_threshold': 0.5, 'images': len(records),
        'positive_images': sum(r['ground_truth'] > 0 for r in records),
        'negative_images': sum(r['ground_truth'] == 0 for r in records),
        'ground_truth_boxes': sum(r['ground_truth'] for r in records),
        'predicted_boxes': sum(r['predictions'] for r in records),
        'true_positive_boxes': sum(r['true_positive_boxes_iou50'] for r in records),
        'false_positive_boxes': sum(r['false_positive_boxes_iou50'] for r in records),
        'missed_ground_truth_boxes': sum(r['missed_ground_truth_boxes_iou50'] for r in records),
        'precision': (sum(r['true_positive_boxes_iou50'] for r in records) /
                      (sum(r['true_positive_boxes_iou50'] for r in records) +
                       sum(r['false_positive_boxes_iou50'] for r in records)))
                     if sum(r['predictions'] for r in records) else None,
        'recall': (sum(r['true_positive_boxes_iou50'] for r in records) /
                   sum(r['ground_truth'] for r in records)) if sum(r['ground_truth'] for r in records) else None,
        'f1': (2 * sum(r['true_positive_boxes_iou50'] for r in records) /
               (2 * sum(r['true_positive_boxes_iou50'] for r in records) +
                sum(r['false_positive_boxes_iou50'] for r in records) +
                sum(r['missed_ground_truth_boxes_iou50'] for r in records)))
              if (2 * sum(r['true_positive_boxes_iou50'] for r in records) +
                  sum(r['false_positive_boxes_iou50'] for r in records) +
                  sum(r['missed_ground_truth_boxes_iou50'] for r in records)) else None,
        'confidence_distribution': {
            'predicted_box_count': len(scores), 'scores': scores,
            'min': min(scores) if scores else None, 'max': max(scores) if scores else None,
            'mean': sum(scores) / len(scores) if scores else None,
            'median': statistics.median(scores) if scores else None,
            'bins_0.1_wide': {f'{low/100:.1f}-{(low+10)/100:.1f}':
                              sum(low / 100 <= score < (low + 10) / 100 for score in scores)
                              for low in range(0, 100, 10)},
        },
        'false_positive_images': [
            {'image': r['image'], 'false_positive_boxes': r['false_positive_boxes_iou50']}
            for r in records if r['false_positive_boxes_iou50']
        ],
        'missed_boxes_by_image': [
            {'image': r['image'], 'missed_ground_truth_boxes': r['missed_ground_truth_boxes_iou50']}
            for r in records if r['missed_ground_truth_boxes_iou50']
        ],
        'caveat': 'Small 10-image held-out sample only; not a production accuracy estimate.',
    }
    (output_dir / 'hrp4k_sample_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Results: {csv_path}')


if __name__ == '__main__':
    main()
