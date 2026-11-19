"""Pothole-specific inference using the locally pinned Ultralytics checkpoint."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

from config import (DETECTION_CONFIDENCE, DETECTION_IMAGE_SIZE, DETECTION_IOU,
                    MODEL_PATH, MODEL_SHA256, MODEL_SIZE_BYTES)


@lru_cache(maxsize=2)
def _model(model_path):
    # Lazy import keeps the app usable when weights or the optional ML runtime
    # are absent. Inference errors are returned as Not evaluated, never faked.
    # Disable Ultralytics' implicit pip installs during inference; model loading
    # must not attempt network/package mutations for optional image codecs.
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    from ultralytics import YOLO
    return YOLO(model_path)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _is_pinned_checkpoint(path, modified_ns, size):
    return size == MODEL_SIZE_BYTES and _sha256(path) == MODEL_SHA256


def _name_for(names, class_id):
    return str(names[class_id] if not isinstance(names, dict) else names.get(class_id, class_id)).strip().lower()


def detect(image_path, output_path=None, confidence=DETECTION_CONFIDENCE, iou=DETECTION_IOU,
           image_size=DETECTION_IMAGE_SIZE):
    """Run real YOLO inference and return pothole boxes plus annotated output.

    Only class names that explicitly identify potholes/road damage are used,
    except for the SHA-verified pinned one-class checkpoint whose dataset
    encoded its sole class as "0". Generic COCO labels are never treated as
    pothole detections.
    """
    configured = Path(os.environ.get('ROADGUARD_YOLO_MODEL', str(MODEL_PATH))).expanduser()
    configured = configured.resolve()
    base = {'boxes': [], 'annotated_path': None, 'model_path': str(configured), 'confidence': None,
            'confidence_threshold': confidence, 'iou_threshold': iou}
    if not 0 < float(confidence) <= 1 or not 0 < float(iou) < 1:
        return {**base, 'result': 'Not evaluated', 'note': 'Confidence must be greater than 0 and at most 1; IoU must be greater than 0 and less than 1.'}
    if not configured.is_file():
        return {**base, 'result': 'Not evaluated', 'note': f'Pothole model weights are missing: {configured}. Install the pinned checkpoint to enable detection.'}

    try:
        pinned = configured == Path(MODEL_PATH).resolve()
        stat = configured.stat()
        if pinned and not _is_pinned_checkpoint(str(configured), stat.st_mtime_ns, stat.st_size):
            return {**base, 'result': 'Not evaluated', 'note': 'Pinned pothole checkpoint SHA-256 does not match the verified source artifact; refusing to load it.'}

        model = _model(str(configured))
        names = model.names
        is_pinned_single_class = pinned and len(names) == 1 and _name_for(names, 0) in {'0', 'pothole'}
        pothole_ids = {
            int(class_id) for class_id in (names.keys() if isinstance(names, dict) else range(len(names)))
            if ('pothole' in _name_for(names, int(class_id))
                or 'road_damage' in _name_for(names, int(class_id))
                or _name_for(names, int(class_id)) == 'd40')
        }
        if is_pinned_single_class:
            pothole_ids.add(0)
        if not pothole_ids:
            return {**base, 'result': 'Not evaluated', 'note': f'Configured model has no pothole/road-damage class ({names}); no detections were inferred.'}

        prediction = model.predict(str(image_path), conf=float(confidence), iou=float(iou),
                                   imgsz=int(image_size), verbose=False, device='cpu')[0]
        detections = []
        for box in prediction.boxes:
            class_id = int(box.cls[0])
            if class_id not in pothole_ids:
                continue
            x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            score = round(float(box.conf[0]), 4)
            detections.append({'class_id': class_id, 'label': 'pothole', 'confidence': score,
                               'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        annotated = None
        if detections:
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError('OpenCV could not read the input image for annotation.')
            for item in detections:
                p1 = (int(item['x1']), int(item['y1']))
                p2 = (int(item['x2']), int(item['y2']))
                label = f"Pothole {item['confidence']:.2f}"
                cv2.rectangle(image, p1, p2, (238, 211, 34), 2)
                text_y = max(22, p1[1] - 8)
                cv2.putText(image, label, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (238, 211, 34), 2, cv2.LINE_AA)
            target = Path(output_path) if output_path else Path(image_path).with_name(Path(image_path).stem + '_annotated.jpg')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), image):
                raise OSError(f'Could not write annotated image to {target}')
            annotated = str(target.resolve())

        best = max((item['confidence'] for item in detections), default=None)
        return {**base, 'result': 'Pothole candidate' if detections else 'No pothole detected',
                'confidence': best, 'boxes': detections, 'annotated_path': annotated,
                'note': (f'Configured pothole model returned {len(detections)} pothole box(es); '
                         f'best confidence {best:.3f} (conf={confidence:.2f}, IoU={iou:.2f}).' if detections else
                         f'Configured pothole model returned no boxes at confidence threshold {confidence:.2f}.')}
    except Exception as exc:
        return {**base, 'result': 'Not evaluated', 'note': f'Pothole inference failed: {type(exc).__name__}: {exc}'}
