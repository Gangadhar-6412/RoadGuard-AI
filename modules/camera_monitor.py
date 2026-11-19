"""Local prerecorded-video sampling wired to the shared report workflow."""
from pathlib import Path
from uuid import uuid4
import cv2

from config import DETECTION_IMAGE_SIZE, UPLOAD_DIR
from database import event
from modules.detection import detect
from modules.automation_engine import process_report


class _FrameUpload:
    """Small file-like adapter so a sampled frame enters normal image intake."""
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        self._stream = open(self.path, 'rb')

    def seek(self, *args):
        return self._stream.seek(*args)

    def read(self, *args):
        return self._stream.read(*args)

    def close(self):
        self._stream.close()


def monitor_video(video_path, camera_id='RG-CAM-01', road='Camera detected road segment',
                  ward='Ward 12', max_samples=12, seconds_between_samples=2,
                  confidence=None, iou=None):
    """Sample frames, detect potholes, and submit positive frames for automation.

    Returns frame records containing the real inference result and optional
    report ID. Missing weights produce Not evaluated records, never fake boxes.
    """
    video_path = Path(video_path)
    event('Camera', f'Camera {camera_id} monitoring started · prerecorded simulation.')
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f'Could not decode video file: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    interval = max(1, int(fps * seconds_between_samples))
    frame_no = 0
    sampled = []
    recent_boxes = []  # per-clip temporal association; never persists across clips
    try:
        while len(sampled) < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % interval == 0:
                ordinal = len(sampled) + 1
                frame_path = UPLOAD_DIR / f'cam_{camera_id}_{uuid4().hex[:10]}_{ordinal:02d}.jpg'
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f'Could not save sampled frame {frame_no} from {video_path}.')
                options = {}
                options['image_size'] = DETECTION_IMAGE_SIZE
                if confidence is not None:
                    options['confidence'] = confidence
                if iou is not None:
                    options['iou'] = iou
                prediction = detect(str(frame_path), **options)
                report_id = None
                duplicate_of = None
                if prediction['result'] == 'Pothole candidate':
                    height, width = frame.shape[:2]
                    current = []
                    novel_boxes = []
                    for box in prediction.get('boxes', []):
                        candidate = tuple(float(box[k]) for k in ('x1', 'y1', 'x2', 'y2'))
                        duplicate_track = next((track for track in recent_boxes
                                                if frame_no - track['frame_no'] <= interval * 3
                                                and _same_object(candidate, track['box'], width, height)), None)
                        if duplicate_track:
                            duplicate_of = duplicate_track.get('report_id') or duplicate_of
                        else:
                            novel_boxes.append(box)
                        current.append({'box': candidate, 'frame_no': frame_no, 'report_id': None})
                    # Keep only a short window, bounded by sampled-frame distance.
                    recent_boxes = [track for track in recent_boxes
                                    if frame_no - track['frame_no'] <= interval * 3] + current
                    if novel_boxes:
                        filtered = {**prediction, 'boxes': novel_boxes,
                                    'confidence': max(b['confidence'] for b in novel_boxes)}
                        upload = _FrameUpload(frame_path)
                        try:
                            report_id = process_report(
                                upload, 'Vehicle camera', road, ward,
                                description=f'Camera {camera_id} · sampled frame {frame_no}; prerecorded simulation source',
                                reporter=camera_id, detection_result=filtered)
                        finally:
                            upload.close()
                        for track in current:
                            if track in recent_boxes:
                                track['report_id'] = report_id
                    else:
                        event('Duplicate frame suppression',
                              f'Camera {camera_id} frame {frame_no}: repeated pothole box suppressed; no additional report/work order created.',
                              duplicate_of)
                event('Frame detection',
                      f'Camera {camera_id} frame {frame_no}: {prediction["result"]} · {prediction["note"]}' +
                      (' · repeated box suppressed' if prediction['result'] == 'Pothole candidate' and report_id is None else ''),
                      report_id)
                sampled.append({'frame_no': frame_no, 'frame_path': str(frame_path),
                                'result': prediction, 'report_id': report_id,
                                'duplicate_of': duplicate_of, 'suppressed': prediction['result'] == 'Pothole candidate' and report_id is None})
            frame_no += 1
    finally:
        cap.release()
    positives = sum(item['result']['result'] == 'Pothole candidate' for item in sampled)
    event('Camera', f'Camera {camera_id} clip processing complete · {len(sampled)} frames sampled · {positives} pothole candidate frames.')
    return sampled


def _same_object(a, b, width, height):
    """Conservative adjacent-frame box match using overlap or normalized center drift."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    overlap = inter / (area_a + area_b - inter) if area_a + area_b > inter else 0.0
    ac = ((a[0] + a[2]) / 2 / width, (a[1] + a[3]) / 2 / height)
    bc = ((b[0] + b[2]) / 2 / width, (b[1] + b[3]) / 2 / height)
    drift = ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
    size_ratio = min(area_a, area_b) / max(area_a, area_b) if max(area_a, area_b) else 0
    return overlap >= 0.30 or (drift <= 0.04 and size_ratio >= 0.5)
