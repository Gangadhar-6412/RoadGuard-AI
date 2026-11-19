from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from PIL import Image, ImageDraw

import database
import modules.automation_engine as automation
import modules.camera_monitor as camera
import modules.work_order_service as work_orders
from config import MODEL_PATH, MODEL_SHA256
from database import init_db, rows
from modules.detection import detect

IMAGE = ROOT / 'tests' / 'evaluation' / 'hrp4k' / '30811.jpg'
REPORT = ROOT / 'reports' / 'final_workflow_test_report.md'
ANNOTATED_CITIZEN = ROOT / 'reports' / 'annotated_citizen_30811.jpg'
ANNOTATED_CAMERA = ROOT / 'reports' / 'annotated_camera_frame_30811.jpg'
SYNTHETIC_EVIDENCE = ROOT / 'reports' / 'TEST_ONLY_synthetic_after_evidence.png'

class Upload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def make_test_evidence():
    image = Image.new('RGB', (640, 360), '#d9e2ec')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 250, 640, 360), fill='#4b5563')
    draw.text((24, 30), 'SYNTHETIC TEST EVIDENCE', fill='#111827')
    draw.text((24, 60), 'NOT A PHOTOGRAPH OF A COMPLETED REPAIR', fill='#b91c1c')
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


def run():
    if not IMAGE.is_file() or not MODEL_PATH.is_file():
        raise SystemExit(f'Required real model fixture missing: image={IMAGE.is_file()}, checkpoint={MODEL_PATH.is_file()}')
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    if sha != MODEL_SHA256:
        raise SystemExit(f'Checkpoint SHA mismatch: expected {MODEL_SHA256}, got {sha}')

    old_db, old_auto_upload, old_camera_upload = database.DB_PATH, automation.UPLOAD_DIR, camera.UPLOAD_DIR
    old_order_upload = work_orders.UPLOAD_DIR
    with tempfile.TemporaryDirectory(prefix='roadguard_final_demo_') as folder:
        base = Path(folder)
        database.DB_PATH = base / 'demo.sqlite3'
        upload_dir = base / 'uploads'
        upload_dir.mkdir()
        automation.UPLOAD_DIR = upload_dir
        camera.UPLOAD_DIR = upload_dir
        work_orders.UPLOAD_DIR = upload_dir
        init_db()
        results = {}
        try:
            # A. Genuine image inference -> intake -> human review -> assignment.
            REPORT.parent.mkdir(parents=True, exist_ok=True)
            image_result = detect(str(IMAGE), output_path=str(ANNOTATED_CITIZEN))
            rid = automation.process_report(
                Upload(IMAGE.read_bytes(), IMAGE.name), source='Citizen', road='TEST ONLY HRP4K Road',
                ward='Ward 12', lat=19.0760, lon=72.8777,
                description='TEST ONLY: attributed held-out HRP4K workflow sample.',
                reporter='TEST ONLY workflow runner')
            pending_before = len(rows('SELECT * FROM work_orders')) == 0
            report = rows('SELECT * FROM reports WHERE id=?', (rid,))[0]
            priority = {'label': report['priority'], 'score': report['priority_score'],
                        'inputs': {'severity': report['severity'], 'traffic': report['traffic'],
                                   'road_importance': report['importance'], 'repeat_reports': report['repeats']}}
            citizen_order = automation.verify_report(
                rid, 'TEST ONLY human reviewer',
                'TEST ONLY: scripted review action to exercise the explicit human verification gate.')
            work_orders.update_work_order(citizen_order, 'In progress', 'TEST ONLY contractor')
            synthetic_bytes = make_test_evidence()
            SYNTHETIC_EVIDENCE.write_bytes(synthetic_bytes)
            evidence = Upload(synthetic_bytes, 'TEST_ONLY_not_a_repair.png')
            evidence_path = work_orders.submit_repair_evidence(citizen_order, evidence, 'TEST ONLY contractor')
            work_orders.approve_closure(citizen_order, 'TEST ONLY admin approver',
                                        'TEST ONLY: synthetic evidence validates persistence and approval gate.')
            citizen_status = rows('SELECT status,after_path,closed_by,closure_reason FROM work_orders WHERE id=?', (citizen_order,))[0]
            results['citizen'] = {
                'detection': image_result['result'], 'confidence': image_result.get('confidence'),
                'boxes': image_result.get('boxes', []), 'annotated': str(ANNOTATED_CITIZEN.resolve()),
                'report_id': rid, 'initially_pending_without_order': pending_before,
                'trust': report['trust_status'], 'trust_score': report['trust_score'],
                'verification': dict(rows('SELECT verification_status,verified_by,verification_reason FROM reports WHERE id=?', (rid,))[0]),
                'priority': priority, 'work_order': citizen_order, 'contractor': rows('SELECT contractor FROM work_orders WHERE id=?', (citizen_order,))[0]['contractor'],
                'final_test_status': citizen_status['status'], 'evidence_path': citizen_status['after_path'],
                'test_evidence_file_exists': Path(evidence_path).is_file(),
                'test_evidence_is_synthetic': True,
            }

            # B. Actual model inference on a short lossless video made from the real sample still.
            frame = cv2.imread(str(IMAGE))
            clip = base / 'TEST_ONLY_prerecorded_still_clip.avi'
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'FFV1'), 1.0,
                                     (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise RuntimeError('OpenCV FFV1 video writer unavailable; camera workflow not completed.')
            writer.write(frame)
            writer.write(frame)
            writer.release()
            sampled = camera.monitor_video(clip, camera_id='TEST-CAM', road='TEST ONLY Camera Road',
                                           ward='Ward 12', max_samples=2, seconds_between_samples=1)
            candidates = [x for x in sampled if x['result']['result'] == 'Pothole candidate']
            report_frames = [x for x in candidates if x.get('report_id')]
            if not report_frames:
                raise RuntimeError('Actual camera inference returned no reportable candidate; camera review/assignment not executed.')
            cam_report_id = report_frames[0]['report_id']
            camera_annotation = report_frames[0]['result'].get('annotated_path')
            if camera_annotation and Path(camera_annotation).is_file():
                ANNOTATED_CAMERA.write_bytes(Path(camera_annotation).read_bytes())
            no_camera_order_before_review = len(rows('SELECT * FROM work_orders WHERE report_id=?', (cam_report_id,))) == 0
            cam_order = automation.verify_report(
                cam_report_id, 'TEST ONLY camera reviewer',
                'TEST ONLY: scripted human review action for the actual model-generated camera candidate.')
            cam_report = rows('SELECT * FROM reports WHERE id=?', (cam_report_id,))[0]
            results['camera'] = {
                'clip_type': 'TEST ONLY prerecorded lossless two-frame clip made from the same held-out road photograph',
                'sampled_frames': len(sampled), 'candidate_frames': len(candidates),
                'submitted_reports': len(report_frames), 'suppressed_frames': sum(bool(x['suppressed']) for x in sampled),
                'detection_results': [{'frame': x['frame_no'], 'result': x['result']['result'],
                                      'confidence': x['result'].get('confidence'), 'boxes': x['result'].get('boxes', [])}
                                     for x in sampled],
                'report_id': cam_report_id, 'pending_without_order_before_review': no_camera_order_before_review,
                'verification': cam_report['verification_status'], 'work_order': cam_order,
                'contractor': rows('SELECT contractor FROM work_orders WHERE id=?', (cam_order,))[0]['contractor'],
            }

            results['persistence'] = {
                'reports': len(rows('SELECT * FROM reports')),
                'work_orders': len(rows('SELECT * FROM work_orders')),
                'events': len(rows('SELECT * FROM events')),
                'report_ids': [r['id'] for r in rows('SELECT id FROM reports ORDER BY id')],
                'work_order_ids': [r['id'] for r in rows('SELECT id FROM work_orders ORDER BY id')],
                'event_stages': [r['stage'] for r in rows('SELECT stage FROM events ORDER BY id')],
                'notification_events_are_simulated': all('Simulated' in r['message'] for r in rows("SELECT message FROM events WHERE stage='Notification'")),
            }
        finally:
            database.DB_PATH, automation.UPLOAD_DIR = old_db, old_auto_upload
            camera.UPLOAD_DIR, work_orders.UPLOAD_DIR = old_camera_upload, old_order_upload

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    when = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    c, v = results['citizen'], results['camera']
    content = f'''# RoadGuard AI — Final Workflow Test Report

Run: {when}  
Database: temporary isolated SQLite database; no existing application data was used or changed.  
Checkpoint: `{MODEL_PATH.name}` ({MODEL_PATH.stat().st_size:,} bytes), SHA-256 `{sha}`.  
Test photo: `{IMAGE.relative_to(ROOT).as_posix()}`. This is a held-out evaluation image, not a newly collected field report.

## Results

| Workflow check | Actual result |
|---|---|
| Citizen photograph inference | {c['detection']}; confidence {c['confidence']}; {len(c['boxes'])} box(es) |
| Citizen intake gate | Pending human decision; no work order before review: {c['initially_pending_without_order']} |
| Citizen review / assignment | TEST ONLY scripted reviewer recorded `{c['verification']['verification_status']}`; assigned `{c['work_order']}` to {c['contractor']} |
| Priority | {c['priority']['label']}, {c['priority']['score']}/100; inputs severity={c['priority']['inputs']['severity']}, traffic={c['priority']['inputs']['traffic']}, importance={c['priority']['inputs']['road_importance']}, repeats={c['priority']['inputs']['repeat_reports']} |
| Citizen evidence / closure | Contractor update and after-evidence file persisted; TEST ONLY admin approval moved order to `{c['final_test_status']}` |
| Camera video inference | {v['sampled_frames']} sampled frames, {v['candidate_frames']} candidate frames, {v['submitted_reports']} report(s), {v['suppressed_frames']} repeated frame(s) suppressed |
| Camera review / assignment | No order before review: {v['pending_without_order_before_review']}; TEST ONLY reviewer assigned `{v['work_order']}` to {v['contractor']} |
| SQLite persistence | {results['persistence']['reports']} reports, {results['persistence']['work_orders']} work orders, {results['persistence']['events']} event records |
| External notifications | Simulated event records only; no email or contractor message was sent |

## Actual model output

Citizen image: `{c['detection']}` at confidence `{c['confidence']}`. Durable annotated outputs: `{ANNOTATED_CITIZEN.relative_to(ROOT).as_posix()}` and `{ANNOTATED_CAMERA.relative_to(ROOT).as_posix()}`. Test-only synthetic evidence is `{SYNTHETIC_EVIDENCE.relative_to(ROOT).as_posix()}` and is labelled in-image as not a real repair. Box data:

```json
{json.dumps(c['boxes'], indent=2)}
```

Camera frame inference:

```json
{json.dumps(v['detection_results'], indent=2)}
```

The video was a TEST ONLY two-frame lossless clip generated from the same still image. It validates the video inference path and duplicate-frame suppression, not live camera performance. Detections are actual configured checkpoint outputs. Reviewer, contractor update, synthetic evidence, and admin approval were scripted TEST ONLY actions in an isolated database; they are not claims of field review or completed road repairs. The synthetic evidence image explicitly says it is not repair evidence from a real site.

## Event and record evidence

Report IDs: `{results['persistence']['report_ids']}`.  
Work order IDs: `{results['persistence']['work_order_ids']}`.  
Recorded event stages in order: `{', '.join(results['persistence']['event_stages'])}`.  
All logged notification events were marked simulated: `{results['persistence']['notification_events_are_simulated']}`.

## Limitations

The detector has documented poor recall on the evaluation sample and was not tuned in this workflow task. A model candidate is only evidence for human review; the pipeline does not dispatch based solely on confidence. Human and admin names are recorded as supplied text; the application has no authentication/authorization system, so the approval step is a workflow gate, not verified identity control. The final `Closed` state above demonstrates persistence logic only with synthetic test evidence and does not represent a completed physical repair.
'''
    REPORT.write_text(content, encoding='utf-8')
    print(f'Report written: {REPORT}')
    print(json.dumps(results, indent=2, default=str))

if __name__ == '__main__':
    run()
