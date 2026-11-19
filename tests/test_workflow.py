"""Deterministic integration test for storage and automation plumbing.

The detector stub below is test-only. It is not an AI prediction and must never
be used by the Streamlit application or cited as model evidence.
"""
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import database
from database import init_db, rows
import modules.automation_engine as workflow
import modules.detection as detection
import modules.camera_monitor as camera_monitor
import modules.work_order_service as work_orders
import modules.demo_reset as demo_reset
from modules.priority_engine import calculate
from config import MODEL_PATH, MODEL_SHA256

# This labeled HRP4K sample is the only included road image where the configured
# replacement produced a box matched to pothole ground truth at IoU >= 0.5.
FIXTURE_IMAGE = Path(__file__).parent / 'evaluation' / 'hrp4k' / '30811.jpg'


class DemoUpload(io.BytesIO):
    name = 'labelled_demo_evidence.png'


def demo_test_detector(_path):
    """Test fixture signal only; no inference or image detection is performed."""
    return {'result': 'Pothole candidate', 'confidence': 0.92,
            'note': 'TEST FIXTURE ONLY: deterministic signal, not model inference.'}


def demo_image():
    image = Image.new('RGB', (320, 240), '#999999')
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 25, 100, 80), fill='#111111')
    draw.ellipse((140, 55, 285, 215), fill='#222222')
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()


class WorkflowIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_db = database.DB_PATH
        database.DB_PATH = Path(self.temp.name) / 'workflow.sqlite3'
        self.old_upload_dir = workflow.UPLOAD_DIR
        workflow.UPLOAD_DIR = Path(self.temp.name) / 'uploads'
        workflow.UPLOAD_DIR.mkdir()
        self.old_camera_upload_dir = camera_monitor.UPLOAD_DIR
        camera_monitor.UPLOAD_DIR = workflow.UPLOAD_DIR
        init_db()
        self.detector = patch.object(workflow, 'detect', demo_test_detector)
        self.detector.start()
        self.image = demo_image()

    def tearDown(self):
        if self.detector:
            self.detector.stop()
        workflow.UPLOAD_DIR = self.old_upload_dir
        camera_monitor.UPLOAD_DIR = self.old_camera_upload_dir
        database.DB_PATH = self.old_db
        self.temp.cleanup()

    def submit(self):
        return workflow.process_report(
            DemoUpload(self.image), source='Citizen', road='Demo Test Road',
            ward='Ward 12', lat=19.0760, lon=72.8777,
            description='Clearly labelled synthetic integration-test report',
            reporter='TEST FIXTURE')

    def test_report_triggers_persisted_automation_and_duplicate_link(self):
        first_id = self.submit()
        first = rows('SELECT * FROM reports WHERE id=?', (first_id,))[0]
        self.assertEqual(first['source'], 'Citizen')
        self.assertEqual(first['trust_status'], 'High')
        self.assertEqual(first['verification_status'], 'Pending')
        self.assertGreater(first['priority_score'], 0)
        self.assertTrue(first['priority'].startswith(('P1', 'P2', 'P3')))
        self.assertEqual(first['priority_score'], 54.8)
        self.assertEqual(calculate(100, 100, 100, 100), (100.0, 'P1 — Urgent'))

        self.assertEqual(len(rows('SELECT * FROM work_orders WHERE report_id=?', (first_id,))), 0,
                         'TrustShield High alone must not dispatch a contractor.')
        work_id = workflow.verify_report(first_id, 'Test municipal reviewer',
                                         'Reviewed the report photo and confirmed the defect.')
        self.assertTrue(work_id.startswith('RG-WO-'))
        first = rows('SELECT * FROM reports WHERE id=?', (first_id,))[0]
        self.assertEqual(first['verification_status'], 'Verified')
        orders = rows('SELECT * FROM work_orders WHERE report_id=?', (first_id,))
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['contractor'], 'Ward 12 Roadworks Team')
        self.assertTrue(orders[0]['id'].startswith('RG-WO-'))

        second_id = self.submit()
        second = rows('SELECT * FROM reports WHERE id=?', (second_id,))[0]
        self.assertEqual(second['duplicate_of'], first_id)
        self.assertEqual(second['verification_status'], 'Duplicate')
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 1,
                         'A duplicate must not create another work order.')
        canonical = rows('SELECT * FROM reports WHERE id=?', (first_id,))[0]
        self.assertEqual(canonical['repeats'], 45)
        self.assertGreater(canonical['priority_score'], 54.8)

        events = rows('SELECT stage,message FROM events WHERE report_id=?', (first_id,))
        stages = {event['stage'] for event in events}
        self.assertTrue({'Detection', 'Duplicate check', 'TrustShield', 'Priority',
                         'Human verification', 'Assignment', 'Notification', 'Duplicate merged'}.issubset(stages))
        self.assertGreater(len(rows('SELECT * FROM events')), 0)

    def test_rejected_report_never_dispatches(self):
        report_id = self.submit()
        self.assertIsNone(workflow.verify_report(report_id, 'TEST ONLY reviewer',
                                                 'TEST ONLY: fixture rejected during workflow test.',
                                                 confirmed=False))
        report = rows('SELECT verification_status,verification_reason FROM reports WHERE id=?', (report_id,))[0]
        self.assertEqual(report['verification_status'], 'Rejected')
        self.assertIn('fixture rejected', report['verification_reason'])
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 0)

    def test_work_order_updates_evidence_and_admin_closure_are_persisted(self):
        report_id = self.submit()
        work_id = workflow.verify_report(report_id, 'TEST ONLY reviewer',
                                         'TEST ONLY: inspect deterministic candidate fixture.')
        work_orders.update_work_order(work_id, 'In progress', 'TEST ONLY contractor')
        image_bytes = demo_image()
        class Evidence(io.BytesIO):
            name = 'test_fixture_only_after.png'
        with self.assertRaisesRegex(ValueError, 'completion evidence'):
            work_orders.approve_closure(work_id, 'TEST ONLY admin', 'TEST ONLY closure check')
        path = work_orders.submit_repair_evidence(work_id, Evidence(image_bytes), 'TEST ONLY contractor')
        saved = rows('SELECT status,after_path FROM work_orders WHERE id=?', (work_id,))[0]
        self.assertEqual(saved['status'], 'Completion requested')
        self.assertEqual(Path(saved['after_path']).resolve(), Path(path))
        self.assertTrue(Path(path).is_file())
        work_orders.approve_closure(work_id, 'TEST ONLY admin',
                                    'TEST ONLY: synthetic evidence validates approval persistence.')
        closed = rows('SELECT status,closed_by,closure_reason,closed_at FROM work_orders WHERE id=?', (work_id,))[0]
        self.assertEqual(closed['status'], 'Closed')
        self.assertEqual(closed['closed_by'], 'TEST ONLY admin')
        self.assertIn('synthetic evidence', closed['closure_reason'])
        self.assertIsNotNone(closed['closed_at'])
        stages = {r['stage'] for r in rows('SELECT stage FROM events WHERE report_id=?', (report_id,))}
        self.assertTrue({'Contractor update', 'Repair evidence', 'Admin approval'}.issubset(stages))

    def test_demo_reset_clears_local_demo_records_and_uploads(self):
        report_id = self.submit()
        workflow.verify_report(report_id, 'TEST ONLY reviewer',
                               'TEST ONLY: confirm fixture for reset verification.')
        data_root = Path(self.temp.name) / 'data'
        upload_root = data_root / 'uploads'
        upload_root.mkdir(parents=True)
        # Repoint only the reset helper's configured roots to this isolated test area.
        old_reset_upload, old_reset_data = demo_reset.UPLOAD_DIR, demo_reset.DATA_DIR
        demo_reset.UPLOAD_DIR, demo_reset.DATA_DIR = upload_root, data_root
        marker = upload_root / 'test_fixture_only.txt'
        marker.write_text('TEST FIXTURE ONLY', encoding='utf-8')
        try:
            result = demo_reset.reset_demo_environment()
        finally:
            demo_reset.UPLOAD_DIR, demo_reset.DATA_DIR = old_reset_upload, old_reset_data
        self.assertEqual(result['deleted_files'], 1)
        self.assertFalse(marker.exists())
        self.assertEqual(len(rows('SELECT * FROM reports')), 0)
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 0)
        self.assertEqual(len(rows('SELECT * FROM contractors')), 3)
        events = rows('SELECT stage FROM events')
        self.assertEqual([r['stage'] for r in events], ['Demo reset'])

    def test_bad_image_is_rejected_before_report_persistence(self):
        class BadUpload(io.BytesIO):
            name = 'not_an_image.png'
        with self.assertRaisesRegex(ValueError, 'not a readable image'):
            workflow.process_report(BadUpload(b'clearly invalid test bytes'),
                                    source='Citizen', road='Test Road', ward='Ward 12')
        self.assertEqual(len(rows('SELECT * FROM reports')), 0)

    def test_missing_yolo_weights_are_reported_as_not_evaluated(self):
        with patch.dict(os.environ, {'ROADGUARD_YOLO_MODEL': str(Path(self.temp.name) / 'absent.pt')}):
            result = detection.detect('unused-test-image.png')
        self.assertEqual(result['result'], 'Not evaluated')
        self.assertIsNone(result['confidence'])

    def test_detection_thresholds_are_passed_to_ultralytics(self):
        from types import SimpleNamespace
        class FakeModel:
            names = {0: 'pothole'}
            def predict(self, _path, **kwargs):
                self.kwargs = kwargs
                return [SimpleNamespace(boxes=[])]
        fake = FakeModel()
        with patch.object(detection, '_model', return_value=fake):
            result = detection.detect(str(FIXTURE_IMAGE), confidence=0.37, iou=0.62)
        self.assertEqual(result['result'], 'No pothole detected')
        self.assertEqual(fake.kwargs['conf'], 0.37)
        self.assertEqual(fake.kwargs['iou'], 0.62)

    def test_invalid_detection_thresholds_do_not_run_inference(self):
        with patch.object(detection, '_model') as model:
            result = detection.detect(str(FIXTURE_IMAGE), confidence=1.2, iou=0.45)
        self.assertEqual(result['result'], 'Not evaluated')
        model.assert_not_called()

    @unittest.skipUnless(MODEL_PATH.is_file() and FIXTURE_IMAGE.is_file(),
                         'Download model weights and test fixture to run real inference.')
    def test_real_pothole_model_returns_boxes_and_annotated_road_photo(self):
        import hashlib
        with open(MODEL_PATH, 'rb') as f:
            self.assertEqual(hashlib.sha256(f.read()).hexdigest(), MODEL_SHA256)
        output = Path(self.temp.name) / 'annotated_test.jpg'
        result = detection.detect(str(FIXTURE_IMAGE), output_path=str(output))
        self.assertEqual(result['result'], 'Pothole candidate', result['note'])
        self.assertGreater(len(result['boxes']), 0)
        self.assertGreater(result['confidence'], 0.25)
        self.assertTrue(output.is_file())
        self.assertEqual(result['annotated_path'], str(output.resolve()))

    @unittest.skipUnless(MODEL_PATH.is_file() and FIXTURE_IMAGE.is_file(),
                         'Download model weights and test fixture to run real inference.')
    def test_real_candidate_enters_automation_and_false_positive_is_held_for_review(self):
        self.detector.stop()  # Run the actual checkpoint through the production workflow.
        self.detector = None
        class RealRoadPhoto(io.BytesIO):
            name = 'attributed_pothole_test.jpg'
        rid = workflow.process_report(
            RealRoadPhoto(FIXTURE_IMAGE.read_bytes()), source='Citizen', road='Model Test Road',
            ward='Ward 12', lat=19.0760, lon=72.8777,
            description='Attributed HRP4K held-out road image', reporter='Real model integration test')
        report = rows('SELECT * FROM reports WHERE id=?', (rid,))[0]
        self.assertEqual(report['detection'], 'Pothole candidate')
        self.assertGreater(report['detection_conf'], 0.25)
        self.assertGreater(len(__import__('json').loads(report['detection_boxes'])), 0)
        self.assertTrue(Path(report['annotated_path']).is_file())
        order = rows('SELECT * FROM work_orders WHERE report_id=?', (rid,))
        self.assertEqual(len(order), 0, 'A false-positive box must not pass the existing trust/assignment gate.')
        self.assertTrue({'Detection', 'Duplicate check', 'TrustShield', 'Priority', 'Human review'}.issubset(
            {event['stage'] for event in rows('SELECT stage FROM events WHERE report_id=?', (rid,))}))

    @unittest.skipUnless(MODEL_PATH.is_file() and FIXTURE_IMAGE.is_file(),
                         'Download model weights and test fixture to run real video inference.')
    def test_real_video_frame_detection_uses_camera_automation_path(self):
        import cv2
        frame = cv2.imread(str(FIXTURE_IMAGE))
        clip = Path(self.temp.name) / 'labelled_pothole_test.avi'
        # Use lossless FFV1 so the 4K small-pothole features survive encoding.
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'FFV1'), 1.0,
                                 (frame.shape[1], frame.shape[0]))
        self.assertTrue(writer.isOpened(), 'OpenCV could not create the deterministic test clip.')
        writer.write(frame)
        writer.write(frame)
        writer.release()

        sampled = camera_monitor.monitor_video(clip, max_samples=2, seconds_between_samples=1)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(sampled[0]['result']['result'], 'Pothole candidate')
        self.assertGreater(len(sampled[0]['result']['boxes']), 0)
        self.assertIsNotNone(sampled[0]['report_id'])
        report = rows('SELECT * FROM reports WHERE id=?', (sampled[0]['report_id'],))[0]
        self.assertEqual(report['source'], 'Vehicle camera')
        self.assertTrue(Path(report['annotated_path']).is_file())
        self.assertEqual(len(rows('SELECT * FROM work_orders WHERE report_id=?', (report['id'],))), 0,
                         'The same model false positive should remain held for review in the video path.')
        self.assertTrue(sampled[1]['suppressed'], 'A repeated frame box should not create another report.')
        self.assertEqual(len(rows('SELECT * FROM reports WHERE source=?', ('Vehicle camera',))), 1)
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 0)

    def test_repeated_video_box_creates_only_one_demo_work_order(self):
        """TEST FIXTURE ONLY: fixed boxes exercise clip dedupe, not AI accuracy."""
        import cv2
        frame = cv2.imread(str(FIXTURE_IMAGE))
        clip = Path(self.temp.name) / 'labelled_duplicate_demo.avi'
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 1.0,
                                 (frame.shape[1], frame.shape[0]))
        self.assertTrue(writer.isOpened())
        writer.write(frame)
        writer.write(frame)
        writer.release()
        fake_candidate = {
            'result': 'Pothole candidate', 'confidence': 0.92,
            'boxes': [{'class_id': 0, 'label': 'pothole', 'confidence': 0.92,
                       'x1': 30, 'y1': 25, 'x2': 100, 'y2': 80}],
            'annotated_path': None, 'model_path': None,
            'note': 'TEST FIXTURE ONLY: deterministic candidate used to validate video deduplication.',
        }
        with patch.object(camera_monitor, 'detect', return_value=fake_candidate):
            sampled = camera_monitor.monitor_video(clip, camera_id='TEST-CAM',
                                                   max_samples=2, seconds_between_samples=1)
        self.assertEqual(len(sampled), 2)
        self.assertIsNotNone(sampled[0]['report_id'])
        self.assertTrue(sampled[1]['suppressed'])
        self.assertEqual(len(rows("SELECT * FROM reports WHERE source='Vehicle camera'")), 1)
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 0,
                         'A camera candidate remains pending until a reviewer confirms it.')
        report_id = sampled[0]['report_id']
        work_id = workflow.verify_report(report_id, 'TEST ONLY reviewer',
                                         'TEST ONLY: confirm this deterministic fixture to validate assignment.')
        self.assertTrue(work_id.startswith('RG-WO-'))
        self.assertEqual(len(rows('SELECT * FROM work_orders')), 1)


if __name__ == '__main__':
    unittest.main()
