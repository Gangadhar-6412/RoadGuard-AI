"""Render every top-level Streamlit route and fail on page exceptions."""
import unittest
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest
import database
import modules.automation_engine as workflow
from database import init_db, rows
from config import MODEL_PATH


APP_FILE = Path(__file__).resolve().parents[1] / 'app.py'
PAGES = [
    'Command Center', 'Live Camera', 'Citizen Report', 'Officer Report',
    'TrustShield AI', 'Damage Map', 'Work Orders', 'Contractor Portal',
    'Repair Verification', 'Analytics & Reports', 'Demo & Settings',
]


class StreamlitNavigationTest(unittest.TestCase):
    def test_every_navigation_page_renders_without_exception(self):
        app = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
        self.assertFalse(app.exception, f'Initial route raised: {app.exception}')
        for page in PAGES:
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception, f'{page} raised: {app.exception}')

    @unittest.skipUnless(MODEL_PATH.is_file(), 'Download the pinned pothole model for live UI inference.')
    def test_citizen_upload_runs_real_model_and_displays_annotation(self):
        # Use the held-out HRP4K image with the replacement's one verified match.
        image = Path(__file__).parent / 'evaluation' / 'hrp4k' / '30811.jpg'
        old_db, old_upload = database.DB_PATH, workflow.UPLOAD_DIR
        with tempfile.TemporaryDirectory() as folder:
            database.DB_PATH = Path(folder) / 'ui.sqlite3'
            workflow.UPLOAD_DIR = Path(folder) / 'uploads'
            workflow.UPLOAD_DIR.mkdir()
            try:
                init_db()
                app = AppTest.from_file(str(APP_FILE), default_timeout=60).run()
                app.radio[0].set_value('Citizen Report').run()
                app.text_input[0].set_value('UI Model Test Road')
                app.file_uploader[0].upload('attributed_pothole_test.jpg', image.read_bytes(), 'image/jpeg')
                app.form_submit_button[0].click().run()

                self.assertFalse(app.exception, f'Citizen upload route raised: {app.exception}')
                self.assertTrue(any('Latest model output' in h.value for h in app.subheader),
                                f"subheaders={[h.value for h in app.subheader]}, errors={[e.value for e in app.error]}, success={[e.value for e in app.success]}")
                reports = rows("SELECT * FROM reports WHERE source='Citizen' AND road='UI Model Test Road'")
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0]['detection'], 'Pothole candidate')
                self.assertTrue(Path(reports[0]['annotated_path']).is_file())
                # The model returns a matched box and a false positive here; the
                # existing trust gate should hold this unverified image for review.
                self.assertEqual(len(rows('SELECT * FROM work_orders WHERE report_id=?', (reports[0]['id'],))), 0)
            finally:
                database.DB_PATH, workflow.UPLOAD_DIR = old_db, old_upload


if __name__ == '__main__':
    unittest.main()
