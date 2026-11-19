"""RoadGuard local configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "roadguard.sqlite3"
MAX_UPLOAD_MB = 12
DUPLICATE_DISTANCE_KM = 0.25
DUPLICATE_HASH_DISTANCE = 8
TARGET_DAYS = 7
MODEL_PATH = ROOT / "models" / "pothole_yolo12s_rdd2022.pt"
MODEL_URL = "https://huggingface.co/rezzzq/yolo12s-road-damage-rdd2022/resolve/main/yolo12s_RDD2022_best.pt"
MODEL_SHA256 = "ccf91e3199d81ee761f4a3a012da752939c9d72e9b1eaa242e508db3163270a0"
MODEL_SIZE_BYTES = 18966618
DETECTION_CONFIDENCE = 0.25
DETECTION_IOU = 0.45
DETECTION_IMAGE_SIZE = 1280
CAMERA_FRAME_INTERVAL_SECONDS = 2.0
CAMERA_MAX_SAMPLES = 12
for directory in (DATA_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)
