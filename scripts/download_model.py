"""Download and verify the pinned RoadGuard YOLOv12s road-damage checkpoint."""
import hashlib
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import MODEL_PATH, MODEL_SHA256, MODEL_SIZE_BYTES, MODEL_URL  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size == MODEL_SIZE_BYTES and sha256(MODEL_PATH) == MODEL_SHA256:
        print(f'Model already verified: {MODEL_PATH} ({MODEL_SIZE_BYTES:,} bytes)')
        return

    partial = MODEL_PATH.with_suffix(MODEL_PATH.suffix + '.download')
    request = Request(MODEL_URL, headers={'User-Agent': 'RoadGuard-AI/1.0 model downloader'})
    size = 0
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=60) as response, open(partial, 'wb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != MODEL_SIZE_BYTES:
            raise ValueError(f'Expected {MODEL_SIZE_BYTES:,} bytes, received {size:,}.')
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError(f'SHA-256 mismatch: {digest.hexdigest()}')
        os.replace(partial, MODEL_PATH)
        print(f'Downloaded and verified: {MODEL_PATH} ({size:,} bytes; SHA-256 {MODEL_SHA256})')
    except Exception:
        partial.unlink(missing_ok=True)
        raise


if __name__ == '__main__':
    main()
