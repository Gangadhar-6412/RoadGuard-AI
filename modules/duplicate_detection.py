import hashlib
from PIL import Image
import imagehash
from config import DUPLICATE_DISTANCE_KM, DUPLICATE_HASH_DISTANCE
from database import rows

def perceptual_hash(path):
    try: return str(imagehash.phash(Image.open(path).convert('RGB')))
    except Exception: return hashlib.sha256(open(path,'rb').read()).hexdigest()[:16]

def match(image_hash, lat, lon):
    # Compare only against open canonical reports. Rejected or already closed
    # cases should not absorb a fresh defect report at the same location.
    for r in rows('''SELECT r.id,r.image_hash,r.lat,r.lon FROM reports r
        WHERE r.duplicate_of IS NULL
          AND r.verification_status!='Rejected'
          AND NOT EXISTS (SELECT 1 FROM work_orders w WHERE w.report_id=r.id AND w.status IN ('Closed','Rejected'))
        ORDER BY r.id DESC'''):
        hash_match = False
        try: hash_match = imagehash.hex_to_hash(image_hash) - imagehash.hex_to_hash(r['image_hash']) <= DUPLICATE_HASH_DISTANCE
        except Exception: hash_match = image_hash == r['image_hash']
        gps_match = False
        if lat is not None and lon is not None and r['lat'] is not None and r['lon'] is not None:
            from math import radians, sin, cos, asin, sqrt
            dlat, dlon = radians(lat-r['lat']), radians(lon-r['lon'])
            a = sin(dlat/2)**2 + cos(radians(lat))*cos(radians(r['lat']))*sin(dlon/2)**2
            gps_match = 6371 * 2 * asin(sqrt(a)) <= DUPLICATE_DISTANCE_KM
        if hash_match or gps_match: return r['id'], ('image similarity' if hash_match else 'nearby coordinates')
    return None, None
