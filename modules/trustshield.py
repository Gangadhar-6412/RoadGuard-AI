from PIL import Image, ImageStat
from database import rows

def assess(path, detection, duplicate_id, lat, lon):
    reasons=[]; score=45
    try:
        im=Image.open(path); im.verify(); im=Image.open(path).convert('RGB')
        if min(im.size)<180: reasons.append('Low resolution'); score-=12
        if sum(ImageStat.Stat(im).stddev)/3 < 12: reasons.append('Low visual detail or near-uniform image'); score-=10
        exif=im.getexif()
        reasons.append('EXIF metadata present' if exif else 'No EXIF metadata found (common after sharing)')
        if exif: score+=4
    except Exception:
        reasons.append('Image could not be inspected'); score-=25
    if detection['result']=='Pothole candidate':
        confidence = detection.get('confidence')
        score += 32 if confidence and confidence >= .7 else 22
        reasons.append('Configured detector found a pothole candidate')
        if confidence is not None and confidence < .5:
            reasons.append(f'Low-confidence model output ({confidence:.3f}; below the 0.50 advisory mark)')
        reasons.append('Human verification is required before any contractor dispatch')
    elif detection['result']=='No pothole detected': score-=18; reasons.append('Configured detector did not find a pothole')
    else: reasons.append('Automated pothole model unavailable; requires human review')
    if lat is None or lon is None: reasons.append('GPS coordinates not supplied')
    else: reasons.append('Reported coordinates recorded; image GPS metadata is not treated as proof')
    if duplicate_id: reasons.append(f'Possible duplicate of report RG-{duplicate_id:04d}'); score-=8
    score=max(0,min(100,score))
    status='High' if score>=70 and detection['result']=='Pothole candidate' else ('Medium' if score>=40 else 'Low')
    return score,status,reasons
