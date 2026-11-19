"""Shared report intake, human verification, and idempotent dispatch workflow."""
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import uuid

from config import MAX_UPLOAD_MB, TARGET_DAYS, UPLOAD_DIR
from database import connect, event, rows
from modules.detection import detect
from modules.duplicate_detection import match, perceptual_hash
from modules.notifications import notify
from modules.priority_engine import calculate
from modules.trustshield import assess


def process_report(upload, source, road, ward, lat=None, lon=None, description='', reporter='Demo user',
                   severity=65, traffic=55, importance=60, repeats=20, detection_result=None,
                   detection_options=None):
    """Persist independent citizen/officer/camera intake; never dispatch at intake."""
    pending_events = []

    def record(stage, message, level='info'):
        pending_events.append((stage, message, level))

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
    safe_name = f'{stamp}_{uuid.uuid4().hex[:8]}_{Path(upload.name).name}'
    path = UPLOAD_DIR / safe_name
    upload.seek(0)
    data = upload.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f'Image exceeds the {MAX_UPLOAD_MB} MB limit.')
    if not data:
        raise ValueError('Uploaded image is empty.')
    path.write_bytes(data)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise ValueError('Upload is not a readable image. Use a valid JPG, PNG, or WEBP file.') from exc

    record('Received', f'{source} report received for {road}, {ward}.')
    detection = detection_result or detect(str(path), **(detection_options or {}))
    record('Detection', detection['note'])
    image_hash = perceptual_hash(str(path))
    duplicate_id, duplicate_reason = match(image_hash, lat, lon)
    record('Duplicate check',
           f'Possible match by {duplicate_reason} with RG-{duplicate_id:04d}.' if duplicate_id
           else 'No matching open road-damage record found.')
    trust_score, trust_status, reasons = assess(str(path), detection, duplicate_id, lat, lon)
    record('TrustShield', f'{trust_status} advisory score ({trust_score}/100): ' + '; '.join(reasons))

    effective_repeats = repeats
    canonical = None
    if duplicate_id:
        canonical = rows('SELECT * FROM reports WHERE id=?', (duplicate_id,))
        if canonical:
            canonical = canonical[0]
            effective_repeats = min(100, int(canonical['repeats']) + 25)
            canonical_score, canonical_priority = calculate(canonical['severity'], canonical['traffic'],
                                                            canonical['importance'], effective_repeats)
        else:
            duplicate_id = None
            duplicate_reason = None

    priority_score, priority = calculate(severity, traffic, importance, effective_repeats)
    record('Priority', f'{priority} · {priority_score}/100 (severity 40%, traffic 25%, road importance 20%, repeat reports 15%).')
    if duplicate_id:
        record('Duplicate merged', f'Report is linked to canonical RG-{duplicate_id:04d}; repeats and canonical priority were updated. No second work order will be created.')
    else:
        record('Human review', 'Pending explicit municipal verification. AI confidence is advisory; no contractor dispatch occurs at intake.', 'warning')

    created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    verification_status = 'Duplicate' if duplicate_id else 'Pending'
    report_status = 'Merged duplicate' if duplicate_id else 'Needs verification'
    verification_reason = f'Merged into canonical RG-{duplicate_id:04d} ({duplicate_reason}).' if duplicate_id else None
    with connect() as connection:
        if duplicate_id:
            connection.execute('UPDATE reports SET repeats=?,priority_score=?,priority=? WHERE id=?',
                               (effective_repeats, canonical_score, canonical_priority, duplicate_id))
        cursor = connection.execute('''INSERT INTO reports(
            created_at,source,reporter,road,ward,lat,lon,description,image_path,image_hash,
            detection,detection_conf,trust_score,trust_status,trust_reasons,severity,traffic,
            importance,repeats,priority_score,priority,status,duplicate_of,detection_model,
            detection_boxes,annotated_path,verification_status,verification_reason)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (created_at, source, reporter, road, ward, lat, lon, description, str(path), image_hash,
             detection['result'], detection.get('confidence'), trust_score, trust_status, '; '.join(reasons),
             severity, traffic, importance, effective_repeats, priority_score, priority, report_status,
             duplicate_id, Path(detection['model_path']).name if detection.get('model_path') else None,
             json.dumps(detection.get('boxes', [])), detection.get('annotated_path'),
             verification_status, verification_reason))
        report_id = cursor.lastrowid

    for stage, message, level in pending_events:
        event(stage, message, report_id, level)
    if duplicate_id:
        event('Duplicate merged', f'Canonical RG-{duplicate_id:04d} repeat count and priority updated; duplicate RG-{report_id:04d} linked.', duplicate_id)
    return report_id


def verify_report(report_id, reviewer, reason, confirmed=True):
    """Record a human decision; a confirmed canonical report dispatches once."""
    reviewer = (reviewer or '').strip()
    reason = (reason or '').strip()
    if not reviewer:
        raise ValueError('Reviewer name is required.')
    if len(reason) < 5:
        raise ValueError('Enter a verification reason of at least five characters.')
    record = rows('SELECT * FROM reports WHERE id=?', (report_id,))
    if not record:
        raise ValueError(f'Report RG-{report_id:04d} does not exist.')
    report = record[0]
    if report['verification_status'] == 'Duplicate' or report['duplicate_of'] is not None:
        raise ValueError(f'Report RG-{report_id:04d} is merged into RG-{report["duplicate_of"]:04d}; review the canonical report instead.')
    if report['verification_status'] != 'Pending':
        raise ValueError(f'Report RG-{report_id:04d} has already been reviewed ({report["verification_status"]}).')

    decision = 'Verified' if confirmed else 'Rejected'
    new_status = 'Verified' if confirmed else 'Rejected'
    verified_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with connect() as connection:
        cursor = connection.execute('''UPDATE reports
            SET verification_status=?,verification_reason=?,verified_by=?,verified_at=?,status=?
            WHERE id=? AND verification_status='Pending' ''',
            (decision, reason, reviewer, verified_at, new_status, report_id))
        if cursor.rowcount != 1:
            raise ValueError('This report was reviewed by another action; refresh and check its current state.')

    event('Human verification',
          f'{decision} by {reviewer}. Reason: {reason}. AI result was {report["detection"]} '
          f'(confidence {report["detection_conf"] if report["detection_conf"] is not None else "not available"}).',
          report_id, 'info' if confirmed else 'warning')
    if not confirmed:
        event('Dispatch held', 'Report rejected by human review; no contractor or work order assigned.', report_id, 'warning')
        return None

    return _assign_verified_report(report_id)


def _assign_verified_report(report_id):
    """Atomically ensure a verified canonical report has no more than one order."""
    now = datetime.now(timezone.utc)
    work_id = None
    contractor = None
    created = False
    no_contractor = False
    with connect() as connection:
        connection.execute('BEGIN IMMEDIATE')
        report = connection.execute('SELECT * FROM reports WHERE id=?', (report_id,)).fetchone()
        if not report or report['verification_status'] != 'Verified' or report['duplicate_of'] is not None:
            raise ValueError('Only a human-verified canonical report can be assigned.')
        existing = connection.execute('SELECT id FROM work_orders WHERE report_id=?', (report_id,)).fetchone()
        if existing:
            return existing['id']
        contractor_row = connection.execute('SELECT name FROM contractors WHERE ward=? ORDER BY name LIMIT 1',
                                            (report['ward'],)).fetchone()
        if not contractor_row:
            no_contractor = True
        else:
            contractor = contractor_row['name']
            work_id = 'RG-WO-' + now.strftime('%y%m%d') + '-' + uuid.uuid4().hex[:5].upper()
            due_at = (now + timedelta(days=TARGET_DAYS)).isoformat(timespec='seconds')
            connection.execute('''INSERT INTO work_orders(id,report_id,contractor,ward,created_at,due_at,status,before_path)
                VALUES(?,?,?,?,?,?,?,?)''',
                (work_id, report_id, contractor, report['ward'], now.isoformat(timespec='seconds'), due_at,
                 'Assigned', report['image_path']))
            connection.execute("UPDATE reports SET status='Assigned' WHERE id=?", (report_id,))
            created = True
    if no_contractor:
        event('Assignment exception', f'RG-{report_id:04d} is verified, but no contractor is configured for {report["ward"]}.', report_id, 'warning')
        return None
    if created:
        event('Assignment', f'Contractor identified: {contractor}. Work order {work_id} generated; target {TARGET_DAYS} days.', report_id)
        notify(report_id, work_id, contractor)
    return work_id
