"""Validated work-order transitions and evidence/approval persistence."""
from datetime import datetime, timezone
from pathlib import Path
import uuid

from config import MAX_UPLOAD_MB, UPLOAD_DIR
from database import connect, event, rows


STATUS_TRANSITIONS = {
    'Assigned': {'Accepted', 'In progress', 'On site'},
    'Accepted': {'In progress', 'On site'},
    'In progress': {'On site'},
    'On site': {'In progress'},
}


def update_work_order(work_order_id, new_status, actor):
    actor = (actor or '').strip()
    if not actor:
        raise ValueError('Contractor/update actor is required.')
    with connect() as connection:
        order = connection.execute('SELECT * FROM work_orders WHERE id=?', (work_order_id,)).fetchone()
        if not order:
            raise ValueError(f'Work order {work_order_id} does not exist.')
        allowed = STATUS_TRANSITIONS.get(order['status'], set())
        if new_status not in allowed:
            raise ValueError(f'Cannot move work order from {order["status"]} to {new_status}.')
        connection.execute('UPDATE work_orders SET status=? WHERE id=?', (new_status, work_order_id))
    event('Contractor update', f'{actor} changed {work_order_id} from {order["status"]} to {new_status}.',
          order['report_id'])


def submit_repair_evidence(work_order_id, upload, actor='Demo contractor'):
    actor = (actor or '').strip()
    if not actor:
        raise ValueError('Evidence submitter name is required.')
    order = rows('SELECT * FROM work_orders WHERE id=?', (work_order_id,))
    if not order:
        raise ValueError(f'Work order {work_order_id} does not exist.')
    order = order[0]
    if order['status'] not in {'In progress', 'On site'}:
        raise ValueError(f'Repair evidence cannot be submitted while status is {order["status"]}.')
    upload.seek(0)
    data = upload.read()
    if not data:
        raise ValueError('Repair evidence upload is empty.')
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f'Repair evidence exceeds the {MAX_UPLOAD_MB} MB limit.')
    ext = Path(upload.name).suffix.lower()
    if ext not in {'.jpg', '.jpeg', '.png', '.webp'}:
        raise ValueError('Repair evidence must be JPG, PNG, or WEBP.')
    target = UPLOAD_DIR / f'{work_order_id}_after_{uuid.uuid4().hex[:10]}{ext}'
    target.write_bytes(data)
    try:
        from PIL import Image
        with Image.open(target) as image:
            image.verify()
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ValueError('Repair evidence is not a readable image.') from exc
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with connect() as connection:
        current = connection.execute('SELECT status FROM work_orders WHERE id=?', (work_order_id,)).fetchone()
        if not current or current['status'] not in {'In progress', 'On site'}:
            target.unlink(missing_ok=True)
            raise ValueError('Work order changed while the evidence was being uploaded; refresh and retry.')
        connection.execute('''UPDATE work_orders SET after_path=?,repair_score=NULL,status='Completion requested',
            completion_requested_at=? WHERE id=?''', (str(target), stamp, work_order_id))
    event('Repair evidence', f'After photograph stored for {work_order_id} by {actor}; awaiting admin review. No repair is marked complete.', order['report_id'])
    return str(target.resolve())


def approve_closure(work_order_id, admin_name, reason):
    admin_name = (admin_name or '').strip()
    reason = (reason or '').strip()
    if not admin_name:
        raise ValueError('Admin approver name is required.')
    if len(reason) < 5:
        raise ValueError('Enter an approval reason of at least five characters.')
    order = rows('SELECT * FROM work_orders WHERE id=?', (work_order_id,))
    if not order:
        raise ValueError(f'Work order {work_order_id} does not exist.')
    order = order[0]
    if order['status'] != 'Completion requested':
        raise ValueError('Admin closure is available only after completion evidence is submitted.')
    if not order['after_path'] or not Path(order['after_path']).is_file():
        raise ValueError('A stored after-repair photograph is required before admin closure.')
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with connect() as connection:
        cursor = connection.execute('''UPDATE work_orders SET status='Closed',closed_at=?,closed_by=?,closure_reason=?
            WHERE id=? AND status='Completion requested' AND after_path IS NOT NULL''',
            (stamp, admin_name, reason, work_order_id))
        if cursor.rowcount != 1:
            raise ValueError('Work order changed during approval; refresh and review its current status.')
    event('Admin approval', f'{work_order_id} closure approved by {admin_name}. Reason: {reason}.', order['report_id'])


def request_reinspection(work_order_id, reviewer, reason):
    reviewer = (reviewer or '').strip()
    reason = (reason or '').strip()
    if not reviewer:
        raise ValueError('Reviewer name is required.')
    if len(reason) < 5:
        raise ValueError('Enter a reinspection reason of at least five characters.')
    order = rows('SELECT * FROM work_orders WHERE id=?', (work_order_id,))
    if not order or order[0]['status'] != 'Completion requested':
        raise ValueError('Reinspection can be requested only for a completion request.')
    with connect() as connection:
        connection.execute("UPDATE work_orders SET status='In progress' WHERE id=? AND status='Completion requested'",
                           (work_order_id,))
    event('Admin review', f'{work_order_id} returned to In progress by {reviewer}. Reason: {reason}.', order[0]['report_id'], 'warning')
