"""Reset only RoadGuard's local demo records and uploaded evidence."""
from config import DATA_DIR, UPLOAD_DIR
from database import connect, event


def reset_demo_environment():
    upload_root = UPLOAD_DIR.resolve()
    data_root = DATA_DIR.resolve()
    if upload_root == data_root or data_root not in upload_root.parents:
        raise RuntimeError('Configured upload directory is outside the RoadGuard data directory.')
    deleted_files = 0
    if UPLOAD_DIR.exists():
        for item in UPLOAD_DIR.iterdir():
            resolved = item.resolve()
            if upload_root not in resolved.parents:
                raise RuntimeError(f'Refusing to remove upload path outside configured directory: {item}')
            if item.is_file():
                item.unlink()
                deleted_files += 1
            elif item.is_dir():
                # The app stores uploads flat; unknown subtrees are left intact.
                continue
    with connect() as connection:
        connection.execute('DELETE FROM events')
        connection.execute('DELETE FROM work_orders')
        connection.execute('DELETE FROM reports')
        connection.execute("DELETE FROM sqlite_sequence WHERE name IN ('events','work_orders','reports')")
    event('Demo reset', f'Reset local reports, work orders, event history and {deleted_files} uploaded file(s). Seeded demo contractors were retained.')
    return {'deleted_files': deleted_files, 'retained_contractors': True}
