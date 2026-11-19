"""SQLite storage and schema for RoadGuard AI."""
import sqlite3
from contextlib import contextmanager
from config import DB_PATH

@contextmanager
def connect():
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
        con.commit()
    finally:
        con.close()

def init_db():
    with connect() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS reports(
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
          source TEXT NOT NULL, reporter TEXT, road TEXT NOT NULL, ward TEXT NOT NULL,
          lat REAL, lon REAL, description TEXT, image_path TEXT, image_hash TEXT,
          detection TEXT NOT NULL, detection_conf REAL, trust_score INTEGER NOT NULL,
          trust_status TEXT NOT NULL, trust_reasons TEXT, severity INTEGER NOT NULL,
          traffic INTEGER NOT NULL, importance INTEGER NOT NULL, repeats INTEGER NOT NULL,
          priority_score REAL NOT NULL, priority TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Verified',
          duplicate_of INTEGER REFERENCES reports(id), detection_model TEXT,
          detection_boxes TEXT, annotated_path TEXT,
          verification_status TEXT NOT NULL DEFAULT 'Pending', verification_reason TEXT,
          verified_by TEXT, verified_at TEXT);
        CREATE TABLE IF NOT EXISTS work_orders(
          id TEXT PRIMARY KEY, report_id INTEGER NOT NULL REFERENCES reports(id),
          contractor TEXT, ward TEXT NOT NULL, created_at TEXT NOT NULL, due_at TEXT NOT NULL,
          status TEXT NOT NULL, before_path TEXT, after_path TEXT, repair_score INTEGER,
          completion_requested_at TEXT, closed_at TEXT, closed_by TEXT, closure_reason TEXT);
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, report_id INTEGER,
          stage TEXT NOT NULL, message TEXT NOT NULL, level TEXT DEFAULT 'info');
        CREATE TABLE IF NOT EXISTS contractors(name TEXT PRIMARY KEY, ward TEXT NOT NULL, contact TEXT NOT NULL);
        ''')
        c.executemany("INSERT OR IGNORE INTO contractors VALUES(?,?,?)", [
          ('Ward 12 Roadworks Team','Ward 12','demo@roadguard.local'),
          ('Northside Civic Works','Ward 4','demo@roadguard.local'),
          ('Central Roads Cooperative','Ward 7','demo@roadguard.local')])
        # Additive migration for existing local databases created before model
        # outputs were persisted. Existing report rows remain intact.
        columns = {r['name'] for r in c.execute('PRAGMA table_info(reports)').fetchall()}
        for name, sql_type in [
            ('detection_model', 'TEXT'), ('detection_boxes', 'TEXT'), ('annotated_path', 'TEXT'),
            ('verification_status', "TEXT NOT NULL DEFAULT 'Pending'"),
            ('verification_reason', 'TEXT'), ('verified_by', 'TEXT'), ('verified_at', 'TEXT'),
        ]:
            if name not in columns:
                c.execute(f'ALTER TABLE reports ADD COLUMN {name} {sql_type}')
        order_columns = {r['name'] for r in c.execute('PRAGMA table_info(work_orders)').fetchall()}
        for name in ('closed_by', 'closure_reason'):
            if name not in order_columns:
                c.execute(f'ALTER TABLE work_orders ADD COLUMN {name} TEXT')

def event(stage, message, report_id=None, level='info'):
    from datetime import datetime, timezone
    with connect() as c:
        c.execute("INSERT INTO events(created_at,report_id,stage,message,level) VALUES(?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(timespec='seconds'), report_id, stage, message, level))

def rows(query, params=()):
    with connect() as c:
        return c.execute(query, params).fetchall()
