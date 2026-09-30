from pathlib import Path
import sqlite3, uuid, datetime
ROOT=Path(__file__).resolve().parents[2]; DB=ROOT/'outputs/appeals.db'
def conn():
    c=sqlite3.connect(DB); c.execute('CREATE TABLE IF NOT EXISTS appeals(id TEXT PRIMARY KEY, applicant_ref TEXT, reason TEXT, status TEXT, reviewer_note TEXT, created_at TEXT, updated_at TEXT)'); return c
def submit(applicant_ref,reason):
    i=str(uuid.uuid4()); now=datetime.datetime.now(datetime.timezone.utc).isoformat(); c=conn(); c.execute('INSERT INTO appeals VALUES(?,?,?,?,?,?,?)',(i,applicant_ref,reason,'submitted','',now,now)); c.commit(); c.close(); return i
def review(i,status,reviewer_note,override=False):
    if status not in ('approved','declined','needs-info'): raise ValueError('invalid status')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat(); c=conn(); c.execute('UPDATE appeals SET status=?, reviewer_note=?, updated_at=? WHERE id=?',(status,reviewer_note,now,i)); c.commit(); c.close(); return {'id':i,'status':status,'override':override}
def list_appeals():
    c=conn(); rows=c.execute('SELECT * FROM appeals ORDER BY created_at DESC').fetchall(); c.close(); return rows
