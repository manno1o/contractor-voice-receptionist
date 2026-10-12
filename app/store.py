"""SQLite call log. Only redacted transcripts are stored."""
import json
import os
import sqlite3
import threading


class CallStore:
    def __init__(self, path: str):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("CREATE TABLE IF NOT EXISTS calls (call_sid TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, data TEXT)")

    def save(self, summary: dict) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO calls (call_sid, data) VALUES (?, ?)",
                            (summary["call_sid"], json.dumps(summary, default=str)))
            self.db.commit()

    def recent(self, limit: int = 20) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT data FROM calls ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]
