from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).parent / "aahvani.db"


class CRM:
    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = str(db_path)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a database connection and always close it after the operation."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT, company TEXT,
                need TEXT, budget TEXT, authority TEXT, timeline TEXT, sentiment TEXT,
                lead_score INTEGER, qualification TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS appointments (
                id TEXT PRIMARY KEY, lead_id TEXT NOT NULL, slot TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Booked', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(lead_id) REFERENCES leads(id)
            )""")

    def save_lead(self, lead: dict[str, Any], lead_id: str | None = None) -> str:
        lead_id = lead_id or str(uuid.uuid4())
        columns = ["name", "email", "phone", "company", "need", "budget", "authority", "timeline", "sentiment", "lead_score", "qualification"]
        values = [lead.get(column, "") for column in columns]
        with self._connect() as conn:
            exists = conn.execute("SELECT 1 FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if exists:
                assignments = ", ".join(f"{column} = ?" for column in columns)
                conn.execute(f"UPDATE leads SET {assignments}, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (*values, lead_id))
            else:
                placeholders = ", ".join("?" for _ in range(len(columns) + 1))
                conn.execute(f"INSERT INTO leads (id, {', '.join(columns)}) VALUES ({placeholders})", (lead_id, *values))
        return lead_id

    def book_appointment(self, lead_id: str, slot: str) -> str:
        appointment_id = str(uuid.uuid4())
        with self._connect() as conn:
            lead_exists = conn.execute("SELECT 1 FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if not lead_exists:
                raise ValueError("Save the lead before booking an appointment.")
            existing = conn.execute("SELECT id FROM appointments WHERE lead_id = ? AND slot = ? AND status = 'Booked'", (lead_id, slot)).fetchone()
            if existing:
                return str(existing["id"])
            conn.execute("INSERT INTO appointments (id, lead_id, slot) VALUES (?, ?, ?)", (appointment_id, lead_id, slot))
        return appointment_id

    def recent_leads(self, limit: int = 20) -> list[list[Any]]:
        with self._connect() as conn:
            rows = conn.execute("""SELECT name, company, email, lead_score, qualification, sentiment, updated_at
                FROM leads ORDER BY updated_at DESC LIMIT ?""", (limit,)).fetchall()
        return [list(row) for row in rows]

    def appointment_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM appointments WHERE status = 'Booked'").fetchone()
        return int(row["total"])

    def dashboard_stats(self) -> tuple[int, int, int]:
        """Return total leads, qualified leads, and booked appointments."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total_leads,
                    COALESCE(SUM(CASE WHEN qualification = 'Qualified' THEN 1 ELSE 0 END), 0) AS qualified_leads,
                    (SELECT COUNT(*) FROM appointments WHERE status = 'Booked') AS booked_appointments
                FROM leads
            """).fetchone()
        return int(row["total_leads"]), int(row["qualified_leads"]), int(row["booked_appointments"])
