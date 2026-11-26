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
            conn.execute("""CREATE TABLE IF NOT EXISTS service_appointments (
                id TEXT PRIMARY KEY,
                lead_id TEXT NOT NULL,
                service_category TEXT NOT NULL,
                service_name TEXT NOT NULL,
                provider_name TEXT NOT NULL DEFAULT 'Provider to be selected',
                city TEXT NOT NULL DEFAULT 'Jalandhar',
                preferred_date TEXT NOT NULL,
                preferred_time TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                customer_phone TEXT NOT NULL DEFAULT '',
                customer_email TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'Pending provider confirmation',
                provider_reference TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'Gemini Live Voice',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
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

    def create_service_appointment_request(
        self,
        *,
        lead_id: str,
        service_category: str,
        service_name: str,
        provider_name: str = "",
        city: str = "Jalandhar",
        preferred_date: str,
        preferred_time: str,
        customer_name: str,
        customer_phone: str = "",
        customer_email: str = "",
        notes: str = "",
        source: str = "Gemini Live Voice",
    ) -> dict[str, Any]:
        """Save a voice-captured service appointment request, pending provider confirmation."""
        required = {
            "lead_id": lead_id,
            "service_category": service_category,
            "service_name": service_name,
            "city": city,
            "preferred_date": preferred_date,
            "preferred_time": preferred_time,
            "customer_name": customer_name,
        }
        missing = [field for field, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError("Missing required appointment details: " + ", ".join(missing))
        if not (customer_phone or customer_email):
            raise ValueError("A phone number or email address is required for provider follow-up.")

        with self._connect() as conn:
            lead_exists = conn.execute(
                "SELECT 1 FROM leads WHERE id = ?", (lead_id,)
            ).fetchone()
            if not lead_exists:
                raise ValueError("Save the customer record before creating an appointment request.")

            # Avoid duplicate requests when Gemini retries the same confirmed tool call.
            contact_value = customer_phone.strip() or customer_email.strip().lower()
            existing = conn.execute(
                """SELECT * FROM service_appointments
                   WHERE lead_id = ? AND service_category = ? AND service_name = ?
                     AND provider_name = ? AND city = ? AND preferred_date = ?
                     AND preferred_time = ? AND status != 'Cancelled'
                   ORDER BY created_at DESC LIMIT 1""",
                (
                    lead_id,
                    service_category.strip(),
                    service_name.strip(),
                    (provider_name or "Provider to be selected").strip(),
                    city.strip(),
                    preferred_date.strip(),
                    preferred_time.strip(),
                ),
            ).fetchone()
            if existing and contact_value in {
                str(existing["customer_phone"] or "").strip(),
                str(existing["customer_email"] or "").strip().lower(),
            }:
                record = dict(existing)
                record["duplicate_prevented"] = True
                return record

            appointment_id = str(uuid.uuid4())
            provider = provider_name.strip() or "Provider to be selected"
            conn.execute(
                """INSERT INTO service_appointments (
                    id, lead_id, service_category, service_name, provider_name, city,
                    preferred_date, preferred_time, customer_name, customer_phone,
                    customer_email, notes, status, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          'Pending provider confirmation', ?)""",
                (
                    appointment_id,
                    lead_id,
                    service_category.strip(),
                    service_name.strip(),
                    provider,
                    city.strip(),
                    preferred_date.strip(),
                    preferred_time.strip(),
                    customer_name.strip(),
                    customer_phone.strip(),
                    customer_email.strip(),
                    notes.strip(),
                    source.strip(),
                ),
            )
            row = conn.execute(
                "SELECT * FROM service_appointments WHERE id = ?", (appointment_id,)
            ).fetchone()
        return dict(row)

    def recent_service_appointments(self, limit: int = 50) -> list[list[Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT customer_name, service_category, service_name, provider_name,
                          city, preferred_date, preferred_time, status, customer_phone,
                          customer_email, id, created_at
                   FROM service_appointments
                   ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [list(row) for row in rows]

    def service_appointment_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total FROM service_appointments"
            ).fetchone()
        return int(row["total"])

    def update_service_appointment_status(
        self,
        appointment_id: str,
        status: str,
        provider_reference: str = "",
    ) -> None:
        allowed = {
            "Pending provider confirmation",
            "Confirmed",
            "Declined",
            "Cancelled",
        }
        if status not in allowed:
            raise ValueError(f"Unsupported appointment status: {status}")
        if status == "Confirmed" and not provider_reference.strip():
            raise ValueError(
                "Add a provider confirmation reference or confirmation note before marking Confirmed."
            )
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE service_appointments
                   SET status = ?, provider_reference = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (status, provider_reference.strip(), appointment_id),
            )
            if cursor.rowcount == 0:
                raise ValueError("Appointment request was not found.")

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
