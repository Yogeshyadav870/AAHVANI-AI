import tempfile
import unittest
from pathlib import Path

from agent import AahvaniAgent
from crm import CRM
from knowledge_base import KnowledgeBase


class AgentCoreTests(unittest.TestCase):
    def setUp(self):
        self.agent = AahvaniAgent(KnowledgeBase())
        self.session = self.agent.new_session()

    def test_lead_scoring(self):
        lead = {
            "need": "Need to manage inbound leads",
            "budget": "₹50,000",
            "authority": "Decision maker",
            "timeline": "this month",
        }
        score, status = self.agent._score(lead)
        self.assertEqual(score, 100)
        self.assertEqual(status, "Qualified")

    def test_fact_extraction(self):
        lead = self.session["lead"]
        self.agent._extract_facts(
            "I am Rahul, founder. Our budget is ₹50,000 and we need a sales CRM this month. Email rahul@example.com",
            lead,
        )
        self.assertEqual(lead["name"], "Rahul")
        self.assertEqual(lead["email"], "rahul@example.com")
        self.assertEqual(lead["budget"], "₹50,000")
        self.assertEqual(lead["authority"], "Decision maker")
        self.assertTrue(lead["timeline"])

    def test_conversation_returns_response_and_updates_memory(self):
        answer, session = self.agent.respond("Hello", self.session)
        self.assertIn("AAHVANI AI", answer)
        self.assertEqual(len(session["messages"]), 2)

    def test_knowledge_retrieval(self):
        results = self.agent.knowledge_base.search("What are the lead management features?", top_k=2)
        self.assertTrue(results)
        self.assertIn("Lead management", results[0]["content"])


class CRMTests(unittest.TestCase):
    def test_save_and_book(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "test.db")
            lead_id = crm.save_lead({"name": "Demo User", "email": "demo@example.com", "lead_score": 70, "qualification": "Qualified"})
            appointment_id = crm.book_appointment(lead_id, "Demo slot 1 — 11:00 AM")
            self.assertTrue(appointment_id)
            self.assertEqual(crm.appointment_count(), 1)
            self.assertEqual(len(crm.recent_leads()), 1)


if __name__ == "__main__":
    unittest.main()
