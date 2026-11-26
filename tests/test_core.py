import tempfile
import unittest
from pathlib import Path

from agent import AahvaniAgent
from crm import CRM
from knowledge_base import KnowledgeBase


class AgentCoreTests(unittest.TestCase):
    def setUp(self):
        self.agent = AahvaniAgent(KnowledgeBase())
        # Keep unit tests deterministic and independent of external API keys.
        self.agent.client = None
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

    def test_buying_intent_is_captured_instead_of_dumping_an_faq(self):
        _, session = self.agent.respond("Hello", self.session)
        reply, session = self.agent.respond(
            "I have to book CRM services for my company", session
        )
        self.assertIn("company", session["lead"]["need"].lower())
        self.assertEqual(session["pending_field"], "budget")
        self.assertIn("budget", reply.lower())
        self.assertNotIn("Product Knowledge Base", reply)
        self.assertNotIn("Security and privacy", reply)

    def test_need_to_store_company_data_is_not_misrouted_to_faq(self):
        reply, session = self.agent.respond(
            "I want to store data of my company", self.session
        )
        self.assertIn("store data", session["lead"]["need"].lower())
        self.assertEqual(session["pending_field"], "budget")
        self.assertIn("budget", reply.lower())
        self.assertNotIn("Security and privacy", reply)

    def test_contextual_flexible_budget_answer_advances_to_authority(self):
        _, session = self.agent.respond("Hello", self.session)
        _, session = self.agent.respond(
            "I have to book CRM services for my company", session
        )
        _, session = self.agent.respond("Yes my bugut is flexible", session)
        self.assertEqual(session["lead"]["budget"], "Flexible (no fixed budget)")
        self.assertEqual(session["pending_field"], "authority")

        reply, session = self.agent.respond("Yes", session)
        self.assertEqual(session["lead"]["authority"], "Decision maker")
        self.assertEqual(session["pending_field"], "timeline")
        self.assertIn("when are you hoping", reply.lower())

    def test_faq_is_answered_briefly_then_discovery_continues(self):
        reply, session = self.agent.respond(
            "What features does Northstar CRM provide?", self.session
        )
        self.assertNotIn("Product Knowledge Base", reply)
        self.assertNotIn("## Product overview", reply)
        self.assertEqual(session["pending_field"], "need")
        self.assertIn("manage or improve", reply.lower())

    def test_numeric_budget_and_flexible_status_are_both_captured(self):
        budget = self.agent._extract_budget("55000 and it's flexible")
        self.assertEqual(budget, "55,000 (flexible)")
        comma_formatted_budget = self.agent._extract_budget("My budget is 55,000 and it's flexible")
        self.assertEqual(comma_formatted_budget, "55,000 (flexible)")

    def test_ceo_and_approve_it_are_recognised_as_decision_makers(self):
        self.assertEqual(
            self.agent._extract_authority("I m the CEO of my company"),
            "Decision maker",
        )
        self.assertEqual(
            self.agent._extract_authority("I will approve it"),
            "Decision maker",
        )

    def test_yes_i_will_advances_past_authority_question(self):
        _, session = self.agent.respond("Hello", self.session)
        _, session = self.agent.respond(
            "I want CRM services for my company", session
        )
        _, session = self.agent.respond("55000 and it's flexible", session)
        self.assertEqual(session["pending_field"], "authority")

        reply, session = self.agent.respond("Yes i will", session)
        self.assertEqual(session["lead"]["authority"], "Decision maker")
        self.assertEqual(session["pending_field"], "timeline")
        self.assertIn("when are you hoping", reply.lower())
        self.assertNotIn("final purchase decision", reply.lower())

    def test_end_to_end_lead_to_booking_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "workflow.db")
            _, session = self.agent.respond("Hello", self.session)
            _, session = self.agent.respond(
                "I am Rahul, founder. We need a CRM to track incoming leads. "
                "Our budget is ₹50,000 and we will implement this month. "
                "Email rahul@example.com",
                session,
            )
            lead = session["lead"]
            self.assertEqual(lead["email"], "rahul@example.com")
            self.assertEqual(lead["qualification"], "Qualified")
            lead_id = crm.save_lead(lead)
            appointment_id = crm.book_appointment(lead_id, "Demo slot 1 — 11:00 AM")
            summary = self.agent.summarize(session)
            self.assertTrue(appointment_id)
            self.assertIn("Rahul", summary)
            self.assertIn("Qualified", summary)
            self.assertEqual(crm.appointment_count(), 1)


class CRMTests(unittest.TestCase):
    def test_save_and_book(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "test.db")
            lead_id = crm.save_lead({"name": "Demo User", "email": "demo@example.com", "lead_score": 70, "qualification": "Qualified"})
            appointment_id = crm.book_appointment(lead_id, "Demo slot 1 — 11:00 AM")
            self.assertTrue(appointment_id)
            self.assertEqual(crm.appointment_count(), 1)
            self.assertEqual(crm.dashboard_stats(), (1, 1, 1))
            self.assertEqual(len(crm.recent_leads()), 1)


if __name__ == "__main__":
    unittest.main()
