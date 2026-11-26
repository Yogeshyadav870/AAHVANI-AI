import tempfile
import unittest
from pathlib import Path

from agent import AahvaniAgent
from crm import CRM
from knowledge_base import KnowledgeBase
from shopping import ShoppingAgent, ShoppingCatalogue


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

    def test_text_chat_routes_dental_appointment_away_from_crm_sales(self):
        reply, session = self.agent.respond(
            "I need a dental appointment tomorrow at 11 AM", self.session
        )
        request = session["appointment_request"]
        self.assertEqual(session["active_flow"], "appointment")
        self.assertEqual(request["service_category"], "Healthcare")
        self.assertEqual(request["service_name"], "Dental consultation")
        self.assertEqual(request["preferred_time"], "11:00 AM")
        self.assertEqual(request["city"], "Jalandhar")
        self.assertIn("name", reply.lower())
        self.assertNotIn("budget", reply.lower())

    def test_text_appointment_requires_readback_confirmation_before_saving(self):
        _, session = self.agent.respond(
            "I need a dental appointment tomorrow at 11 AM", self.session
        )
        reply, session = self.agent.respond(
            "My name is Yogesh and my phone is +91 9876543210", session
        )
        self.assertIn("should i submit", reply.lower())
        self.assertNotIn("appointment_to_save", session)

        reply, session = self.agent.respond("yes", session)
        self.assertIn("submitting", reply.lower())
        self.assertEqual(session["appointment_to_save"]["customer_name"], "Yogesh")
        self.assertEqual(
            session["appointment_to_save"]["customer_phone"], "+91 9876543210"
        )

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

    def test_job_title_is_not_saved_as_customer_name(self):
        lead = self.session["lead"]
        self.agent._extract_facts("I m the CEO of my company", lead)
        self.assertEqual(lead["name"], "")
        self.assertEqual(lead["authority"], "Decision maker")

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
    def test_save_service_appointment_request_to_crm(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "appointments.db")
            lead_id = crm.save_lead({
                "name": "Rahul",
                "phone": "9876543210",
                "need": "Healthcare: Dental consultation",
            })
            record = crm.create_service_appointment_request(
                lead_id=lead_id,
                service_category="Healthcare",
                service_name="Dental consultation",
                provider_name="City Dental Clinic",
                city="Jalandhar",
                preferred_date="2026-12-02",
                preferred_time="11:30 AM",
                customer_name="Rahul",
                customer_phone="9876543210",
                notes="Routine dental check-up",
            )
            self.assertEqual(record["status"], "Pending provider confirmation")
            self.assertEqual(record["city"], "Jalandhar")
            self.assertEqual(record["customer_name"], "Rahul")
            self.assertEqual(crm.service_appointment_count(), 1)
            self.assertEqual(len(crm.recent_service_appointments()), 1)

    def test_duplicate_service_appointment_request_is_not_inserted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "appointments.db")
            lead_id = crm.save_lead({"name": "Rahul", "phone": "9876543210"})
            args = {
                "lead_id": lead_id,
                "service_category": "Healthcare",
                "service_name": "Dental consultation",
                "provider_name": "City Dental Clinic",
                "city": "Jalandhar",
                "preferred_date": "2026-12-02",
                "preferred_time": "11:30 AM",
                "customer_name": "Rahul",
                "customer_phone": "9876543210",
            }
            first = crm.create_service_appointment_request(**args)
            second = crm.create_service_appointment_request(**args)
            self.assertEqual(first["id"], second["id"])
            self.assertTrue(second["duplicate_prevented"])
            self.assertEqual(crm.service_appointment_count(), 1)

    def test_service_appointment_requires_a_customer_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "appointments.db")
            lead_id = crm.save_lead({"name": "Rahul"})
            with self.assertRaisesRegex(ValueError, "phone number or email"):
                crm.create_service_appointment_request(
                    lead_id=lead_id,
                    service_category="Healthcare",
                    service_name="Dental consultation",
                    city="Jalandhar",
                    preferred_date="2026-12-02",
                    preferred_time="11:30 AM",
                    customer_name="Rahul",
                )

    def test_appointment_status_can_be_updated_after_provider_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "appointments.db")
            lead_id = crm.save_lead({"name": "Rahul", "phone": "9876543210"})
            record = crm.create_service_appointment_request(
                lead_id=lead_id,
                service_category="Vehicle service",
                service_name="Car inspection",
                city="Jalandhar",
                preferred_date="2026-12-03",
                preferred_time="10:00 AM",
                customer_name="Rahul",
                customer_phone="9876543210",
            )
            crm.update_service_appointment_status(
                record["id"], "Confirmed", "PROVIDER-REF-123"
            )
            updated = crm.recent_service_appointments()[0]
            self.assertEqual(updated[7], "Confirmed")

    def test_confirmed_status_requires_provider_confirmation_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "appointments.db")
            lead_id = crm.save_lead({"name": "Rahul", "phone": "9876543210"})
            record = crm.create_service_appointment_request(
                lead_id=lead_id,
                service_category="Healthcare",
                service_name="Dental consultation",
                city="Jalandhar",
                preferred_date="2026-12-04",
                preferred_time="12:00 PM",
                customer_name="Rahul",
                customer_phone="9876543210",
            )
            with self.assertRaisesRegex(ValueError, "provider confirmation"):
                crm.update_service_appointment_status(record["id"], "Confirmed")

    def test_appointment_details_are_included_in_session_summary(self):
        agent = AahvaniAgent(KnowledgeBase())
        session = agent.new_session()
        session["appointments"] = [{
            "appointment_id": "abc12345-0000",
            "service_category": "Healthcare",
            "service_name": "Dental consultation",
            "provider_name": "Provider to be selected",
            "city": "Jalandhar",
            "preferred_date": "2026-12-04",
            "preferred_time": "12:00 PM",
            "status": "Pending provider confirmation",
        }]
        summary = agent.summarize(session)
        self.assertIn("Dental consultation", summary)
        self.assertIn("Jalandhar", summary)
        self.assertIn("Pending provider confirmation", summary)

    def test_save_and_book(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "test.db")
            lead_id = crm.save_lead({"name": "Demo User", "email": "demo@example.com", "lead_score": 70, "qualification": "Qualified"})
            appointment_id = crm.book_appointment(lead_id, "Demo slot 1 — 11:00 AM")
            self.assertTrue(appointment_id)
            self.assertEqual(crm.appointment_count(), 1)
            self.assertEqual(crm.dashboard_stats(), (1, 1, 1))
            self.assertEqual(len(crm.recent_leads()), 1)


class ShoppingAgentTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = ShoppingCatalogue()
        self.agent = ShoppingAgent(self.catalogue)
        self.session = self.agent.new_session()

    def test_detects_multiple_general_product_categories(self):
        self.assertEqual(self.catalogue.detect_category("I want to buy a phone"), "smartphone")
        self.assertEqual(self.catalogue.detect_category("Suggest a gaming laptop"), "laptop")
        self.assertEqual(self.catalogue.detect_category("Need wireless headphones"), "headphones")
        self.assertEqual(self.catalogue.detect_category("Looking for a smartwatch"), "smartwatch")

    def test_parses_indian_budget_formats(self):
        self.assertEqual(self.catalogue.parse_budget("phone under ₹15,000"), 15000)
        self.assertEqual(self.catalogue.parse_budget("15k ke andar"), 15000)
        self.assertEqual(self.catalogue.parse_budget("my budget is 60000"), 60000)
        self.assertEqual(self.catalogue.parse_budget("1.2 lakh"), 120000)

    def test_product_search_obeys_category_budget_and_limit(self):
        products = self.catalogue.search(
            category="laptop", budget_max=60000, use_case="gaming", limit=3
        )
        self.assertLessEqual(len(products), 3)
        self.assertTrue(products)
        self.assertTrue(all(item["category"] == "laptop" for item in products))
        self.assertTrue(all(item["price_inr"] <= 60000 for item in products))

    def test_text_chat_collects_requirements_and_recommends_phone(self):
        reply, session = self.agent.respond(
            "I need a smartphone under ₹15,000", self.session
        )
        self.assertIn("camera", reply.lower())
        reply, session = self.agent.respond("Camera is important", session)
        self.assertIn("brand", reply.lower())
        reply, session = self.agent.respond("Samsung", session)
        self.assertIn("matching options", reply.lower())
        self.assertTrue(session["shopping"]["recommendations"])
        self.assertTrue(
            all(p["price_inr"] <= 15000 for p in session["shopping"]["recommendations"])
        )

    def test_text_chat_recommends_gaming_laptop_within_budget(self):
        _, session = self.agent.respond(
            "I want a gaming laptop under ₹60,000", self.session
        )
        reply, session = self.agent.respond("Gaming", session)
        self.assertIn("brand", reply.lower())
        reply, session = self.agent.respond("Any brand", session)
        self.assertIn("matching options", reply.lower())
        self.assertTrue(session["shopping"]["recommendations"])
        self.assertTrue(
            all(p["price_inr"] <= 60000 for p in session["shopping"]["recommendations"])
        )

    def test_selected_product_requires_separate_confirmation(self):
        _, session = self.agent.respond(
            "I need a phone under ₹15,000 for camera", self.session
        )
        _, session = self.agent.respond("Any brand", session)
        self.assertTrue(session["shopping"]["recommendations"])
        reply, session = self.agent.respond("option 1", session)
        self.assertTrue(session["shopping"]["awaiting_confirmation"])
        self.assertIn("shall i place", reply.lower())
        self.assertNotIn("demo_order_to_create", session)

        reply, session = self.agent.respond("yes", session)
        self.assertIn("simulated order", reply.lower())
        self.assertIn("demo_order_to_create", session)
        self.assertEqual(
            session["demo_order_to_create"]["product"]["id"],
            session["shopping"]["selected_product"]["id"],
        )

    def test_brand_no_match_falls_back_to_other_brands(self):
        _, session = self.agent.respond(
            "I need a phone under ₹20,000 for battery", self.session
        )
        reply, session = self.agent.respond("Apple", session)
        self.assertIn("alternatives", reply.lower())
        self.assertTrue(session["shopping"]["recommendations"])
        self.assertTrue(
            all(p["price_inr"] <= 20000 for p in session["shopping"]["recommendations"])
        )


class DemoOrderTests(unittest.TestCase):
    def test_create_demo_order_and_update_to_delivered(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "orders.db")
            catalogue = ShoppingCatalogue()
            product = catalogue.get("phone-samsung-a")
            self.assertIsNotNone(product)
            order = crm.create_demo_order(product=product, quantity=2)
            self.assertTrue(order["order_ref"].startswith("AAH-"))
            self.assertEqual(order["status"], "Pending")
            self.assertEqual(order["quantity"], 2)
            self.assertEqual(order["total_price"], product["price_inr"] * 2)
            self.assertEqual(crm.demo_order_count(), 1)

            crm.update_demo_order_status(order["order_ref"], "Delivered")
            updated = crm.get_demo_order(order["order_ref"])
            self.assertEqual(updated["status"], "Delivered")
            self.assertEqual(crm.recent_demo_orders()[0][0], order["order_ref"])

    def test_demo_order_idempotency_prevents_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "orders.db")
            product = ShoppingCatalogue().get("laptop-hp-victus")
            first = crm.create_demo_order(
                product=product, idempotency_key="session-1:laptop-hp-victus"
            )
            second = crm.create_demo_order(
                product=product, idempotency_key="session-1:laptop-hp-victus"
            )
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(crm.demo_order_count(), 1)

    def test_invalid_quantity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            crm = CRM(Path(directory) / "orders.db")
            product = ShoppingCatalogue().get("phone-samsung-a")
            with self.assertRaisesRegex(ValueError, "quantity"):
                crm.create_demo_order(product=product, quantity=0)



if __name__ == "__main__":
    unittest.main()
