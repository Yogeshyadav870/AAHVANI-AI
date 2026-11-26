import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app


class FakeInteractions:
    def __init__(self, first_steps, final_text):
        self.first_steps = first_steps
        self.final_text = final_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            if self.first_steps:
                return SimpleNamespace(id="interaction-first", steps=self.first_steps)
            return SimpleNamespace(
                id="interaction-first",
                output_text=self.final_text,
                steps=[],
            )
        return SimpleNamespace(
            id="interaction-final",
            output_text=self.final_text,
            steps=[
                SimpleNamespace(
                    type="model_output",
                    content=[SimpleNamespace(type="text", text=self.final_text)],
                )
            ],
        )


class GeminiInteractionsTests(unittest.TestCase):
    def test_legacy_25_model_setting_is_migrated_to_38_flash(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}):
            self.assertEqual(app._gemini_text_model_name(), "gemini-3.8-flash")

    def test_custom_supported_model_setting_is_preserved(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-3.8-flash"}):
            self.assertEqual(app._gemini_text_model_name(), "gemini-3.8-flash")

    def test_interactions_api_executes_catalogue_tool_and_continues_stateful_turn(self):
        tool_call = SimpleNamespace(
            type="function_call",
            id="call-search-1",
            name="search_products",
            arguments={"category": "smartphone"},
            signature=None,
        )
        fake_interactions = FakeInteractions(
            first_steps=[tool_call],
            final_text="I found some options for you.",
        )
        fake_client = SimpleNamespace(interactions=fake_interactions)

        session = app.initial_state()
        session["shopping"].update({
            "category": "smartphone",
            "budget_max": 25000,
            "brand": "Any",
            "use_case": "battery",
        })

        with patch("app._get_gemini_text_client", return_value=fake_client), patch.dict(
            os.environ, {"GEMINI_MODEL": "gemini-2.5-flash"}
        ):
            answer = app.gemini_shopping_reply("Please find a phone for me.", session)

        self.assertEqual(answer, "I found some options for you.")
        self.assertEqual(fake_interactions.calls[0]["model"], "gemini-3.8-flash")
        self.assertEqual(
            fake_interactions.calls[1]["previous_interaction_id"],
            "interaction-first",
        )
        self.assertEqual(
            fake_interactions.calls[1]["input"][0]["type"],
            "function_result",
        )
        # The Interactions API accepts the tool result as a normal JSON object.
        self.assertIsInstance(fake_interactions.calls[1]["input"][0]["result"], dict)
        self.assertTrue(fake_interactions.calls[1]["input"][0]["result"]["ok"])
        self.assertTrue(fake_interactions.calls[0]["store"])
        self.assertTrue(fake_interactions.calls[1]["store"])
        self.assertEqual(
            session["gemini_previous_interaction_id"],
            "interaction-final",
        )
        self.assertEqual(session["_text_tool_events"], [])

    def test_follow_up_turn_always_includes_new_user_input_and_previous_id(self):
        fake_interactions = FakeInteractions(
            first_steps=[],
            final_text="Of course. Is Samsung your preferred brand?",
        )
        fake_client = SimpleNamespace(interactions=fake_interactions)
        session = app.initial_state()
        session["gemini_previous_interaction_id"] = "previous-interaction-123"

        with patch("app._get_gemini_text_client", return_value=fake_client):
            answer = app.gemini_shopping_reply(
                "I want a Samsung phone under 25000.",
                session,
            )

        self.assertEqual(answer, "Of course. Is Samsung your preferred brand?")
        self.assertEqual(
            fake_interactions.calls[0]["previous_interaction_id"],
            "previous-interaction-123",
        )
        self.assertEqual(
            fake_interactions.calls[0]["input"],
            "I want a Samsung phone under 25000.",
        )
        self.assertEqual(
            session["gemini_previous_interaction_id"],
            "interaction-first",
        )

    def test_checkout_tool_cannot_place_order_in_same_turn_as_selection(self):
        product = {
            "id": "TEST-PHONE-001",
            "title": "Test Phone",
            "brand": "Test Brand",
            "price_inr": 19999,
            "specs": ["Test specification"],
            "recommendation_reason": "Test match",
        }
        select_call = SimpleNamespace(
            type="function_call",
            id="call-select-1",
            name="select_product",
            arguments={"product_id": product["id"]},
            signature=None,
        )
        place_call = SimpleNamespace(
            type="function_call",
            id="call-order-1",
            name="place_demo_order",
            arguments={"product_id": product["id"], "quantity": 1},
            signature=None,
        )
        fake_interactions = FakeInteractions(
            first_steps=[select_call, place_call],
            final_text="Shall I place this simulated order?",
        )
        fake_client = SimpleNamespace(interactions=fake_interactions)

        session = app.initial_state()
        session["shopping"].update({
            "category": "smartphone",
            "budget_max": 25000,
            "brand": "Any",
            "use_case": "battery",
            "recommendations": [product],
            "awaiting_selection": True,
        })

        with patch("app._get_gemini_text_client", return_value=fake_client):
            answer = app.gemini_shopping_reply("option 1", session)

        self.assertIn("Shall I place", answer)
        self.assertIsNone(session["shopping"].get("order"))
        self.assertTrue(session["shopping"]["awaiting_confirmation"])
        self.assertTrue(session["text_confirmation_ready"])


if __name__ == "__main__":
    unittest.main()
