from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from google import genai
from google.genai import types

from crm import CRM
from shopping import ShoppingCatalogue

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Kore"
_AVAILABLE_CATEGORIES = (
    "smartphone", "laptop", "headphones", "tablet", "smartwatch",
    "monitor", "camera", "home appliance", "speaker", "gaming console",
)


def _audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    array = np.asarray(audio)
    if array.ndim > 1:
        array = np.squeeze(array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype("<i2")
    else:
        array = array.astype("<i2", copy=False)
    return np.ascontiguousarray(array).tobytes()


class AahvaniLiveHandler(AsyncStreamHandler):
    """One Gemini Live session for both typed and spoken shopping conversations."""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.crm = CRM()
        self.catalogue = ShoppingCatalogue()
        self.session_state: dict[str, Any] = {
            "messages": [],
            "shopping": {
                "category": "", "budget_max": None, "brand": "", "use_case": "",
                "recommendations": [], "selected_product": None,
                "awaiting_selection": False, "awaiting_confirmation": False,
                "order": None,
            },
        }
        self.chat_history: list[dict[str, str]] = []
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=96)
        self.stop_event = asyncio.Event()
        self.session = None
        self._session_context = None
        self._client = None
        self._tasks: list[asyncio.Task] = []
        self._assistant_transcript_buffer: list[str] = []
        self._last_input_text = ""
        self._last_input_at = 0.0
        self._last_output_text = ""
        self._last_output_at = 0.0
        self._assistant_is_speaking = False
        self._session_id = str(uuid.uuid4())
        self.status_message = "Ready. Start the conversation to connect to Gemini Live."
        self.recommendations_markdown = "Your product recommendations will appear here."
        self.order_success_markdown = "No demo order has been placed in this conversation yet."
        self._audio_frames_sent = 0
        self._last_typed_text = ""
        self._last_typed_at = 0.0

    def copy(self) -> "AahvaniLiveHandler":
        return AahvaniLiveHandler()

    def set_args(self, args: list[Any]) -> None:
        """FastRTC's integrated textbox supplies WebRTCData(textbox, audio, webrtc_id)."""
        super().set_args(args)
        typed_text = ""
        for value in args:
            if isinstance(value, dict):
                typed_text = str(value.get("textbox") or "").strip()
            else:
                typed_text = str(getattr(value, "textbox", "") or "").strip()
            if typed_text:
                break
        if not typed_text:
            return

        now = time.monotonic()
        if typed_text == self._last_typed_text and now - self._last_typed_at < 1.5:
            return
        self._last_typed_text = typed_text
        self._last_typed_at = now

        loop = getattr(self, "_loop", None)
        if loop is None or self.session is None:
            self.status_message = "Start the live conversation first, then type or speak in the same chat."
            return

        future = asyncio.run_coroutine_threadsafe(self.send_text(typed_text), loop)

        def report_send_result(done_future) -> None:
            try:
                done_future.result()
            except Exception:
                logger.exception("Could not send typed text to Gemini Live")

        future.add_done_callback(report_send_result)

    def _system_instruction(self) -> str:
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        return f"""
You are AAHVANI AI, a friendly, natural-sounding personal shopping and product recommendation assistant.
Today is {today}. The user's currency is Indian rupees (INR). Speak in the user's language;
understand English, Hindi and natural Hinglish. Keep replies concise and ask one useful
question at a time.

YOUR JOB
Help users discover and compare general consumer products from the AAHVANI demo catalogue.
Supported categories: smartphones, laptops, headphones, tablets, smartwatches, monitors,
cameras, home appliances, speakers and gaming consoles. Detect the product category from
natural language, then gather missing requirements such as budget, primary use and preferred brand.
Do not force irrelevant questions: use the category's needs to ask a helpful follow-up.

PRODUCT SEARCH
A Python tool named search_products is available. Call it only after you know the category,
budget when the customer has one, and enough about use-case/brand to make a useful match.
If brand is unknown or the customer has no preference, use "Any". If a budget is not specified,
ask the customer rather than inventing one. The tool returns up to three real entries from the
local demo catalogue. Recommend ONLY returned products and repeat their exact demo names and prices.
Never invent a model, specification, product, discount or price. Clearly say these are illustrative
catalogue listings and not live retailer stock or offers.
If no result is found, politely ask whether to increase the budget or relax the brand preference.

SELECTION AND ORDER
When the customer chooses an option, call select_product with the returned product ID. Read the
selected product and exact demo price back, and ask: "Shall I place this simulated order?"
Only after the customer explicitly says yes/confirm/go ahead may you call place_demo_order.
Do not call place_demo_order on the same turn as select_product. Never infer confirmation.
The backend verifies the confirmation independently.
After success, say the simulated order is placed, provide the order reference and Pending status.
No payment is taken, no retailer is contacted, and no actual package is shipped.

FOLLOW-UP
If the customer asks about an existing demo order, call get_demo_order_status with the order reference.
Explain the returned status. Do not pretend a real retailer or delivery partner exists.
Allow the customer to change category, budget, brand or use-case at any point and restart matching.

BRAND AND TRUST
Use the name AAH-vuh-nee AI (आह्वनी AI). Be transparent that this is a simulated shopping
environment based on a curated catalogue. Do not ask for card details, UPI PINs, passwords,
addresses or other payment credentials. Do not provide payment or delivery guarantees.
""".strip()

    @staticmethod
    def _search_products_declaration():
        return types.FunctionDeclaration(
            name="search_products",
            description=(
                "Search the local illustrative shopping catalogue and return up to three matching "
                "products. Use only after the customer's product category and enough preferences are known."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(_AVAILABLE_CATEGORIES)},
                    "budget_max": {"type": "integer", "description": "Maximum budget in whole Indian rupees. Use 0 if the customer has not stated a budget."},
                    "brand": {"type": "string", "description": "Preferred brand, or Any if not specified/no preference."},
                    "primary_use": {"type": "string", "description": "Primary use such as gaming, coding, study, work, camera, battery, music, fitness or everyday."},
                },
                "required": ["category", "brand", "primary_use"],
            },
        )

    @staticmethod
    def _select_product_declaration():
        return types.FunctionDeclaration(
            name="select_product",
            description="Select one product from the most recent options after the customer chooses an option. This does not place an order.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Exact product id returned by search_products."},
                },
                "required": ["product_id"],
            },
        )

    @staticmethod
    def _place_order_declaration():
        return types.FunctionDeclaration(
            name="place_demo_order",
            description="Create a simulated demo order after the assistant has read back the selected product and the customer explicitly confirmed in a separate turn.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "Exact product id the customer selected."},
                    "quantity": {"type": "integer", "description": "Quantity from 1 to 10. Default 1."},
                },
                "required": ["product_id"],
            },
        )

    def _has_explicit_order_confirmation(self) -> bool:
        last_user = ""
        last_assistant = ""
        for item in reversed(self.chat_history):
            if item.get("role") == "user" and not last_user:
                last_user = str(item.get("content", "")).strip().lower()
            elif item.get("role") == "assistant" and not last_assistant:
                last_assistant = str(item.get("content", "")).strip().lower()
            if last_user and last_assistant:
                break
        prompt_seen = any(
            phrase in last_assistant
            for phrase in ("shall i place this simulated order", "should i place this order",
                           "shall i place the order", "confirm this order")
        )
        confirmed = bool(re.fullmatch(
            r"(?:yes|yeah|yep|sure|go ahead|confirm|confirmed|place it|buy it|order it|yes please|please do|do it|haan|han|ji haan|jee haan|haan ji|kar do|order kar do|haan kar do)[.! ]*",
            re.sub(r"[,!?]", "", last_user).strip(),
        ))
        return prompt_seen and confirmed

    async def _execute_tool_calls(self, tool_call) -> None:
        responses = []
        for function_call in getattr(tool_call, "function_calls", None) or []:
            name = str(getattr(function_call, "name", "") or "")
            call_id = getattr(function_call, "id", None)
            args = getattr(function_call, "args", None) or {}
            if not isinstance(args, dict):
                args = {}

            shopping = self.session_state["shopping"]
            if name == "search_products":
                category = str(args.get("category") or shopping.get("category") or "").lower()
                if category not in _AVAILABLE_CATEGORIES:
                    category = self.catalogue.detect_category(category)
                budget_raw = args.get("budget_max")
                try:
                    budget = int(budget_raw) if budget_raw not in (None, "", 0, "0") else shopping.get("budget_max")
                except (ValueError, TypeError):
                    budget = shopping.get("budget_max")
                brand = str(args.get("brand") or shopping.get("brand") or "Any").strip()
                primary_use = str(args.get("primary_use") or shopping.get("use_case") or "everyday").lower()
                products = self.catalogue.search(
                    category=category, budget_max=budget, brand=brand,
                    use_case=primary_use, limit=3,
                )
                brand_note = ""
                if not products and brand and brand.lower() != "any":
                    products = self.catalogue.search(
                        category=category, budget_max=budget, use_case=primary_use, limit=3
                    )
                    if products:
                        brand_note = f"No {brand} match was found; alternatives are shown."
                shopping.update({
                    "category": category, "budget_max": budget, "brand": brand,
                    "use_case": primary_use, "recommendations": products,
                    "selected_product": None, "awaiting_selection": bool(products),
                    "awaiting_confirmation": False, "order_idempotency_key": "",
                })
                self.recommendations_markdown = self.catalogue.format_recommendations(products)
                await self._publish_ui()
                result = {
                    "ok": bool(products),
                    "count": len(products),
                    "brand_note": brand_note,
                    "products": [
                        {
                            "id": p["id"], "title": p["title"], "brand": p["brand"],
                            "price_inr": int(p["price_inr"]), "specs": p.get("specs", []),
                            "recommendation_reason": p.get("recommendation_reason", ""),
                        }
                        for p in products
                    ],
                    "note": "All listings and prices are illustrative demo catalogue data, not live retailer stock.",
                }
            elif name == "select_product":
                product_id = str(args.get("product_id") or "").strip()
                product = next(
                    (p for p in shopping.get("recommendations", []) if p["id"] == product_id),
                    None,
                )
                if not product:
                    result = {"ok": False, "error": "Product is not part of the latest recommendations. Search again or ask the customer to choose one of the displayed options."}
                else:
                    shopping["selected_product"] = product
                    shopping["awaiting_selection"] = False
                    shopping["awaiting_confirmation"] = True
                    shopping["order_idempotency_key"] = (
                        f"{self._session_id}:{uuid.uuid4()}"
                    )
                    result = {
                        "ok": True, "product_id": product["id"],
                        "title": product["title"], "brand": product["brand"],
                        "price_inr": int(product["price_inr"]),
                        "next_step": "Read the selection and exact demo price back. Ask whether to place this simulated order. Do not order yet.",
                    }
                    await self._publish_ui()
            elif name == "place_demo_order":
                product_id = str(args.get("product_id") or "").strip()
                selected = shopping.get("selected_product") or {}
                if not self._has_explicit_order_confirmation():
                    result = {"ok": False, "needs_confirmation": True, "error": "The customer has not explicitly confirmed the separately read-back order. Ask for explicit yes/confirmation; do not create an order."}
                elif not selected or selected.get("id") != product_id:
                    result = {"ok": False, "error": "The requested product does not match the product the customer selected. Confirm the correct selection first."}
                else:
                    try:
                        order = self.crm.create_demo_order(
                            product=selected,
                            quantity=int(args.get("quantity") or 1),
                            customer_name="",
                            idempotency_key=shopping.get("order_idempotency_key") or f"{self._session_id}:{uuid.uuid4()}",
                        )
                        shopping["order"] = order
                        self.order_success_markdown = (
                            "## ✅ Demo order placed successfully!\n\n"
                            f"**Order ID:** {order['order_ref']}\n\n"
                            f"**Product:** {order['product_title']}\n\n"
                            f"**Quantity:** {order['quantity']}\n\n"
                            f"**Total:** ₹{int(order['total_price']):,}\n\n"
                            f"**Status:** {order['status']}\n\n"
                            "Simulated order only — no payment was collected and no real shipment was created."
                        )
                        shopping["awaiting_confirmation"] = False
                        shopping["awaiting_selection"] = False
                        result = {
                            "ok": True, "order_ref": order["order_ref"],
                            "product_title": order["product_title"],
                            "brand": order["brand"], "quantity": order["quantity"],
                            "total_price": order["total_price"], "status": order["status"],
                            "simulation_only": True,
                            "message": "Simulated order created in local SQLite. No payment was taken and no actual shipment was created.",
                        }
                        self.status_message = f"Demo order {order['order_ref']} placed successfully — status Pending."
                        await self._publish_ui()
                    except (ValueError, TypeError) as exc:
                        result = {"ok": False, "error": str(exc)}
                    except Exception as exc:
                        logger.exception("Could not create demo order")
                        result = {"ok": False, "error": f"Order creation failed: {type(exc).__name__}: {exc}"}
            elif name == "get_demo_order_status":
                order_reference = str(args.get("order_reference") or "").strip()
                order = self.crm.get_demo_order(order_reference)
                if order:
                    self.order_success_markdown = (
                        "## Order status lookup\\n\\n"
                        f"**Order ID:** {order['order_ref']}\\n\\n"
                        f"**Product:** {order['product_title']}\\n\\n"
                        f"**Total:** ₹{int(order['total_price']):,}\\n\\n"
                        f"**Current status:** {order['status']}\\n\\n"
                        "Simulated order only — no real shipment exists."
                    )
                    await self._publish_ui()
                    result = {
                        "ok": True,
                        "order_reference": order["order_ref"],
                        "product_title": order["product_title"],
                        "total_price": order["total_price"],
                        "status": order["status"],
                        "simulation_only": True,
                    }
                else:
                    result = {
                        "ok": False,
                        "error": "I could not find that order reference in the AAHVANI demo order history.",
                    }
            else:
                result = {"ok": False, "error": f"Unsupported function: {name}"}

            responses.append(types.FunctionResponse(id=call_id, name=name, response=result))

        if responses:
            await self.session.send_tool_response(function_responses=responses)

    @staticmethod
    def _get_order_status_declaration():
        return types.FunctionDeclaration(
            name="get_demo_order_status",
            description="Look up the status of an existing simulated AAHVANI order by its order reference.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "order_reference": {"type": "string", "description": "Order reference such as AAH-12AB34CD."},
                },
                "required": ["order_reference"],
            },
        )

    async def start_up(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.status_message = "Connecting to AAHVANI's live shopping assistant…"
        await self._publish_ui()
        try:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "Live Voice needs GEMINI_API_KEY in the local .env file. Restart the app after adding it."
                )
            requested_model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
            voice = os.getenv("GEMINI_LIVE_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                system_instruction=self._system_instruction(),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                tools=[types.Tool(function_declarations=[
                    self._search_products_declaration(),
                    self._select_product_declaration(),
                    self._place_order_declaration(),
                    self._get_order_status_declaration(),
                ])],
            )
            models_to_try = [requested_model]
            if requested_model == "gemini-3.8-live":
                models_to_try.append("gemini-3.1-flash-live-preview")
            last_error = None
            connected_model = None
            for candidate in models_to_try:
                context = self._client.aio.live.connect(model=candidate, config=config)
                try:
                    session = await asyncio.wait_for(context.__aenter__(), timeout=30)
                    self._session_context = context
                    self.session = session
                    connected_model = candidate
                    break
                except Exception as exc:
                    last_error = exc
                    try:
                        await context.__aexit__(type(exc), exc, exc.__traceback__)
                    except Exception:
                        logger.debug("Live context cleanup failed", exc_info=True)
            if self.session is None:
                raise RuntimeError(
                    f"Gemini Live could not connect. Last error: {type(last_error).__name__}: {last_error}"
                ) from last_error

            self.status_message = f"Connected to {connected_model}. AAHVANI is greeting you…"
            await self._publish_ui()
            self._tasks = [
                asyncio.create_task(self._send_audio_loop(), name="aahvani-live-send"),
                asyncio.create_task(self._receive_loop(), name="aahvani-live-receive"),
            ]
            await self.session.send_realtime_input(
                text=(
                    "Start the shopping conversation. Say exactly: "
                    "'Hi, I'm AAH-vuh-nee AI. What are you looking to buy today?' "
                    "Then listen for the customer's request."
                )
            )
        except Exception as exc:
            logger.exception("AAHVANI Live Voice startup failed")
            self.status_message = f"Live connection failed — {type(exc).__name__}: {exc}"
            await self._publish_ui()
            raise

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        sample_rate, audio = frame
        if audio is None or self.stop_event.is_set():
            return
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            self.status_message = "Microphone supplied an invalid audio sample rate."
            await self._publish_ui()
            return
        array = np.asarray(audio)
        if array.ndim > 1:
            array = np.squeeze(array)
        if sample_rate != INPUT_SAMPLE_RATE and array.size:
            if not np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float32) / 32768.0
            source_positions = np.arange(array.size, dtype=np.float64) / sample_rate
            target_count = max(1, int(round(array.size * INPUT_SAMPLE_RATE / sample_rate)))
            target_positions = np.arange(target_count, dtype=np.float64) / INPUT_SAMPLE_RATE
            array = np.interp(target_positions, source_positions, array).astype(np.float32)
        pcm = _audio_to_pcm_bytes(array)
        if not pcm:
            return
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.input_queue.put(pcm)

    async def send_text(self, text: str) -> None:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return
        if self.session is None:
            self.status_message = "Connect the microphone/live session first, then type or speak in this same chat."
            await self._publish_ui()
            return
        # Record before sending so a very fast Gemini function call can verify
        # the explicit customer confirmation against the latest conversation turn.
        await self._record_customer_message(cleaned)
        await self.session.send_realtime_input(text=cleaned)
        self.status_message = "Message sent to AAHVANI. Preparing a spoken reply…"
        await self._publish_ui()

    async def emit(self):
        return await wait_for_item(self.output_queue)

    async def _send_audio_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    pcm = await asyncio.wait_for(self.input_queue.get(), timeout=0.25)
                except asyncio.TimeoutError:
                    continue
                if not pcm or self.session is None:
                    continue
                await self.session.send_realtime_input(
                    audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}")
                )
                self._audio_frames_sent += 1
                if self._audio_frames_sent == 1:
                    self.status_message = "Microphone audio is reaching Gemini Live. Speak naturally."
                    await self._publish_ui()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini Live audio sender failed")
            self.status_message = f"Microphone send failed: {type(exc).__name__}: {exc}"
            await self._publish_ui()

    async def _receive_loop(self) -> None:
        try:
            # A Live receive iterator completes at the end of a model turn.
            # Re-enter it for each turn to keep the session alive for the whole conversation.
            while not self.stop_event.is_set() and self.session is not None:
                received_event = False
                async for response in self.session.receive():
                    received_event = True
                    if self.stop_event.is_set():
                        break
                    tool_call = getattr(response, "tool_call", None)
                    server = getattr(response, "server_content", None)
                    # Some Live events can carry input transcription and a function
                    # call together. Record the customer turn before validating tools.
                    if server is not None:
                        input_transcription = getattr(server, "input_transcription", None)
                        input_text = getattr(input_transcription, "text", None) if input_transcription else None
                        if input_text:
                            await self._record_customer_message(input_text)
                    if tool_call is not None:
                        await self._execute_tool_calls(tool_call)
                        if server is None:
                            continue
                    if server is None:
                        continue
                    output_transcription = getattr(server, "output_transcription", None)
                    output_text = getattr(output_transcription, "text", None) if output_transcription else None
                    if output_text:
                        self._assistant_transcript_buffer.append(output_text)
                    model_turn = getattr(server, "model_turn", None)
                    if model_turn:
                        for part in getattr(model_turn, "parts", None) or []:
                            inline = getattr(part, "inline_data", None)
                            audio_bytes = getattr(inline, "data", None) if inline else None
                            if audio_bytes:
                                audio = np.frombuffer(audio_bytes, dtype="<i2").copy()
                                if audio.size:
                                    if not self._assistant_is_speaking:
                                        self._assistant_is_speaking = True
                                        self.status_message = "AAHVANI is speaking — you can interrupt."
                                        await self._publish_ui()
                                    await self.output_queue.put((OUTPUT_SAMPLE_RATE, audio.reshape(1, -1)))
                    if bool(getattr(server, "interrupted", False)):
                        self._assistant_is_speaking = False
                        await self._flush_assistant_transcript()
                        self.status_message = "Your turn — speak or type in the chat."
                        await self._publish_ui()
                    elif bool(getattr(server, "turn_complete", False)) or bool(getattr(server, "generation_complete", False)):
                        self._assistant_is_speaking = False
                        await self._flush_assistant_transcript()
                        self.status_message = "Listening — continue speaking or type your next message."
                        await self._publish_ui()
                if not received_event and not self.stop_event.is_set():
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Gemini Live receive loop failed")
            self.status_message = f"Live session error: {type(exc).__name__}: {exc}"
            await self._publish_ui()

    async def _record_customer_message(self, text: str) -> None:
        cleaned = " ".join(str(text).split()).strip()
        if not cleaned:
            return
        now = time.monotonic()
        if cleaned == self._last_input_text and now - self._last_input_at < 2.0:
            return
        self._last_input_text, self._last_input_at = cleaned, now
        self.chat_history.append({"role": "user", "content": cleaned})
        self.session_state.setdefault("messages", []).append({"role": "user", "content": cleaned})

    async def _flush_assistant_transcript(self) -> None:
        cleaned = " ".join(" ".join(self._assistant_transcript_buffer).split()).strip()
        self._assistant_transcript_buffer.clear()
        if not cleaned:
            return
        now = time.monotonic()
        if cleaned == self._last_output_text and now - self._last_output_at < 2.0:
            return
        self._last_output_text, self._last_output_at = cleaned, now
        self.chat_history.append({"role": "assistant", "content": cleaned})
        self.session_state.setdefault("messages", []).append({"role": "assistant", "content": cleaned})

    async def _publish_ui(self) -> None:
        shopping = self.session_state.get("shopping", {})
        values = (
            self.status_message,
            list(self.chat_history),
            self.recommendations_markdown,
            self.order_success_markdown,
            self.crm.recent_demo_orders(),
        )
        await self.output_queue.put(AdditionalOutputs(*values))

    async def _close_session(self) -> None:
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(None, None, None)
            except Exception:
                logger.debug("Gemini Live session already closed", exc_info=True)
            self._session_context = None
            self.session = None

    async def shutdown(self) -> None:
        self.stop_event.set()
        await self._close_session()
