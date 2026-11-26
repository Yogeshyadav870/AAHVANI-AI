from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item
from google import genai
from google.genai import types

from agent import AahvaniAgent
from crm import CRM

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Kore"


def _audio_to_pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert a mono FastRTC frame to little-endian PCM16 bytes."""
    array = np.asarray(audio)
    if array.ndim > 1:
        array = np.squeeze(array)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype("<i2")
    else:
        array = array.astype("<i2", copy=False)
    return np.ascontiguousarray(array).tobytes()


_ACTIVE_LIVE_HANDLER = None


class AahvaniLiveHandler(AsyncStreamHandler):
    """Per-connection Gemini Live voice handler with automatic CRM lead capture."""

    def __init__(self) -> None:
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.agent = AahvaniAgent()
        self.crm = CRM()
        self.session_state = self.agent.new_session()
        self.chat_history: list[dict[str, str]] = []
        self.last_response_text = ""
        self.status_message = "Ready. Start the microphone to connect."

        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=96)
        self.stop_event = asyncio.Event()
        self.session = None
        self._session_context = None
        self._client = None
        self._tasks: list[asyncio.Task] = []
        self._last_input_text = ""
        self._last_input_at = 0.0
        self._assistant_transcript_buffer: list[str] = []
        self._last_output_text = ""
        self._last_output_at = 0.0
        self._assistant_is_speaking = False
        self._last_typed_text = ""
        self._last_typed_at = 0.0

    def copy(self) -> "AahvaniLiveHandler":
        return AahvaniLiveHandler()

    def set_args(self, args: list[Any]) -> None:
        """Receive the integrated FastRTC textbox submission and route it to Gemini Live."""
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
        if typed_text == self._last_typed_text and now - self._last_typed_at < 2.0:
            return
        self._last_typed_text = typed_text
        self._last_typed_at = now

        loop = getattr(self, "_loop", None)
        if loop is None or self.session is None:
            self.status_message = "Connect the microphone before sending a text message."
            return

        future = asyncio.run_coroutine_threadsafe(self.send_text(typed_text), loop)

        def report_send_result(done_future) -> None:
            try:
                done_future.result()
            except Exception:
                logger.exception("Could not send typed text to the Gemini Live session")

        future.add_done_callback(report_send_result)

    def _system_instruction(self) -> str:
        knowledge = self.agent.knowledge_base.documents
        approved_product_knowledge = "\n\n".join(knowledge)
        current_local_date = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
        return f"""
You are AAHVANI AI, a warm, professional, natural-sounding AI appointment assistant for Jalandhar.
You help customers request appointments across service categories such as healthcare,
salons, vehicle services, professional consultations, fitness, education and other
local services. Keep spoken replies concise and conversational. Ask one useful
question at a time. Match the customer's language, including English, Hindi, or
natural Hinglish when appropriate.
Current local date: {current_local_date}. Time zone: Asia/Kolkata. Interpret relative
dates using this local date, and always read back the resulting calendar date.

BRAND NAME PRONUNCIATION — IMPORTANT
The brand name AAHVANI is pronounced "AAH-vuh-nee" (आह्वनी), followed by
"AI". Say it as three syllables: AAH / vuh / nee. Do not pronounce it as
"A-Vee-Yaani", do not spell out the letters, and do not change the brand name.
Whenever you introduce yourself or say the brand name aloud, use the
pronunciation "AAH-vuh-nee AI".

OPENING GREETING — FIRST TURN
At the start of every new live session, your first spoken sentence must be:
"Hi, I'm AAH-vuh-nee AI. How can I help you?"
Pronounce AAH-vuh-nee as three syllables: AAH / vuh / nee (आह्वनी).
Do not add a long introduction. After this one sentence, stop and listen.

PRIMARY TASK — APPOINTMENT REQUESTS
Help the customer request an appointment in Jalandhar. Support a broad range of
service categories: healthcare, dental, salon/spa, vehicle service, legal or
professional consultation, fitness, education and other services.
Do not assume the request is for Northstar CRM. First understand what service
the customer wants and acknowledge it naturally.

APPOINTMENT DETAILS
Collect only the missing details, one question at a time:
1. Service category and exact service/appointment type.
2. Provider or business name, if the customer has selected one. If not selected,
   record "Provider to be selected"; do not invent providers.
3. City/area. Default to Jalandhar when the customer has not specified another city.
4. Preferred date and time.
5. Customer name.
6. A phone number or email address for provider follow-up.
7. Any short notes or special requirements.
Remember details already given in the conversation and never ask for them again.
A vague preference such as "tomorrow morning" must be clarified to a specific
date and time before saving the request. Do not invent provider availability.
Collect only the information needed to arrange the appointment. Do not ask for
medical history, diagnoses, financial account details or other sensitive details.
Do not provide medical, legal or other professional advice; help arrange the
appropriate appointment instead.

BOOKING CONFIRMATION AND TOOL USE
A server-side tool named save_appointment_request is available. It saves an
appointment request and the customer details into the local AAHVANI CRM.
Never call it until all required details are collected and you have read back
the service, provider (or provider-to-be-selected), city, date, time and contact
to the customer. Then ask exactly: "To confirm, should I submit this appointment
request to the CRM now?" Call the tool only after the customer explicitly says
yes, confirms, or says go ahead. A bare request to find or discuss an appointment
is not confirmation to submit it.
After the tool succeeds, clearly say the request was saved and give the request
reference. Its initial status is "Pending provider confirmation". Do NOT say
the external appointment is confirmed or booked. A real provider/calendar
integration must confirm availability before that status can become Confirmed.
If the tool rejects the request because details or confirmation are missing,
ask only for the missing information or confirmation.

REAL-WORLD PROVIDER LIMITS
This version has no live Jalandhar provider directory and no external provider
calendar integrations yet. Never invent clinic, business or professional names,
opening hours, prices, or available slots. If the customer has not selected a
provider, save "Provider to be selected" and explain that a provider still needs
to confirm the request. The preferred date/time is a request, not verified
availability. Use today's date in Asia/Kolkata when interpreting relative dates.

EXISTING CRM DEMO
If a customer explicitly asks about Northstar CRM, answer using the approved
product knowledge below. The Northstar CRM demo remains fictional sample data.
Do not turn ordinary appointment requests into a CRM product pitch.

APPROVED PRODUCT KNOWLEDGE
{approved_product_knowledge}

GROUNDING AND HONESTY
Never invent providers, provider availability, prices, opening hours, guarantees
or external booking confirmations. Only state that a request is saved after the
save_appointment_request tool returns success. Clearly distinguish a saved
request from a provider-confirmed appointment. For Northstar CRM questions,
use only the approved product knowledge above; this product is fictional demo data.

VOICE BEHAVIOUR
Sound helpful, confident and human. Keep most replies to one or two sentences.
If the customer interrupts, stop the previous point and respond to the new
question. Never claim to be a human salesperson.
""".strip()

    @staticmethod
    def _appointment_request_tool():
        return types.FunctionDeclaration(
            name="save_appointment_request",
            description=(
                "Save a customer-confirmed local service appointment request into the "
                "AAHVANI SQLite CRM. Call only after every required field is collected, "
                "the assistant has read the details back, and the customer explicitly "
                "confirmed. This creates a request pending provider confirmation; it "
                "does not reserve an external provider calendar slot."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "service_category": {
                        "type": "string",
                        "description": "Service category, e.g. healthcare, salon, vehicle service, legal, fitness or education.",
                    },
                    "service_name": {
                        "type": "string",
                        "description": "Specific service or appointment type requested.",
                    },
                    "provider_name": {
                        "type": "string",
                        "description": "Chosen clinic, business or professional; use Provider to be selected if the customer has no preference.",
                    },
                    "city": {
                        "type": "string",
                        "description": "Appointment city. Use Jalandhar when the customer has not specified another city.",
                    },
                    "preferred_date": {
                        "type": "string",
                        "description": "Specific calendar date confirmed by the customer, preferably YYYY-MM-DD.",
                    },
                    "preferred_time": {
                        "type": "string",
                        "description": "Specific preferred local appointment time, e.g. 11:30 AM.",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Customer's name.",
                    },
                    "customer_phone": {
                        "type": "string",
                        "description": "Customer phone number, if provided.",
                    },
                    "customer_email": {
                        "type": "string",
                        "description": "Customer email address, if provided.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional reason for visit or special requirements.",
                    },
                },
                "required": [
                    "service_category",
                    "service_name",
                    "city",
                    "preferred_date",
                    "preferred_time",
                    "customer_name",
                ],
            },
        )

    def _has_explicit_appointment_confirmation(self) -> bool:
        last_user = ""
        last_assistant = ""
        for item in reversed(self.chat_history):
            role = item.get("role")
            if role == "user" and not last_user:
                last_user = str(item.get("content", "")).strip().lower()
            elif role == "assistant" and not last_assistant:
                last_assistant = str(item.get("content", "")).strip().lower()
            if last_user and last_assistant:
                break

        assistant_requested_confirmation = any(
            phrase in last_assistant
            for phrase in (
                "shall i submit",
                "should i submit",
                "would you like me to submit",
                "shall i save",
                "should i save",
                "please confirm",
                "is that correct",
                "would you like me to book",
                "shall i book",
                "shall i place",
                "to confirm, should i",
            )
        )
        normalized_user = re.sub(r"[,!?]", "", last_user).strip()
        customer_confirmed = bool(re.fullmatch(
            r"(?:yes|yeah|yep|sure|please do|go ahead|confirm|confirmed|"
            r"book it|submit it|do it|that's correct|that is correct|"
            r"yes please|yes i confirm|i confirm|yes confirm|yes book it|"
            r"please book it|yes go ahead|please proceed)[. ]*",
            normalized_user,
        ))
        return assistant_requested_confirmation and customer_confirmed

    async def _execute_live_tool_call(self, tool_call) -> None:
        function_responses = []
        for function_call in (getattr(tool_call, "function_calls", None) or []):
            name = str(getattr(function_call, "name", "") or "")
            call_id = getattr(function_call, "id", None)
            args = getattr(function_call, "args", None) or {}
            if not isinstance(args, dict):
                args = {}

            if name != "save_appointment_request":
                result = {"ok": False, "error": f"Unsupported function: {name}"}
            elif not self._has_explicit_appointment_confirmation():
                result = {
                    "ok": False,
                    "needs_confirmation": True,
                    "error": (
                        "The customer has not explicitly confirmed the read-back. "
                        "Do not save yet. Read back the details and ask the customer "
                        "to say yes, confirm, or go ahead."
                    ),
                }
            else:
                try:
                    lead = self.session_state.setdefault("lead", {})
                    customer_name = str(args.get("customer_name") or lead.get("name") or "").strip()
                    customer_phone = str(args.get("customer_phone") or lead.get("phone") or "").strip()
                    customer_email = str(args.get("customer_email") or lead.get("email") or "").strip()
                    service_category = str(args.get("service_category") or "").strip()
                    service_name = str(args.get("service_name") or "").strip()
                    city = str(args.get("city") or "Jalandhar").strip()
                    provider_name = str(args.get("provider_name") or "").strip()
                    if provider_name.lower() in {"", "any", "any provider", "no preference", "not sure", "none"}:
                        provider_name = "Provider to be selected"

                    if not lead.get("name"):
                        lead["name"] = customer_name
                    if not lead.get("phone"):
                        lead["phone"] = customer_phone
                    if not lead.get("email"):
                        lead["email"] = customer_email
                    lead["need"] = f"{service_category}: {service_name}".strip(": ")
                    lead["lead_score"], lead["qualification"] = self.agent._score(lead)
                    self.session_state["lead_id"] = self.crm.save_lead(
                        lead, self.session_state.get("lead_id")
                    )

                    record = self.crm.create_service_appointment_request(
                        lead_id=self.session_state["lead_id"],
                        service_category=service_category,
                        service_name=service_name,
                        provider_name=provider_name,
                        city=city,
                        preferred_date=str(args.get("preferred_date") or "").strip(),
                        preferred_time=str(args.get("preferred_time") or "").strip(),
                        customer_name=customer_name,
                        customer_phone=customer_phone,
                        customer_email=customer_email,
                        notes=str(args.get("notes") or "").strip(),
                        source="Gemini Live Voice",
                    )
                    saved_appointments = self.session_state.setdefault("appointments", [])
                    if not any(
                        item.get("appointment_id") == record["id"]
                        for item in saved_appointments
                    ):
                        saved_appointments.append({
                            "appointment_id": record["id"],
                            "service_category": record["service_category"],
                            "service_name": record["service_name"],
                            "provider_name": record["provider_name"],
                            "city": record["city"],
                            "preferred_date": record["preferred_date"],
                            "preferred_time": record["preferred_time"],
                            "status": record["status"],
                        })
                    result = {
                        "ok": True,
                        "appointment_id": record["id"],
                        "request_reference": record["id"][:8].upper(),
                        "service": record["service_name"],
                        "provider": record["provider_name"],
                        "city": record["city"],
                        "preferred_date": record["preferred_date"],
                        "preferred_time": record["preferred_time"],
                        "status": record["status"],
                        "provider_confirmed": False,
                        "message": (
                            "Appointment request successfully saved in the AAHVANI CRM. "
                            "It is pending provider confirmation; no external calendar has "
                            "been reserved."
                        ),
                        "duplicate_prevented": bool(record.get("duplicate_prevented", False)),
                    }
                    self.status_message = (
                        f"Appointment request {result['request_reference']} saved to CRM — "
                        "pending provider confirmation."
                    )
                    await self._publish_ui()
                except (KeyError, TypeError, ValueError) as exc:
                    result = {"ok": False, "error": str(exc)}
                except Exception as exc:
                    logger.exception("Could not save the appointment request")
                    result = {"ok": False, "error": f"CRM save failed: {type(exc).__name__}: {exc}"}

            function_responses.append(
                types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response=result,
                )
            )

        if function_responses:
            await self.session.send_tool_response(function_responses=function_responses)

    async def start_up(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.status_message = "Microphone connected. Preparing AAHVANI's voice session…"
        await self._publish_ui()

        try:
            api_key = os.getenv("GEMINI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "Live Voice Mode needs GEMINI_API_KEY. Add it to the local "
                    ".env file and restart AAHVANI AI. The key stays on the server."
                )

            requested_model = (
                os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
            )
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
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
                input_audio_transcription=types.AudioTranscriptionConfig(),
                output_audio_transcription=types.AudioTranscriptionConfig(),
                tools=[types.Tool(function_declarations=[self._appointment_request_tool()])],
            )

            # Gemini 3.8 Live has recently shown intermittent realtime-audio
            # failures. Keep it as the preferred model, but retry setup with
            # Gemini 3.1 Flash Live Preview if the first handshake fails.
            models_to_try = [requested_model]
            if requested_model == "gemini-3.8-live":
                models_to_try.append("gemini-3.1-flash-live-preview")

            last_error = None
            connected_model = None
            for candidate_model in models_to_try:
                self.status_message = f"Connecting to {candidate_model}…"
                await self._publish_ui()
                session_context = self._client.aio.live.connect(
                    model=candidate_model,
                    config=config,
                )
                try:
                    session = await asyncio.wait_for(
                        session_context.__aenter__(),
                        timeout=30,
                    )
                    self._session_context = session_context
                    self.session = session
                    connected_model = candidate_model
                    break
                except Exception as exc:
                    last_error = exc
                    try:
                        await session_context.__aexit__(
                            type(exc), exc, exc.__traceback__
                        )
                    except Exception:
                        logger.debug(
                            "Gemini Live context cleanup failed", exc_info=True
                        )
                    logger.warning(
                        "Could not connect to %s: %s: %s",
                        candidate_model,
                        type(exc).__name__,
                        exc,
                    )

            if self.session is None or connected_model is None:
                raise RuntimeError(
                    "Gemini Live could not establish a session. "
                    f"Last error: {type(last_error).__name__}: {last_error}"
                ) from last_error

            global _ACTIVE_LIVE_HANDLER
            _ACTIVE_LIVE_HANDLER = self

            if connected_model != requested_model:
                self.status_message = (
                    f"Connected using fallback model {connected_model}. "
                    f"{requested_model} did not complete its connection."
                )
            else:
                self.status_message = f"Connected to {connected_model}. AAHVANI is greeting you…"
            await self._publish_ui()

            self._tasks = [
                asyncio.create_task(self._send_audio_loop(), name="aahvani-live-send"),
                asyncio.create_task(self._receive_loop(), name="aahvani-live-receive"),
            ]

            await self.session.send_realtime_input(
                text=(
                    "A customer has just started a new live conversation. Speak this "
                    "exact opening sentence aloud now: 'Hi, I'm AAH-vuh-nee AI. "
                    "How can I help you?' Pronounce AAH-vuh-nee as AAH / vuh / nee "
                    "(आह्वनी). Do not add any extra introduction. Then stop and wait "
                    "for the customer to speak."
                )
            )
        except Exception as exc:
            logger.exception("AAHVANI Live Voice startup failed")
            self.status_message = (
                f"Live connection failed — {type(exc).__name__}: {exc}"
            )
            try:
                await self._publish_ui()
            except Exception:
                logger.exception("Could not publish the Live Voice error to the UI")
            raise

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        sample_rate, audio = frame
        if audio is None or self.stop_event.is_set():
            return
        if int(sample_rate) != INPUT_SAMPLE_RATE:
            self.status_message = (
                f"Audio sample-rate mismatch: received {sample_rate} Hz; "
                f"expected {INPUT_SAMPLE_RATE} Hz."
            )
            await self._publish_ui()
            return

        pcm = _audio_to_pcm_bytes(audio)
        if not pcm:
            return

        # Keep the newest microphone frames if the network/model falls behind.
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.input_queue.put(pcm)

    async def send_text(self, text: str) -> None:
        """Send a typed customer message through the active Gemini Live session."""
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return
        if self.session is None:
            raise RuntimeError("Start the microphone before sending a text message.")

        await self.session.send_realtime_input(text=cleaned)
        await self._record_customer_transcript(cleaned)
        self.status_message = "Message sent. AAHVANI is preparing a spoken reply…"
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
                    audio=types.Blob(
                        data=pcm,
                        mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("AAHVANI Gemini Live audio sender failed")
            self.status_message = f"Live audio connection failed: {exc}"
            await self._publish_ui()

    async def _receive_loop(self) -> None:
        try:
            async for response in self.session.receive():
                if self.stop_event.is_set():
                    break
                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    await self._execute_live_tool_call(tool_call)
                    continue

                server_content = getattr(response, "server_content", None)
                if server_content is None:
                    continue

                # Gemini sends finalized input transcription for the utterance.
                input_transcription = getattr(server_content, "input_transcription", None)
                input_text = (
                    getattr(input_transcription, "text", None)
                    if input_transcription else None
                )
                if input_text:
                    await self._record_customer_transcript(input_text)

                # Output transcription arrives in fragments; collect it into one turn.
                output_transcription = getattr(server_content, "output_transcription", None)
                output_text = (
                    getattr(output_transcription, "text", None)
                    if output_transcription else None
                )
                if output_text:
                    self._assistant_transcript_buffer.append(output_text)

                # Process every audio part in this server event.
                model_turn = getattr(server_content, "model_turn", None)
                if model_turn:
                    for part in (getattr(model_turn, "parts", None) or []):
                        inline_data = getattr(part, "inline_data", None)
                        audio_bytes = getattr(inline_data, "data", None) if inline_data else None
                        if not audio_bytes:
                            continue
                        audio = np.frombuffer(audio_bytes, dtype="<i2").copy()
                        if audio.size:
                            if not self._assistant_is_speaking:
                                self._assistant_is_speaking = True
                                self.status_message = (
                                    "🔊 AAHVANI is speaking — you can interrupt at any time."
                                )
                                await self._publish_ui()
                            await self.output_queue.put(
                                (OUTPUT_SAMPLE_RATE, audio.reshape(1, -1))
                            )

                interrupted = bool(getattr(server_content, "interrupted", False))
                turn_complete = bool(getattr(server_content, "turn_complete", False))
                generation_complete = bool(
                    getattr(server_content, "generation_complete", False)
                )

                if interrupted:
                    # Discard audio still queued on the server when the customer barges in.
                    queued_items = []
                    while not self.output_queue.empty():
                        try:
                            item = self.output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if not (isinstance(item, tuple) and len(item) == 2):
                            queued_items.append(item)
                    for item in queued_items:
                        await self.output_queue.put(item)
                    self._assistant_is_speaking = False
                    await self._flush_agent_transcript(interrupted=True)
                    self.status_message = (
                        "🎤 Your turn — AAHVANI is listening. Speak or type below."
                    )
                    await self._publish_ui()
                elif turn_complete or generation_complete:
                    self._assistant_is_speaking = False
                    await self._flush_agent_transcript(interrupted=False)
                    self.status_message = (
                        "🎤 Your turn — speak now or type a message below."
                    )
                    await self._publish_ui()

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("AAHVANI Gemini Live receive loop failed")
            self.status_message = f"Live connection ended: {exc}"
            await self._publish_ui()

    def _apply_contextual_lead_answer(self, text: str, lead: dict[str, Any]) -> None:
        """Interpret short live-voice answers against the previous assistant question."""
        previous_question = next(
            (
                item.get("content", "").lower()
                for item in reversed(self.chat_history)
                if item.get("role") == "assistant"
            ),
            "",
        )
        answer = text.strip().lower()
        is_yes = self.agent._is_simple_yes(text)
        is_no = self.agent._is_simple_no(text)

        if (
            not lead.get("authority")
            and "decision" in previous_question
            and ("approve" in previous_question or "final" in previous_question)
        ):
            if is_yes:
                lead["authority"] = "Decision maker"
            elif is_no:
                lead["authority"] = "Influencer / requires approval"

        if (
            not lead.get("budget")
            and "budget" in previous_question
            and re.search(r"\bflexib\w*\b|\bno fixed\b", answer)
        ):
            lead["budget"] = "Flexible (no fixed budget)"

        if (
            not lead.get("timeline")
            and any(word in previous_question for word in ("when", "timeline", "implement", "start using"))
            and re.search(r"\bsoon\b|\basap\b|\bas soon as possible\b", answer)
        ):
            lead["timeline"] = "Soon"

        lead["lead_score"], lead["qualification"] = self.agent._score(lead)

    async def _record_customer_transcript(self, text: str) -> None:
        cleaned = " ".join(str(text).split()).strip()
        if not cleaned:
            return
        now = time.monotonic()
        if cleaned == self._last_input_text and now - self._last_input_at < 2.0:
            return
        self._last_input_text = cleaned
        self._last_input_at = now

        self.session_state.setdefault("messages", []).append(
            {"role": "user", "content": cleaned}
        )
        lead = self.session_state.setdefault("lead", {})
        self.agent._extract_facts(cleaned, lead)
        self._apply_contextual_lead_answer(cleaned, lead)
        self.session_state["lead_id"] = self.crm.save_lead(
            lead, self.session_state.get("lead_id")
        )
        self.chat_history.append({"role": "user", "content": cleaned})
        self.status_message = (
            "Customer speech recognised. Extracted lead details synced to the local CRM."
        )
        await self._publish_ui()

    async def _flush_agent_transcript(self, interrupted: bool) -> None:
        cleaned = " ".join(" ".join(self._assistant_transcript_buffer).split()).strip()
        self._assistant_transcript_buffer.clear()
        if not cleaned:
            return

        now = time.monotonic()
        if cleaned == self._last_output_text and now - self._last_output_at < 2.0:
            return
        self._last_output_text = cleaned
        self._last_output_at = now
        self.last_response_text = cleaned

        self.session_state.setdefault("messages", []).append(
            {"role": "assistant", "content": cleaned}
        )
        self.chat_history.append({"role": "assistant", "content": cleaned})
        self.status_message = (
            "AAHVANI's reply was interrupted."
            if interrupted
            else "AAHVANI finished speaking. You can continue the conversation."
        )
        await self._publish_ui()

    async def _publish_ui(self) -> None:
        lead = self.session_state.get("lead", {})
        values = (
            self.status_message,
            list(self.chat_history),
            self.session_state,
            self.last_response_text,
            lead.get("name", ""),
            lead.get("company", ""),
            lead.get("email", ""),
            lead.get("phone", ""),
            lead.get("lead_score", 0),
            lead.get("qualification", "Nurture"),
            str(lead.get("sentiment", "neutral")).title(),
            self.crm.recent_leads(),
            *self.crm.dashboard_stats(),
            self.crm.recent_service_appointments(),
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
        global _ACTIVE_LIVE_HANDLER
        self.stop_event.set()
        await self._close_session()
        if _ACTIVE_LIVE_HANDLER is self:
            _ACTIVE_LIVE_HANDLER = None
