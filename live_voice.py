from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

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
        return f"""
You are AAHVANI AI, a warm, professional, natural-sounding AI sales assistant.
You are speaking to a prospective customer. Keep spoken replies concise and
conversational. Ask only one useful question at a time. Match the customer's
language, including English, Hindi, or natural Hinglish when appropriate.

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

BUSINESS GOAL
Help small and medium businesses understand Northstar CRM, discover their
sales and customer-management needs, qualify the prospect, and guide them to
a product demonstration.

DISCOVERY AND CONVERSATION STATE
Run a natural, consultative sales conversation. Keep track of information the
customer has already shared and NEVER ask for the same field again.
Collect only missing details, one question at a time, in this order:
1. Need — what they want the CRM to help their company manage or improve.
2. Budget — an approximate range or whether the budget is flexible.
3. Authority — whether they make the final decision or need approval.
4. Timeline — when they want to start using the CRM.
5. Contact — an email address or phone number for demo follow-up.
Skip any item already answered by the customer.

Treat statements such as "I need a CRM", "I want to store company data",
or "we need to manage customer records" as buying requirements. Acknowledge
the need and continue with the next missing discovery question. Do not mistake
these statements for product-information questions just because they contain
words that also appear in the knowledge base.

Interpret short replies using the immediately preceding question. For example,
"yes" after asking whether the customer makes the final decision means they
are the decision maker; "no" means someone else must approve. If the customer
says only "yes" after a budget question, clarify whether the budget is flexible
or ask for an approximate range. If they say the budget is flexible, record it
and move on. Never repeat a question whose answer is already known.

PRODUCT QUESTIONS
Answer product questions only when the customer explicitly asks about a
feature, capability, integration, pricing, security or another product detail.
Answer in one or two concise spoken sentences using only the relevant approved
knowledge. Paraphrase naturally: NEVER read Markdown headings, recite the
whole knowledge-base section, or dump a long list. After answering, return to
the next missing discovery question. Do not make up unsupported capabilities
or pricing.

APPROVED PRODUCT KNOWLEDGE
{approved_product_knowledge}

GROUNDING AND HONESTY
Use only the approved product knowledge above. Never invent prices, guarantees,
integrations, customer references or product capabilities. This product data is
fictional demonstration content. If you do not know an answer, say so and
offer a human follow-up.

LEAD AND BOOKING WORKFLOW
The application receives finalized speech transcripts and extracts available
lead details into its local CRM. Do not claim a lead was saved unless the
application confirms it. When the customer is ready for a demo, explain that
the dashboard offers sample slots and ask for their preference.
A demo is NOT booked until the customer explicitly selects a slot and presses
"Confirm demo booking" in the AAHVANI dashboard. Sample slots are not live
calendar availability.

VOICE BEHAVIOUR
Sound helpful, confident and human. Keep most replies to one or two sentences.
If the customer interrupts, stop the previous point and respond to the new
question. Never claim to be a human salesperson.
""".strip()

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
        is_yes = bool(re.fullmatch(
            r"(?:yes|yeah|yep|yup|correct|sure|that's right|that is right)[.! ]*",
            answer,
        ))
        is_no = bool(re.fullmatch(
            r"(?:no|nope|not really|i don't|i do not)[.! ]*",
            answer,
        ))

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
