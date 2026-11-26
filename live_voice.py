from __future__ import annotations

import asyncio
import logging
import os
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

    def copy(self) -> "AahvaniLiveHandler":
        return AahvaniLiveHandler()

    def _system_instruction(self) -> str:
        knowledge = self.agent.knowledge_base.documents
        approved_product_knowledge = "\n\n".join(knowledge)
        return f"""
You are AAHVANI AI, a warm, professional, natural-sounding AI sales assistant.
You are speaking to a prospective customer. Keep spoken replies concise and
conversational. Ask only one useful question at a time. Match the customer's
language, including English, Hindi, or natural Hinglish when appropriate.

BUSINESS GOAL
Help small and medium businesses understand Northstar CRM, discover their
sales and customer-management needs, qualify the prospect, and guide them to
a product demonstration.

DISCOVERY
Naturally learn the prospect's:
- Need: what sales/customer-management problem they want to solve.
- Budget: approximate budget or whether it is flexible.
- Authority: whether they make the final decision or need approval.
- Timeline: when they want to implement a solution.
Also capture name, company, email and phone when the customer volunteers them.
Do not interrogate the customer or ask for every field at once.

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
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "Live Voice Mode needs GEMINI_API_KEY. Add your key to the local "
                ".env file, then restart AAHVANI AI. The key stays on the server."
            )

        model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        voice = os.getenv("GEMINI_LIVE_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE

        # Gemini Live is served on v1beta; google-genai 3.x defaults to v1.
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

        self.status_message = f"Connecting to {model}…"
        self._session_context = self._client.aio.live.connect(
            model=model,
            config=config,
        )
        self.session = await self._session_context.__aenter__()
        self.status_message = (
            "Connected. Speak naturally; you can interrupt AAHVANI while it is talking."
        )
        await self._publish_ui()

        self._tasks = [
            asyncio.create_task(self._send_audio_loop(), name="aahvani-live-send"),
            asyncio.create_task(self._receive_loop(), name="aahvani-live-receive"),
        ]

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
                    await self._flush_agent_transcript(interrupted=True)
                    self.status_message = (
                        "Listening — AAHVANI was interrupted and is ready for you."
                    )
                    await self._publish_ui()
                elif turn_complete or generation_complete:
                    await self._flush_agent_transcript(interrupted=False)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("AAHVANI Gemini Live receive loop failed")
            self.status_message = f"Live connection ended: {exc}"
            await self._publish_ui()

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
        self.stop_event.set()
        await self._close_session()
