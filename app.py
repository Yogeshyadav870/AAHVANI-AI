from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agent import AahvaniAgent
from crm import CRM
from shopping import ShoppingAgent, ShoppingCatalogue

try:
    from fastrtc import WebRTC
    from live_voice import AahvaniLiveHandler
    LIVE_VOICE_AVAILABLE = True
except ImportError as live_import_error:
    WebRTC = None
    AahvaniLiveHandler = None
    LIVE_VOICE_AVAILABLE = False
    LIVE_VOICE_IMPORT_ERROR = str(live_import_error)

APP_NAME = "AAHVANI AI"
_WHISPER_MODEL = None
agent = AahvaniAgent()
crm = CRM()
shopping_catalogue = ShoppingCatalogue()
shopping_agent = ShoppingAgent(shopping_catalogue)
SAMPLE_SLOTS = ["Demo slot 1 — 11:00 AM", "Demo slot 2 — 2:00 PM", "Demo slot 3 — 4:30 PM"]


def initial_state():
    session = shopping_agent.new_session()
    session["session_id"] = str(uuid.uuid4())
    return session


def shopping_chat(message, history, session):
    session = session or initial_state()
    session.setdefault("session_id", str(uuid.uuid4()))
    order_match = re.search(r"\bAAH-[A-Z0-9]{8}\b", str(message or ""), re.I)
    if order_match and re.search(r"\b(order|status|track|delivery|delivered)\b", str(message or ""), re.I):
        record = crm.get_demo_order(order_match.group(0))
        if record:
            answer = (
                f"Your demo order **{record['order_ref']}** for **{record['product_title']}** "
                f"is currently **{record['status']}**. This is a simulated order; no real shipment exists."
            )
            order_panel = (
                "## Order status lookup\n\n"
                f"**Order ID:** {record['order_ref']}\n\n"
                f"**Product:** {record['product_title']}\n\n"
                f"**Total:** ₹{int(record['total_price']):,}\n\n"
                f"**Current status:** {record['status']}\n\n"
                "Simulated order only — no real shipment exists."
            )
        else:
            answer = f"I couldn't find demo order **{order_match.group(0)}** in the order history."
            order_panel = "No matching demo order was found."
        history = list(history or [])
        history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": answer}])
        return (
            history, session,
            shopping_catalogue.format_recommendations(session.get("shopping", {}).get("recommendations", [])),
            order_panel,
            crm.recent_demo_orders(),
            "",
        )

    answer, session = shopping_agent.respond(message, session)
    order_request = session.pop("demo_order_to_create", None)

    if order_request:
        product = order_request["product"]
        try:
            record = crm.create_demo_order(
                product=product,
                quantity=int(order_request.get("quantity") or 1),
                customer_name="",
                idempotency_key=order_request.get("idempotency_key") or f"{session['session_id']}:{uuid.uuid4()}",
            )
            session.setdefault("shopping", {})["order"] = record
            session["shopping"]["awaiting_confirmation"] = False
            session["shopping"]["awaiting_selection"] = False
            answer = shopping_agent.format_order_confirmation(record)
        except Exception as exc:
            answer = (
                "I couldn't create the simulated order. Your selection is still available. "
                f"Please try again. Details: {type(exc).__name__}: {exc}"
            )
            session.setdefault("shopping", {})["awaiting_confirmation"] = True

        if session.get("messages") and session["messages"][-1].get("role") == "assistant":
            session["messages"][-1]["content"] = answer

    history = list(history or [])
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    shopping = session.get("shopping", {})
    recommendations = shopping.get("recommendations", [])
    recommendation_panel = (
        shopping_catalogue.format_recommendations(recommendations)
        if recommendations else
        "Your recommendations will appear here once AAHVANI understands what you need."
    )
    order = shopping.get("order") or {}
    order_confirmation = (
        shopping_agent.format_order_confirmation(order)
        if order else "No demo order has been placed in this conversation yet."
    )
    return (
        history,
        session,
        recommendation_panel,
        order_confirmation,
        crm.recent_demo_orders(),
        "",
    )


def update_demo_order_status(order_ref, status):
    try:
        reference = str(order_ref or "").strip()
        crm.update_demo_order_status(reference, status)
        record = crm.get_demo_order(reference)
        receipt = (
            "## Order status updated\\n\\n"
            f"**Order ID:** {record['order_ref']}\\n\\n"
            f"**Product:** {record['product_title']}\\n\\n"
            f"**Total:** ₹{int(record['total_price']):,}\\n\\n"
            f"**Current status:** {record['status']}\\n\\n"
            "Simulated order only — no real shipment exists."
        ) if record else "No demo order has been placed in this conversation yet."
        return (
            crm.recent_demo_orders(),
            f"Order **{reference}** updated to **{status}**.",
            receipt,
        )
    except ValueError as exc:
        return crm.recent_demo_orders(), f"**Could not update order:** {exc}", "No order status update was made."


def refresh_demo_orders():
    return crm.recent_demo_orders()



def speak_text(text):
    if not text or not text.strip():
        return None, "No assistant response is available to speak yet."
    try:
        import edge_tts
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        output_path = handle.name
        handle.close()
        voice = os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural")
        asyncio.run(edge_tts.Communicate(text.strip(), voice=voice).save(output_path))
        return output_path, "Generated speech using the optional online Edge TTS service."
    except ImportError:
        return None, "Install edge-tts to enable spoken responses. Text chat remains available."
    except Exception as exc:
        return None, f"Text-to-speech could not run: {exc}"


def speak_last_response(history):
    if not history:
        return None, "Start a conversation before requesting audio."
    for item in reversed(history):
        if item.get("role") == "assistant" and isinstance(item.get("content"), str):
            return speak_text(item["content"])
    return None, "No assistant response is available to speak yet."


def transcribe_audio(audio_path):
    global _WHISPER_MODEL
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
        if _WHISPER_MODEL is None:
            model_name = os.getenv("WHISPER_MODEL", "base")
            _WHISPER_MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = _WHISPER_MODEL.transcribe(audio_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text or "No speech detected. Please try again or type your message."
    except ImportError:
        return "Voice transcription is optional. Install requirements-voice.txt and restart. You can continue using text chat."
    except Exception as exc:
        return f"Could not transcribe audio: {exc}"


def voice_conversation(transcript, history, session):
    transcript = (transcript or "").strip()
    failure_prefixes = ("Could not transcribe audio:", "Voice transcription is optional", "No speech detected")
    if not transcript or transcript.startswith(failure_prefixes):
        return (*([gr.skip()] * 16), None, "Record a voice message first, then send it here.")

    result = chat(transcript, history, session)
    audio_path, speech_status = speak_text(result[11])
    return (*result, audio_path, f"Recognised: {transcript}\n\n{speech_status}")


def update_lead_fields(name, company, email, phone, session):
    session = session or initial_state()
    lead = session.setdefault("lead", {})
    lead.update({
        "name": name or lead.get("name", ""),
        "company": company or lead.get("company", ""),
        "email": email or lead.get("email", ""),
        "phone": phone or lead.get("phone", ""),
    })
    lead["lead_score"], lead["qualification"] = agent._score(lead)
    session["lead_id"] = crm.save_lead(lead, session.get("lead_id"))
    return session, lead["lead_score"], lead["qualification"], crm.recent_leads(), "Lead saved to the local CRM.", *crm.dashboard_stats()


def book_demo(slot, session):
    session = session or initial_state()
    lead = session.setdefault("lead", {})
    if not lead.get("email") and not lead.get("phone"):
        return session, crm.recent_leads(), "Add an email address or phone number before booking.", *crm.dashboard_stats()
    session["lead_id"] = crm.save_lead(lead, session.get("lead_id"))
    try:
        crm.book_appointment(session["lead_id"], slot)
    except ValueError as exc:
        return session, crm.recent_leads(), str(exc), *crm.dashboard_stats()
    return session, crm.recent_leads(), f"Demo appointment saved for {slot}. This is a local prototype booking, not a live calendar reservation.", *crm.dashboard_stats()


def make_summary(session):
    return agent.summarize(session or initial_state())


def refresh_crm():
    return crm.recent_leads(), *crm.dashboard_stats()


def refresh_service_appointments():
    return crm.recent_service_appointments()


def update_service_appointment_status(request_id, status, provider_reference):
    try:
        crm.update_service_appointment_status(
            str(request_id or "").strip(),
            status,
            str(provider_reference or "").strip(),
        )
        return (
            crm.recent_service_appointments(),
            f"Appointment status updated to **{status}**.",
        )
    except ValueError as exc:
        return crm.recent_service_appointments(), f"**Could not update:** {exc}"


CUSTOM_CSS = """
:root, :host, html, body, gradio-app, .gradio-container,
body.dark, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    --page-bg: #dfcfbf !important;
    --card-bg: #fffdf9 !important;
    --inner-bg: #f6eee5 !important;
    --inner-bg-soft: #faf5ef !important;
    --ink: #211a14 !important;
    --muted: #74685c !important;
    --gold: #a97617 !important;
    --gold-deep: #87590f !important;
    --gold-soft: #e8d4ae !important;
    --line: #e5d6c7 !important;

    /* Gradio theme tokens: explicitly force the light palette. */
    --body-background-fill: #dfcfbf !important;
    --body-text-color: #211a14 !important;
    --background-fill-primary: #fffdf9 !important;
    --background-fill-secondary: #f6eee5 !important;
    --block-background-fill: #fffdf9 !important;
    --block-border-color: #e5d6c7 !important;
    --block-label-text-color: #6e6052 !important;
    --input-background-fill: #f7f0e8 !important;
    --input-border-color: #e3d3c2 !important;
    --input-placeholder-color: #897b6d !important;
    --button-primary-background-fill: #332316 !important;
    --button-primary-background-fill-hover: #49321f !important;
    --button-primary-text-color: #fffdf9 !important;
    --button-secondary-background-fill: #f5ede3 !important;
    --button-secondary-text-color: #654719 !important;
    --border-color-primary: #e5d6c7 !important;
    --border-color-accent: #d8c6b3 !important;
}

html, body, body.dark {
    min-height: 100%;
    background: #dfcfbf !important;
    color: #211a14 !important;
}
body, body.dark, gradio-app {
    background:
        radial-gradient(ellipse at 12% 8%, rgba(255, 250, 243, .48), transparent 34%),
        radial-gradient(ellipse at 92% 92%, rgba(245, 222, 194, .42), transparent 32%),
        linear-gradient(135deg, #e2d1c0 0%, #ddcbb9 48%, #e6d4c2 100%) fixed !important;
}
#ambient-background { display: none !important; }

.gradio-container, body.dark .gradio-container, gradio-app .gradio-container {
    position: relative !important;
    z-index: 1 !important;
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 28px 30px 54px !important;
    background: transparent !important;
    color: #211a14 !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}
.gradio-container main, .gradio-container .main, .gradio-container .contain,
.gradio-container .wrap, .gradio-container .form,
.gradio-container [data-testid="block"] {
    color: #211a14 !important;
}

#hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 190px;
    padding: 30px 40px;
    margin: 0 0 24px;
    border: 1px solid rgba(255,255,255,.92);
    border-radius: 28px;
    background:
        radial-gradient(ellipse at 80% 100%, rgba(224, 199, 159, .18), transparent 38%),
        linear-gradient(112deg, #fbf8f3 0%, #f8f4ed 56%, #f5efe5 100%) !important;
    box-shadow: 0 14px 34px rgba(91, 65, 38, .10), inset 0 1px 0 #fff;
}
#hero:before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 370px;
    height: 140px;
    right: 4%;
    bottom: -95px;
    border: 1px solid rgba(182, 135, 54, .15);
    border-radius: 50%;
    box-shadow: 0 -16px 0 -15px rgba(182, 135, 54, .12), 0 -32px 0 -31px rgba(182, 135, 54, .10);
    transform: rotate(-10deg);
}
.hero-copy { position: relative; z-index: 1; }
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 13px;
    color: #936515;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.55px;
    text-transform: uppercase;
}
.hero-dot {
    width: 8px;
    height: 8px;
    display: inline-block;
    border-radius: 50%;
    background: #bf8d31;
    box-shadow: 0 0 0 4px rgba(191, 141, 49, .12);
}
.hero-tag {
    padding: 5px 11px;
    margin-left: 5px;
    border: 1px solid #e8ddd0;
    border-radius: 99px;
    color: #75685a;
    background: rgba(255,255,255,.62);
    font-size: 9px;
    letter-spacing: 1px;
}
#hero h1 {
    margin: 0;
    color: #211a14;
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(38px, 4.1vw, 54px);
    font-weight: 700;
    line-height: 1.06;
    letter-spacing: .2px;
}
#hero h1 span { color: #987022; text-shadow: 0 2px 0 rgba(255,255,255,.35); }
#hero p { margin: 11px 0 0; color: #716456; font-size: 15px; line-height: 1.6; }
.hero-brandmark {
    position: relative;
    z-index: 1;
    display: block;
    width: 178px;
    height: 142px;
    flex: 0 0 178px;
    margin: 0 0 0 28px;
    overflow: visible;
}
.hero-brandmark svg { width: 100%; height: 100%; overflow: visible; }

.kpi-row { gap: 16px !important; margin: 0 0 22px !important; }
.kpi-card {
    min-height: 126px !important;
    padding: 19px 22px 13px !important;
    border: 1px solid #eee5da !important;
    border-radius: 24px !important;
    background: linear-gradient(145deg, #fffefa 0%, #fffdf9 100%) !important;
    box-shadow: 0 10px 25px rgba(83, 59, 35, .09), inset 0 1px 0 #fff !important;
}
.kpi-card > *, .kpi-card .wrap, .kpi-card .form,
.kpi-card [data-testid="block"], .kpi-card .container {
    background: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
.kpi-card label, .kpi-card span {
    color: #76695b !important;
    font-size: 10px !important;
    font-weight: 800 !important;
    letter-spacing: 1.45px !important;
    text-transform: uppercase;
}
.kpi-card input {
    padding-left: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #241b14 !important;
    font-family: Georgia, "Times New Roman", serif !important;
    font-size: 34px !important;
    font-weight: 700 !important;
}

.workspace-grid, .insights-grid { gap: 20px !important; align-items: stretch !important; }
.surface-card, .panel {
    background: linear-gradient(145deg, #fffefa 0%, #fffdf9 100%) !important;
    color: #211a14 !important;
    border: 1px solid rgba(255,255,255,.96) !important;
    border-radius: 25px !important;
    padding: 21px !important;
    box-shadow: 0 13px 30px rgba(83, 59, 35, .10), inset 0 1px 0 #fff !important;
    backdrop-filter: blur(12px);
}
#lead-panel, #booking-panel, #summary-panel, #crm-panel {
    min-width: 0;
    color: #211a14 !important;
    background: #fffdf9 !important;
}
#lead-panel .wrap, #booking-panel .wrap,
#lead-panel > .wrap, #booking-panel > .wrap,
#lead-panel [data-testid="block"], #booking-panel [data-testid="block"] {
    background: #fffdf9 !important;
    color: #211a14 !important;
    border-color: #eee5da !important;
}
.right-rail { gap: 16px !important; }
.section-head { display: flex; align-items: flex-start; gap: 12px; margin: 0 0 16px; }
.section-symbol {
    display: grid;
    place-items: center;
    width: 37px;
    height: 37px;
    flex: 0 0 37px;
    border-radius: 12px;
    color: #916313 !important;
    background: linear-gradient(145deg, #f7eddd, #f2e7d6) !important;
    border: 1px solid #ebddc9 !important;
    font-size: 17px;
}
.section-head h2 { margin: 0; color: #241c15 !important; font-size: 18px; font-weight: 780; letter-spacing: -.2px; }
.section-head p { margin: 4px 0 0; color: #776a5d !important; font-size: 12px; line-height: 1.5; }

#conversation-chat, #conversation-chat > *, #conversation-chat .wrap,
#conversation-chat .container, #conversation-chat .message-wrap,
#conversation-chat [data-testid="chatbot"] {
    background: linear-gradient(150deg, #f5ede4 0%, #f8f2eb 100%) !important;
    color: #30251b !important;
    border-color: #e8d9c9 !important;
}
#conversation-chat {
    overflow: hidden;
    border: 1px solid #e8d9c9 !important;
    border-radius: 18px !important;
    box-shadow: inset 0 1px 5px rgba(83, 59, 35, .035);
}
#conversation-chat .message { border-radius: 15px !important; }
#message-input, #message-input > *, #message-input .wrap,
#message-input textarea, textarea, input, select {
    border: 1px solid #e3d3c2 !important;
    border-radius: 13px !important;
    background: linear-gradient(145deg, #f7f0e8, #f9f4ee) !important;
    color: #30251b !important;
    box-shadow: inset 0 1px 2px rgba(83, 59, 35, .025) !important;
}
textarea::placeholder, input::placeholder { color: #897b6d !important; }
label, .label-wrap span { color: #6e6052 !important; font-weight: 700 !important; }
.gradio-container .prose h3 { color: #2a2119 !important; }
.gradio-container .prose, .gradio-container p { color: #74675a; }

.gradio-container button {
    min-height: 44px;
    border-radius: 13px !important;
    font-weight: 720 !important;
    transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
}
.gradio-container button:hover { filter: brightness(1.035); transform: translateY(-1px); }
.gradio-container button.primary, .gradio-container .primary {
    border: 1px solid rgba(255,255,255,.3) !important;
    background: linear-gradient(135deg, #332316 0%, #1e1711 100%) !important;
    color: #fffdf9 !important;
    box-shadow: 0 7px 17px rgba(58, 38, 20, .17) !important;
}
.gradio-container button.secondary {
    border: 1px solid #e1d1bf !important;
    background: linear-gradient(135deg, #fffdf9, #f5ede3) !important;
    color: #654719 !important;
    box-shadow: 0 4px 12px rgba(83, 59, 35, .07) !important;
}
#voice-send-button {
    min-height: 52px !important;
    color: #fffdf8 !important;
    background: linear-gradient(110deg, #b98932 0%, #98671c 100%) !important;
    border: 1px solid rgba(255,255,255,.55) !important;
    box-shadow: 0 8px 20px rgba(139, 94, 26, .19) !important;
    letter-spacing: .1px;
}
.slot-picker {
    padding: 12px 14px !important;
    border: 1px solid #e8d9c9 !important;
    border-radius: 15px !important;
    background: #f8f2eb !important;
}
.status-note { color: #776a5d !important; font-size: 12px; line-height: 1.55; }
#voice-note {
    padding: 11px 14px;
    border-left: 3px solid #b58a42;
    border-radius: 0 11px 11px 0;
    background: #f6eee4 !important;
    color: #786a5a !important;
    font-size: 12px;
    line-height: 1.5;
}
#live-voice-panel {
    margin: 0 0 22px !important;
    background: linear-gradient(145deg, #fffefa 0%, #fffdf9 100%) !important;
}
#live-voice-panel .live-mode-note {
    margin: -4px 0 14px 49px;
    padding: 10px 13px;
    border-left: 3px solid #b58a42;
    border-radius: 0 10px 10px 0;
    background: #f6eee4;
    color: #786a5a;
    font-size: 12px;
    line-height: 1.5;
}
#aahvani-shopping-webrtc {
    border: 1px solid #e8d9c9 !important;
    border-radius: 17px !important;
    background: linear-gradient(150deg, #f5ede4 0%, #f8f2eb 100%) !important;
    padding: 12px !important;
}
#aahvani-shopping-webrtc .input-container {
    align-items: center !important;
    gap: 9px !important;
    padding: 10px 12px !important;
    border: 1px solid #e7d5c0 !important;
    border-radius: 18px !important;
    background: #fffdf9 !important;
}
#aahvani-shopping-webrtc .input-container textarea {
    min-height: 46px !important;
    border: 1px solid #e7d5c0 !important;
    border-radius: 13px !important;
    background: #fffaf3 !important;
    color: #30251b !important;
}
#aahvani-shopping-webrtc .input-container button {
    width: 46px !important;
    min-width: 46px !important;
    height: 46px !important;
    border: 1px solid #d8b77f !important;
    border-radius: 50% !important;
    background: #f6ead8 !important;
    color: #855b16 !important;
    cursor: pointer !important;
    transition: transform .15s ease, background .15s ease, box-shadow .15s ease !important;
}
#aahvani-shopping-webrtc .input-container button:hover {
    background: #edd7b6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(112, 75, 21, .16) !important;
}
#aahvani-shopping-webrtc .input-container button svg {
    width: 23px !important;
    height: 23px !important;
}
#speech-output { border-radius: 14px !important; }
#crm-table {
    color-scheme: light !important;
    border-radius: 14px;
    overflow: hidden;
    background: #fffdf9 !important;
    color: #30251b !important;
}
#crm-table .table-wrap,
#crm-table .dataframe,
#crm-table table,
#crm-table thead,
#crm-table tbody,
#crm-table tr,
#crm-table th,
#crm-table td,
#crm-table [role="grid"],
#crm-table [role="row"],
#crm-table [role="gridcell"],
#crm-table [role="columnheader"] {
    background-color: #fffdf9 !important;
    color: #30251b !important;
    border-color: #e7d8c8 !important;
}
#crm-table thead th,
#crm-table [role="columnheader"] {
    background-color: #efe1cf !important;
    color: #563d20 !important;
    font-weight: 750 !important;
}
#crm-table tbody tr:nth-child(even) td,
#crm-table tbody [role="row"]:nth-child(even) [role="gridcell"] {
    background-color: #f8f2ea !important;
}
#crm-table tbody tr:hover td,
#crm-table tbody [role="row"]:hover [role="gridcell"] {
    background-color: #f1e5d7 !important;
}
#crm-table input {
    background: #fffdf9 !important;
    color: #30251b !important;
}
.live-text-row { align-items: stretch !important; gap: 10px !important; margin-top: 10px !important; }
#live-text-input textarea {
    min-height: 44px !important;
    background: #fffdf9 !important;
}
#live-text-send {
    min-width: 145px;
}
.live-control-hint {
    margin-top: 5px;
    color: #776a5d !important;
    font-size: 12px;
    line-height: 1.5;
}
#shopping-chat, #shopping-message, #shopping-recommendations, #order-confirmation,
#demo-orders-table, #aahvani-shopping-webrtc {
    border-radius: 16px !important;
}
#shopping-chat {
    background: #fffdf9 !important;
    border: 1px solid #e7d8c8 !important;
}
#shopping-recommendations, #order-confirmation {
    color: #30251b !important;
    line-height: 1.65 !important;
}
#shopping-recommendations h3 { color: #563d20 !important; margin-bottom: 6px !important; }
#shopping-recommendations hr { border-color: #e7d8c8 !important; }
#demo-orders-table { overflow: hidden; background: #fffdf9 !important; }
footer { opacity: .48 !important; }

@media (max-width: 850px) {
    .gradio-container { padding: 14px 12px 32px !important; }
    #hero { min-height: 150px; padding: 23px 21px; }
    #hero h1 { font-size: 36px; }
    .hero-brandmark { width: 88px; height: 80px; flex-basis: 88px; margin-left: 8px; }
    .surface-card, .panel { padding: 15px !important; }
    .kpi-card { min-height: 108px !important; padding: 13px 14px 8px !important; }
}
@media (max-width: 520px) {
    .hero-brandmark { width: 62px; height: 62px; flex-basis: 62px; }
    #hero { padding: 20px 17px; }
    .hero-tag { display: none; }
    #hero p { font-size: 13px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; scroll-behavior: auto !important; } }
"""

with gr.Blocks(
    title=APP_NAME,
    theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange", neutral_hue="stone"),
    css=CUSTOM_CSS,
) as demo:
    session_state = gr.State(initial_state())
    gr.HTML("""
    <header id="hero">
        <div class="hero-copy">
            <div class="hero-kicker"><span class="hero-dot"></span> AI SHOPPING & ORDER WORKSPACE <span class="hero-tag">DEMO STORE</span></div>
            <h1>AAHVANI <span>AI</span></h1>
            <p>Your personal AI shopping assistant — discover, compare and order products through natural conversation.</p>
        </div>
        <div class="hero-brandmark" aria-label="AAHVANI AI headset and soundwave logo">
            <svg viewBox="0 0 220 160" role="img" aria-hidden="true">
                <defs>
                    <linearGradient id="aahvaniGold" x1="0" y1="0" x2="1" y2="1">
                        <stop offset="0" stop-color="#d4a447"/>
                        <stop offset=".48" stop-color="#a8781d"/>
                        <stop offset="1" stop-color="#765015"/>
                    </linearGradient>
                </defs>
                <path d="M43 83 C43 28 79 9 113 9 C149 9 177 34 177 83"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="12" stroke-linecap="round"/>
                <path d="M43 77 C36 77 31 83 31 91 L31 113 C31 122 37 128 46 128 L51 128 L51 77 Z"
                      fill="url(#aahvaniGold)"/>
                <path d="M177 77 L169 77 L169 128 L176 128 C185 128 191 122 191 113 L191 91 C191 83 185 77 177 77 Z"
                      fill="url(#aahvaniGold)"/>
                <g fill="url(#aahvaniGold)">
                    <rect x="73" y="81" width="8" height="20" rx="4"/>
                    <rect x="88" y="68" width="8" height="45" rx="4"/>
                    <rect x="103" y="52" width="8" height="70" rx="4"/>
                    <rect x="118" y="72" width="8" height="35" rx="4"/>
                    <rect x="133" y="82" width="8" height="18" rx="4"/>
                    <rect x="148" y="76" width="7" height="29" rx="3.5"/>
                </g>
                <path d="M181 117 C181 136 166 143 150 142 C143 142 138 139 133 135"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="6" stroke-linecap="round"/>
                <circle cx="131" cy="134" r="7" fill="url(#aahvaniGold)"/>
                <path d="M28 145 C76 157 139 151 190 116 C205 106 210 94 211 83"
                      fill="none" stroke="url(#aahvaniGold)" stroke-width="4" stroke-linecap="round"/>
            </svg>
        </div>
    </header>
    """)

    with gr.Row(elem_classes=["workspace-grid"]):
        with gr.Column(scale=8, elem_classes=["surface-card", "conversation-panel"]):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">✦</div>
                <div>
                    <h2>Chat with AAHVANI</h2>
                    <p>Tell me what you want to buy. Type or speak in the same conversation.</p>
                </div>
            </div>
            """)
            chatbot = gr.Chatbot(
                value=[{"role": "assistant", "content": "Hi, I'm AAH-vuh-nee AI. What are you looking to buy today?"}],
                height=520,
                label="Shopping conversation",
                type="messages",
                elem_id="shopping-chat",
            )
            if LIVE_VOICE_AVAILABLE and os.getenv("GEMINI_API_KEY", "").strip():
                live_audio = WebRTC(
                    label="Type a message or use the microphone",
                    mode="send-receive",
                    modality="audio",
                    variant="textbox",
                    icon_button_color="#b98527",
                    pulse_color="#d5a24d",
                    elem_id="aahvani-shopping-webrtc",
                )
                live_status = gr.Markdown(
                    "Ready. Start the live chat with the microphone, then speak or type in the same box.",
                    elem_classes=["status-note"],
                )
                gr.Markdown(
                    "The text box and microphone share one Gemini Live session and the same conversation memory. "
                    "Use headphones for clearer voice interaction.",
                    elem_classes=["live-control-hint"],
                )
                live_handler = AahvaniLiveHandler()
                live_audio.stream(
                    live_handler,
                    inputs=[live_audio],
                    outputs=[live_audio],
                    time_limit=900,
                    concurrency_limit=3,
                    send_input_on="submit",
                )
            else:
                message = gr.Textbox(
                    placeholder="Try: I need a gaming laptop under ₹60,000...",
                    label="Message AAHVANI",
                    lines=2,
                    elem_id="shopping-message",
                )
                send = gr.Button("Send message", variant="primary", elem_id="shopping-send")
                live_status = gr.Markdown(
                    "Text shopping is ready. Add GEMINI_API_KEY and install requirements-live.txt for live voice + typing in one box.",
                    elem_classes=["status-note"],
                )

            with gr.Row():
                refresh_orders_btn = gr.Button("Refresh my orders", variant="secondary")
                refresh_status = gr.Markdown("Choose a product in the conversation to begin.")

        with gr.Column(scale=5, elem_classes=["right-rail"]):
            with gr.Group(elem_classes=["surface-card"], elem_id="recommendations-panel"):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">✧</div>
                    <div><h2>Recommended for you</h2><p>Up to three matching products from the demo catalogue.</p></div>
                </div>
                """)
                recommendations_panel = gr.Markdown(
                    "Your recommendations will appear here once AAHVANI understands what you need.",
                    elem_id="shopping-recommendations",
                )

            with gr.Group(elem_classes=["surface-card"], elem_id="order-success-panel"):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">✓</div>
                    <div><h2>Order confirmation</h2><p>Simulated checkout — no real payment or shipment.</p></div>
                </div>
                """)
                order_confirmation = gr.Markdown(
                    "No demo order has been placed in this conversation yet.",
                    elem_id="order-confirmation",
                )

    with gr.Row(elem_classes=["insights-grid"]):
        with gr.Column(elem_classes=["surface-card"], elem_id="orders-panel"):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">▦</div>
                <div><h2>My orders</h2><p>Track the demo orders created during your shopping conversations.</p></div>
            </div>
            """)
            orders_table = gr.Dataframe(
                headers=[
                    "Order reference", "Product", "Brand", "Category", "Qty",
                    "Unit price", "Total", "Status", "Customer", "Created", "Order ID",
                ],
                datatype=["str", "str", "str", "str", "number", "number", "number", "str", "str", "str", "str"],
                value=crm.recent_demo_orders(),
                interactive=False,
                label="Order history",
                elem_id="demo-orders-table",
            )
            gr.Markdown(
                "For demonstration only, update an order to show the order lifecycle. "
                "These status changes do not correspond to a real shipment."
            )
            with gr.Row():
                order_reference = gr.Textbox(
                    label="Order reference",
                    placeholder="e.g. AAH-12AB34CD",
                )
                order_status = gr.Dropdown(
                    choices=["Pending", "Processing", "Shipped", "Delivered", "Cancelled"],
                    value="Processing",
                    label="Demo order status",
                )
            update_order_btn = gr.Button("Update demo order status", variant="primary")
            order_update_message = gr.Markdown(elem_classes=["status-note"])

            if LIVE_VOICE_AVAILABLE and os.getenv("GEMINI_API_KEY", "").strip():
                live_audio.on_additional_outputs(
                    lambda *values: values,
                    outputs=[
                        live_status, chatbot, recommendations_panel,
                        order_confirmation, orders_table,
                    ],
                    queue=False,
                    show_progress="hidden",
                )

    if not (LIVE_VOICE_AVAILABLE and os.getenv("GEMINI_API_KEY", "").strip()):
        send.click(
            shopping_chat,
            [message, chatbot, session_state],
            [chatbot, session_state, recommendations_panel, order_confirmation, orders_table, message],
        )
        message.submit(
            shopping_chat,
            [message, chatbot, session_state],
            [chatbot, session_state, recommendations_panel, order_confirmation, orders_table, message],
        )
    refresh_orders_btn.click(refresh_demo_orders, [], [orders_table])
    update_order_btn.click(
        update_demo_order_status,
        [order_reference, order_status],
        [orders_table, order_update_message, order_confirmation],
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
