from __future__ import annotations

import asyncio
import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agent import AahvaniAgent
from crm import CRM

APP_NAME = "AAHVANI AI"
_WHISPER_MODEL = None
agent = AahvaniAgent()
crm = CRM()
SAMPLE_SLOTS = ["Demo slot 1 — 11:00 AM", "Demo slot 2 — 2:00 PM", "Demo slot 3 — 4:30 PM"]


def initial_state():
    return agent.new_session()


def display_lead(session):
    lead = (session or {}).get("lead", {})
    return (
        lead.get("name", ""), lead.get("company", ""), lead.get("email", ""),
        lead.get("phone", ""), lead.get("lead_score", 0),
        lead.get("qualification", "Nurture"), lead.get("sentiment", "neutral").title()
    )


def chat(message, history, session):
    session = session or initial_state()
    answer, session = agent.respond(message, session)
    history = list(history or [])
    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    name, company, email, phone, score, status, sentiment = display_lead(session)
    session["lead_id"] = crm.save_lead(session.get("lead", {}), session.get("lead_id"))
    return history, session, "", name, company, email, phone, score, status, sentiment, crm.recent_leads(), answer, *crm.dashboard_stats()


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
        return (*([gr.skip()] * 15), None, "Record a voice message first, then send it here.")

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


CUSTOM_CSS = """
:root {
    --aahvani-ink: #eaf4ff;
    --aahvani-muted: #9fb3c9;
    --aahvani-line: rgba(154, 190, 211, .16);
    --aahvani-panel: rgba(12, 27, 45, .82);
    --aahvani-teal: #42d6c4;
    --aahvani-blue: #5b9dff;
}

html, body {
    min-height: 100%;
    background: #071321 !important;
}

body {
    background:
        radial-gradient(ellipse at 8% 5%, rgba(21, 146, 145, .22), transparent 30%),
        radial-gradient(ellipse at 92% 8%, rgba(53, 105, 190, .23), transparent 32%),
        radial-gradient(ellipse at 50% 100%, rgba(17, 91, 111, .15), transparent 38%),
        linear-gradient(135deg, #071321 0%, #0a1728 48%, #0b1d30 100%) fixed !important;
    color: var(--aahvani-ink) !important;
}


/* Decorative ambient layer: visual only, never intercepts UI interaction. */
#ambient-background {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    background:
        radial-gradient(ellipse at 18% 45%, rgba(45, 202, 183, .15), transparent 36%),
        radial-gradient(ellipse at 86% 60%, rgba(76, 139, 237, .17), transparent 38%);
}

.ambient-network {
    position: absolute;
    inset: 0;
    z-index: 1;
    width: 100%;
    height: 100%;
    opacity: .86;
    filter: drop-shadow(0 0 7px rgba(67, 210, 197, .25));
}

#ambient-background:before {
    content: "";
    position: absolute;
    inset: 0;
    opacity: .40;
    background-image:
        radial-gradient(circle, rgba(126, 231, 218, .86) 1px, transparent 1.5px),
        radial-gradient(circle, rgba(113, 169, 246, .60) .8px, transparent 1.4px),
        radial-gradient(circle, rgba(176, 220, 244, .42) .7px, transparent 1.3px);
    background-size: 86px 86px, 137px 137px, 193px 193px;
    background-position: 12px 16px, 47px 63px, 91px 24px;
    mask-image: linear-gradient(to bottom, black 0%, rgba(0,0,0,.65) 50%, transparent 100%);
}
#ambient-background:after {
    content: "";
    position: absolute;
    inset: 0;
    opacity: .23;
    background-image:
        linear-gradient(90deg, transparent 0 47%, rgba(74, 204, 193, .24) 47.2% 47.6%, transparent 47.8%),
        linear-gradient(0deg, transparent 0 47%, rgba(74, 204, 193, .20) 47.2% 47.6%, transparent 47.8%),
        linear-gradient(90deg, transparent 0 72%, rgba(101, 159, 239, .18) 72.2% 72.5%, transparent 72.7%),
        linear-gradient(0deg, transparent 0 26%, rgba(101, 159, 239, .18) 26.2% 26.5%, transparent 26.7%);
    background-size: 210px 210px, 210px 210px, 280px 280px, 280px 280px;
    background-position: -95px -78px, -95px -78px, calc(100% + 90px) calc(100% + 90px), calc(100% + 90px) calc(100% + 90px);
    mask-image: radial-gradient(ellipse at top left, black 0%, transparent 56%);
}

.gradio-container {
    position: relative !important;
    z-index: 1 !important;
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 24px 28px 48px !important;
    background: transparent !important;
    color: var(--aahvani-ink) !important;
    font-family: Inter, "Segoe UI", Arial, sans-serif !important;
}

#hero {
    position: relative;
    isolation: isolate;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 190px;
    padding: 32px 38px;
    margin: 0 0 24px 0;
    border: 1px solid rgba(153, 218, 226, .23);
    border-radius: 26px;
    background:
        linear-gradient(112deg, rgba(12, 46, 67, .98) 0%, rgba(16, 59, 78, .96) 48%, rgba(24, 48, 86, .97) 100%);
    box-shadow: 0 22px 65px rgba(0, 0, 0, .27), inset 0 1px 0 rgba(255,255,255,.08);
}

#hero:before {
    content: "";
    position: absolute;
    z-index: -1;
    width: 410px;
    height: 410px;
    right: 8%;
    top: -230px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(67, 217, 196, .28), rgba(67, 217, 196, 0) 68%);
}

#hero:after {
    content: "";
    position: absolute;
    z-index: -1;
    width: 330px;
    height: 330px;
    right: -55px;
    bottom: -245px;
    border: 1px solid rgba(111, 213, 218, .24);
    border-radius: 50%;
    box-shadow: 0 0 0 28px rgba(111, 213, 218, .035), 0 0 0 58px rgba(111, 213, 218, .025);
}

.hero-copy { position: relative; z-index: 1; }
.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: #91eee1;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 13px;
}
.hero-dot {
    width: 8px;
    height: 8px;
    display: inline-block;
    border-radius: 50%;
    background: #45dec7;
    box-shadow: 0 0 14px rgba(69, 222, 199, .95);
}
.hero-tag {
    padding: 5px 9px;
    margin-left: 5px;
    border: 1px solid rgba(162, 221, 231, .25);
    border-radius: 99px;
    color: #c5d9e8;
    font-size: 9px;
    letter-spacing: 1.2px;
}
#hero h1 {
    margin: 0;
    color: #f4fbff;
    font-size: clamp(34px, 4vw, 48px);
    font-weight: 850;
    line-height: 1.08;
    letter-spacing: 1.4px;
}
#hero h1 span {
    color: #65e0d0;
    text-shadow: 0 0 26px rgba(101, 224, 208, .2);
}
#hero p {
    margin: 11px 0 0;
    color: #c1d3e5;
    font-size: 15px;
    line-height: 1.6;
}
.hero-orbit {
    position: relative;
    width: 126px;
    height: 126px;
    flex: 0 0 126px;
    margin: 0 42px 0 24px;
    border: 1px solid rgba(113, 222, 218, .4);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(69, 210, 195, .18), rgba(53, 129, 168, .05) 62%, transparent 64%);
    box-shadow: 0 0 42px rgba(59, 210, 194, .13), inset 0 0 30px rgba(77, 196, 207, .1);
}
.hero-orbit:before, .hero-orbit:after {
    content: "";
    position: absolute;
    inset: 15px;
    border: 1px solid rgba(150, 230, 229, .36);
    border-radius: 50%;
}
.hero-orbit:after { inset: 31px; border-color: rgba(150, 230, 229, .24); }
.hero-orbit-core {
    position: absolute;
    inset: 48px;
    border-radius: 50%;
    background: #8af2df;
    box-shadow: 0 0 22px rgba(138, 242, 223, .9);
}


.kpi-row { gap: 16px !important; margin: 0 0 20px !important; }
.kpi-card {
    padding: 14px 18px 10px !important;
    border: 1px solid rgba(113, 213, 213, .20) !important;
    border-radius: 18px !important;
    background: linear-gradient(135deg, rgba(16, 48, 66, .92), rgba(17, 38, 63, .88)) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,.16), inset 0 1px 0 rgba(255,255,255,.04) !important;
}
.kpi-card label { color: #8fe5dd !important; font-size: 10px !important; letter-spacing: 1.4px !important; }
.kpi-card input { color: #f2fbff !important; font-size: 25px !important; font-weight: 800 !important; }
#voice-send-button {
    background: linear-gradient(110deg, #0d8c83, #168fa8 58%, #376ebc) !important;
    border: 1px solid rgba(107, 235, 218, .45) !important;
    box-shadow: 0 8px 26px rgba(19, 158, 157, .24) !important;
    min-height: 52px !important;
    letter-spacing: .2px;
}

.workspace-grid, .insights-grid { gap: 20px !important; align-items: flex-start !important; }
.surface-card, .panel {
    background: var(--aahvani-panel) !important;
    border: 1px solid var(--aahvani-line) !important;
    border-radius: 22px !important;
    padding: 20px !important;
    box-shadow: 0 16px 42px rgba(0, 0, 0, .20), inset 0 1px 0 rgba(255,255,255,.035) !important;
    backdrop-filter: blur(16px);
}
.right-rail { gap: 16px !important; }
.section-head {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 0 0 16px;
}
.section-symbol {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    flex: 0 0 36px;
    border-radius: 12px;
    color: #77ebdc;
    background: rgba(65, 205, 190, .12);
    border: 1px solid rgba(104, 225, 210, .18);
    font-size: 17px;
}
.section-head h2 {
    margin: 0;
    color: #edf7ff;
    font-size: 17px;
    font-weight: 750;
    letter-spacing: .1px;
}
.section-head p {
    margin: 4px 0 0;
    color: var(--aahvani-muted);
    font-size: 12px;
    line-height: 1.5;
}

#conversation-chat {
    overflow: hidden;
    border: 1px solid rgba(148, 186, 210, .16) !important;
    border-radius: 18px !important;
    background: rgba(5, 16, 29, .78) !important;
    box-shadow: inset 0 1px 12px rgba(0,0,0,.16);
}
#conversation-chat .message-wrap, #conversation-chat .wrap {
    background: transparent !important;
}
#conversation-chat .message {
    border-radius: 16px !important;
}
#message-input textarea, textarea, input, select {
    border: 1px solid rgba(150, 188, 210, .17) !important;
    border-radius: 13px !important;
    background: rgba(7, 20, 35, .78) !important;
    color: #eaf4ff !important;
    box-shadow: none !important;
}
textarea::placeholder, input::placeholder { color: #7890a9 !important; }
label, .label-wrap span {
    color: #bfd0df !important;
    font-weight: 650 !important;
}
.gradio-container .prose h3 {
    color: #e9f5ff !important;
}
.gradio-container button {
    min-height: 44px;
    border-radius: 13px !important;
    font-weight: 700 !important;
    transition: transform .16s ease, filter .16s ease, box-shadow .16s ease;
}
.gradio-container button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}
.gradio-container button.primary, .gradio-container .primary {
    border: 1px solid rgba(111, 231, 216, .28) !important;
    background: linear-gradient(115deg, #137e83 0%, #168f9b 48%, #316eb8 100%) !important;
    color: #f4ffff !important;
    box-shadow: 0 8px 22px rgba(19, 143, 153, .20) !important;
}
.gradio-container button.secondary {
    border: 1px solid rgba(131, 183, 205, .24) !important;
    background: rgba(37, 64, 85, .72) !important;
    color: #dff6f7 !important;
}
#lead-panel, #booking-panel, #summary-panel, #crm-panel { min-width: 0; }
.status-note {
    color: #9fb5c9;
    font-size: 12px;
    line-height: 1.55;
}
#voice-note {
    padding: 10px 13px;
    border-left: 2px solid rgba(72, 211, 194, .65);
    border-radius: 0 10px 10px 0;
    background: rgba(44, 125, 130, .09);
    color: #a8c0d0;
    font-size: 12px;
    line-height: 1.5;
}
#speech-output { border-radius: 14px !important; }
#crm-table { border-radius: 14px; overflow: hidden; }
footer { opacity: .55 !important; }

@media (max-width: 850px) {
    .gradio-container { padding: 14px 12px 32px !important; }
    #hero { min-height: 150px; padding: 25px 22px; }
    .hero-orbit { width: 76px; height: 76px; flex-basis: 76px; margin-right: 5px; }
    .hero-orbit:before { inset: 10px; }
    .hero-orbit:after { inset: 20px; }
    .hero-orbit-core { inset: 30px; }
    .surface-card, .panel { padding: 15px !important; }
}
@media (prefers-reduced-motion: reduce) {
    * { transition: none !important; scroll-behavior: auto !important; }
}
"""

with gr.Blocks(title=APP_NAME) as demo:
    session_state = gr.State(initial_state())
    last_response_state = gr.State("")
    gr.HTML("""
    <div id="ambient-background" aria-hidden="true">
      <svg class="ambient-network" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="networkTeal" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#50e1d0" stop-opacity=".82"/>
            <stop offset="1" stop-color="#50e1d0" stop-opacity=".10"/>
          </linearGradient>
          <linearGradient id="networkBlue" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stop-color="#67aaff" stop-opacity=".12"/>
            <stop offset="1" stop-color="#67aaff" stop-opacity=".82"/>
          </linearGradient>
          <radialGradient id="networkGlow">
            <stop offset="0" stop-color="#3bd8c8" stop-opacity=".26"/>
            <stop offset="1" stop-color="#3bd8c8" stop-opacity="0"/>
          </radialGradient>
        </defs>
        <circle cx="55" cy="490" r="250" fill="url(#networkGlow)"/>
        <circle cx="1545" cy="390" r="290" fill="url(#networkGlow)"/>
        <circle cx="1545" cy="390" r="190" fill="none" stroke="#55d9ce" stroke-opacity=".30" stroke-width="1.2"/>
        <circle cx="1545" cy="390" r="225" fill="none" stroke="#55d9ce" stroke-opacity=".18" stroke-width="1"/>
        <circle cx="55" cy="490" r="155" fill="none" stroke="#62aaff" stroke-opacity=".24" stroke-width="1.2"/>
        <circle cx="55" cy="490" r="195" fill="none" stroke="#62aaff" stroke-opacity=".16" stroke-width="1"/>
        <g fill="none" stroke="url(#networkTeal)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M0 235 L70 235 L112 277 L172 277 L218 323"/>
          <path d="M0 330 L58 330 L104 376 L165 376 L205 416 L248 416"/>
          <path d="M0 595 L68 595 L118 545 L173 545 L228 490"/>
          <path d="M0 690 L58 690 L106 642 L160 642 L210 592 L260 592"/>
          <path d="M75 0 L75 78 L122 125 L122 185 L170 233"/>
          <path d="M180 1000 L180 920 L226 874 L226 805 L277 754"/>
        </g>
        <g fill="none" stroke="url(#networkBlue)" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M1600 175 L1534 175 L1485 224 L1425 224 L1378 270"/>
          <path d="M1600 278 L1542 278 L1495 325 L1438 325 L1393 370 L1352 370"/>
          <path d="M1600 610 L1538 610 L1487 559 L1430 559 L1382 510"/>
          <path d="M1600 730 L1542 730 L1490 678 L1434 678 L1380 624 L1335 624"/>
          <path d="M1510 0 L1510 74 L1465 119 L1465 180 L1415 230"/>
          <path d="M1410 1000 L1410 920 L1364 874 L1364 814 L1312 762"/>
        </g>
        <g fill="#83f0df" stroke="#071321" stroke-width="4">
          <circle cx="70" cy="235" r="5"/><circle cx="112" cy="277" r="4"/>
          <circle cx="172" cy="277" r="5"/><circle cx="104" cy="376" r="4"/>
          <circle cx="165" cy="376" r="5"/><circle cx="118" cy="545" r="5"/>
          <circle cx="173" cy="545" r="4"/><circle cx="106" cy="642" r="4"/>
          <circle cx="160" cy="642" r="5"/><circle cx="122" cy="125" r="4"/>
          <circle cx="226" cy="874" r="4"/>
        </g>
        <g fill="#86baff" stroke="#071321" stroke-width="4">
          <circle cx="1534" cy="175" r="5"/><circle cx="1485" cy="224" r="4"/>
          <circle cx="1425" cy="224" r="5"/><circle cx="1495" cy="325" r="4"/>
          <circle cx="1438" cy="325" r="5"/><circle cx="1487" cy="559" r="5"/>
          <circle cx="1430" cy="559" r="4"/><circle cx="1490" cy="678" r="4"/>
          <circle cx="1434" cy="678" r="5"/><circle cx="1465" cy="119" r="4"/>
          <circle cx="1364" cy="874" r="4"/>
        </g>
        <g fill="none" stroke="#7de9dc" stroke-opacity=".34" stroke-width="1.2">
          <circle cx="172" cy="277" r="13"/><circle cx="165" cy="376" r="13"/>
          <circle cx="118" cy="545" r="15"/><circle cx="160" cy="642" r="13"/>
        </g>
        <g fill="none" stroke="#8abaff" stroke-opacity=".34" stroke-width="1.2">
          <circle cx="1425" cy="224" r="13"/><circle cx="1438" cy="325" r="13"/>
          <circle cx="1487" cy="559" r="15"/><circle cx="1434" cy="678" r="13"/>
        </g>
      </svg>
    </div>
    """)

    gr.HTML("""
    <header id="hero">
        <div class="hero-copy">
            <div class="hero-kicker"><span class="hero-dot"></span> AI SALES WORKSPACE <span class="hero-tag">LOCAL MVP</span></div>
            <h1>AAHVANI <span>AI</span></h1>
            <p>Intelligent sales conversations, qualified leads, and effortless demo scheduling.</p>
        </div>
        <div class="hero-orbit" aria-hidden="true"><div class="hero-orbit-core"></div></div>
    </header>
    """)

    total_leads_now, qualified_leads_now, bookings_now = crm.dashboard_stats()
    with gr.Row(elem_classes=["kpi-row"]):
        with gr.Column(elem_classes=["kpi-card"]):
            lead_count = gr.Number(label="TOTAL LEADS", value=total_leads_now, interactive=False)
        with gr.Column(elem_classes=["kpi-card"]):
            qualified_count = gr.Number(label="QUALIFIED LEADS", value=qualified_leads_now, interactive=False)
        with gr.Column(elem_classes=["kpi-card"]):
            appointment_count = gr.Number(label="DEMO BOOKINGS", value=bookings_now, interactive=False)

    with gr.Row(elem_classes=["workspace-grid"]):
        with gr.Column(scale=7, elem_classes=["surface-card", "conversation-panel"]):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">✦</div>
                <div><h2>Live customer conversation</h2><p>Talk naturally. AAHVANI captures intent and guides the next step.</p></div>
            </div>
            """)
            chatbot = gr.Chatbot(height=420, label="Conversation", elem_id="conversation-chat")
            with gr.Row(elem_classes=["composer-row"]):
                message = gr.Textbox(
                    placeholder="Start with: Hi, I run a small agency and need help tracking leads...",
                    show_label=False, scale=8, lines=2, elem_id="message-input"
                )
                send = gr.Button("Send  ↗", variant="primary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice input", scale=3)
                transcribed = gr.Textbox(label="Transcription", placeholder="Speech will appear here...", scale=4, lines=2)
            with gr.Row():
                voice_send_btn = gr.Button("Send voice message & hear reply", variant="primary", elem_id="voice-send-button")
            with gr.Row():
                use_transcript = gr.Button("Use transcription", variant="secondary")
                speak_btn = gr.Button("Speak AI response", variant="secondary")
            speech_output = gr.Audio(label="AI voice response", type="filepath", interactive=False, elem_id="speech-output")
            speech_status = gr.Markdown(elem_classes=["status-note"])
            gr.HTML("<div id='voice-note'>Voice features are optional. Text chat works without microphone access or additional speech packages.</div>")

        with gr.Column(scale=5, elem_classes=["right-rail"]):
            with gr.Group(elem_classes=["surface-card"], elem_id="lead-panel"):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">◈</div>
                    <div><h2>Lead intelligence</h2><p>Customer profile and buying-readiness signals.</p></div>
                </div>
                """)
                lead_name = gr.Textbox(label="Customer name", placeholder="Name")
                lead_company = gr.Textbox(label="Company", placeholder="Business or organisation")
                lead_email = gr.Textbox(label="Email", placeholder="name@company.com")
                lead_phone = gr.Textbox(label="Phone", placeholder="+91 ...")
                with gr.Row():
                    lead_score = gr.Number(label="Lead score / 100", value=0, interactive=False)
                    lead_status = gr.Textbox(label="Qualification", value="Nurture", interactive=False)
                sentiment = gr.Textbox(label="Sentiment", value="Neutral", interactive=False)
                save_lead_btn = gr.Button("Save lead to CRM", variant="secondary")
                save_status = gr.Markdown(elem_classes=["status-note"])

            with gr.Group(elem_classes=["surface-card"], elem_id="booking-panel"):
                gr.HTML("""
                <div class="section-head">
                    <div class="section-symbol">◷</div>
                    <div><h2>Schedule a product demo</h2><p>Choose a sample slot to complete the workflow.</p></div>
                </div>
                """)
                slot = gr.Radio(choices=SAMPLE_SLOTS, value=SAMPLE_SLOTS[0], label="Choose a demo slot", elem_classes=["slot-picker"])
                book_btn = gr.Button("Confirm demo booking", variant="primary")
                booking_status = gr.Markdown(elem_classes=["status-note"])

    with gr.Row(elem_classes=["insights-grid"]):
        with gr.Column(scale=5, elem_classes=["surface-card"], elem_id="summary-panel"):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">▤</div>
                <div><h2>Call summary</h2><p>Key customer details and the conversation outcome.</p></div>
            </div>
            """)
            summary = gr.Textbox(lines=9, label="Conversation summary", interactive=False)
            summary_btn = gr.Button("Generate summary", variant="secondary")
        with gr.Column(scale=7, elem_classes=["surface-card"], elem_id="crm-panel"):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">▦</div>
                <div><h2>Local CRM records</h2><p>Leads and demo bookings saved in the local database.</p></div>
            </div>
            """)
            crm_table = gr.Dataframe(
                headers=["Name", "Company", "Email", "Score", "Status", "Sentiment", "Updated"],
                datatype=["str", "str", "str", "number", "str", "str", "str"],
                value=crm.recent_leads(), interactive=False, label="Recent leads", elem_id="crm-table"
            )
            refresh_btn = gr.Button("Refresh CRM & metrics", variant="secondary")

    outputs = [chatbot, session_state, message, lead_name, lead_company, lead_email, lead_phone, lead_score, lead_status, sentiment, crm_table, last_response_state, lead_count, qualified_count, appointment_count]
    send.click(chat, [message, chatbot, session_state], outputs)
    message.submit(chat, [message, chatbot, session_state], outputs)
    audio_input.change(transcribe_audio, [audio_input], [transcribed])
    use_transcript.click(lambda text: text, [transcribed], [message])
    voice_send_btn.click(voice_conversation, [transcribed, chatbot, session_state], outputs + [speech_output, speech_status])
    speak_btn.click(speak_text, [last_response_state], [speech_output, speech_status])
    save_lead_btn.click(update_lead_fields, [lead_name, lead_company, lead_email, lead_phone, session_state], [session_state, lead_score, lead_status, crm_table, save_status, lead_count, qualified_count, appointment_count])
    book_btn.click(book_demo, [slot, session_state], [session_state, crm_table, booking_status, lead_count, qualified_count, appointment_count])
    summary_btn.click(make_summary, [session_state], [summary])
    refresh_btn.click(refresh_crm, [], [crm_table, lead_count, qualified_count, appointment_count])


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "127.0.0.1"),
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan", neutral_hue="slate"),
        css=CUSTOM_CSS,
    )
