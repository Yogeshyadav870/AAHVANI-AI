from __future__ import annotations

import asyncio
import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agent import AahvaniAgent
from crm import CRM

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
#aahvani-live-webrtc {
    border: 1px solid #e8d9c9 !important;
    border-radius: 17px !important;
    background: linear-gradient(150deg, #f5ede4 0%, #f8f2eb 100%) !important;
    padding: 12px !important;
}
#aahvani-live-webrtc .input-container {
    align-items: center !important;
    gap: 9px !important;
    padding: 10px 12px !important;
    border: 1px solid #e7d5c0 !important;
    border-radius: 18px !important;
    background: #fffdf9 !important;
}
#aahvani-live-webrtc .input-container textarea {
    min-height: 46px !important;
    border: 1px solid #e7d5c0 !important;
    border-radius: 13px !important;
    background: #fffaf3 !important;
    color: #30251b !important;
}
#aahvani-live-webrtc .input-container button {
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
#aahvani-live-webrtc .input-container button:hover {
    background: #edd7b6 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(112, 75, 21, .16) !important;
}
#aahvani-live-webrtc .input-container button svg {
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
    last_response_state = gr.State("")
    gr.HTML("""
    <header id="hero">
        <div class="hero-copy">
            <div class="hero-kicker"><span class="hero-dot"></span> AI SALES WORKSPACE <span class="hero-tag">LOCAL MVP</span></div>
            <h1>AAHVANI <span>AI</span></h1>
            <p>Intelligent sales conversations, qualified leads, and effortless demo scheduling.</p>
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

    total_leads_now, qualified_leads_now, bookings_now = crm.dashboard_stats()
    with gr.Row(elem_classes=["kpi-row"]):
        with gr.Column(elem_classes=["kpi-card"]):
            lead_count = gr.Number(label="TOTAL LEADS", value=total_leads_now, interactive=False)
        with gr.Column(elem_classes=["kpi-card"]):
            qualified_count = gr.Number(label="QUALIFIED LEADS", value=qualified_leads_now, interactive=False)
        with gr.Column(elem_classes=["kpi-card"]):
            appointment_count = gr.Number(label="DEMO BOOKINGS", value=bookings_now, interactive=False)

    with gr.Group(elem_classes=["surface-card"], elem_id="live-voice-panel"):
        gr.HTML("""
        <div class="section-head">
            <div class="section-symbol">🎙</div>
            <div>
                <h2>Speak Up — Live Voice Mode</h2>
                <p>Talk naturally with AAHVANI. It listens continuously, responds with voice, and can be interrupted.</p>
            </div>
        </div>
        <div class="live-mode-note">
            Click the <strong>microphone icon</strong> to start a live conversation. AAHVANI will greet you automatically.
            Speak naturally, or type in the input and press Enter. While connected, the microphone button mutes/unmutes you,
            the speaker button mutes/unmutes AAHVANI, and the square button ends the conversation.
            Customer transcripts are automatically saved to the local CRM. Use headphones for the clearest conversation.
        </div>
        """)
        if LIVE_VOICE_AVAILABLE:
            live_audio = WebRTC(
                label="Speak Up — type or use the microphone",
                mode="send-receive",
                modality="audio",
                variant="textbox",
                icon_button_color="#b98527",
                pulse_color="#d5a24d",
                elem_id="aahvani-live-webrtc",
            )
            live_status = gr.Markdown(
                "Ready. Click the microphone icon to connect; AAHVANI will greet you automatically.",
                elem_classes=["status-note"],
            )
            gr.Markdown(
                "Controls: microphone = start / mute yourself; speaker = mute or unmute AAHVANI; "
                "square = end conversation. To send text, type in the built-in input and press Enter. "
                "Demo appointments still require selecting a slot and clicking **Confirm demo booking**.",
                elem_classes=["live-control-hint"],
            )
            live_handler = AahvaniLiveHandler()
            live_audio.stream(
                live_handler,
                inputs=[live_audio],
                outputs=[live_audio],
                time_limit=900,
                concurrency_limit=1,
            )
        else:
            gr.Markdown(
                "Live Voice Mode is not installed yet. In the project terminal, run "
                "`pip install -r requirements-live.txt`, then restart AAHVANI AI. "
                "You will also need a Gemini API key in your local `.env` file. "
                f"\n\nOptional dependency detail: {LIVE_VOICE_IMPORT_ERROR}",
                elem_classes=["status-note"],
            )

    with gr.Row(elem_classes=["workspace-grid"]):
        with gr.Column(scale=7, elem_classes=["surface-card", "conversation-panel"]):
            gr.HTML("""
            <div class="section-head">
                <div class="section-symbol">✦</div>
                <div><h2>Live customer conversation</h2><p>Talk naturally. AAHVANI captures intent and guides the next step.</p></div>
            </div>
            """)
            chatbot = gr.Chatbot(height=420, label="Conversation", type="messages", elem_id="conversation-chat")
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
    if LIVE_VOICE_AVAILABLE:
        live_audio.on_additional_outputs(
            lambda *values: values,
            outputs=[
                live_status, chatbot, session_state, last_response_state,
                lead_name, lead_company, lead_email, lead_phone,
                lead_score, lead_status, sentiment, crm_table,
                lead_count, qualified_count, appointment_count,
            ],
            queue=False,
            show_progress="hidden",
        )
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
    )
