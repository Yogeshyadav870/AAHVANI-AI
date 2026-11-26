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
    return history, session, "", name, company, email, phone, score, status, sentiment, crm.recent_leads()


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
    if not audio_path:
        return ""
    try:
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL", "base")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text or "No speech detected. Please try again or type your message."
    except ImportError:
        return "Voice transcription is optional. Install requirements-voice.txt and restart. You can continue using text chat."
    except Exception as exc:
        return f"Could not transcribe audio: {exc}"


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
    return session, lead["lead_score"], lead["qualification"], crm.recent_leads(), "Lead saved to the local CRM."


def book_demo(slot, session):
    session = session or initial_state()
    lead = session.setdefault("lead", {})
    if not lead.get("email") and not lead.get("phone"):
        return session, crm.recent_leads(), "Add an email address or phone number before booking."
    session["lead_id"] = crm.save_lead(lead, session.get("lead_id"))
    try:
        crm.book_appointment(session["lead_id"], slot)
    except ValueError as exc:
        return session, crm.recent_leads(), str(exc)
    return session, crm.recent_leads(), f"Demo appointment saved for {slot}. This is a local prototype booking, not a live calendar reservation."


def make_summary(session):
    return agent.summarize(session or initial_state())


def refresh_crm():
    rows = crm.recent_leads()
    return rows, len(rows), crm.appointment_count()


CUSTOM_CSS = """
.gradio-container {max-width: 1280px !important; margin: auto !important;}
#hero {padding: 24px 26px; border-radius: 18px; background: linear-gradient(115deg, #101a35, #203b62); color: white; margin-bottom: 14px;}
#hero h1 {font-size: 34px; margin: 0 0 6px 0; letter-spacing: .4px;}
#hero p {margin: 0; color: #d4e0ef;}
.panel {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px;}
"""

with gr.Blocks(title=APP_NAME) as demo:
    session_state = gr.State(initial_state())
    gr.HTML("<div id='hero'><h1>AAHVANI AI</h1><p>Autonomous sales conversations · Lead qualification · Demo scheduling</p></div>")
    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown("### Live customer conversation")
            chatbot = gr.Chatbot(height=430, label="Conversation")
            with gr.Row():
                message = gr.Textbox(placeholder="Try: Hi, I run a small real-estate agency and need help tracking leads...", show_label=False, scale=8, lines=2)
                send = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Optional voice input", scale=3)
                transcribed = gr.Textbox(label="Transcription", placeholder="Record speech or type here", scale=4, lines=2)
            use_transcript = gr.Button("Use transcription")
            speak_btn = gr.Button("Speak last AI response")
            speech_output = gr.Audio(label="AI voice response", type="filepath", interactive=False)
            speech_status = gr.Markdown()
            gr.Markdown("*Speech input uses optional Faster-Whisper. Speech output uses optional Edge TTS and requires internet. Text chat works without either.*")
        with gr.Column(scale=5):
            gr.Markdown("### Lead intelligence")
            with gr.Group(elem_classes=["panel"]):
                lead_name = gr.Textbox(label="Customer name")
                lead_company = gr.Textbox(label="Company")
                lead_email = gr.Textbox(label="Email")
                lead_phone = gr.Textbox(label="Phone")
                lead_score = gr.Number(label="Lead score / 100", value=0, interactive=False)
                lead_status = gr.Textbox(label="Qualification", value="Nurture", interactive=False)
                sentiment = gr.Textbox(label="Sentiment", value="Neutral", interactive=False)
                save_lead_btn = gr.Button("Save lead to CRM", variant="secondary")
                save_status = gr.Markdown()
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### Schedule a product demo")
                slot = gr.Dropdown(choices=SAMPLE_SLOTS, value=SAMPLE_SLOTS[0], label="Available sample slots")
                book_btn = gr.Button("Book demo", variant="primary")
                booking_status = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Call summary")
            summary = gr.Textbox(lines=10, label="Conversation summary", interactive=False)
            summary_btn = gr.Button("Generate call summary")
        with gr.Column():
            gr.Markdown("### Local CRM records")
            crm_table = gr.Dataframe(
                headers=["Name", "Company", "Email", "Score", "Status", "Sentiment", "Updated"],
                datatype=["str", "str", "str", "number", "str", "str", "str"],
                value=crm.recent_leads(), interactive=False, label="Recent leads"
            )
            with gr.Row():
                lead_count = gr.Number(label="Saved leads", value=len(crm.recent_leads()), interactive=False)
                appointment_count = gr.Number(label="Demo bookings", value=crm.appointment_count(), interactive=False)
            refresh_btn = gr.Button("Refresh CRM")

    outputs = [chatbot, session_state, message, lead_name, lead_company, lead_email, lead_phone, lead_score, lead_status, sentiment, crm_table]
    send.click(chat, [message, chatbot, session_state], outputs)
    message.submit(chat, [message, chatbot, session_state], outputs)
    audio_input.change(transcribe_audio, [audio_input], [transcribed])
    use_transcript.click(lambda text: text, [transcribed], [message])
    speak_btn.click(speak_last_response, [chatbot], [speech_output, speech_status])
    save_lead_btn.click(update_lead_fields, [lead_name, lead_company, lead_email, lead_phone, session_state], [session_state, lead_score, lead_status, crm_table, save_status])
    book_btn.click(book_demo, [slot, session_state], [session_state, crm_table, booking_status])
    summary_btn.click(make_summary, [session_state], [summary])
    refresh_btn.click(refresh_crm, [], [crm_table, lead_count, appointment_count])


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("HOST", "127.0.0.1"), server_port=int(os.getenv("PORT", "7860")), share=False, theme=gr.themes.Soft(), css=CUSTOM_CSS)
