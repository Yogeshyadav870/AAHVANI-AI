# AAHVANI AI

**Autonomous AI Sales Voice Agent** — a functional prototype for customer conversations, lead qualification, product-knowledge retrieval, demo booking, CRM capture and call summaries.

## What the MVP demonstrates

- Interactive Gradio customer conversation interface.
- Session-level conversational memory.
- Local FAQ retrieval using a lightweight TF-IDF-style lexical scoring baseline.
- Optional Gemini response generation (`GEMINI_API_KEY`); deterministic local fallback when no key is configured or the API is unavailable.
- BANT-inspired lead qualification with a transparent 0–100 heuristic score.
- Basic keyword-based sentiment signal.
- SQLite CRM persistence for leads and sample demo appointments.
- Call summary generation.
- Optional microphone transcription using Faster-Whisper.
- Optional spoken AI responses using Edge TTS (internet required).

## Demo product data

The agent demonstrates a fictional product called **Northstar CRM**. The FAQ does not contain approved commercial pricing, and the agent is instructed not to invent prices. Sample slots are local demo entries, not actual calendar availability. The project does not connect to an external CRM or telephony provider.

## Run locally

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install and start:

```bash
pip install -r requirements.txt
# Copy .env.example to .env; Gemini API key is optional.
python app.py
```

Open the local Gradio URL printed in the terminal. To enable optional speech input/output:

```bash
pip install -r requirements-voice.txt
```

The first Faster-Whisper run may download the selected speech model. Edge TTS requires internet. The application runs with `share=False` and binds to `127.0.0.1` by default.

## Suggested live demo

1. Type: `Hi, I run a small real-estate agency and need help tracking incoming leads.`
2. Answer questions about budget, decision authority and implementation timeline.
3. Enter synthetic contact details in the lead panel and save the lead.
4. Select one of the sample demo slots and book it.
5. Generate the call summary and refresh the CRM table.
6. Ask: `What features does Northstar CRM provide?` to demonstrate knowledge retrieval.

## Repository layout

```text
AAHVANI-AI/
├── app.py
├── agent.py
├── crm.py
├── knowledge_base.py
├── data/product_faq.md
├── architecture/
│   ├── system_architecture.md
│   └── round4_talking_points.md
├── tests/test_core.py
├── project_report.md
├── DEMO_WALKTHROUGH.md
├── requirements.txt
├── requirements-voice.txt
└── .env.example
```

## Scope and limitations

This is an evaluation MVP, not a production telephony agent. Speech features are optional adapters, not a live full-duplex phone system. The lead score and sentiment are heuristic baselines, not validated predictive models. The local SQLite database and session state are not designed for production-scale concurrent workloads. The architecture document describes a path to support 100,000 registered users; that capacity has not been implemented or load-tested.

## Security

- Never commit `.env` or API keys.
- Use synthetic data for demonstrations.
- Do not upload real call recordings or personal customer information.
- A production version requires consent, retention and deletion controls, access control, encryption and a security review.

## Architecture

See [System Architecture](architecture/system_architecture.md) and the [Round 4 technical talking points](architecture/round4_talking_points.md).
