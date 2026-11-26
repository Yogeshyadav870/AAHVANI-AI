# AAHVANI AI — One-Page Project Report

**Project:** Autonomous AI Sales Voice Agent  
**Candidate:** Yogesh Yadav  
**Challenge:** FORTIVEX AI — Round 3, Advanced AI Project Challenge & Technical Discussion

## Business problem and solution

Small and medium businesses receive inquiries through websites, social media and other channels. Sales teams spend time answering repeated questions, collecting prospect details, qualifying leads and arranging introductory demos. AAHVANI AI is an individual-built MVP that demonstrates an automated first-line sales workflow: it converses with a prospect, retrieves approved product information, captures buying signals, scores the lead, records a sample demo booking and produces a call summary.

## End-to-end workflow

Customer text or optional microphone audio → Gradio interface → session-aware Python sales agent → local FAQ retrieval → customer fact extraction → BANT-inspired lead score and keyword sentiment → user-confirmed sample appointment → SQLite CRM → call summary and dashboard metrics. Optional voice input uses Faster-Whisper for local speech-to-text; optional Edge TTS creates spoken responses online.

## Technical implementation

The application is built with Python and Gradio. Conversation messages and lead context are held in Gradio session state. The local Markdown product FAQ is split into sections and ranked by a custom TF-IDF-style lexical scoring baseline; no embedding model or vector database is used. Gemini response generation is optional. The default local demo uses a deterministic fallback because no Gemini API key is configured. Lead details are extracted using regular expressions and keyword rules. The transparent BANT-inspired score allocates Need 30, Budget 25, Authority up to 25 and Timeline up to 20 points. Sentiment is a keyword-based baseline. SQLite stores leads and sample appointments in separate tables. Automated `unittest` checks cover scoring, extraction, memory, retrieval and CRM workflow.

## Commercial feasibility

The initial target customers are agencies and other small or medium businesses with recurring inbound inquiries. A potential model is a monthly SaaS subscription with usage-based charges for voice minutes or conversation volume. Commercial validation should measure qualified-lead rate, demo conversion, response latency, cost per qualified lead, customer satisfaction and human handoff rate. Variable costs include speech recognition, speech synthesis, optional LLM tokens, telephony, compute, storage and bandwidth. Exact pricing should be set only after pilot usage and unit-cost measurement.

## Scale, infrastructure, deployment and security

The current MVP runs locally with Gradio, in-process session state and SQLite; it has not been load-tested for 100,000 users. The proposed production design uses stateless containerised API services behind an API gateway, separate scalable voice workers, PostgreSQL for transactions, Redis for sessions, a managed vector database for semantic retrieval, queues for background jobs, monitoring and automated backups. Capacity planning must use peak concurrent calls and audio/request throughput, not registered users alone.

Production security would include TLS, encryption at rest, tenant isolation, authentication, role-based access, secret management, explicit recording consent, audit logs and retention/deletion controls. A staged deployment would begin with a limited customer pilot, then add integrations, observability, concurrency testing and autoscaling before wider release.

## Scope and limitations

This prototype demonstrates the core business workflow. It does not include live telephony, a real calendar, an external CRM, full-duplex voice, validated end-to-end multilingual support, production authentication or a 100,000-user deployment. Gemini and voice features are optional adapters; the demo fallback remains usable without a Gemini key. The accompanying repository includes source code, an architecture diagram and a practical demo walkthrough.
