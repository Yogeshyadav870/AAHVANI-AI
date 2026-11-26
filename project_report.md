# AAHVANI AI — One-Page Project Report

**Project:** AI Voice Assistant for Sales & Service Appointment Requests  
**Candidate:** Yogesh Yadav  
**Challenge:** FORTIVEX AI — Round 3, Advanced AI Project Challenge & Technical Discussion

## Business problem and solution

Small and medium businesses receive inquiries through websites, social media and other channels. Sales teams spend time answering repeated questions, collecting prospect details, qualifying leads and arranging introductory demos. AAHVANI AI is an individual-built conversational MVP that demonstrates sales lead qualification and voice-led local service appointment requests. It can gather service details, confirm the customer's intent, save an appointment request in CRM and produce a structured summary.

## End-to-end workflow

Customer voice → FastRTC/WebRTC → Gemini Live speech-to-speech → conversational collection of service, provider preference, city, date/time and contact → read-back and explicit confirmation → Gemini function call → Python validation and SQLite service appointment request → CRM dashboard and call summary. The existing text sales workflow and sample demo booking remain available. A separate batch voice fallback uses local Faster-Whisper and online Edge TTS.

## Technical implementation

The application is built with Python and Gradio. Conversation messages and lead context are held in Gradio session state. The local Markdown product FAQ is split into sections and ranked by a custom TF-IDF-style lexical scoring baseline; no embedding model or vector database is used. Text chat uses deterministic local response logic by default, with optional Gemini text generation. The optional Speak Up mode uses FastRTC/WebRTC for continuous browser audio and the Gemini Live API (`v1beta`) for native speech-to-speech, turn-taking and interruption. Gemini Live input transcription feeds the existing Python fact extractor, BANT-inspired score and SQLite CRM; the API key remains in the server-side local `.env` file. Lead details are extracted using regular expressions and keyword rules. The transparent BANT-inspired score allocates Need 30, Budget 25, Authority up to 25 and Timeline up to 20 points. Sentiment is a keyword-based baseline. SQLite stores sales leads, sample demo bookings and service appointment requests in separate tables. Gemini Live uses a declared server-side function tool to write service appointment requests only after explicit customer confirmation. Automated `unittest` checks cover scoring, extraction, memory, retrieval and CRM workflow.

## Commercial feasibility

The initial target customers are agencies and other small or medium businesses with recurring inbound inquiries. A potential model is a monthly SaaS subscription with usage-based charges for voice minutes or conversation volume. Commercial validation should measure qualified-lead rate, demo conversion, response latency, cost per qualified lead, customer satisfaction and human handoff rate. Variable costs include speech recognition, speech synthesis, optional LLM tokens, telephony, compute, storage and bandwidth. Exact pricing should be set only after pilot usage and unit-cost measurement.

## Scale, infrastructure, deployment and security

The current MVP runs locally with Gradio, in-process session state, SQLite and an optional single-user browser WebRTC voice session; it has not been load-tested for 100,000 users. The proposed production design uses stateless containerised API services behind an API gateway, separate scalable voice workers, PostgreSQL for transactions, Redis for sessions, a managed vector database for semantic retrieval, queues for background jobs, monitoring and automated backups. Capacity planning must use peak concurrent calls and audio/request throughput, not registered users alone.

Production security would include TLS, encryption at rest, tenant isolation, authentication, role-based access, secret management, explicit recording consent, audit logs and retention/deletion controls. A staged deployment would begin with a limited customer pilot, then add integrations, observability, concurrency testing and autoscaling before wider release.

## Scope and limitations

This prototype demonstrates the core business workflow. It does not include PSTN telephony, a live Jalandhar provider directory, provider availability search, external provider/calendar booking APIs, an external CRM, production authentication or a 100,000-user deployment. Voice booking currently creates a request with status Pending provider confirmation; a real provider slot is not confirmed until a provider confirms it. The optional Gemini Live mode supports browser-based continuous audio and appointment function calling, not a phone number or telephony integration. End-to-end multilingual quality has not been validated. Gemini and voice features are optional adapters; the demo fallback remains usable without a Gemini key. The accompanying repository includes source code, an architecture diagram and a practical demo walkthrough.
