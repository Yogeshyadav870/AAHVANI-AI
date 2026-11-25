# AAHVANI AI — One-Page Project Report

**Project:** Autonomous AI Sales Voice Agent  
**Author:** Yogesh Yadav

## Problem and solution
Sales teams spend time answering repeated product questions, collecting prospect details, qualifying leads and coordinating introductory demos. AAHVANI AI demonstrates an assistant that conducts a contextual conversation, retrieves approved product information, captures buying signals, offers a demo slot, saves the lead in a CRM and prepares a call summary.

## Core workflow
Customer message or optional microphone audio → optional speech transcription → conversational sales agent → session context and FAQ retrieval → lead qualification and sentiment signal → local appointment booking → SQLite CRM record → call summary. The interface also supports optional spoken responses.

## Objectives and implementation
- **Conversation:** Gradio UI with session-level conversation history.
- **Knowledge retrieval:** local Markdown FAQ ranked with a lightweight term-frequency/inverse-document-frequency-style lexical scoring baseline.
- **Response generation:** optional Gemini API; deterministic local fallback keeps the demonstration usable without a key.
- **Lead qualification:** transparent BANT-inspired heuristic using Budget, Authority, Need and Timeline signals.
- **Sentiment:** simple keyword baseline for positive, neutral or negative language.
- **Booking and CRM:** Python functions and SQLite persist leads and sample demo appointments.
- **Voice:** optional Faster-Whisper speech-to-text and Edge TTS speech output.

## Technology stack
Python, Gradio, Gemini API (optional), custom lexical retrieval, SQLite, Faster-Whisper (optional), Edge TTS (optional), and pytest.

## Evaluation
The repository includes automated tests for lead scoring, fact extraction, conversation memory, FAQ retrieval and CRM booking/persistence. The prototype uses fictional product information and sample appointment slots. It is not connected to a live telephony system, external CRM or real calendar.

## Scalability, security and commercial feasibility
For 100,000 registered users, a production design should separate stateless API services from streaming voice workers; use PostgreSQL for durable transactional data, Redis for session state, a managed vector database for semantic retrieval, queues for asynchronous work, and horizontally scaled containers behind an API gateway. Capacity planning must use peak concurrent calls and audio throughput rather than registered users alone. The production service should implement tenant isolation, TLS, encryption at rest, role-based access, secret management, explicit recording consent, audit logs, and retention/deletion controls.

A SaaS model could combine subscription tiers with usage-based voice minutes. Key business metrics include qualified-lead rate, appointment conversion, response latency, cost per qualified lead and human handoff rate.

## Limitations and next steps
This is an evaluation MVP; it has not been load-tested for 100,000 users. The qualification and sentiment methods are heuristic baselines. Next steps include live telephony, real calendar and CRM integrations, multilingual evaluation, authentication, monitoring, privacy controls and load testing.
