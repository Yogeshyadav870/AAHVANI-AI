# AAHVANI AI — Project Report

## Project overview

AAHVANI AI is an AI-powered autonomous sales voice-agent prototype designed to automate the first stages of customer engagement. It listens to or receives a customer inquiry, understands the requirement, retrieves relevant product information, qualifies the lead, offers an appointment, records the outcome in a CRM, and generates a concise call summary.

## Problem statement

Sales teams spend significant time answering repetitive product questions, collecting customer details, identifying purchase readiness, and coordinating introductory appointments. Delayed responses and inconsistent qualification can reduce conversion opportunities. AAHVANI AI demonstrates how a conversational agent can streamline this workflow while preserving the option for human intervention.

## Objectives

1. Provide a conversational interface for customer inquiries.
2. Maintain context during a customer session.
3. Answer product questions using a retrieval-augmented knowledge base.
4. Qualify prospects using budget, authority, need, and timeline signals.
5. Detect basic sentiment cues and prioritize follow-up.
6. Book a demo slot and persist the lead and appointment.
7. Generate a structured conversation summary.

## Technology stack

- **Python:** application logic and orchestration
- **Gradio:** interactive web interface
- **Gemini API (optional):** LLM-based response generation
- **TF-IDF and cosine similarity:** lightweight local retrieval
- **SQLite:** lead, appointment, and interaction persistence
- **Faster-Whisper (optional):** local speech-to-text
- **Edge TTS (optional):** spoken response generation

## End-to-end workflow

Customer inquiry → speech transcription (optional) → conversational sales agent → contextual history and FAQ retrieval → lead qualification and sentiment signal → appointment booking → SQLite CRM record → call summary and follow-up recommendation.

## Implementation and evaluation

The prototype includes a Gradio application, session-level conversation history, a local product FAQ retriever, transparent BANT-inspired scoring, basic sentiment rules, a booking action, SQLite storage, optional voice adapters, and a call-summary generator. Automated core tests cover retrieval, qualification, appointment booking, CRM persistence, and summary generation. The demonstration should use synthetic customer information.

## Architecture and scalability

The current MVP runs locally and uses SQLite for simplicity. A production design for up to 100,000 registered users should use stateless FastAPI services behind a load balancer, containerized workers with horizontal autoscaling, PostgreSQL for transactional data, Redis for short-lived session state and caching, a managed vector database for retrieval, a message queue for asynchronous tasks, and a dedicated streaming voice/telephony layer. Capacity should be planned around concurrent calls and audio throughput, not registered-user count alone.

## Security and commercial feasibility

A production system should obtain explicit recording consent, minimize collected personal data, encrypt data in transit and at rest, apply role-based access, manage secrets outside source control, define retention/deletion policies, and log tool actions. Commercially, the product could be offered as SaaS with subscription tiers and usage-based voice minutes, targeting businesses with repetitive inbound sales inquiries. Key metrics include response time, qualified-lead rate, appointment conversion, cost per qualified lead, and human handoff rate.

## Limitations and next steps

This MVP does not include production telephony, live calendar integration, enterprise CRM connectors, robust multilingual evaluation, or production-grade concurrent scaling. Next steps are to integrate a telephony provider, add real calendar/CRM connectors, improve language and sentiment evaluation, implement authentication and observability, and load-test the deployment architecture.
