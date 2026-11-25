# System Architecture

## Current MVP

```mermaid
flowchart TD
    Customer[Customer] --> UI[Gradio UI]
    UI --> STT[Optional local STT]
    STT --> Agent[Conversation Agent]
    UI --> Agent
    Agent <--> Memory[Session conversation history]
    Agent --> Retriever[Local FAQ retriever]
    Retriever --> KB[Product FAQ Markdown]
    Agent --> Qualifier[BANT-inspired qualifier]
    Agent --> Booking[Appointment booking action]
    Booking --> DB[(SQLite CRM)]
    Agent --> TTS[Optional TTS]
    TTS --> Customer
    DB --> Summary[Call summary and insights]
```

## Request flow

1. The UI receives a typed message or optional audio input.
2. Speech input, when enabled, is transcribed before entering the same agent flow.
3. The agent appends the message to session history.
4. Retrieval searches the approved FAQ and returns relevant passages.
5. The response generator uses the LLM when configured; otherwise a deterministic fallback keeps the demo functional.
6. The agent updates extracted customer signals and a transparent BANT-inspired lead score.
7. When the customer confirms a slot, the booking action writes the lead and appointment to SQLite.
8. A call summary is generated from the interaction and stored CRM state.
9. Optional TTS converts the response to audio.

## Production architecture for 100,000 registered users

Registered users and concurrent voice sessions are different capacity dimensions. A production design should estimate peak simultaneous calls, average call length, audio bitrate, model latency, and peak requests per second before sizing infrastructure.

- **Ingress:** CDN/WAF, API gateway, authentication, request throttling.
- **Application:** stateless FastAPI services deployed in containers and scaled horizontally.
- **Voice:** streaming telephony/WebRTC gateway, streaming STT, interruption-aware orchestration, streaming TTS.
- **Agent orchestration:** tool registry with strict schemas, timeouts, retries, idempotency keys, and human-handoff policies.
- **Session state:** Redis for ephemeral conversation state; durable conversation metadata in PostgreSQL.
- **RAG:** document ingestion pipeline, embeddings, managed vector database, metadata filters, versioning, and citations.
- **Transactional data:** PostgreSQL with indexes, backups, connection pooling, and read replicas when justified.
- **Async work:** queue plus workers for summaries, CRM synchronization, analytics, and notifications.
- **Observability:** centralized logs, metrics, traces, token/audio usage, latency, quality, and failure-rate monitoring.
- **Security:** TLS, encryption at rest, tenant isolation, secret manager, RBAC, retention and deletion controls, consent management, and audit logs.

## Reliability and safety controls

- Never book an appointment without explicit customer confirmation.
- Validate tool arguments on the server; do not trust model-generated values.
- Use idempotency keys to prevent duplicate bookings.
- Provide clear fallback and human handoff paths.
- Restrict RAG to approved company content and treat retrieved documents as untrusted data.
- Do not store raw recordings by default; define consent and retention before enabling recording.
- Keep customer data out of application logs and public repositories.

## Cost model

Operational costs depend on concurrent call minutes, speech-to-text and text-to-speech usage, LLM tokens, retrieval volume, storage, bandwidth, and support. Measure cost per completed conversation and cost per qualified lead. Apply quotas, caching, model routing, and usage-based plans.
