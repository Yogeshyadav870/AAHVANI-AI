# AAHVANI AI — System Architecture

![AAHVANI AI current MVP system architecture](system_architecture.svg)

## 1. Current MVP architecture

```mermaid
flowchart TD
    Customer[Customer: text or microphone] --> UI[Gradio interface]
    UI -->|Typed text| Agent[Python sales agent]
    UI -->|Batch audio fallback| STT[Faster-Whisper: local STT]
    STT --> Agent
    UI <-->|Speak Up: WebRTC audio| Live[FastRTC live handler]
    Live <-->|Bidirectional audio| GeminiLive[Gemini Live API]
    Live -->|Input transcription| Agent
    Live -->|Function call: save_appointment_request| BookingTool[Python booking tool]
    BookingTool --> ServiceAppointments[(SQLite service_appointments)]
    ServiceAppointments --> Dashboard
    Agent <--> Memory[Gradio session state]
    Agent --> Retriever[Custom lexical FAQ retriever]
    Retriever --> KB[Local Markdown product FAQ]
    Agent -. optional generation .-> Gemini[Gemini API]
    Agent --> Qualifier[BANT-inspired lead scoring]
    Agent --> Sentiment[Keyword sentiment baseline]
    Qualifier --> CRM[(SQLite CRM)]
    Sentiment --> CRM
    Agent --> Booking[User-confirmed sample booking]
    Booking --> CRM
    Agent --> Summary[Call summary]
    CRM --> Dashboard[Dashboard metrics and CRM table]
    Agent -. optional speech .-> TTS[Edge TTS: online]
    TTS --> Customer
```

### Live Voice Mode (optional)

The dashboard includes a Speak Up control backed by FastRTC/WebRTC. Each connection gets a dedicated asynchronous handler and Gemini Live session. The browser streams microphone audio to the Python server; the server forwards mono PCM16 audio at 16 kHz to Gemini Live. The model returns native PCM16 audio at 24 kHz, which FastRTC streams back to the browser. Gemini Live handles voice activity and can interrupt its generated speech when the customer starts speaking.

Input and output audio transcription are enabled. Customer transcripts are passed through the Python lead extractor and CRM. The LiveConnectConfig also declares a server-side `save_appointment_request` function. When the customer requests an appointment, Gemini gathers the service category, service, provider preference, city, requested date/time and contact details, reads them back, and waits for explicit confirmation. The server independently verifies that the previous assistant turn asked for confirmation and that the customer's latest utterance is affirmative before writing anything. It then saves the customer lead and a row in SQLite `service_appointments`, sends the function result back to Gemini, and lets Gemini speak the request reference and status. The dashboard displays the new request and the session summary includes its details.

A newly saved service appointment starts as **Pending provider confirmation**. This release has no live Jalandhar provider directory, provider availability search, provider notification, external calendar reservation or automatic provider confirmation. An operator can update the CRM status after contacting the provider and entering a confirmation note/reference. This prevents a preferred time from being misrepresented as a confirmed appointment.

The Live API key is read from the server-side `.env` setting `GEMINI_API_KEY`; it is never embedded in browser code. The live model and voice can be changed using `GEMINI_LIVE_MODEL` and `GEMINI_LIVE_VOICE`. Current defaults are `gemini-3.8-live` and `Kore`.

## 2. Implemented request flow

1. The Gradio UI accepts typed text, optional batch microphone audio, or continuous Speak Up audio over WebRTC.
2. In the batch voice fallback, Faster-Whisper transcribes audio locally when the optional voice dependencies are installed. In Live Voice Mode, FastRTC streams audio to Gemini Live and native audio responses stream back through WebRTC.
3. The agent appends the message to the session history and extracts available facts such as name, email, phone, company, budget, authority and timeline using regular expressions and keyword rules.
4. The local Markdown FAQ is divided into sections. A custom term-frequency/inverse-document-frequency-style lexical scoring function ranks relevant sections and returns the top matches.
5. If Gemini is configured, the retrieved context can be supplied to the response-generation prompt. The default demo has no Gemini key configured and therefore uses deterministic local response logic.
6. The agent calculates a BANT-inspired score and a simple keyword-based sentiment signal.
7. The lead is saved to SQLite. A user can select a sample demo slot; the application validates that a contact method exists and persists the sample appointment linked to that lead.
8. In Live Voice Mode, Gemini can call `save_appointment_request` after explicit customer confirmation. Python validates the required fields and confirmation state, saves a service appointment request linked to the customer lead, and returns the request status/reference to Gemini.
9. Service appointment requests appear in the CRM table. The structured summary includes appointment category, service, provider preference, city, requested date/time and status. SQL queries update the lead and demo-booking counters.
9. For a batch voice turn, Edge TTS (online) converts the generated reply into an MP3 file. In Live Voice Mode, Gemini Live generates native streaming audio directly.

## 3. Important implementation boundaries

- **Context:** Session messages and extracted lead fields persist during the active Gradio session. The default deterministic fallback is not a full LLM conversational planner.
- **Retrieval:** This is local lexical FAQ retrieval, not embedding-based semantic search or a vector database. The optional Gemini path can use retrieved text as generation context.
- **Lead scoring:** The score is a transparent heuristic, not a trained conversion prediction model.
- **Sentiment:** The current classifier is keyword-based, not a trained sentiment model.
- **Voice:** The optional Live Voice Mode supports browser-based bidirectional audio, turn-taking, interruptions and server-side appointment function calling. It is not a PSTN phone connection. The batch Faster-Whisper + Edge TTS path remains available.
- **Appointment booking:** The current voice tool stores an appointment request as pending provider confirmation. It does not search providers, check live availability, call providers, or reserve external calendars.
- **Multilingual:** Whisper supports multilingual speech recognition, but the complete multilingual experience, including language-specific response generation and TTS voice selection, has not been validated in this MVP.
- **Integrations:** The demo uses fictional Northstar CRM content and sample slots. It is not connected to an external CRM or real calendar. The live audio connection is browser WebRTC, not a phone provider.

## 4. Production design for up to 100,000 registered users

Registered users are not the same as concurrent active conversations. Capacity planning must estimate peak simultaneous voice sessions, average call length, audio throughput, request rate, model latency and service-level targets.

| Layer | Proposed production component | Purpose |
|---|---|---|
| Edge / ingress | CDN, WAF, API gateway, rate limits | Traffic protection and routing |
| Application | Stateless FastAPI services in containers | Horizontally scalable business APIs |
| Voice | Telephony/WebRTC gateway and separate STT/TTS workers | Isolate audio workloads from API requests |
| Agent orchestration | Tool registry, timeouts, retries and human handoff | Reliable workflow execution |
| Session state | Redis | Fast ephemeral conversation state |
| Transactional data | PostgreSQL | Durable leads, bookings and account data |
| Knowledge retrieval | Embedding pipeline and managed vector database | Semantic search with metadata filters |
| Async processing | Message queue and worker pool | Summaries, notifications and CRM sync |
| Observability | Central logs, metrics, traces and alerts | Latency, quality, cost and failure monitoring |
| Delivery | CI pipeline, container registry and managed container platform | Repeatable deployment and rollback |

### Deployment approach

**Current MVP:** Run locally with a Python virtual environment, install `requirements.txt`, run the unit tests, then start `python app.py`. Voice and Gemini packages are installed separately from their optional requirements files. The application binds to localhost by default.

**Production path:** Package the API and worker services into containers; deploy behind an authenticated API gateway; provision managed PostgreSQL, Redis, object storage and a queue; store credentials in a secrets manager; use CI/CD with staged rollout and rollback; configure autoscaling, health checks, backups, dashboards and alerts. Voice services should be scaled using concurrent audio sessions and processing latency, not the total user count.

## 5. Security and privacy

The MVP is a local evaluation prototype and does not implement production authentication, multi-tenant isolation or enterprise security controls. Use synthetic demo data and never commit secrets, real customer records or recordings.

Before production:
- Use TLS in transit and encryption at rest.
- Enforce authentication, role-based access and tenant-level data isolation.
- Store API keys in a secrets manager; never in source control.
- Obtain explicit consent before recording or retaining calls.
- Define data minimisation, retention, export and deletion policies.
- Keep personal data out of application logs; maintain audit trails for sensitive actions.
- Validate tool inputs server-side and require explicit user confirmation before booking.
- Add backups, restore drills, dependency scanning and incident response procedures.

## 6. Commercial feasibility and operating-cost model

**Target customers:** Agencies, real-estate teams, education consultancies and other small or medium businesses with recurring inbound inquiries.

**Potential revenue model:** Monthly SaaS subscription plus usage-based charges for voice minutes or conversation volume. A pilot should validate willingness to pay, qualification quality and time saved before setting prices.

**Variable cost drivers:** LLM tokens when Gemini or another hosted model is enabled; STT and TTS minutes; telephony minutes; compute; database and object storage; network egress; monitoring; customer support.

A simple unit-economics view is:

```text
Contribution per conversation
= revenue attributed to the conversation
  - (LLM + STT + TTS + telephony + compute + storage + bandwidth)

Monthly operating result
= subscription revenue + usage revenue
  - fixed infrastructure, support and operating costs
  - variable conversation costs
```

Track cost per completed conversation and cost per qualified lead alongside demo conversion, response latency and human handoff rate. Use model routing, quotas, caching and asynchronous processing to control costs. Exact pricing and infrastructure sizing require pilot traffic and measured usage; this repository does not claim benchmarked production costs.

## 7. Reliability and next steps

Use idempotency keys for booking APIs, bounded retries, clear failure messages and human handoff. Treat retrieved documents as untrusted content and restrict retrieval to approved product data. Next steps are real telephony and calendar/CRM integrations, validated multilingual flows, improved extraction and semantic retrieval, production security, load testing and measured unit economics.
