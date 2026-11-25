# Round 4 — Technical Talking Points

## Why is this an agent and not only a chatbot?
The system can invoke explicit application actions: retrieve product information, calculate lead qualification, create an appointment, persist CRM records, and produce a summary. The model interprets the conversation, while deterministic Python functions validate and execute the business actions.

## What is RAG?
Retrieval-Augmented Generation retrieves relevant passages from an external knowledge source and supplies them as context to a language model. In this MVP, TF-IDF vectors and cosine similarity provide lightweight local retrieval. A production system may use embedding models and a vector database.

## What is an embedding?
An embedding is a numeric vector representation of content. Similarity between vectors can help retrieve semantically related passages. The MVP's TF-IDF representation is lexical rather than a modern semantic embedding.

## How is memory implemented?
The current application maintains conversation history in the Gradio session. It is short-lived session context, not persistent long-term memory. A production system should store only necessary conversation state with consent and retention controls.

## How is a lead qualified?
The prototype uses BANT-inspired signals: Budget, Authority, Need, and Timeline. The score is a transparent heuristic, not a validated predictive model. A production scoring model should be calibrated against conversion outcomes.

## How do STT and TTS fit?
STT turns audio into text so the agent can use the same conversational pipeline. TTS converts the generated text response to speech. The optional local adapters are separate from the core text workflow.

## How do you control hallucinations?
Retrieve approved product information, instruct the model to say when information is unavailable, validate tool inputs, and keep booking and CRM actions deterministic. Production requires evaluation, grounding checks, audit logs, and human escalation.

## How will you support 100,000 users?
Clarify that registered user count is not concurrent call count. Use a stateless horizontally scaled API, Redis session state, PostgreSQL, a vector store, asynchronous queues, and a streaming voice gateway. Load-test realistic concurrent sessions and measure p95 latency and cost per call.

## Security and privacy
Obtain recording consent, minimize personal data, encrypt it in transit and at rest, apply tenant isolation and role-based access, keep secrets outside source control, define retention/deletion, and avoid logging raw audio or sensitive customer details.

## Commercial feasibility
Offer subscription tiers with usage-based voice minutes. Track lead qualification rate, appointment conversion, response time, cost per qualified lead, and human handoff rate. Start with a narrow customer segment and validate conversion lift.
