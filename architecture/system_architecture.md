# AAHVANI AI — System Architecture

## 1. Current MVP architecture

~~~mermaid
flowchart TD
    Customer["Customer: text or microphone"] --> UI["Gradio shopping workspace"]
    UI --> Chat["Unified chat and conversation history"]
    Chat --> Live["Gemini Live API (voice mode)"]
    Chat --> Text["Python text fallback"]
    Live <--> RTC["FastRTC / WebRTC"]
    Live --> Tools["Gemini function calls"]
    Text --> Shopping["Shopping Agent"]
    Tools --> Shopping
    Shopping --> Parser["Category, budget, brand and use-case extraction"]
    Parser --> Catalogue["Local JSON product catalogue"]
    Catalogue --> Rank["Python filter and ranking"]
    Rank --> Recs["Up to three recommendations"]
    Recs --> Select["Customer product selection"]
    Select --> Confirm["Read-back and explicit confirmation"]
    Confirm --> OrderTool["Python simulated order function"]
    OrderTool --> DB[("SQLite Demo Orders")]
    DB --> Receipt["Order success receipt"]
    DB --> Tracking["Order history and status tracking"]
    Tracking --> UI
~~~

### Unified Live Voice + Text

The Live Voice panel uses FastRTC's integrated textbox variant. It exposes a text composer and microphone in the same component. FastRTC supplies typed text through WebRTCData while microphone frames flow through the audio stream. The handler forwards typed text and microphone audio to the same Gemini Live session and stores both in one conversation history.

Gemini Live receives a shopping-specific system instruction and server-side function declarations:
- search_products: search the local catalogue and return up to three matches.
- select_product: select one of the latest recommendations and request customer confirmation.
- place_demo_order: create a simulated order only after a separate explicit customer confirmation.
- get_demo_order_status: retrieve the current status for a demo order reference.

The order function verifies that the previous assistant turn asked for confirmation, the latest customer turn is an explicit affirmative response, and the selected product matches the product being ordered. It creates a local SQLite record with a unique AAH order reference and initial Pending status. It never charges a payment method or contacts a retailer.

### Product catalogue and recommendation

The catalogue is stored in data/shopping_catalogue.json. Each record includes a stable ID, category, product title, brand, illustrative INR price, specifications, intended-use tags and a short recommendation rationale. The Python recommendation engine filters by category, maximum budget and brand, then ranks results using use-case matches. It returns at most three products.

This is structured catalogue retrieval and deterministic ranking. The current MVP does not use embeddings or a vector database and does not query real retailer inventory or prices.

### Order tracking

The demo_orders SQLite table stores order reference, product ID/title, category, brand, unit price, quantity, total, status, customer label and timestamps. New orders begin as Pending. The dashboard can update a demo order to Processing, Shipped, Delivered or Cancelled. Both text and voice can look up an order by its AAH reference.

## 2. Implemented request flow

1. Customer types or speaks in the unified shopping chat.
2. Gemini Live (or the deterministic text fallback) identifies the product category and gathers missing preferences.
3. The Python ShoppingCatalogue parses the maximum budget, brand and primary use.
4. The catalogue filters and ranks available demo listings, returning no more than three.
5. The customer selects an option by speaking or typing its number or product name.
6. AAHVANI reads back the chosen product and illustrative price and requests explicit confirmation.
7. After a separate affirmative response, the Python order function persists a Pending demo order in SQLite.
8. The interface shows a success receipt and order reference. The customer can ask for status, and the dashboard can simulate the order lifecycle.

## 3. Important implementation boundaries

- **Catalogue:** Curated local demo data; no live retailer API, current inventory, verified pricing or promotions.
- **Transactions:** No payment, retailer order submission, courier integration or real shipment is performed.
- **Confirmation:** The order function enforces a separate explicit confirmation. The text fallback follows the same conversation state machine.
- **Conversation:** Live Voice uses Gemini Live; text fallback uses deterministic Python conversation logic. The UI uses one shared input when Live mode and the API key are configured.
- **Recommendation:** Structured Python filtering/ranking, not embedding-based RAG or a vector database.
- **Scale:** This local Gradio/SQLite prototype has not been load-tested for 100,000 users. The production architecture below is a proposal.

## 4. Production design for up to 100,000 registered users

Registered users are not the same as concurrent active sessions. Capacity planning must estimate peak simultaneous voice conversations, average duration, audio throughput, request rate, model latency and service-level targets.

| Layer | Proposed production component | Purpose |
|---|---|---|
| Edge / ingress | CDN, WAF, API gateway and rate limits | Traffic protection and routing |
| Application | Stateless containerised API services | Horizontal scale |
| Voice | WebRTC/telephony gateway and voice workers | Isolate audio workloads |
| Agent orchestration | Tool registry, retries, timeouts and human handoff | Reliable tool execution |
| Session state | Redis | Ephemeral conversation state |
| Transactional data | PostgreSQL | Durable orders, users and audit records |
| Product data | Product catalogue service and retailer connectors | Synchronised price and stock data |
| Async processing | Queue and worker pool | Notifications and order updates |
| Observability | Central logs, metrics, traces and alerts | Latency, quality, cost and reliability |
| Delivery | CI/CD, container registry and managed platform | Repeatable rollout and rollback |

## 5. Deployment and security

**Current MVP:** Run locally with a Python virtual environment, install requirements.txt, run the unit tests and start python app.py. Install requirements-live.txt and configure the local .env only for Gemini Live voice/text.

Before production:
- Use TLS in transit and encryption at rest.
- Enforce authentication, role-based access and tenant isolation.
- Store API keys in a secrets manager; never commit .env.
- Do not request or store card numbers, CVVs, UPI PINs or passwords.
- Obtain consent before recording or retaining voice conversations.
- Define data minimisation, retention, export and deletion policies.
- Validate all function inputs server-side and require explicit order confirmation.
- Keep audit records for order creation and status changes.
- Add backups, restore drills, dependency scanning and incident response procedures.

## 6. Commercial feasibility and cost model

Potential customers include e-commerce retailers, electronics stores and direct-to-consumer brands. Potential revenue models include monthly SaaS subscription, usage-based voice minutes or assisted-conversion pricing.

Track recommendation selection rate, assisted conversion, average order value, abandonment, response latency, customer satisfaction and cost per completed conversation. Variable costs include Gemini usage, voice transport, compute, database, storage, network and support. Exact pricing and capacity require pilot measurements; this repository does not claim production benchmarks.
