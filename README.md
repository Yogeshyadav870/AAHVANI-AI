<div align="center">
  <img src="assets/aahvani_logo.svg" alt="AAHVANI AI headset and waveform logo" width="130"/>
  <h1>AAHVANI AI</h1>
  <p><strong>Multimodal AI Shopping & Product Recommendation Assistant</strong></p>
  <p>Intelligence that speaks. Shopping that feels personal.</p>
</div>

AAHVANI AI is a voice-and-text shopping assistant built around one shared conversation. Customers describe what they want to buy, answer a few useful questions, compare up to three catalogue matches, select a product and place a simulated order. The dashboard keeps the existing warm beige, cream and gold visual identity and combines the chat, recommendations, order receipt and order history in one workspace.

> **Round 3 project:** Advanced AI Project Challenge & Technical Discussion — FORTIVEX AI PRIVATE LIMITED. This is an e-commerce sales-agent use case: the assistant understands buying intent, qualifies preferences, recommends relevant products and records the order outcome.

## Quick links for evaluators

| Material | File |
|---|---|
| One-page project report | [project_report.md](project_report.md) |
| End-to-end shopping workflow | [architecture/workflow.md](architecture/workflow.md) |
| Visual architecture diagram | [architecture/system_architecture.svg](architecture/system_architecture.svg) |
| Architecture, scale, security and cost plan | [architecture/system_architecture.md](architecture/system_architecture.md) |
| Practical demo script | [DEMO_WALKTHROUGH.md](DEMO_WALKTHROUGH.md) |
| Automated tests | [tests/test_core.py](tests/test_core.py) |

## What the current MVP implements

| Capability | Current implementation |
|---|---|
| Unified chat | One Gradio conversation area; in Live mode its integrated composer accepts typed text and microphone audio in the same session |
| Product categories | Smartphones, laptops, headphones, tablets, smartwatches, monitors, cameras, home appliances, speakers and gaming consoles |
| Preference gathering | Category-aware prompts for budget, primary use and brand preference |
| Product retrieval | Python filters and ranks a curated local JSON catalogue and returns up to three matching products |
| Product selection | Customer can say or type option 1, 2 or 3; Gemini Live also has a server-side selection tool |
| Simulated checkout | Explicit confirmation is required before an order is written to SQLite |
| Order receipt | Success confirmation, unique AAH order reference, product, quantity, total and initial Pending status |
| Order tracking | Demo statuses: Pending, Processing, Shipped, Delivered and Cancelled; text and voice can look up an order reference |
| Live voice | FastRTC/WebRTC with Gemini Live native audio, input/output transcription and server-side function tools |
| Text fallback | Deterministic local shopping conversation uses the same catalogue and CRM order functions when Live Voice is not configured |
| Persistence | SQLite stores simulated orders, status changes and order references |
| Tests | unittest covers existing lead/CRM logic, shopping recommendations, selection confirmation and demo order lifecycle |

## Example customer journey

1. Customer: "I need a gaming laptop under ₹60,000."
2. AAHVANI asks about primary use or any missing preference (for example, brand).
3. The Python recommendation engine searches the local demo catalogue and displays up to three options.
4. Customer says or types "option 2".
5. AAHVANI reads the selection and demo price back and asks for explicit confirmation.
6. Customer says "yes". The order is created in SQLite and the chat shows a success receipt with an AAH-XXXXXXXX reference and Pending status.
7. The customer can ask for the order status using the reference. The dashboard also provides a demo status control to move an order through Processing, Shipped and Delivered.

Smartphone, headphone, tablet, watch, monitor, camera, appliance, speaker and gaming-console conversations use the same reusable flow with category-specific follow-up questions.

## Important demo boundaries

- The catalogue is curated sample data. Product titles, prices and specifications are labelled as illustrative demo listings; they are not live retailer inventory, current offers or verified stock.
- No payment is collected. AAHVANI does not request card details, UPI PINs or payment credentials.
- No retailer, courier or external order system is contacted. Order status changes are simulated inside the local dashboard.
- Gemini Live uses function calling for product search, selection, demo-order creation and demo-order status lookup. The Python server validates inputs and requires a separate explicit confirmation before creating an order.
- The deterministic text fallback uses the same product catalogue and order logic. It is available without a Gemini key; Live Voice requires the optional dependencies and a server-side GEMINI_API_KEY.
- The current catalogue matching is structured filtering/ranking, not an embedding-based vector database. No real retailer APIs are integrated.

## Technology stack

- **Python:** product matching, conversation fallback, order validation and workflow.
- **Gradio:** dashboard, conversation transcript, integrated voice/text composer and order panels.
- **SQLite:** persistent simulated order history and status updates.
- **JSON catalogue:** editable demo product listings across ten categories.
- **Gemini Live API:** optional real-time speech-to-speech conversation and function calling.
- **FastRTC/WebRTC:** browser microphone, speaker and integrated textbox transport.
- **NumPy:** microphone audio conversion/resampling support.
- **unittest:** deterministic checks for agent and database logic.

## Run locally on Windows

Python 3.11+ is recommended. Gradio is pinned to 5.49.1 to remain compatible with FastRTC 0.0.34.

~~~powershell
git clone https://github.com/Yogeshyadav870/AAHVANI-AI.git
cd AAHVANI-AI

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python app.py
~~~

Open the local Gradio URL printed in the terminal (normally http://127.0.0.1:7860).

### Enable the unified Live Voice + Text chat

Install the optional Live dependencies:

~~~powershell
python -m pip install -r requirements-live.txt
copy .env.example .env
~~~

Edit .env locally and set:

~~~dotenv
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_LIVE_MODEL=gemini-3.8-live
GEMINI_LIVE_VOICE=Kore
~~~

Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey). Keep it only in the local .env file; never commit it or paste it into browser-side code.

Restart with python app.py. In the dashboard, start the Live conversation using the integrated microphone control. AAHVANI greets the customer; then the same input area accepts either typed messages or speech. Both methods use the same Gemini Live session and conversation history. Use headphones to reduce acoustic echo.

### Text-only fallback

When GEMINI_API_KEY or the Live dependencies are not available, the dashboard still provides a text composer. It uses the deterministic Python shopping flow and the same demo catalogue, order database and order tracking.

## Testing

Run:

~~~powershell
python -m unittest discover -s tests -v
~~~

The repository includes tests for the existing agent/CRM logic, category detection, Indian budget formats, category/budget-constrained product retrieval, phone and laptop recommendation flows, product-selection confirmation, demo-order persistence, idempotency and status updates. GitHub Actions is configured to run the suite on pushes and pull requests. These unit tests do not validate browser microphone access, external network quality or production-scale load.

## Proposed production architecture

The current application is a local prototype using Gradio, in-process conversation state and SQLite. It has not been deployed or load-tested for 100,000 users. A production design would add an API gateway, authenticated stateless application services, managed PostgreSQL, Redis for sessions, scalable voice workers, queues, observability, backups and autoscaling. Capacity planning must use peak concurrent conversations and audio throughput rather than registered users alone.

## Security and commercial feasibility

The MVP uses demo catalogue records and simulated orders. It does not store payment credentials or make real purchases. A production service would require authenticated user accounts, TLS, encryption at rest, role-based access, secure secrets management, explicit consent and retention controls, audit logging, retailer integrations and payment-provider compliance.

Potential business models include an e-commerce sales assistant subscription or usage-based conversational commerce service. Pilot metrics should include recommendation click/selection rate, assisted conversion, average order value, abandonment rate, response latency, cost per completed conversation and customer satisfaction. Real retailer prices, inventory, delivery estimates and integrations must be validated before offering real transactions.
