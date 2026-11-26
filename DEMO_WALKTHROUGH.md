# AAHVANI AI — Round 3 Practical Demo Walkthrough

**Purpose:** Demonstrate one complete end-to-end e-commerce sales workflow.  
**Demo data:** Curated sample products and illustrative INR prices.  
**Safety rule:** No payment is taken and no real order is submitted to a retailer.

## Before the evaluator arrives

- Pull the latest main branch and install dependencies.
- Run the unit tests with python -m unittest discover -s tests -v.
- Start the app with python app.py and open the local Gradio URL.
- For the voice demo, install requirements-live.txt, configure the server-side .env key and verify the browser microphone and speaker before the interview.
- Keep the GitHub repository and architecture diagram ready.
- Use only the local demo catalogue. Explain that products, prices and order statuses are simulated.

## Main scenario: smartphone recommendation

### 1. Start the conversation

Choose the Text Chat tab for typing or the Live Speaking tab for microphone-based Gemini Live conversation.

**Customer:** "Hi, I want to buy a smartphone under ₹20,000."

AAHVANI should understand the product category and budget, then ask about the use case.

**Customer:** "Camera and battery are important. I want it for everyday use."

AAHVANI should ask for a preferred brand or whether it should compare all brands.

**Customer:** "Any brand is fine."

### 2. Show product recommendations

AAHVANI searches the local catalogue and displays up to three matching products with:
- Product name and brand.
- Illustrative demo price.
- Key specifications.
- A short reason the option fits the customer's requirements.

Explain that the recommendation engine is deterministic Python filtering/ranking over a curated JSON catalogue. Gemini handles the natural conversation and invokes the server-side search tool in Live mode.

### 3. Select and confirm

**Customer:** "I want option 2."

AAHVANI should identify the selected product, read back its exact demo price and ask whether to place the simulated order.

**Customer:** "Yes, place it."

The backend should create a local SQLite order only after this separate explicit confirmation. The chat should show a success check, AAH order reference, product, quantity, total and Pending status.

Explain that no payment is collected and no retailer is contacted.

### 4. Track the order

Ask: "What is the status of order AAH-XXXXXXXX?"

AAHVANI should look up the reference and report the current status. In the dashboard, use the demo status control to move it through Processing, Shipped and Delivered. Refresh the order history to show persistence.

## Alternative scenario: gaming laptop

**Customer:** "I need a gaming laptop under ₹60,000."

**Customer follow-up:** "Gaming is the main use."

**Customer follow-up:** "Any brand."

AAHVANI should retrieve laptop options priced within the stated demo budget, show up to three matches, accept "option 1/2/3", and require a separate confirmation before creating a demo order.

## Voice and text behaviour

Text Chat and Live Speaking are separate tabs. Text Chat uses the local deterministic shopping workflow. In Live Speaking, start the FastRTC microphone session; AAHVANI greets the customer and listens for speech. The transcript is shown in the Live Speaking chat history and Gemini's audio response plays automatically. Use headphones to reduce echo.

If Live Voice is not configured, the dashboard provides a text-only fallback with the same product catalogue, selection workflow and SQLite order functions.

## What to explain technically

- FastRTC/WebRTC streams browser microphone audio to the Python server and returns Gemini's native spoken response.
- Gemini Live receives a shopping-specific system instruction and function declarations.
- Python searches the local catalogue and returns only products stored in that catalogue.
- Product selection is separate from order creation.
- The order tool independently verifies the selection and explicit confirmation before it creates a Pending demo order.
- SQLite stores the order reference and status; the dashboard can display and update the simulated lifecycle.

## Honest limitations

The demo does not search live retailer inventory, validate real-time prices, reserve an item, collect payment, send a retailer order or dispatch a package. Its catalogue is curated illustrative data. Production use would require retailer APIs, stock/price synchronisation, customer authentication, payment-provider integration, fulfilment APIs, audit controls and load testing.
