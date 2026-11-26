# AAHVANI AI — One-Page Project Report

**Project:** Multimodal AI Shopping & Product Recommendation Assistant  
**Candidate:** Yogesh Yadav  
**Challenge:** FORTIVEX AI — Round 3, Advanced AI Project Challenge & Technical Discussion

## Business problem and solution

Online shoppers often know the type of product they need but are unsure which model, brand or configuration is right for their budget. AAHVANI AI demonstrates a conversational e-commerce sales agent that understands buying intent, asks relevant questions, compares catalogue options and guides the customer through a simulated checkout.

A customer can say or type: "I need a gaming laptop under ₹60,000." AAHVANI asks about primary use and brand preference, retrieves matching items from a local catalogue, recommends up to three options and lets the customer select one through the same conversation.

## End-to-end workflow

Customer text or speech → unified Gradio chat → Gemini Live API (voice mode) or deterministic Python text flow → category and preference gathering → Python catalogue filtering/ranking → up to three product recommendations → customer selects an option → AAHVANI reads back the product and demo price → explicit customer confirmation → Python order tool → SQLite simulated order → success receipt and AAH order reference → order history and status tracking.

The same reusable flow supports smartphones, laptops, headphones, tablets, smartwatches, monitors, cameras, home appliances, speakers, gaming consoles, televisions, printers, footwear, backpacks and furniture.

## Technical implementation

The application uses Python and Gradio. FastRTC/WebRTC streams browser audio and provides an integrated textbox in Live mode, so typed and spoken messages reach the same Gemini Live session. Gemini Live uses server-side function calling for product search, product selection, simulated order creation and order-status lookup.

A local JSON catalogue stores each product's category, brand, illustrative price, specifications, tags and intended use cases. A Python recommendation engine applies category, maximum-budget and brand filters, then ranks candidates using use-case matches. The system returns at most three catalogue entries; it does not invent products or retailer stock.

Text mode uses a deterministic conversational fallback with the same catalogue and order functions. Both modes require a separate explicit confirmation before creating an order. SQLite persists the order reference, product, quantity, total, creation time and status. Each simulated order starts as Pending and may be advanced in the dashboard through Processing, Shipped, Delivered or Cancelled.

## Commercial feasibility

The target market includes online retailers, electronics sellers and direct-to-consumer brands that want conversational product discovery and assisted sales. Potential commercial models include a monthly SaaS subscription, usage-based voice minutes or a fee linked to assisted conversions. A pilot should measure recommendation selection rate, assisted conversion, average order value, conversation abandonment, response latency, customer satisfaction and cost per completed conversation.

## Scale, infrastructure, deployment and security

The current MVP runs locally using Gradio, in-process conversation state and SQLite. It has not been deployed or load-tested for 100,000 users. A production design would introduce an API gateway, authentication, stateless containerised services, managed PostgreSQL, Redis for sessions, scalable voice workers, queues, observability, backups and autoscaling. Capacity planning must be based on peak concurrent voice sessions and audio throughput.

Production security would require TLS, encryption at rest, role-based access, tenant isolation, secrets management, consent controls, audit logging, retention/deletion policies and secure retailer/payment integrations. AAHVANI never asks customers for payment credentials.

## Scope and limitations

The product catalogue and prices are illustrative demo data. The system does not connect to Amazon, Flipkart or another retailer, check live inventory, accept payment, create a real shipment or guarantee delivery. Order status changes are simulated locally. The core workflow demonstrates conversational selling, structured recommendation, explicit order confirmation, SQLite persistence and order tracking; real retailer, payment and fulfilment APIs are future integrations.
