# AAHVANI AI — Shopping Workflow

~~~mermaid
flowchart TD
    A["Customer"] --> B{"Voice or Text"}
    B --> C["Unified Gradio Chat"]
    C --> D["Gemini Live API or Text Fallback"]
    D --> E["Understand product category and buying intent"]
    E --> F["Ask missing preferences: budget, use case, brand"]
    F --> G["Python Shopping Catalogue"]
    G --> H["Filter by category, budget and brand"]
    H --> I["Rank by use-case fit"]
    I --> J["Show up to 3 matching products"]
    J --> K["Customer selects an option"]
    K --> L["Read back product and demo price"]
    L --> M["Ask explicit order confirmation"]
    M --> N{"Customer confirms?"}
    N -->|No| O["Keep order unplaced"]
    N -->|Yes| P["Gemini Function Call / Python Order Tool"]
    P --> Q["Validate selected product and confirmation"]
    Q --> R[("SQLite Demo Orders")]
    R --> S["Order success receipt + AAH reference"]
    R --> T["Order History and Tracking"]
    T --> U["Pending → Processing → Shipped → Delivered"]
~~~

**Current implementation:** The same shopping journey works through text fallback and Gemini Live voice/text. The local product catalogue contains illustrative demo listings across fifteen categories, including electronics, home goods, footwear, bags and furniture. Python performs filtering/ranking and stores simulated orders. Live Gemini function tools search products, select a product, create a demo order after explicit confirmation and look up an order status.

**Boundary:** The system does not connect to live retailers, check live inventory, collect payments or create real shipments. Order statuses are simulated in the local dashboard.
