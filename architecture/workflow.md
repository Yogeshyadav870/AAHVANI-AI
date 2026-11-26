# AAHVANI AI — End-to-End Workflow

```mermaid
flowchart TD
    A["Customer"] --> B{"Voice or Text"}
    B --> C["WebRTC + FastRTC"]
    B --> D["Gradio Chat"]
    C --> E["Gemini Live API"]
    D --> F["AAHVANI Agent + Knowledge Base"]
    E --> G["Conversation & Transcription"]
    F --> G
    G --> H["Lead Information Extraction"]
    H --> I["BANT Lead Scoring"]
    I --> J["Lead Qualification"]
    J --> K[("SQLite CRM")]
    K --> L["CRM Dashboard"]
    L --> M["Demo Booking & Call Summary"]
```

**Core flow:** Customer Input → AI Processing → Lead Extraction → BANT Scoring → Qualification → CRM → Demo Booking.

**Implementation note:** Voice and text are separate input paths that converge on lead extraction and CRM workflows. The current demo uses local SQLite storage and sample demo slots; it does not connect to an external CRM or live calendar.
