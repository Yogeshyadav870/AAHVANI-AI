# AAHVANI AI — Sales & Service Appointment Workflow

```mermaid
flowchart TD
    A["Customer"] --> B{"Voice or Text"}
    B --> C["WebRTC + FastRTC"]
    B --> D["Gradio Chat"]
    C --> E["Gemini Live API"]
    D --> F["AAHVANI Python Agent"]
    E --> G["Conversation & Transcription"]
    F --> G

    G --> H{"Customer intent"}
    H -->|CRM sales enquiry| I["Knowledge Base + Lead Discovery"]
    I --> J["BANT Lead Scoring"]
    J --> K["Lead Record"]
    K --> M[("SQLite CRM")]

    H -->|Service appointment| N["Collect service, provider, date, time and contact"]
    N --> O["Read back details"]
    O --> P["Customer explicitly confirms"]
    P --> Q["Gemini Function Call"]
    Q --> R["Python validates and saves request"]
    R --> S["Service Appointment Record"]
    S --> M

    M --> T["CRM Dashboard"]
    T --> U["Appointment Summary"]
    S --> V["Pending provider confirmation"]
    V --> W["Provider confirms out of band"]
    W --> X["Operator updates status + reference"]
    X --> T
```

**Appointment voice flow:** Customer speaks → Gemini Live understands the request → asks for missing details → reads back the appointment → waits for explicit confirmation → calls the Python booking function → saves the request and customer data in SQLite → updates the CRM table and call summary.

**Current implementation boundary:** Gemini can create a structured appointment request for Jalandhar or another city selected by the customer. The request starts as **Pending provider confirmation**. This release does not yet include a live provider directory or external provider/calendar APIs, so it cannot verify availability or claim a real-world appointment is confirmed. An operator may update the status after receiving confirmation from the provider.
