# AAHVANI AI — Round 3 Practical Demo Walkthrough

**Purpose:** Demonstrate one complete, end-to-end sales workflow.  
**Demo product:** Northstar CRM (fictional sample product).  
**Data rule:** Use synthetic prospect information only.

## Before the evaluator arrives

- Start the local app with `python app.py` and open the Gradio URL.
- Check that the dashboard, chat, lead panel, booking panel, summary and CRM table load.
- Keep the GitHub repository and [system architecture diagram](architecture/system_architecture.svg) ready in another browser tab.
- Confirm the microphone and audio output work only if you intend to demonstrate the optional voice path.
- Avoid relying on first-time model downloads during the timed demonstration.

## Scenario

A founder of a small digital marketing agency receives leads through Instagram and the company website. The agency needs a better way to manage incoming leads and follow-ups and is evaluating Northstar CRM.

## Recommended live sequence

### 1. Introduce the workspace

Point out the three KPI cards:
- Total leads
- Qualified leads
- Demo bookings

Explain that these counters are read from the local SQLite database.

### 2. Demonstrate the conversation and product FAQ

Send:

```text
Hello
```

Then send:

```text
What features does Northstar CRM provide for a small marketing agency?
```

Explain that the agent searches the local Markdown product FAQ and uses the relevant retrieved section. In the default local configuration, the response uses deterministic fallback logic; Gemini generation is optional and requires an API key.

### 3. Demonstrate customer fact extraction and qualification

Send this synthetic prospect message:

```text
I'm Rahul, founder at BrightWave Agency. We need a CRM to manage leads and follow-ups. Our budget is ₹50,000, I make the final decision, and we want to implement it this month. My email is rahul@example.com.
```

Show the lead intelligence panel. Explain that the backend extracts the name, company, email, budget, authority, timeline and need using regex and keyword rules. The lead score is a BANT-inspired heuristic:

- Need: 30
- Budget: 25
- Decision maker: 25
- Timeline: 20

Expected score for these signals: **100/100 — Qualified**. This is a transparent rule-based score, not an ML prediction.

### 4. Save the lead and book a sample demo

Click **Save lead to CRM**. Choose one of the displayed sample slots and click **Confirm demo booking**.

Explain that the booking is persisted in the SQLite appointments table and linked to the saved lead. The slots are local demonstration values, not real calendar availability.

### 5. Show the business output

Click **Generate summary**, then click **Refresh CRM & metrics**.

Point out the lead record, qualification status, score and dashboard counts. This completes the end-to-end workflow from customer inquiry to CRM record and sample appointment.

## Optional Speak Up — Live Voice Mode

Use this only after installing `requirements-live.txt`, configuring a valid local `GEMINI_API_KEY`, and testing the microphone and speakers.

1. Open **Speak Up — Live Voice Mode**.
2. Allow browser microphone access and click the integrated microphone icon. AAHVANI should greet the customer automatically.
3. Say: "Hello, I'm Rahul. I run a digital marketing agency and want to improve how we follow up with leads."
4. Continue with a natural follow-up exchange. Let AAHVANI ask about needs, budget, authority and timeline. You can also type in the component's built-in textbox and press Enter; this sends the message to the same active Gemini Live session.
5. Use the built-in controls: microphone toggles the customer's microphone, speaker toggles AAHVANI's audio, and the square button ends the session. Watch the status line: it changes to **AAHVANI is speaking** while audio arrives and **Your turn** after the response. Interrupt AAHVANI while it is speaking with a short clarification to demonstrate barge-in/turn-taking.
6. Point to the conversation transcript and Lead Intelligence panel. Input transcripts and typed messages are passed to the existing lead extractor, BANT-inspired score and SQLite CRM.
7. End the live stream. Use the existing dashboard controls to inspect the lead, select a sample slot and explicitly confirm the demo booking.

Explain the stages accurately: FastRTC/WebRTC streams browser audio to the Python server; Gemini Live processes audio in the cloud and returns native speech audio. The server also receives input/output transcripts. The local API key is never sent to browser JavaScript. Live mode uses the Gemini API and is subject to current quota and usage pricing. This is browser-based live voice, not PSTN telephony or a connected phone number.

## Gemini Live — Jalandhar service appointment request

Use this scenario to demonstrate Gemini function calling and the new CRM appointment table. Use synthetic contact information only.

**Example voice request:**

```text
Hi, I need a dental consultation in Jalandhar next Wednesday at 11:30 AM.
I don't have a clinic preference. My name is Rahul and my phone number is 9876543210.
```

1. Start Live Voice Mode and speak the request naturally.
2. AAHVANI should collect any missing details (service category, service type, provider preference, city, specific date/time, customer name, and phone/email).
3. AAHVANI must read the complete request back and ask: **"To confirm, should I submit this appointment request to the CRM now?"**
4. Say **"Yes, go ahead."** Only after this explicit confirmation should Gemini call the server-side `save_appointment_request` function.
5. Show the new row in **Jalandhar service appointment requests**. It contains the service, provider preference, city, requested date/time, customer contact, request ID and status.
6. Click **Generate summary** to show the appointment details in the call summary.
7. The initial status is **Pending provider confirmation**. This is a saved request, not a confirmed provider booking. After contacting the provider and receiving confirmation, paste the request ID into the CRM status control, choose **Confirmed**, and add the provider reference or confirmation note.

**Explain the integration:** Gemini Live returns a structured function call. The Python server validates the customer confirmation and required fields, saves the customer as a CRM lead, creates a row in `service_appointments`, and sends the tool result back to Gemini. Gemini then speaks the result to the customer. A duplicate tool retry returns the existing request instead of creating another row.

**Current limitation:** The application does not yet search a live Jalandhar provider directory, check real provider availability, contact the provider automatically, or reserve an external calendar slot. Those need provider directory data and provider/calendar integrations. Never present a pending request as an appointment confirmation.

## Batch voice fallback

If Live Voice Mode is unavailable, use the existing optional batch voice path:

1. Record a short customer request using the microphone input.
2. Wait for Faster-Whisper to transcribe the recording locally.
3. Review the transcription and click **Send voice message & hear reply**.
4. Play the Edge TTS response.

Faster-Whisper runs locally (the first model load may take longer); Edge TTS uses an online service. This is a one-turn workflow, not a live conversation.

## If something fails

- **Gemini/API unavailable:** Continue with the deterministic local fallback; Gemini is optional.
- **Microphone/STT unavailable:** Complete the business workflow with text chat.
- **TTS/internet unavailable:** Show the textual response and explain that Edge TTS is an optional online adapter.
- **Unexpected demo data:** Use the CRM refresh action and continue with a clean synthetic prospect.

Do not claim live telephony, a real calendar, an external CRM, fully validated multilingual communication, embedding-based RAG or production-scale deployment. Those are future extensions documented in the architecture plan.
