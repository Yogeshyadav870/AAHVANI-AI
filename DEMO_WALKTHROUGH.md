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

## Optional voice demonstration

Only demonstrate this when the packages and microphone have already been tested:

1. Record a short customer request using the microphone input.
2. Wait for Faster-Whisper to transcribe the recording.
3. Review the transcription and click **Send voice message & hear reply**.
4. Play the generated response.

Explain the stages accurately: Faster-Whisper runs locally (the first model load may take longer); the transcript uses the same sales-agent workflow as text; Edge TTS uses an online service to generate the spoken reply. This is a one-turn voice workflow, not a live telephony or full-duplex calling system.

## If something fails

- **Gemini/API unavailable:** Continue with the deterministic local fallback; Gemini is optional.
- **Microphone/STT unavailable:** Complete the business workflow with text chat.
- **TTS/internet unavailable:** Show the textual response and explain that Edge TTS is an optional online adapter.
- **Unexpected demo data:** Use the CRM refresh action and continue with a clean synthetic prospect.

Do not claim live telephony, a real calendar, an external CRM, fully validated multilingual communication, embedding-based RAG or production-scale deployment. Those are future extensions documented in the architecture plan.
