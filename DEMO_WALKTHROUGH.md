# AAHVANI AI — Demo Walkthrough

Use synthetic details only. This script is designed to demonstrate one complete sales workflow.

## Scenario

A prospective customer is evaluating a customer-management platform for a small digital marketing agency.

## Suggested conversation

1. Customer: "We run a digital marketing agency and need a better way to manage customers."
2. Agent: asks what kind of customer-management problem needs solving.
3. Customer: "We need lead tracking and automated follow-ups."
4. Agent: asks about team size, budget, decision authority, and buying timeline.
5. Customer: "We have 12 employees, a budget of ₹50,000, I make the decision, and we want to start this month."
6. Agent: retrieves relevant product details from the FAQ, summarizes the fit, and offers available demo slots.
7. Customer: selects one of the offered slots.
8. Agent: confirms booking, stores the lead and appointment in SQLite, and produces a call summary.

## What to show evaluators

- The agent asks follow-up questions rather than returning a one-shot answer.
- Context is preserved across turns.
- The product answer is grounded in the local FAQ.
- The lead score is explained by need, budget, authority, and timeline signals.
- Booking creates a persistent CRM record.
- The final summary includes customer requirements and next steps.

## Demonstration checklist

- [ ] Start the application and show the Gradio interface.
- [ ] Run the scripted inquiry.
- [ ] Ask a question covered by the product FAQ.
- [ ] Complete lead qualification.
- [ ] Select and book a demo slot.
- [ ] Show the saved lead/appointment details.
- [ ] Generate the call summary.
- [ ] Explain the optional voice modules and production architecture.

## Fallback

If a network API is unavailable, use the local deterministic response mode. If microphone or TTS setup fails, complete the same end-to-end business workflow with the text interface. Do not claim optional voice features are live unless you demonstrate them.
