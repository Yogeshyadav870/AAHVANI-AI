from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from knowledge_base import KnowledgeBase


@dataclass
class Lead:
    name: str = ""
    email: str = ""
    phone: str = ""
    company: str = ""
    need: str = ""
    budget: str = ""
    authority: str = ""
    timeline: str = ""
    sentiment: str = "neutral"
    lead_score: int = 0
    qualification: str = "Nurture"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AahvaniAgent:
    """Sales qualification agent with optional Gemini generation and deterministic fallback."""

    MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def __init__(self, knowledge_base: KnowledgeBase | None = None) -> None:
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.client = None
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=api_key)
            except Exception:
                self.client = None

    @staticmethod
    def new_session() -> dict[str, Any]:
        return {"lead": Lead().to_dict(), "messages": [], "lead_id": None}

    @staticmethod
    def _extract_email(text: str) -> str:
        match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I)
        return match.group(0) if match else ""

    @staticmethod
    def _extract_phone(text: str) -> str:
        match = re.search(r"(?:\+?\d[\d\s()-]{7,}\d)", text)
        return re.sub(r"\s+", " ", match.group(0)).strip() if match else ""

    @staticmethod
    def _extract_name(text: str) -> str:
        prefix = re.search(r"\b(?:my name is|i am|i'm|this is)\s+", text, re.I)
        if not prefix:
            return ""
        words = re.findall(r"[A-Za-z]+", text[prefix.end():])
        if not words:
            return ""
        name_parts = [words[0].capitalize()]
        connectors = {"and", "email", "phone", "my", "i", "we", "our", "from", "at", "the", "working"}
        if len(words) > 1 and words[1].lower() not in connectors:
            raw_remainder = text[prefix.end():]
            second = re.match(r"[A-Za-z]+\s+([A-Z][a-z]+|[A-Z]{2,})\b", raw_remainder)
            if second:
                name_parts.append(second.group(1).capitalize())
        return " ".join(name_parts)

    @staticmethod
    def _extract_budget(text: str) -> str:
        patterns = [
            r"(?:₹\s*|rs\.?\s*)(\d[\d,]*(?:\.\d+)?)\s*(lakhs?|lacs?|lakh|lac|crores?|crore|k|m)?",
            r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|crore|crores)\b",
            r"\b(no fixed budget|not decided|budget is flexible|flexible budget)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(0).strip()
        return ""

    @staticmethod
    def _extract_timeline(text: str) -> str:
        pattern = r"\b(immediately|asap|right away|this week|next week|this month|next month|within \d+\s*(?:days?|weeks?|months?)|in \d+\s*(?:days?|weeks?|months?)|just researching|no specific timeline)\b"
        match = re.search(pattern, text, re.I)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_authority(text: str) -> str:
        lower = text.lower()
        if any(term in lower for term in ["i am the owner", "i'm the owner", "founder", "co-founder", "decision maker", "final decision", "i decide", "i make the decision"]):
            return "Decision maker"
        if any(term in lower for term in ["my manager", "my boss", "management decides", "need approval", "not the decision maker"]):
            return "Influencer / requires approval"
        if any(term in lower for term in ["team lead", "department head", "director", "manager"]):
            return "Manager / evaluator"
        return ""

    @staticmethod
    def _sentiment(text: str) -> str:
        lower = text.lower()
        positive = ["great", "good", "interested", "love", "helpful", "excellent", "excited", "looking forward"]
        negative = ["angry", "frustrated", "disappointed", "bad", "issue", "problem", "expensive", "not happy"]
        p = sum(term in lower for term in positive)
        n = sum(term in lower for term in negative)
        if p > n:
            return "positive"
        if n > p:
            return "negative"
        return "neutral"

    @staticmethod
    def _score(lead: dict[str, Any]) -> tuple[int, str]:
        score = 0
        if lead.get("need"):
            score += 30
        if lead.get("budget"):
            score += 25
        if lead.get("authority"):
            score += 25 if lead["authority"] == "Decision maker" else 12
        if lead.get("timeline"):
            score += 20 if lead["timeline"].lower() not in {"just researching", "no specific timeline"} else 8
        status = "Qualified" if score >= 70 else "Warm" if score >= 40 else "Nurture"
        return score, status

    def _extract_facts(self, text: str, lead: dict[str, Any]) -> None:
        for key, value in {
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "name": self._extract_name(text),
            "budget": self._extract_budget(text),
            "timeline": self._extract_timeline(text),
            "authority": self._extract_authority(text),
        }.items():
            if value:
                lead[key] = value

        lower = text.lower()
        if not lead.get("company"):
            company_match = re.search(
                r"\b(?:at|from|with)\s+([A-Z][A-Za-z0-9& .-]{1,40})(?:,|\.|\bwhere\b|\bwe\b)",
                text,
            )
            if company_match:
                lead["company"] = company_match.group(1).strip()

        # Buying-intent statements should become a lead need, not an FAQ lookup.
        need_markers = (
            "need", "looking for", "want to", "interested in", "help us",
            "manage", "automate", "customer", "sales", "leads", "calls",
            "crm", "store", "storing", "company data", "customer data",
            "records", "follow-up", "follow up", "book crm",
        )
        if not lead.get("need") and any(marker in lower for marker in need_markers) and len(text.split()) > 2:
            lead["need"] = text.strip()

        lead["sentiment"] = self._sentiment(text)
        lead["lead_score"], lead["qualification"] = self._score(lead)

    @staticmethod
    def _is_product_question(text: str) -> bool:
        """Only route explicit product-information questions to FAQ retrieval."""
        cleaned = (text or "").strip().lower()
        question_starters = (
            r"^(?:what|which|how|does|do|can|could|is|are|will|tell me|explain|describe)\b",
            r"^(?:give me|show me)\s+(?:details|information|an overview)\b",
            r"^i want to know\s+(?:about|whether|if|how|what)\b",
        )
        return "?" in cleaned or any(re.search(pattern, cleaned) for pattern in question_starters)

    @staticmethod
    def _fallback_knowledge_answer(document: str) -> str:
        """Turn a retrieved FAQ section into a short, natural spoken-style answer."""
        raw_lines = [line.strip() for line in document.splitlines() if line.strip()]
        bullets = [
            re.sub(r"^[-*]\s*", "", line)
            for line in raw_lines
            if line.startswith(("-", "*"))
        ]
        if bullets:
            return "Northstar CRM includes " + "; ".join(bullets[:2]).rstrip(".") + "."

        paragraphs = [
            re.sub(r"^#{1,6}\s*", "", line)
            for line in raw_lines
            if not line.startswith(("-", "*", "#"))
        ]
        cleaned = " ".join(paragraphs)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        return " ".join(sentences[:2]).strip() or "I don't have an approved answer for that detail yet."

    def _next_discovery_question(
        self, lead: dict[str, Any], session: dict[str, Any]
    ) -> str:
        """Ask the next missing qualification question once, in a stable order."""
        questions = (
            ("need", "What would you like the CRM to help your company manage or improve?"),
            ("budget", "What approximate budget should I note, or is your budget flexible?"),
            ("authority", "Will you make the final purchase decision, or will someone else need to approve it?"),
            ("timeline", "When are you hoping to start using the CRM?"),
            ("email", "What email address should we use for the demo confirmation and follow-up?"),
        )
        for key, question in questions:
            if not lead.get(key) and not (key == "email" and session.get("contact_skipped")):
                session["pending_field"] = key
                return question

        session["pending_field"] = "demo"
        return (
            f"Thanks, {lead.get('name') or 'there'}. I have the key details for your CRM requirement. "
            f"Your current lead score is {lead.get('lead_score', 0)}/100 "
            f"({lead.get('qualification', 'Nurture')}). Choose a sample demo slot in the dashboard "
            "and press Confirm demo booking. These are demonstration slots, not live calendar availability."
        )

    @staticmethod
    def _is_simple_yes(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*(?:yes|yeah|yep|yup|correct|sure|i do|i will|that's right|that is right)[.! ]*\s*",
            text,
            re.I,
        ))

    @staticmethod
    def _is_simple_no(text: str) -> bool:
        return bool(re.fullmatch(
            r"\s*(?:no|nope|not really|i don't|i do not)[.! ]*\s*",
            text,
            re.I,
        ))

    def _gemini_answer(self, user_text: str, history: list[dict[str, str]], context: str) -> str | None:
        if self.client is None:
            return None
        transcript = "\n".join(f"{item['role']}: {item['content']}" for item in history[-8:])
        prompt = f"""You are AAHVANI AI, a transparent AI sales assistant for a fictional CRM and sales automation product called Northstar CRM.
Keep answers concise, natural, helpful and grounded only in the provided knowledge. If the knowledge does not contain an answer, say you do not have that detail and offer a human follow-up. Never invent prices, guarantees or product capabilities. Do not claim an appointment is booked unless the booking tool confirms it.

Brand pronunciation: AAHVANI is pronounced "AAH-vuh-nee" (आह्वनी), followed by "AI". If generating wording for speech, use this pronunciation and never say "A-Vee-Yaani" or spell the brand letters out.

Retrieved product knowledge:
{context or 'No relevant knowledge retrieved.'}

Recent conversation:
{transcript}

Customer's latest message: {user_text}
Write the next assistant response in 2-4 sentences. Ask at most one useful follow-up question."""
        try:
            response = self.client.models.generate_content(model=self.MODEL, contents=prompt)
            answer = getattr(response, "text", None)
            return answer.strip() if answer else None
        except Exception:
            return None

    def respond(self, user_text: str, session: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        session = session or self.new_session()
        lead = session.setdefault("lead", Lead().to_dict())
        messages = session.setdefault("messages", [])
        text = (user_text or "").strip()
        if not text:
            return "Please type or record a message so I can help.", session

        pending_field = session.get("pending_field")
        messages.append({"role": "user", "content": text})
        self._extract_facts(text, lead)
        lower = text.lower()

        # Interpret short answers using the field AAHVANI asked about last.
        if pending_field == "budget" and not lead.get("budget"):
            if re.search(r"\bflexib\w*\b|\bno fixed budget\b|\bnot decided\b", lower):
                lead["budget"] = "Flexible (no fixed budget)"
            elif self._is_simple_yes(text) or self._is_simple_no(text):
                reply = (
                    "Sure — should I record the budget as flexible, or is there an "
                    "approximate amount or range you have in mind?"
                )
                session["pending_field"] = "budget"
                messages.append({"role": "assistant", "content": reply})
                lead["lead_score"], lead["qualification"] = self._score(lead)
                return reply, session

        if pending_field == "authority" and not lead.get("authority"):
            if self._is_simple_yes(text):
                lead["authority"] = "Decision maker"
            elif self._is_simple_no(text):
                lead["authority"] = "Influencer / requires approval"

        if pending_field == "timeline" and not lead.get("timeline"):
            if re.search(r"\bsoon\b|\bas soon as possible\b", lower):
                lead["timeline"] = "Soon"

        if pending_field == "email" and not lead.get("email"):
            if re.search(r"\b(skip|prefer not|not now)\b", lower):
                session["contact_skipped"] = True

        greeting_only = len(text.split()) <= 3 and any(
            word in lower for word in ("hi", "hello", "hey", "namaste")
        )
        if greeting_only:
            session["pending_field"] = "need"
            reply = (
                "Hello! I'm AAHVANI AI, your sales assistant. What would you like "
                "your company to manage or improve with a CRM?"
            )
        elif self._is_product_question(text):
            retrieved = self.knowledge_base.search(text, top_k=1)
            if retrieved and float(retrieved[0].get("score", 0)) >= 0.25:
                context = retrieved[0]["content"]
                answer = self._gemini_answer(text, messages, context)
                if not answer:
                    answer = self._fallback_knowledge_answer(context)
                follow_up = self._next_discovery_question(lead, session)
                reply = (
                    f"{answer}\n\nTo understand your requirement better, "
                    f"{follow_up[0].lower() + follow_up[1:]}"
                )
            else:
                follow_up = self._next_discovery_question(lead, session)
                reply = (
                    "I don't have an approved product detail for that yet, so I don't want "
                    "to guess. I can arrange a human follow-up. "
                    f"To understand your requirement better, "
                    f"{follow_up[0].lower() + follow_up[1:]}"
                )
        else:
            reply = self._next_discovery_question(lead, session)

        lead["lead_score"], lead["qualification"] = self._score(lead)
        messages.append({"role": "assistant", "content": reply})
        return reply, session

    @staticmethod
    def summarize(session: dict[str, Any]) -> str:
        lead = session.get("lead", {})
        messages = session.get("messages", [])
        customer_messages = [m["content"] for m in messages if m["role"] == "user"]
        lines = [
            f"Lead: {lead.get('name') or 'Not captured'}",
            f"Company: {lead.get('company') or 'Not captured'}",
            f"Requirement: {lead.get('need') or 'Not captured'}",
            f"Budget: {lead.get('budget') or 'Not captured'}",
            f"Decision authority: {lead.get('authority') or 'Not captured'}",
            f"Timeline: {lead.get('timeline') or 'Not captured'}",
            f"Contact: {lead.get('email') or 'Not captured'} | {lead.get('phone') or 'Not captured'}",
            f"Sentiment: {lead.get('sentiment', 'neutral').title()}",
            f"Qualification: {lead.get('qualification', 'Nurture')} ({lead.get('lead_score', 0)}/100)",
            f"Conversation turns: {len(customer_messages)}",
        ]
        if customer_messages:
            lines.append("Customer statements: " + " | ".join(customer_messages[-3:]))
        return "\n".join(lines)
