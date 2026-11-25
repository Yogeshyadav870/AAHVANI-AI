from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


DEFAULT_KNOWLEDGE = """# Northstar CRM — Product Knowledge Base

## Product overview
Northstar CRM is a fictional demo product used to demonstrate AAHVANI AI. It helps small and medium businesses organize inbound leads, track sales conversations, manage follow-ups and monitor their sales pipeline.

## Core features
- Lead management: capture contact details, source, requirements and qualification status.
- Sales pipeline: organize leads by stages such as New, Contacted, Qualified, Demo Scheduled and Won.
- Follow-up reminders: help sales teams remember the next action for each lead.
- Reporting: view lead volume, qualification status, response time and conversion trends.
- Integrations: an implementation can connect to a company's existing CRM through authenticated APIs. The demo does not connect to a real external CRM.

## Ideal customers
Small and medium businesses, service providers, education consultancies, real-estate teams and agencies that receive regular customer inquiries.

## Pricing
The demo knowledge base does not contain approved commercial pricing. AAHVANI AI must not invent prices. It should offer a human sales follow-up for a formal quotation.

## Security and privacy
A production deployment should use TLS in transit, encryption at rest, role-based access control, audit logging, retention controls and explicit consent for call recording. The prototype stores its local demo records in SQLite and should use synthetic customer data only.

## Demo booking
The prototype can reserve one of the sample demo slots displayed in the interface. These are local demonstration slots, not real calendar availability.
"""


class KnowledgeBase:
    """Tiny dependency-free TF-IDF retriever for a small, local product knowledge base."""

    STOPWORDS = {
        "a", "an", "the", "and", "or", "is", "are", "to", "of", "in", "on", "for", "with", "it", "this", "that", "we", "you", "i", "our", "your", "can", "do", "does", "how", "what", "about", "please", "me", "my", "be", "as", "at", "by", "from", "will", "would", "could", "should", "have", "has", "had", "not", "no", "yes", "but", "if", "when", "where", "who", "which", "their", "they", "them", "he", "she", "his", "her", "was", "were", "been", "being", "into", "than", "then", "also", "there", "here"
    }

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(__file__).parent / "data" / "product_faq.md"
        raw = self.path.read_text(encoding="utf-8") if self.path.exists() else DEFAULT_KNOWLEDGE
        self.documents = self._split(raw)
        self.doc_tokens = [self._tokens(doc) for doc in self.documents]
        self.idf = self._build_idf(self.doc_tokens)

    @staticmethod
    def _split(text: str) -> list[str]:
        chunks = re.split(r"\n(?=##? )", text.strip())
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [word for word in words if word not in cls.STOPWORDS and len(word) > 1]

    @staticmethod
    def _build_idf(documents: list[list[str]]) -> dict[str, float]:
        count = len(documents) or 1
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(doc))
        return {word: math.log(1 + count / (1 + freq)) + 1 for word, freq in df.items()}

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str | float]]:
        terms = self._tokens(query)
        if not terms:
            return []
        query_counts = Counter(terms)
        scored: list[tuple[float, str]] = []
        for document, tokens in zip(self.documents, self.doc_tokens):
            if not tokens:
                continue
            counts = Counter(tokens)
            score = sum((1 + math.log(query_counts[t])) * self.idf.get(t, 0) * (1 + math.log(counts[t])) for t in query_counts if counts[t])
            score /= math.sqrt(len(tokens))
            if score > 0:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": round(score, 4), "content": doc} for score, doc in scored[:top_k]]
