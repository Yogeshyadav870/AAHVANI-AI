from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any


CATALOGUE_PATH = Path(__file__).parent / "data" / "shopping_catalogue.json"


class ShoppingCatalogue:
    """Curated demo catalogue. All prices and listings are illustrative, not live offers."""

    ALIASES = {
        "smartphone": ("smartphone", "phone", "mobile", "iphone", "android"),
        "laptop": ("laptop", "notebook", "computer", "gaming laptop"),
        "headphones": ("headphones", "headphone", "earbuds", "earphones", "headset"),
        "tablet": ("tablet", "ipad"),
        "smartwatch": ("smartwatch", "smart watch", "fitness band", "watch"),
        "monitor": ("monitor", "display", "computer screen"),
        "camera": ("camera", "dslr", "mirrorless"),
        "home appliance": ("washing machine", "refrigerator", "fridge", "microwave", "vacuum cleaner", "air purifier"),
        "speaker": ("speaker", "bluetooth speaker", "soundbar"),
        "gaming console": ("gaming console", "playstation", "xbox", "console"),
    }
    BRANDS = (
        "Samsung", "Apple", "Xiaomi", "Redmi", "OnePlus", "Motorola", "Nothing",
        "Realme", "Vivo", "Oppo", "Poco", "Google", "Asus", "Acer", "Lenovo",
        "HP", "Dell", "MSI", "JBL", "Sony", "boAt", "Noise", "Amazfit",
        "LG", "Canon", "Nikon", "Whirlpool", "IFB", "Haier", "Bose", "Nintendo",
        "Microsoft", "Logitech",
    )
    USE_CASES = {
        "gaming": ("gaming", "game", "games", "gamer", "gpu", "fps", "khel"),
        "coding": ("coding", "programming", "developer", "development", "software"),
        "study": ("study", "student", "college", "classes", "online class", "notes", "padhai"),
        "work": ("work", "office", "professional", "productivity", "business", "kaam"),
        "camera": ("camera", "photography", "photos", "photo", "selfie", "video recording"),
        "battery": ("battery", "long battery", "backup", "battery life"),
        "performance": ("performance", "fast", "multitasking", "processor", "speed"),
        "music": ("music", "songs", "audio", "bass"),
        "calls": ("calls", "calling", "meetings", "microphone", "voice"),
        "fitness": ("fitness", "health", "workout", "running", "steps"),
        "content": ("editing", "video editing", "design", "creative", "content creation"),
        "movies": ("movies", "streaming", "watch videos", "entertainment"),
        "travel": ("travel", "portraits", "portrait photography", "landscape"),
        "creative": ("drawing", "art", "creative work", "illustration", "design"),
        "capacity": ("capacity", "family size", "large family", "load capacity"),
        "efficiency": ("energy efficient", "energy efficiency", "low power", "save electricity"),
        "compact": ("compact", "small kitchen", "space saving", "small space"),
        "party": ("party", "parties", "gathering", "loud music"),
        "smart features": ("smart features", "smart functions", "connected features"),
        "everyday": ("daily", "everyday", "general use", "all round", "all-round", "normal use"),
    }

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else CATALOGUE_PATH
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("Shopping catalogue must contain at least one item.")
        required = {"id", "category", "title", "brand", "price_inr", "specs", "use_cases"}
        for product in raw:
            missing = required.difference(product)
            if missing:
                raise ValueError(
                    f"Catalogue product {product.get('id', '<unknown>')} missing: {', '.join(sorted(missing))}"
                )
        self.products = raw
        self.by_id = {item["id"]: item for item in raw}

    @classmethod
    def detect_category(cls, text: str) -> str:
        lower = (text or "").lower()
        aliases = sorted(
            ((alias, category) for category, values in cls.ALIASES.items() for alias in values),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        matches = []
        for alias, category in aliases:
            match = re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", lower)
            if match:
                matches.append((match.start(), -len(alias), category))
        return min(matches)[2] if matches else ""

    @staticmethod
    def parse_budget(text: str) -> int | None:
        raw = (text or "").lower().replace(",", "").replace("₹", " rs ")
        patterns = (
            r"\b(?:under|below|within|max(?:imum)?|budget(?:\s+of)?|upto|up\s+to|less\s+than|ke\s+andar|tak)\s*(?:rs\.?\s*)?(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?\b",
            r"\b(?:rs\.?\s*)(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)?\b",
            r"\b(\d+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|thousand)\b",
        )
        multipliers = {
            "k": 1_000, "thousand": 1_000, "lakh": 100_000, "lac": 100_000,
            "cr": 10_000_000, "crore": 10_000_000,
        }
        for pattern in patterns:
            match = re.search(pattern, raw, re.I)
            if match:
                amount = float(match.group(1))
                unit = (match.group(2) or "").lower()
                value = int(round(amount * multipliers.get(unit, 1)))
                if value >= 1_000:
                    return value
        bare = re.search(r"\b(?:budget\s*(?:is|of)?\s*)?(\d{4,8})\b", raw)
        if bare:
            return int(bare.group(1))
        return None

    @classmethod
    def detect_brand(cls, text: str) -> str:
        lower = (text or "").lower()
        for brand in cls.BRANDS:
            if re.search(r"(?<!\w)" + re.escape(brand.lower()) + r"(?!\w)", lower):
                return brand
        if re.search(r"\b(any brand|no preference|any company|all brands|you decide|doesn't matter)\b", lower):
            return "Any"
        return ""

    @classmethod
    def detect_use_case(cls, text: str) -> str:
        lower = (text or "").lower()
        # Match whole words/phrases, not arbitrary substrings: e.g. "smartphone"
        # must not accidentally match "art" and trigger the creative-use path.
        for use_case, markers in cls.USE_CASES.items():
            for marker in markers:
                if re.search(r"(?<!\w)" + re.escape(marker) + r"(?!\w)", lower):
                    return use_case
        if re.search(r"\b(no preference|anything|all round|all-round|general use)\b", lower):
            return "everyday"
        return ""

    def search(
        self,
        *,
        category: str,
        budget_max: int | None = None,
        brand: str = "",
        use_case: str = "",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        candidates = [p for p in self.products if p["category"] == category]
        if budget_max is not None:
            candidates = [p for p in candidates if int(p["price_inr"]) <= int(budget_max)]
        if brand and brand.lower() != "any":
            candidates = [p for p in candidates if p["brand"].lower() == brand.lower()]

        def score(product: dict[str, Any]) -> tuple[int, int]:
            overlap = 4 if use_case and use_case in product.get("use_cases", []) else 0
            words = re.findall(r"[a-z0-9]+", (use_case or "").lower())
            text = " ".join([product["title"], product["brand"], *product.get("tags", [])]).lower()
            return overlap + sum(token in text for token in words), -int(product["price_inr"])

        return sorted(candidates, key=score, reverse=True)[:max(1, min(limit, 3))]

    def format_recommendations(self, products: list[dict[str, Any]]) -> str:
        if not products:
            return "I couldn't find a match in the current demo catalogue. Try increasing the budget or changing the brand."
        cards = []
        for index, product in enumerate(products, start=1):
            specs = " · ".join(product.get("specs", [])[:4])
            cards.append(
                f"### {product.get('icon', '🛍️')} Option {index} — {product['title']}\n"
                f"**{product['brand']} · ₹{int(product['price_inr']):,}**\n\n"
                f"{specs}\n\n*Why it fits:* {product.get('recommendation_reason', 'Matches your requirements.')}"
            )
        return (
            f"## I found {len(products)} matching options\n\n"
            + "\n\n---\n\n".join(cards)
            + "\n\nChoose one by saying or typing **option 1**, **option 2**, or **option 3**. "
            "Listings and prices are illustrative demo data, not live store offers."
        )

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self.by_id.get(product_id)


class ShoppingAgent:
    """Conversational fallback for text chat; uses the same catalogue as Gemini Live."""

    YES = re.compile(
        r"^(?:yes|yeah|yep|sure|confirm|confirmed|go ahead|place it|buy it|order it|please do|yes please|do it)[.! ]*$",
        re.I,
    )
    YES = re.compile(
        r"^(?:yes|yeah|yep|sure|confirm|confirmed|go ahead|place it|buy it|order it|please do|yes please|do it|haan|han|ji haan|jee haan|haan ji|kar do|order kar do|haan kar do)[.! ]*$",
        re.I,
    )
    NO = re.compile(r"^(?:no|nope|cancel|not now|don't place it|do not place it|nahi|nahin|abhi nahi)[.! ]*$", re.I)
    CATEGORY_SWITCH = re.compile(
        r"\b(?:actually|instead|change to|switch to|i want (?:a|an|the)|i need (?:a|an|the)|looking for (?:a|an|the)|buy (?:a|an|the))\b",
        re.I,
    )

    def __init__(self, catalogue: ShoppingCatalogue | None = None) -> None:
        self.catalogue = catalogue or ShoppingCatalogue()

    @staticmethod
    def new_session() -> dict[str, Any]:
        return {
            "messages": [],
            "shopping": {
                "category": "", "budget_max": None, "brand": "", "use_case": "",
                "recommendations": [], "awaiting_selection": False,
                "selected_product": None, "awaiting_confirmation": False,
            },
        }

    @staticmethod
    def _format_product(product: dict[str, Any], index: int) -> str:
        return (
            f"**Option {index} — {product['title']}**\n"
            f"{product['brand']} · ₹{int(product['price_inr']):,}\n"
            + " · ".join(product.get("specs", [])[:4])
        )

    def _selected(self, text: str, recommendations: list[dict[str, Any]]) -> dict[str, Any] | None:
        lower = (text or "").lower()
        match = re.search(r"\b(?:option|number|no\.?)\s*([1-3])\b", lower)
        if match:
            index = int(match.group(1)) - 1
            return recommendations[index] if index < len(recommendations) else None
        if re.fullmatch(r"\s*[1-3]\s*[.!]?\s*", lower):
            index = int(lower.strip(" .!")) - 1
            return recommendations[index] if index < len(recommendations) else None
        for product in recommendations:
            if product["id"].lower() in lower or product["title"].lower() in lower:
                return product
        return None

    @staticmethod
    def format_order_confirmation(order: dict[str, Any]) -> str:
        return (
            "## ✅ Demo order placed successfully!\n\n"
            f"**Order ID:** {order['order_ref']}\n\n"
            f"**Product:** {order['product_title']}\n\n"
            f"**Quantity:** {order['quantity']}\n\n"
            f"**Total:** ₹{int(order['total_price']):,}\n\n"
            f"**Status:** {order['status']}\n\n"
            "This is a simulated order. No payment was collected and no real shipment was created."
        )

    def respond(self, user_text: str, session: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        session = session or self.new_session()
        text = " ".join((user_text or "").split()).strip()
        if not text:
            return "Tell me what you are looking to buy, and I will help you compare options.", session

        session.setdefault("messages", []).append({"role": "user", "content": text})
        shopping = session.setdefault("shopping", self.new_session()["shopping"])
        lower = text.lower()

        requested_category = self.catalogue.detect_category(text)
        if (
            requested_category
            and requested_category != shopping.get("category")
            and self.CATEGORY_SWITCH.search(text)
        ):
            shopping.update({
                "category": requested_category, "budget_max": None, "brand": "",
                "use_case": "", "recommendations": [], "awaiting_selection": False,
                "selected_product": None, "awaiting_confirmation": False,
            })

        if shopping.get("awaiting_confirmation"):
            if self.YES.fullmatch(lower):
                session["demo_order_to_create"] = {
                    "product": dict(shopping["selected_product"]),
                    "quantity": 1,
                    "idempotency_key": f"{session.get('session_id', 'shopping')}:{uuid.uuid4()}",
                }
                shopping["awaiting_confirmation"] = False
                reply = "Great. I'm placing your simulated order now. No payment will be taken."
            elif self.NO.fullmatch(lower):
                shopping["awaiting_confirmation"] = False
                shopping["selected_product"] = None
                shopping["awaiting_selection"] = True
                reply = "No problem. I have not placed an order. You can choose another option or change your requirements."
            else:
                reply = "Please say yes to place this simulated order, or no to cancel it."
            session["messages"].append({"role": "assistant", "content": reply})
            return reply, session

        if shopping.get("awaiting_selection"):
            chosen = self._selected(text, shopping.get("recommendations", []))
            if chosen:
                shopping["selected_product"] = chosen
                shopping["awaiting_selection"] = False
                shopping["awaiting_confirmation"] = True
                reply = (
                    f"You selected **{chosen['title']}** by {chosen['brand']} for "
                    f"**₹{int(chosen['price_inr']):,}**. This is a simulated order only; "
                    "there is no payment or real delivery. Shall I place this demo order?"
                )
            else:
                reply = "I didn't catch which product you chose. Please say option 1, option 2 or option 3."
            session["messages"].append({"role": "assistant", "content": reply})
            return reply, session

        category = self.catalogue.detect_category(text)
        current_category = shopping.get("category", "")
        if category and (
            not current_category or category == current_category or self.CATEGORY_SWITCH.search(text)
        ):
            if category != current_category:
                shopping.update({
                    "category": category, "budget_max": None, "brand": "", "use_case": "",
                    "recommendations": [], "awaiting_selection": False,
                    "selected_product": None, "awaiting_confirmation": False,
                })

        if not shopping.get("category"):
            reply = (
                "Of course! What are you looking to buy — a smartphone, laptop, tablet, "
                "headphones, smartwatch, monitor, camera, speaker or a home appliance?"
            )
            session["messages"].append({"role": "assistant", "content": reply})
            return reply, session

        budget = self.catalogue.parse_budget(text)
        if budget:
            shopping["budget_max"] = budget
        brand = self.catalogue.detect_brand(text)
        if brand:
            shopping["brand"] = brand
        use_case = self.catalogue.detect_use_case(text)
        if use_case:
            shopping["use_case"] = use_case

        if not shopping.get("budget_max"):
            reply = f"Sure — what is your maximum budget for the {shopping['category']}?"
        elif not shopping.get("use_case"):
            questions = {
                "smartphone": "What matters most — camera, gaming performance, battery life or balanced everyday use?",
                "laptop": "What will you mainly use it for — gaming, coding, studying, office work or editing?",
                "headphones": "Will you mainly use them for music, calls/meetings or gaming?",
                "tablet": "Will you mainly use it for studying, entertainment, drawing or work?",
                "smartwatch": "Is your priority fitness tracking, battery life or everyday smart features?",
                "monitor": "Will you mainly use it for gaming, coding/work or content creation?",
                "camera": "Will you mainly use it for travel, portraits, video or professional work?",
                "home appliance": "What matters most — capacity, energy efficiency, compact size or smart features?",
                "speaker": "Will you mainly use it for music, parties, movies or a desktop setup?",
                "gaming console": "Which matters more — exclusive games, online multiplayer or family gaming?",
            }
            reply = questions.get(shopping["category"], "What is the main use you have in mind?")
        elif not shopping.get("brand"):
            reply = "Do you have a preferred brand, or should I compare options from all brands?"
        else:
            products = self.catalogue.search(
                category=shopping["category"], budget_max=shopping["budget_max"],
                brand=shopping["brand"], use_case=shopping["use_case"], limit=3,
            )
            prefix = ""
            if not products and shopping["brand"] and shopping["brand"] != "Any":
                products = self.catalogue.search(
                    category=shopping["category"], budget_max=shopping["budget_max"],
                    use_case=shopping["use_case"], limit=3,
                )
                if products:
                    prefix = f"I couldn't find a {shopping['brand']} listing within that budget, so here are alternatives.\n\n"
            if not products:
                reply = (
                    f"I couldn't find a {shopping['category']} in our demo catalogue under "
                    f"₹{shopping['budget_max']:,}. Increase the budget or ask me to compare the category."
                )
            else:
                shopping["recommendations"] = products
                shopping["awaiting_selection"] = True
                shopping["selected_product"] = None
                shopping["awaiting_confirmation"] = False
                reply = prefix + self.catalogue.format_recommendations(products)

        session["messages"].append({"role": "assistant", "content": reply})
        return reply, session
