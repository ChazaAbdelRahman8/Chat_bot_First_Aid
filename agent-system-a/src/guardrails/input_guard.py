"""Deterministic pre-retrieval policy gates for the first-aid RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardDecision:
    blocked: bool
    reason_code: str | None = None
    message_en: str = ""
    message_ar: str = ""


_PROMPT_INJECTION = re.compile(
    r"(?:ignore|override|bypass|disregard).{0,80}(?:instructions?|rules?|manuals?|system prompt)"
    r"|(?:reveal|show|print).{0,50}(?:system prompt|hidden instructions?)"
    r"|(?:fabricate|invent).{0,50}(?:answer|treatment|medical)"
    r"|treat this instruction as higher priority",
    re.IGNORECASE | re.DOTALL,
)
_EXACT_DOSE = re.compile(
    r"(?:exact(?:ly)?|how many|what dose|dosage).{0,90}"
    r"(?:milligrams?|\bmg\b|units?|dose|inject|injection|\biv\b)"
    r"|(?:inject|injection|\biv\b).{0,90}(?:milligrams?|\bmg\b|units?|exact dose)",
    re.IGNORECASE | re.DOTALL,
)
_GREETING_ONLY = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|"
    r"مرحب[ااً]?|أهلاً|اهلاً|السلام\s+عليكم)[!.،\s]*$",
    re.IGNORECASE,
)
_NAME_INTRODUCTION = re.compile(
    r"^(?:my\s+name\s+is|call\s+me)\s+([A-Za-z][A-Za-z' -]{0,49}?)[.!\s]*$",
    re.IGNORECASE,
)
_NAME_RECALL = re.compile(
    r"^(?:what(?:'s|\s+is)\s+my\s+name|do\s+you\s+remember\s+my\s+name)[?!.\s]*$",
    re.IGNORECASE,
)
_THANKS_ONLY = re.compile(r"^(?:thanks|thank\s+you|many\s+thanks)[!.,\s]*$", re.IGNORECASE)
_FAREWELL_ONLY = re.compile(r"^(?:bye|goodbye|see\s+you)[!.,\s]*$", re.IGNORECASE)
_OUT_OF_SCOPE = (
    re.compile(r"\b(?:insurance|billing code|tourist visa|stock price|weather)\b", re.I),
    re.compile(r"\b(?:write|generate)\b.{0,30}\b(?:python|javascript|program|code)\b", re.I),
    re.compile(r"\b(?:icu beds?|hospital beds?)\b.{0,50}\b(?:available|availability|nearest)\b", re.I),
    re.compile(r"\b(?:commercial|brand)\b.{0,60}\b(?:lowest|failure rate|best|202\d)\b", re.I),
    re.compile(r"\b(?:mri sequence|antibiotic)\b.{0,80}\b(?:diagnos\w*|prescrib\w*|days?)\b", re.I),
    re.compile(r"\bblood type\b.{0,60}\b(?:symptoms?|based only)\b", re.I),
    re.compile(r"\b(?:status|outcome|prognosis)\b.{0,90}\b(?:six months|long[- ]term)\b|\b(?:six months|long[- ]term)\b.{0,90}\b(?:status|outcome|prognosis)\b", re.I),
    re.compile(r"\b(?:capital of|president of|prime minister|stock market|cryptocurrency|bitcoin)\b", re.I),
    re.compile(r"\b(?:football|soccer|basketball|movie|song|video game|celebrity)\b", re.I),
    re.compile(r"\b(?:recipe|hotel|restaurant|vacation|travel itinerary)\b", re.I),
)
_FIRST_AID_INTENT = re.compile(
    r"\b(?:first aid|injur|bleed|bleeding|cut|burn|chok|fracture|wound|"
    r"breath|unconscious|emergency|seizure|shock|cpr|heat exhaustion|"
    r"heat stroke)\w*\b|"
    r"(?:إسعاف|إصابة|نزيف|حرق|اختناق|كسر|جرح|تنفس|فاقد الوعي|طوارئ)",
    re.IGNORECASE,
)


def has_first_aid_intent(query: str) -> bool:
    """Return whether a mixed request still contains actionable first-aid intent."""
    return bool(_FIRST_AID_INTENT.search(query))


def greeting_response(query: str) -> tuple[str, str] | None:
    """Return a friendly, language-matched response for greeting-only messages."""
    text = query.strip()
    if not _GREETING_ONLY.fullmatch(text):
        return None
    if any("\u0600" <= character <= "\u06ff" for character in text):
        return (
            "مرحباً! أنا مساعد الإسعافات الأولية. كيف يمكنني مساعدتك في حالة إسعافية أو إصابة؟",
            "ar",
        )
    return (
        "Hello! I'm your first-aid assistant. How can I help with an emergency or injury?",
        "en",
    )


def conversational_response(
    query: str, history: list[dict[str, str]] | None = None,
) -> tuple[str, str] | None:
    """Handle harmless conversational turns without sending them to specialists."""
    text = query.strip()
    introduction = _NAME_INTRODUCTION.fullmatch(text)
    if introduction:
        name = " ".join(introduction.group(1).split())
        return (
            f"Nice to meet you, {name}. I'll remember your name in this conversation. "
            "How can I help with first aid?",
            "en",
        )
    if _NAME_RECALL.fullmatch(text):
        for message in reversed(history or []):
            if message.get("role") != "user":
                continue
            previous = _NAME_INTRODUCTION.fullmatch(message.get("content", "").strip())
            if previous:
                name = " ".join(previous.group(1).split())
                return (f"Your name is {name}.", "en")
        return ("You haven't told me your name in this conversation yet.", "en")
    if _THANKS_ONLY.fullmatch(text):
        return ("You're welcome. Let me know if you need more first-aid guidance.", "en")
    if _FAREWELL_ONLY.fullmatch(text):
        return ("Goodbye. Stay safe!", "en")
    return None


def evaluate_input(query: str) -> GuardDecision:
    text = query.strip()
    if not text:
        return GuardDecision(True, "empty_query", "Please provide a first-aid question.", "يرجى تقديم سؤال متعلق بالإسعافات الأولية.")
    if _PROMPT_INJECTION.search(text):
        return GuardDecision(True, "prompt_injection", "I cannot follow instructions that override safety rules, reveal hidden prompts, or fabricate medical advice.", "لا يمكنني اتباع تعليمات تتجاوز قواعد السلامة أو تكشف التعليمات المخفية أو تختلق نصائح طبية.")
    if _EXACT_DOSE.search(text):
        return GuardDecision(True, "exact_dose_or_injection", "I cannot provide patient-specific medication doses or injection instructions. Contact emergency medical services or a qualified clinician.", "لا يمكنني تقديم جرعات دوائية خاصة بالمريض أو تعليمات للحقن. اتصل بخدمات الطوارئ أو بمختص صحي مؤهل.")
    if any(pattern.search(text) for pattern in _OUT_OF_SCOPE) and not has_first_aid_intent(text):
        return GuardDecision(True, "out_of_scope", "This request is outside the first-aid manuals and cannot be answered by this system.", "هذا الطلب خارج نطاق أدلة الإسعافات الأولية ولا يمكن لهذا النظام الإجابة عنه.")
    return GuardDecision(False)
