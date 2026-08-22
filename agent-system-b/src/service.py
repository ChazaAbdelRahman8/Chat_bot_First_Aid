"""Deterministic appointment tools used by the Google ADK agent."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from store import new_appointment_id, utcnow


CRISIS = re.compile(
    r"\b(?:suicid|kill myself|hurt myself|self[- ]harm|end my life|immediate danger)\w*\b|"
    r"(?:Ø§Ù†ØªØ­Ø§Ø±|Ø£Ù‚ØªÙ„ Ù†ÙØ³ÙŠ|Ø£Ø¤Ø°ÙŠ Ù†ÙØ³ÙŠ)", re.I,
)
BOOK = re.compile(r"\b(?:book|confirm|reserve|schedule)\b|(?:Ø§Ø­Ø¬Ø²|Ø£ÙƒØ¯ Ø§Ù„Ø­Ø¬Ø²)", re.I)
ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
RELATIVE_TIME = re.compile(
    r"\b(?:after|in)\s+(\d{1,4})\s*(?:minute|minutes|min|mins)\b", re.I,
)
LANGUAGES = {"arabic": "Arabic", "english": "English", "french": "French"}
SPECIALIZATIONS = ("anxiety", "stress", "grief", "trauma", "family", "sleep")
ONLINE_MODE = re.compile(
    r"\b(?:online|remote|virtual|video(?:\s+call)?|telehealth)\b", re.I,
)
IN_PERSON_MODE = re.compile(
    r"\b(?:in[- ]person|face[- ]to[- ]face|on[- ]?site|at\s+the\s+clinic)\b", re.I,
)


def _selection_index(query: str) -> int | None:
    """Return a zero-based match selection without misreading other numbers.

    Bare digits are accepted only when the complete reply is a selection phrase,
    so values in requests such as ``in 10 minutes`` or ``under $50`` cannot
    accidentally book a provider.
    """
    normalized = " ".join(query.lower().strip().rstrip(".!?,;:").split())
    numeric = re.fullmatch(
        r"(?:(?:book|select|choose|confirm)\s+)?"
        r"(?:(?:the\s+)?(?:option|number)\s+)?#?([1-3])",
        normalized,
    )
    if numeric:
        return int(numeric.group(1)) - 1
    return next(
        (
            value
            for token, value in ORDINALS.items()
            if re.search(rf"\b{re.escape(token)}\b", normalized)
        ),
        None,
    )


def synthetic_profiles(now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or utcnow()
    soon = (now + timedelta(minutes=6)).replace(second=0, microsecond=0)
    tomorrow = (now + timedelta(days=1)).replace(hour=13, minute=0, second=0, microsecond=0)
    later = (now + timedelta(days=2)).replace(hour=17, minute=30, second=0, microsecond=0)
    return [
        {"psychologist_id": "PSY-001", "name": "Dr. Maya Haddad", "languages": ["Arabic", "English"], "specializations": ["anxiety", "stress"], "consultation_modes": ["online"], "price_usd": 45, "available_slots": [soon, tomorrow], "synthetic": True},
        {"psychologist_id": "PSY-002", "name": "Dr. Karim Nasser", "languages": ["Arabic", "French"], "specializations": ["trauma", "grief"], "consultation_modes": ["online", "in_person"], "price_usd": 60, "available_slots": [tomorrow, later], "synthetic": True},
        {"psychologist_id": "PSY-003", "name": "Dr. Lina Saad", "languages": ["English", "French"], "specializations": ["family", "sleep"], "consultation_modes": ["in_person"], "price_usd": 40, "available_slots": [later], "synthetic": True},
        {"psychologist_id": "PSY-004", "name": "Dr. Rami Daher", "languages": ["Arabic", "English", "French"], "specializations": ["anxiety", "family"], "consultation_modes": ["online", "in_person"], "price_usd": 50, "available_slots": [tomorrow, later], "synthetic": True},
    ]


class AppointmentService:
    def __init__(self, store: Any, *, timezone_name: str = "Asia/Beirut") -> None:
        self.store = store
        self.timezone_name = timezone_name
        self.store.seed_profiles(synthetic_profiles())

    def handle(self, query: str, conversation_id: str, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())
        if CRISIS.search(query):
            result = self._result(
                request_id, conversation_id, "crisis_escalation",
                "This may be an immediate mental-health crisis. Contact local emergency services or a trusted person now. I will not continue normal appointment booking while immediate danger may be present.",
            )
            self.store.save_request(request_id, result)
            return result

        state = self.store.get_state(conversation_id)
        extracted = self._preferences(query)
        ambiguous_mode = bool(extracted.pop("consultation_mode_ambiguous", False))
        preferences = {**state.get("preferences", {}), **extracted}
        if ambiguous_mode:
            preferences.pop("consultation_mode", None)
            self.store.save_state(
                conversation_id,
                {**state, "preferences": preferences},
            )
            result = self._result(
                request_id,
                conversation_id,
                "needs_information",
                "You mentioned both online and in-person consultation. Please choose one.",
                missing_fields=["consultation_mode"],
            )
            if self._is_arabic(query):
                result["message"] = (
                    "ذكرت موعداً أونلاين وحضورياً معاً. يرجى اختيار طريقة واحدة."
                )
            self.store.save_request(request_id, result)
            return result
        selects_previous_match = (
            bool(state.get("last_matches"))
            and _selection_index(query) is not None
        )
        if BOOK.search(query) or selects_previous_match:
            result = self._book(query, conversation_id, request_id, state, preferences)
        else:
            result = self._search(conversation_id, request_id, preferences)
        if self._is_arabic(query):
            result["message"] = self._arabic_message(result)
        self.store.save_request(request_id, result)
        return result

    @staticmethod
    def _preferences(query: str) -> dict[str, Any]:
        lower = query.lower()
        preferences: dict[str, Any] = {}
        for token, value in LANGUAGES.items():
            if token in lower:
                preferences["language"] = value
        if "العربية" in query or "عربي" in query:
            preferences["language"] = "Arabic"
        elif "الفرنسية" in query or "فرنسي" in query:
            preferences["language"] = "French"
        elif "الإنجليزية" in query or "انجليزي" in query:
            preferences["language"] = "English"
        online_requested = bool(ONLINE_MODE.search(lower)) or any(
            value in query for value in ("أونلاين", "اونلاين", "عن بعد")
        )
        in_person_requested = bool(IN_PERSON_MODE.search(lower)) or any(
            value in query for value in ("حضورياً", "حضوريا", "في العيادة")
        )
        if online_requested and in_person_requested:
            preferences["consultation_mode_ambiguous"] = True
        elif online_requested:
            preferences["consultation_mode"] = "online"
        elif in_person_requested:
            preferences["consultation_mode"] = "in_person"
        budget = re.search(r"(?:under|below|max(?:imum)?|budget(?: of)?|\$)\s*\$?(\d{2,4})|\$(\d{2,4})", lower)
        if budget:
            preferences["max_budget_usd"] = int(next(value for value in budget.groups() if value))
        arabic_budget = re.search(r"(?:ميزانية|بحد أقصى).{0,25}?(\d{2,4})", query)
        if arabic_budget:
            preferences["max_budget_usd"] = int(arabic_budget.group(1))
        elif "خمسين" in query:
            preferences["max_budget_usd"] = 50
        for specialization in SPECIALIZATIONS:
            if specialization in lower:
                preferences["specialization"] = specialization
        relative = RELATIVE_TIME.search(lower)
        if relative:
            minutes = int(relative.group(1))
            if 1 <= minutes <= 43_200:
                requested = (utcnow() + timedelta(minutes=minutes)).replace(microsecond=0)
                preferences["requested_start_time_utc"] = requested.isoformat()
        return preferences

    @staticmethod
    def _is_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06ff]", text))

    @staticmethod
    def _arabic_message(result: dict[str, Any]) -> str:
        status = result.get("status")
        if status == "matches_found":
            summary = "; ".join(
                f"{index + 1}. {row['name']} ({row['price_usd']} دولار، "
                f"{', '.join(row['languages'])}، {row['consultation_modes'][0]})"
                for index, row in enumerate(result.get("matches", []))
            )
            return f"وجدت هذه الخيارات التجريبية: {summary}. أخبرني أي خيار تريد حجزه."
        if status == "needs_information":
            return "أخبرني باللغة المفضلة وما إذا كنت تريد موعداً أونلاين أو حضورياً."
        if status == "no_matches":
            return "لم أجد أخصائياً تجريبياً يطابق كل الشروط. يمكنك تعديل اللغة أو طريقة الجلسة أو الميزانية."
        if status == "awaiting_confirmation":
            return "يرجى تأكيد رقم الأخصائي التجريبي الذي تريد حجزه."
        if status == "booked":
            appointment = result.get("appointment") or {}
            return (
                f"تم حجز موعدك التجريبي مع {appointment.get('psychologist_name', 'الأخصائي')} "
                f"في {appointment.get('start_time', '')}. سيتم إرسال تذكير قبل الموعد بخمس دقائق."
            )
        return str(result.get("message", ""))

    def _search(self, conversation_id: str, request_id: str, preferences: dict[str, Any]) -> dict[str, Any]:
        missing = [field for field in ("language", "consultation_mode") if not preferences.get(field)]
        if missing:
            self.store.save_state(conversation_id, {"preferences": preferences})
            return self._result(
                request_id, conversation_id, "needs_information",
                "Please tell me your preferred language and whether you want an online or in-person synthetic appointment.",
                missing_fields=missing,
            )
        profiles = self.store.list_profiles()
        matches = []
        for row in profiles:
            if preferences["language"] not in row["languages"]:
                continue
            if preferences["consultation_mode"] not in row["consultation_modes"]:
                continue
            if row["price_usd"] > preferences.get("max_budget_usd", 10_000):
                continue
            score = 55
            score += 20 if preferences.get("specialization") in row["specializations"] else 0
            score += max(0, 15 - abs(row["price_usd"] - preferences.get("max_budget_usd", row["price_usd"])))
            matches.append({**row, "match_score": score})
        matches.sort(key=lambda item: (-item["match_score"], item["price_usd"], item["name"]))
        serializable = [self._public_profile(item) for item in matches[:3]]
        self.store.save_state(conversation_id, {"preferences": preferences, "last_matches": serializable})
        if not serializable:
            return self._result(request_id, conversation_id, "no_matches", "No synthetic psychologist matches all of those preferences. Try changing the language, mode, or budget.")
        summary = "; ".join(
            f"{index + 1}. {row['name']} ({row['price_usd']} USD, {', '.join(row['languages'])}, {row['consultation_modes'][0]})"
            for index, row in enumerate(serializable)
        )
        return self._result(request_id, conversation_id, "matches_found", f"I found these synthetic matches: {summary}. Tell me which option you want to book.", matches=serializable)

    def _book(self, query: str, conversation_id: str, request_id: str, state: dict[str, Any], preferences: dict[str, Any]) -> dict[str, Any]:
        matches = state.get("last_matches", [])
        if not matches:
            return self._result(request_id, conversation_id, "needs_information", "I need to search for matching synthetic psychologists before booking.", missing_fields=["psychologist_selection"])
        lower = query.lower()
        index = _selection_index(query)
        if index is None:
            selected_id = next((row["psychologist_id"] for row in matches if row["psychologist_id"].lower() in lower), None)
            index = next((i for i, row in enumerate(matches) if row["psychologist_id"] == selected_id), None)
        if index is None or index >= len(matches):
            self.store.save_state(
                conversation_id,
                {**state, "preferences": preferences, "last_matches": matches},
            )
            return self._result(request_id, conversation_id, "awaiting_confirmation", "Please confirm which numbered synthetic psychologist you want to book.", matches=matches)
        selected = matches[index]
        requested_start = preferences.get("requested_start_time_utc")
        slot = datetime.fromisoformat(requested_start) if requested_start else datetime.fromisoformat(selected["available_slots"][0])
        start_utc = slot.astimezone(timezone.utc)
        if start_utc <= utcnow():
            return self._result(
                request_id, conversation_id, "error",
                "That requested synthetic appointment time has already passed. Please choose a future time.",
            )
        local_start = start_utc.astimezone(ZoneInfo(self.timezone_name))
        appointment = {
            "appointment_id": new_appointment_id(), "conversation_id": conversation_id,
            "psychologist_id": selected["psychologist_id"], "psychologist_name": selected["name"],
            "start_time_utc": start_utc, "start_time": local_start.isoformat(),
            "timezone": self.timezone_name, "consultation_mode": preferences.get("consultation_mode"),
            "language": preferences.get("language"), "price_usd": selected["price_usd"],
            "reminder_at_utc": start_utc - timedelta(minutes=5), "reminder_status": "scheduled",
            "reminder_attempts": 0, "status": "booked", "synthetic": True, "created_at": utcnow(),
        }
        if not self.store.book(appointment):
            return self._result(request_id, conversation_id, "error", "That synthetic slot is no longer available. Please choose another option.")
        public = self._public_appointment(appointment)
        self.store.save_state(conversation_id, {**state, "preferences": preferences, "last_appointment": public})
        return self._result(
            request_id, conversation_id, "booked",
            f"Your synthetic appointment with {selected['name']} is booked for {local_start.isoformat()} ({self.timezone_name}). A reminder is scheduled five minutes before it starts.",
            appointment=public,
        )

    @staticmethod
    def _public_profile(row: dict[str, Any]) -> dict[str, Any]:
        return {**row, "available_slots": [value.isoformat() if isinstance(value, datetime) else value for value in row["available_slots"]]}

    @staticmethod
    def _public_appointment(row: dict[str, Any]) -> dict[str, Any]:
        return {key: (value.isoformat() if isinstance(value, datetime) else value) for key, value in row.items() if key not in {"reminder_error"}}

    @staticmethod
    def _result(request_id: str, conversation_id: str, status: str, message: str, **extra: Any) -> dict[str, Any]:
        return {
            "request_id": request_id, "conversation_id": conversation_id,
            "status": status, "message": message, "matches": [], "appointment": None,
            "missing_fields": [], "framework": "google-adk", "execution_mode": "deterministic_tool",
            "synthetic": True, **extra,
        }
