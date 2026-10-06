"""Natural-language time ranges + date-filtered semantic retrieval.

"What changed on Tuesday?" needs two things a vector search alone can't do:
interpret *Tuesday* relative to "now" in the user's timezone, and restrict the
search to that window. ``parse_time_range`` handles the first; the index's ID
selector handles the second.

Weekday rule: a bare weekday ("Tuesday") means the most recent such day,
including today; "last Tuesday" means the most recent one strictly before today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.agent.indexer import Hit, VectorIndex

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_UNITS = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}


@dataclass(frozen=True)
class TimeRange:
    """Half-open UTC interval [start, end) plus the phrase it came from."""

    start: datetime
    end: datetime
    label: str

    def describe(self) -> str:
        return f"{self.label}: {self.start.isoformat()} to {self.end.isoformat()}"


def _day_start(d: datetime, tz: ZoneInfo) -> datetime:
    return datetime.combine(d.date(), time.min, tzinfo=tz)


def parse_time_range(text: str, now: datetime, tz_name: str = "UTC") -> TimeRange | None:
    """Parse the first recognised time expression in ``text`` (None if none).

    ``now`` must be timezone-aware. Day boundaries are computed in ``tz_name``
    and the result is returned in UTC.
    """
    tz = ZoneInfo(tz_name)
    local = now.astimezone(tz)
    today = _day_start(local, tz)
    q = text.lower()

    def out(start: datetime, end: datetime, label: str) -> TimeRange:
        return TimeRange(start.astimezone(UTC), end.astimezone(UTC), label)

    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q):
        start = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=tz)
        return out(start, start + timedelta(days=1), m[0])
    if m := re.search(r"\b(?:past|last)\s+(\d+)\s+(minute|hour|day|week)s?\b", q):
        n, unit = int(m[1]), m[2]
        return out(local - timedelta(seconds=n * _UNITS[unit]), local, m[0])
    if re.search(r"\blast night\b", q):
        return out(today - timedelta(hours=6), today + timedelta(hours=6), "last night")
    if re.search(r"\byesterday\b", q):
        return out(today - timedelta(days=1), today, "yesterday")
    if re.search(r"\btoday\b|\bthis morning\b", q):
        return out(today, local, "today")
    if re.search(r"\b(?:past|last)\s+(?:24 hours|day)\b", q):
        return out(local - timedelta(days=1), local, "last 24 hours")
    week_start = today - timedelta(days=local.weekday())
    if re.search(r"\blast week\b", q):
        return out(week_start - timedelta(days=7), week_start, "last week")
    if re.search(r"\bthis week\b", q):
        return out(week_start, local, "this week")
    if re.search(r"\blast month\b", q):
        first = today.replace(day=1)
        prev_first = (first - timedelta(days=1)).replace(day=1)
        return out(prev_first, first, "last month")
    if re.search(r"\bthis month\b", q):
        return out(today.replace(day=1), local, "this month")
    if m := re.search(r"\b(last\s+)?(" + "|".join(WEEKDAYS) + r")\b", q):
        target = WEEKDAYS.index(m[2])
        back = (local.weekday() - target) % 7
        if m[1] and back == 0:
            back = 7
        day = today - timedelta(days=back)
        return out(day, day + timedelta(days=1), m[0].strip())
    return None


@dataclass
class Retrieval:
    """Retrieved context for a question."""

    time_range: TimeRange | None
    hits: list[Hit]
    widened: bool  # True if the date filter matched nothing and we searched all time


class Retriever:
    """Semantic search with automatic date filtering from the question text."""

    def __init__(self, index: VectorIndex, tz_name: str = "UTC") -> None:
        self.index = index
        self.tz_name = tz_name

    def retrieve(
        self, session: Session, question: str, k: int = 8, now: datetime | None = None
    ) -> Retrieval:
        now = now or datetime.now(UTC)
        self.index.ensure_fresh(session)
        tr = parse_time_range(question, now, self.tz_name)
        if tr is None:
            return Retrieval(None, self.index.search(session, question, k), False)
        hits = self.index.search(session, question, k, tr.start, tr.end)
        if hits:
            return Retrieval(tr, hits, False)
        # Nothing in that window: say so, but still give the model nearby context.
        return Retrieval(tr, self.index.search(session, question, k), True)
