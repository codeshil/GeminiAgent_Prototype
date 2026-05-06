"""The three Visual Intent Detection agents, plus a small helper for the
plain text answer used in both columns.

Each function is independently callable from a Python REPL — useful when the
user wants to debug one agent without spinning up Streamlit.
"""

from __future__ import annotations

import os
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

from google import genai
from google.genai import types as genai_types
from googleapiclient.discovery import build as build_youtube_client

import prompts


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VisualIntent(BaseModel):
    has_visual_intent: bool
    format: Literal["short", "long", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class TimestampPick(BaseModel):
    start_seconds: int = Field(ge=0)
    end_seconds: int = Field(ge=0)
    reasoning: str


class YouTubeResult(BaseModel):
    video_id: str
    title: str
    channel: str
    duration: str       # ISO 8601 raw (e.g. "PT3M21S") — kept for the trace
    duration_secs: int  # parsed seconds — used for Short detection
    thumbnail_url: str
    video_url: str
    views: int = 0
    likes: Optional[int] = None  # often hidden by creators; may be None
    search_query_used: str = ""  # the optimized query sent to YouTube
    is_short: bool = False       # True when duration ≤ 90 s


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
_MAX_ATTEMPTS = 3


def _with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying up to _MAX_ATTEMPTS times on
    transient Gemini errors (503 overload, 429 rate-limit) with exponential
    backoff (2 s → 4 s → 8 s)."""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            retryable = any(code in msg for code in _RETRYABLE)
            if retryable and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2.0 * (2 ** attempt))  # 2 s, 4 s, 8 s
                continue
            raise


# ---------------------------------------------------------------------------
# Clients (built lazily so import-time has no side effects)
# ---------------------------------------------------------------------------


def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=api_key)


def _youtube_client():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        raise RuntimeError("YOUTUBE_API_KEY is not set")
    return build_youtube_client("youtube", "v3", developerKey=api_key)


# ---------------------------------------------------------------------------
# Agent 1: classify_visual_intent
# ---------------------------------------------------------------------------


def classify_visual_intent(query: str) -> VisualIntent:
    """Classify whether the query has visual intent and which video format fits."""
    client = _gemini_client()
    response = _with_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompts.build_visual_intent_user_prompt(query),
        config=genai_types.GenerateContentConfig(
            system_instruction=prompts.VISUAL_INTENT_SYSTEM,
            response_mime_type="application/json",
            response_schema=VisualIntent,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, VisualIntent):
        return parsed
    return VisualIntent.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Agent 2: find_youtube_video
# ---------------------------------------------------------------------------

_ISO_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
_SHORT_THRESHOLD_SECS = 90   # videos ≤ this are treated as YouTube Shorts
_SHORT_BOOST = 3.0           # engagement multiplier for Shorts when format="short"


def _parse_duration_secs(iso: str) -> int:
    m = _ISO_DURATION_RE.match(iso)
    if not m:
        return 0
    h, mins, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mins * 60 + s


def _optimize_search_query(query: str, format: Literal["short", "long"]) -> str:
    """Rewrite the raw user query into a tight YouTube search string."""
    client = _gemini_client()
    response = _with_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompts.build_search_query_prompt(query, format),
    )
    optimized = (response.text or query).strip().strip('"').strip("'")
    return optimized if optimized else query


def find_youtube_video(
    query: str, format: Literal["short", "long"]
) -> Optional[YouTubeResult]:
    """Search YouTube and return the best video for the requested format.

    For "short": filter videoDuration=short (<4 min), rank by relevance + view count.
    For "long":  filter videoDuration=medium|long, rank by relevance + view count + recency.
    Returns None when nothing usable comes back.
    """
    youtube = _youtube_client()
    search_query = _optimize_search_query(query, format)
    # For short-form intent, appending "shorts" makes YouTube's search
    # surface actual Shorts rather than burying them under longer tutorials.
    if format == "short":
        search_query = search_query + " shorts"

    # videoDuration only accepts one value at a time; for "long" we ask
    # for "medium" first and fall back to "long".
    duration_filters = ["short"] if format == "short" else ["medium", "long"]

    candidates: list[dict] = []
    for duration in duration_filters:
        search = (
            youtube.search()
            .list(
                q=search_query,
                part="snippet",
                type="video",
                videoDuration=duration,
                maxResults=10,
                order="relevance",
                safeSearch="moderate",
            )
            .execute()
        )
        for item in search.get("items", []):
            vid = item["id"].get("videoId")
            if vid:
                candidates.append({"id": vid, "snippet": item["snippet"]})

    if not candidates:
        return None

    # Hydrate with statistics + contentDetails so we can rank by views.
    ids = ",".join(c["id"] for c in candidates)
    details = (
        youtube.videos()
        .list(part="statistics,contentDetails,snippet", id=ids)
        .execute()
    )

    enriched = []
    for v in details.get("items", []):
        stats = v.get("statistics", {})
        try:
            views = int(stats.get("viewCount", "0"))
        except ValueError:
            views = 0
        likes: Optional[int] = None
        if "likeCount" in stats:
            try:
                likes = int(stats["likeCount"])
            except ValueError:
                pass
        iso_dur = v["contentDetails"]["duration"]
        dur_secs = _parse_duration_secs(iso_dur)
        enriched.append(
            {
                "video_id": v["id"],
                "title": v["snippet"]["title"],
                "channel": v["snippet"]["channelTitle"],
                "duration": iso_dur,
                "duration_secs": dur_secs,
                "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                "published_at": v["snippet"].get("publishedAt", ""),
                "views": views,
                "likes": likes,
            }
        )

    if not enriched:
        return None

    # Ranking: compete only within the top 5 relevance results so a viral
    # unrelated video ranked #8 can't beat the #1 relevant result.
    #
    # For "short" format, genuine Shorts (≤ 90 s) get a 1.5× view-count boost
    # so they win over similar-quality regular videos — but a tutorial with 10×
    # more views still beats a low-engagement Short.
    top5 = enriched[:5]
    for item in top5:
        is_short_video = 0 < item["duration_secs"] <= _SHORT_THRESHOLD_SECS
        boost = _SHORT_BOOST if (format == "short" and is_short_video) else 1.0
        item["score"] = item["views"] * boost

    if format == "long":
        top5.sort(key=lambda x: (x["score"], x["published_at"]), reverse=True)
    else:
        top5.sort(key=lambda x: x["score"], reverse=True)

    top = top5[0]
    is_short = 0 < top["duration_secs"] <= _SHORT_THRESHOLD_SECS
    return YouTubeResult(
        video_id=top["video_id"],
        title=top["title"],
        channel=top["channel"],
        duration=top["duration"],
        duration_secs=top["duration_secs"],
        thumbnail_url=top["thumbnail_url"],
        video_url=f"https://www.youtube.com/watch?v={top['video_id']}",
        views=top["views"],
        likes=top["likes"],
        search_query_used=search_query,
        is_short=is_short,
    )


# ---------------------------------------------------------------------------
# Agent 3: find_timestamp
# ---------------------------------------------------------------------------


def find_timestamp(query: str, video_url: str) -> TimestampPick:
    """Ask Gemini to identify the start/end seconds inside the video that
    answer the query. Caps processing at the first 10 minutes via VideoMetadata
    to control cost.
    """
    client = _gemini_client()
    response = _with_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=genai_types.Content(
            parts=[
                genai_types.Part(
                    file_data=genai_types.FileData(file_uri=video_url),
                    video_metadata=genai_types.VideoMetadata(
                        start_offset="0s",
                        end_offset="600s",
                    ),
                ),
                genai_types.Part(text=prompts.build_timestamp_prompt(query)),
            ]
        ),
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TimestampPick,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, TimestampPick):
        return parsed
    return TimestampPick.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Plain text answer (used in BOTH Control and Treatment columns)
# ---------------------------------------------------------------------------


def generate_text_answer(query: str) -> str:
    client = _gemini_client()
    response = _with_retry(
        client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompts.build_text_answer_prompt(query),
    )
    return (response.text or "").strip()
