"""AI agents for Visual Intent Detection."""
from __future__ import annotations

import os
import re
import time
from functools import wraps
from typing import Optional

import requests
from google import genai
from google.genai import types
from pydantic import BaseModel


# ── Pydantic models ──────────────────────────────────────────────────────────

class VisualIntent(BaseModel):
    has_visual_intent: bool
    format: str          # "short" | "long" | "none"
    confidence: float
    rationale: str


class YouTubeResult(BaseModel):
    video_id: str
    video_url: str
    title: str
    channel: str
    views: int
    likes: Optional[int] = None
    duration_secs: int
    is_short: bool
    search_query_used: str


class TimestampPick(BaseModel):
    start_seconds: int
    end_seconds: int
    reasoning: str


class VerificationResult(BaseModel):
    is_good_match: bool
    verdict: str                              # "strong_match" | "partial_match" | "poor_match"
    confidence: float
    explanation: str
    timestamp_is_correct: bool                # does the proposed timestamp actually answer the query?
    timestamp_explanation: str                # one sentence on why / why not
    corrected_start_seconds: Optional[int] = None   # agent's suggested better start if timestamp is wrong


# ── Gemini client ────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=key)


# ── Retry helper ─────────────────────────────────────────────────────────────

def _retry(max_attempts: int = 3, base_delay: float = 2.0):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        time.sleep(base_delay * (2 ** attempt))
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ── ISO 8601 duration → seconds ──────────────────────────────────────────────

def _iso_to_secs(duration: str) -> int:
    h = int((re.search(r"(\d+)H", duration) or (None, 0))[1])
    m = int((re.search(r"(\d+)M", duration) or (None, 0))[1])
    s = int((re.search(r"(\d+)S", duration) or (None, 0))[1])
    return h * 3600 + m * 60 + s


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Text Answer
# ══════════════════════════════════════════════════════════════════════════════

@_retry()
def generate_text_answer(query: str) -> str:
    client = _get_client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are Gemini, Google's AI assistant. Answer the user's query "
                "thoroughly and clearly, formatted the way Gemini answers in the "
                "consumer app: use ## headings to organize sections when useful, "
                "**bold** for key terms, numbered lists for step-by-step "
                "instructions, bulleted lists for parallel items, and short "
                "explanatory paragraphs. Include a 'Pro Tip:' line when there is "
                "a useful nuance worth flagging. Do not suggest looking for a "
                "video — give the complete answer in text."
            ),
            # Generous budget — thinking tokens AND visible response tokens share
            # the output budget on Gemini 2.5/3 family models. 3000 leaves comfortable
            # room for thinking + a full structured answer with sections and lists.
            max_output_tokens=3000,
        ),
    )
    return response.text or ""


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Visual Intent Classifier
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_EXAMPLES = """
Examples (few-shot):
Q: "how do I replace a bike tire"        → has_visual_intent=true,  format="long",  confidence=0.97, rationale="Multi-step physical task requiring sight of hand positions and tools."
Q: "show me a kettlebell swing"          → has_visual_intent=true,  format="short", confidence=0.96, rationale="Form-check movement best shown in a 30–60 s clip."
Q: "what is the capital of France"       → has_visual_intent=false, format="none",  confidence=0.99, rationale="Single-word factual answer; no visual component."
Q: "how does a transistor work"          → has_visual_intent=true,  format="long",  confidence=0.85, rationale="Abstract electronics concept benefits from animated diagram explainer."
Q: "what causes inflation"               → has_visual_intent=false, format="none",  confidence=0.88, rationale="Economic concept explained well in text; no spatial component."
Q: "how to fold a fitted sheet"          → has_visual_intent=true,  format="short", confidence=0.96, rationale="Classic short physical demo — ideal for a 60–90 s clip."
Q: "tell me about sourdough"             → has_visual_intent=false, format="none",  confidence=0.72, rationale="Open-ended; text is sufficient."
Q: "how to do a pivot table in Excel"   → has_visual_intent=true,  format="long",  confidence=0.93, rationale="UI walkthrough requires seeing each step."
Q: "tie a Windsor knot"                  → has_visual_intent=true,  format="short", confidence=0.95, rationale="Precise hand movements need visual demonstration."
Q: "history of the Roman Empire"        → has_visual_intent=false, format="none",  confidence=0.91, rationale="Historical narrative; text or reading is the right medium."
"""


class _IntentSchema(BaseModel):
    has_visual_intent: bool
    format: str
    confidence: float
    rationale: str


@_retry()
def classify_visual_intent(query: str) -> VisualIntent:
    client = _get_client()
    prompt = (
        f"{_INTENT_EXAMPLES}\n\n"
        f'Now classify: Q: "{query}"\n\n'
        "Return JSON with has_visual_intent (bool), format ('short'|'long'|'none'), "
        "confidence (0–1), rationale (one sentence)."
    )
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_IntentSchema,
        ),
    )
    data = _IntentSchema.model_validate_json(resp.text)
    return VisualIntent(**data.model_dump())


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — YouTube Search
# ══════════════════════════════════════════════════════════════════════════════

def _optimize_query(query: str, fmt: str) -> str:
    """Build the YouTube search string.

    Strategy:
      - Pass-through for normal-length queries (≤10 words) — just append a
        format hint. No LLM call, no risk of over-compression.
      - Only call the LLM to summarize genuinely long, conversational queries
        (>10 words), and even then with low temperature and a validation
        guard so the model can't degenerate the query into garbage
        (e.g. "Windsor knot" → "Wind" — the failure we hit on 2026-05-14).
    """
    suffix = "shorts" if fmt == "short" else "tutorial"
    clean = query.rstrip("?.!,;: ").strip()

    # Pass-through for normal-length queries — no LLM, no risk.
    if len(clean.split()) <= 10:
        return f"{clean} {suffix}"

    # Long query — distill via LLM with guardrails.
    try:
        client = _get_client()
        resp = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=(
                f'Summarize this question into a 4–7 word YouTube search query. '
                f'Keep all important nouns. Append "{suffix}". '
                f'Question: "{clean}". Return ONLY the search string, no quotes.'
            ),
            config=types.GenerateContentConfig(
                max_output_tokens=30,
                temperature=0.2,  # low randomness — we want obedience, not creativity
            ),
        )
        rewritten = resp.text.strip().strip('"')
        # Validation guard: if the rewrite degenerates, fall back to the raw query.
        if len(rewritten.split()) >= 3:
            return rewritten
    except Exception:
        pass
    return f"{clean} {suffix}"


@_retry()
def find_youtube_videos(query: str, fmt: str, n: int = 3) -> list[YouTubeResult]:
    """Fetch top-N candidate videos for the query, sorted by score (best first).

    Score = views × (3 if format is short and video is short, else 1) — pure
    view-count weighted with a short-form boost. The carousel UI consumes this
    list; the legacy `find_youtube_video` below wraps this for the n=1 case.
    """
    yt_key = os.environ.get("YOUTUBE_API_KEY")
    if not yt_key:
        raise ValueError("YOUTUBE_API_KEY environment variable is not set.")

    search_q = _optimize_query(query, fmt)
    dur_filter = "short" if fmt == "short" else "any"

    # Search — YouTube returns up to 5 candidates ranked by its own relevance.
    search = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params={
            "part": "snippet",
            "q": search_q,
            "type": "video",
            "videoDuration": dur_filter,
            "maxResults": 5,
            "relevanceLanguage": "en",
            "key": yt_key,
        },
        timeout=10,
    )
    search.raise_for_status()
    items = search.json().get("items", [])
    if not items:
        return []

    # Fetch stats + duration for all candidates in one batch call.
    ids = [it["id"]["videoId"] for it in items]
    stats_resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={
            "part": "statistics,contentDetails,snippet",
            "id": ",".join(ids),
            "key": yt_key,
        },
        timeout=10,
    )
    stats_resp.raise_for_status()
    stats_map = {it["id"]: it for it in stats_resp.json().get("items", [])}

    # Score every candidate, keep the top N (by score, descending).
    scored: list[tuple[int, YouTubeResult]] = []
    for item in items:
        vid_id = item["id"]["videoId"]
        if vid_id not in stats_map:
            continue
        st = stats_map[vid_id]
        views = int(st["statistics"].get("viewCount", 0))
        likes_raw = st["statistics"].get("likeCount")
        likes = int(likes_raw) if likes_raw else None
        dur = _iso_to_secs(st["contentDetails"]["duration"])
        is_short = dur <= 90
        score = views * (3 if fmt == "short" and is_short else 1)
        scored.append((score, YouTubeResult(
            video_id=vid_id,
            video_url=f"https://www.youtube.com/watch?v={vid_id}",
            title=item["snippet"]["title"],
            channel=item["snippet"]["channelTitle"],
            views=views,
            likes=likes,
            duration_secs=dur,
            is_short=is_short,
            search_query_used=search_q,
        )))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:n]]


def find_youtube_video(query: str, fmt: str) -> Optional[YouTubeResult]:
    """Backward-compat wrapper: returns just the top-scored video, or None.

    Used by the Control column's keyword-triggered embed. The main pipeline
    calls `find_youtube_videos` directly to populate the carousel.
    """
    videos = find_youtube_videos(query, fmt, n=1)
    return videos[0] if videos else None


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — Timestamp Picker
# ══════════════════════════════════════════════════════════════════════════════

class _TsSchema(BaseModel):
    start_seconds: int
    end_seconds: int
    reasoning: str


@_retry()
def find_timestamp(query: str, video_url: str) -> TimestampPick:
    client = _get_client()
    prompt = (
        f'Watch this video (first 10 min only). Find the 30–120 second segment '
        f'that best answers: "{query}". '
        f"Return start_seconds, end_seconds, and a one-sentence reasoning."
    )
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Part(
                file_data=types.FileData(file_uri=video_url),
                # Cap ingestion at the first 10 minutes. Without this, Gemini
                # processes the full video — the dominant latency cost for long
                # ones. The prompt's "first 10 min only" wording does nothing on
                # its own; this is the API-level enforcement.
                video_metadata=types.VideoMetadata(end_offset="600s"),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_TsSchema,
        ),
    )
    data = _TsSchema.model_validate_json(resp.text)
    return TimestampPick(**data.model_dump())


# ══════════════════════════════════════════════════════════════════════════════
# Agent 5 — Verification Agent
# ══════════════════════════════════════════════════════════════════════════════

class _VerifySchema(BaseModel):
    is_good_match: bool
    verdict: str                          # "strong_match" | "partial_match" | "poor_match"
    confidence: float
    explanation: str
    timestamp_is_correct: bool            # does the clip at start_seconds actually answer the query?
    timestamp_explanation: str            # one sentence explaining the timestamp verdict
    corrected_start_seconds: Optional[int] = None   # agent's better start if timestamp is wrong


@_retry()
def verify_video_match(
    query: str,
    yt: YouTubeResult,
    ts: Optional[TimestampPick] = None,
) -> VerificationResult:
    """
    Quality-gate agent: watches the actual video at the proposed timestamp
    using Gemini's native video ingestion, then independently verifies:
      1. Is this the right video for the query?
      2. Is the proposed timestamp actually the right moment?
         If not, it suggests a corrected start time.
    """
    client = _get_client()

    def _fmt(secs: int) -> str:
        return f"{secs // 60}:{secs % 60:02d}"

    if ts:
        ts_meta = (
            f"\nProposed clip  : {_fmt(ts.start_seconds)} → {_fmt(ts.end_seconds)}"
            f" ({ts.start_seconds}s – {ts.end_seconds}s)"
            f"\nTimestamp note : {ts.reasoning}"
        )
        timestamp_instruction = (
            "Watch the video. Specifically, check what is happening at the proposed "
            f"clip ({_fmt(ts.start_seconds)} → {_fmt(ts.end_seconds)}). "
            "Decide whether that exact moment genuinely answers the query. "
            "If the content at that timestamp is an intro, unrelated topic, or not yet "
            "at the relevant moment, set timestamp_is_correct=false and provide a "
            "corrected_start_seconds that points to the better moment. "
            "If the clip is correct, set timestamp_is_correct=true and "
            "corrected_start_seconds=null."
        )
    else:
        ts_meta = "\nNo timestamp proposed — video will play from 0:00."
        timestamp_instruction = (
            "Watch the video. Identify the best start time (in seconds) for the query. "
            "Set timestamp_is_correct=false and corrected_start_seconds=<best start>. "
            "If the video genuinely starts at the right moment from 0:00, "
            "set timestamp_is_correct=true and corrected_start_seconds=null."
        )

    prompt = f"""You are a quality-verification agent for a video recommendation system.
Your job is to watch the actual video and verify both the video relevance AND the timestamp accuracy.

User query     : "{query}"
Video title    : "{yt.title}"
Channel        : "{yt.channel}"
Duration       : {yt.duration_secs}s{ts_meta}

TASK 1 — Video relevance:
Rate whether this video answers the query:
- "strong_match"  : clearly addresses the query; format fits the task.
- "partial_match" : related but may not fully cover the specific query.
- "poor_match"    : off-topic, misleading, or wrong format.

TASK 2 — Timestamp accuracy:
{timestamp_instruction}

Return JSON:
  is_good_match           : true if verdict is strong_match or partial_match
  verdict                 : "strong_match" | "partial_match" | "poor_match"
  confidence              : 0.0–1.0
  explanation             : one sentence on video relevance
  timestamp_is_correct    : bool
  timestamp_explanation   : one sentence on whether the timestamp points to the right moment
  corrected_start_seconds : int or null
"""
    # Watch the actual video — same capability used by the timestamp picker.
    # Cap at first 10 minutes (see find_timestamp for rationale).
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=[
            types.Part(
                file_data=types.FileData(file_uri=yt.video_url),
                video_metadata=types.VideoMetadata(end_offset="600s"),
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_VerifySchema,
        ),
    )
    data = _VerifySchema.model_validate_json(resp.text)
    return VerificationResult(**data.model_dump())
