"""Streamlit UI + orchestrator for the Visual Intent Detection prototype."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _load_secret(key: str) -> None:
    if os.environ.get(key):
        return
    try:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
    except Exception:
        pass


for _k in ("GEMINI_API_KEY", "YOUTUBE_API_KEY"):
    _load_secret(_k)

import agents  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_QUERY_LEN = 500

EXAMPLE_QUERIES = [
    ("🎬  Fold a fitted sheet", "how to fold a fitted sheet"),
    ("🔧  Change a car tire", "how to change a car tire"),
    ("💬  What causes inflation", "what causes inflation"),
    ("📡  How does GPS work", "how does GPS know where you are"),
]

BADGE_CSS = """
<style>
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-blue   { background: #4285F4; color: white; }
.badge-green  { background: #34A853; color: white; }
.badge-grey   { background: #5f6368; color: white; }
.badge-yellow { background: #FBBC04; color: #333; }

.col-control   { border-left: 3px solid #5f6368; padding-left: 16px; }
.col-treatment { border-left: 3px solid #4285F4; padding-left: 16px; }

.pipeline-card {
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    padding: 12px 14px;
    height: 100%;
}
.pipeline-card-active { border-top: 3px solid #4285F4; }
.pipeline-card-inactive { border-top: 3px solid #5f6368; opacity: 0.6; }
.pipeline-card-error { border-top: 3px solid #EA4335; }
</style>
"""

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults = {
        "query_input": "",
        "do_run": False,
        "last_result": None,
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def has_explicit_video_word(query: str) -> bool:
    return re.search(r"\bvideos?\b", query, re.IGNORECASE) is not None


def fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def fmt_ts(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def build_video_framing(yt, intent_format: str, ts) -> str:
    stats_bits = [f"{fmt_count(yt.views)} views"]
    if yt.likes:
        stats_bits.append(f"{fmt_count(yt.likes)} likes")
    stats = ", ".join(stats_bits)

    format_label = "YouTube Short" if yt.is_short else "video"
    head = f"I found a helpful {format_label} — **{yt.title}** by *{yt.channel}* ({stats})."

    if intent_format == "long" and ts:
        reasoning = ts.reasoning.strip().rstrip(".")
        if reasoning:
            reasoning = reasoning[0].lower() + reasoning[1:]
        return (
            f"{head} The clip below runs from **{fmt_ts(ts.start_seconds)}** "
            f"to **{fmt_ts(ts.end_seconds)}** — {reasoning}."
        )
    if intent_format == "long":
        return f"{head} Watch from the start for the full walkthrough."
    return f"{head} Quick demonstration below."


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(query: str) -> dict:
    trace: dict = {"query": query, "errors": {}, "timings_ms": {}}
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        text_future = pool.submit(agents.generate_text_answer, query)
        intent_future = pool.submit(agents.classify_visual_intent, query)

        try:
            text_answer = text_future.result()
        except Exception as e:
            text_answer = ""
            trace["errors"]["text_answer"] = repr(e)

        try:
            intent = intent_future.result()
            trace["visual_intent"] = intent.model_dump()
        except Exception as e:
            intent = None
            trace["errors"]["visual_intent"] = repr(e)

    trace["text_answer"] = text_answer

    youtube_result = None
    timestamp_pick = None
    if intent and intent.has_visual_intent and intent.format in ("short", "long"):
        try:
            youtube_result = agents.find_youtube_video(query, intent.format)
            trace["youtube"] = youtube_result.model_dump() if youtube_result else None
        except Exception as e:
            trace["errors"]["youtube"] = repr(e)

        # Run timestamp agent whenever the video is longer than 60 s.
        # Short-format queries can still have intros worth skipping; only
        # genuine Shorts (≤ 60 s) are short enough to play from 0:00.
        needs_timestamp = youtube_result and youtube_result.duration_secs > 60
        if needs_timestamp:
            try:
                timestamp_pick = agents.find_timestamp(query, youtube_result.video_url)
                trace["timestamp"] = timestamp_pick.model_dump()
                if not (0 <= timestamp_pick.start_seconds < timestamp_pick.end_seconds):
                    trace["errors"]["timestamp"] = "invalid range — rendering without jump"
                    timestamp_pick = None
            except Exception as e:
                trace["errors"]["timestamp"] = repr(e)
                timestamp_pick = None

    trace["timings_ms"]["total"] = round((time.perf_counter() - t_start) * 1000)
    trace["youtube_result"] = youtube_result
    trace["timestamp_pick"] = timestamp_pick
    trace["intent"] = intent
    return trace


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🔍 Visual Intent Detection")
        st.markdown("*A Gemini feature prototype*")
        st.divider()

        st.markdown("**The problem**")
        st.markdown(
            "Gemini only embeds a video when you literally type the word "
            "*\"video\"* — missing the much larger set of queries where a "
            "video would be the better answer."
        )

        st.markdown("**The fix**")
        st.markdown(
            "Three AI agents collaborate to detect *visual intent* and "
            "surface the right video at the right moment:"
        )
        st.markdown(
            "1. **Intent Classifier** — is video the right format? short or long?\n"
            "2. **YouTube Search** — finds the best video for that format\n"
            "3. **Timestamp Picker** — jumps to the exact segment that answers the query"
        )

        st.divider()
        st.markdown("**How to read the demo**")
        st.markdown(
            "- **Control (left):** how Gemini works today — text only, video only if you type *\"video\"*\n"
            "- **Treatment (right):** the new feature — video appears when the query needs it"
        )

        # Query history
        history = st.session_state.get("history", [])
        if history:
            st.divider()
            st.markdown("**Recent queries**")
            for h in history:
                icon = "🎥" if h["had_video"] else "📝"
                st.markdown(f"{icon} _{h['query'][:42]}_")
                st.caption(f"Intent: {h['intent']}")


# ---------------------------------------------------------------------------
# Render: chips + input
# ---------------------------------------------------------------------------


def render_search_ui() -> None:
    st.markdown(BADGE_CSS, unsafe_allow_html=True)

    st.markdown("**Try an example:**")
    chip_cols = st.columns(len(EXAMPLE_QUERIES))
    for col, (label, q) in zip(chip_cols, EXAMPLE_QUERIES):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state.query_input = q
                st.session_state.do_run = True
                st.rerun()

    st.markdown("")
    inp_col, btn_col = st.columns([6, 1])
    with inp_col:
        st.text_input(
            "query",
            key="query_input",
            max_chars=MAX_QUERY_LEN,
            placeholder="Or ask anything...",
            label_visibility="collapsed",
        )
    with btn_col:
        if st.button("Search", type="primary", use_container_width=True):
            st.session_state.do_run = True


# ---------------------------------------------------------------------------
# Render: columns
# ---------------------------------------------------------------------------


def render_control_column(query: str, text_answer: str) -> None:
    st.markdown('<div class="col-control">', unsafe_allow_html=True)
    st.markdown(
        f"### Control &nbsp; {badge('Gemini today', 'grey')}",
        unsafe_allow_html=True,
    )
    st.write(text_answer or "_(no answer)_")

    if has_explicit_video_word(query):
        st.caption("Keyword 'video' detected — embedding a video.")
        try:
            yt = agents.find_youtube_video(query, "short")
            if yt:
                st.video(yt.video_url)
            else:
                st.info("No video found.")
        except Exception as e:
            st.warning(f"YouTube lookup failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def _intent_badge(intent) -> str:
    if not intent or not intent.has_visual_intent:
        return badge("text only", "grey")
    if intent.format == "short":
        return badge("short video", "green")
    return badge("long video + timestamp", "blue")


def render_treatment_column(result: dict) -> None:
    intent = result.get("intent")
    yt = result.get("youtube_result")
    ts = result.get("timestamp_pick")

    st.markdown('<div class="col-treatment">', unsafe_allow_html=True)
    st.markdown(
        f"### Treatment &nbsp; {_intent_badge(intent)}",
        unsafe_allow_html=True,
    )
    st.write(result["text_answer"] or "_(no answer)_")

    if not intent:
        st.warning("Visual intent classifier failed — see agent pipeline.")
    elif not intent.has_visual_intent or intent.format == "none":
        st.caption(f"No visual intent detected ({intent.confidence:.0%} confidence). Text is the right medium.")
    elif not yt:
        st.info("No relevant video found.")
    else:
        st.markdown(build_video_framing(yt, intent.format, ts))
        if ts:
            src = (
                f"https://www.youtube.com/embed/{yt.video_id}"
                f"?start={ts.start_seconds}&end={ts.end_seconds}&autoplay=0&rel=0"
            )
            st.components.v1.iframe(src, height=340)
        else:
            st.video(yt.video_url)

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render: agent pipeline
# ---------------------------------------------------------------------------


def _pipeline_card(title: str, state: str, body_lines: list[str], detail: dict | None = None) -> None:
    """state: 'active' | 'inactive' | 'error'"""
    st.markdown(
        f'<div class="pipeline-card pipeline-card-{state}">',
        unsafe_allow_html=True,
    )
    icons = {"active": "✅", "inactive": "—", "error": "❌"}
    st.markdown(f"**{icons[state]} {title}**")
    for line in body_lines:
        st.markdown(line)
    if detail:
        with st.expander("JSON", expanded=False):
            st.json(detail)
    st.markdown("</div>", unsafe_allow_html=True)


def render_agent_pipeline(result: dict) -> None:
    with st.expander("🔬  Agent pipeline", expanded=False):
        intent_data = result.get("visual_intent")
        yt_data = result.get("youtube")
        ts_data = result.get("timestamp")
        errors = result["errors"]

        c1, arrow1, c2, arrow2, c3 = st.columns([4, 0.4, 4, 0.4, 4])

        with c1:
            if intent_data:
                _pipeline_card(
                    "Intent Classifier",
                    "active",
                    [
                        f"Format: `{intent_data['format'].upper()}`",
                        f"Confidence: `{intent_data['confidence']:.0%}`",
                        f"_{intent_data['rationale']}_",
                    ],
                    intent_data,
                )
            else:
                _pipeline_card(
                    "Intent Classifier",
                    "error",
                    [f"`{errors.get('visual_intent', 'unknown error')[:120]}`"],
                )

        with arrow1:
            st.markdown(
                "<div style='text-align:center;font-size:22px;padding-top:28px'>→</div>",
                unsafe_allow_html=True,
            )

        with c2:
            if yt_data:
                dur = yt_data.get("duration_secs", 0)
                dur_label = f"{dur}s {'· ✨ Short' if yt_data.get('is_short') else ''}"
                _pipeline_card(
                    "YouTube Search",
                    "active",
                    [
                        f"_{yt_data['title']}_",
                        f"Channel: {yt_data['channel']}",
                        f"Duration: `{dur_label.strip()}`",
                        f"Search used: `{yt_data.get('search_query_used', '—')}`",
                        f"{fmt_count(yt_data['views'])} views"
                        + (f" · {fmt_count(yt_data['likes'])} likes" if yt_data.get("likes") else ""),
                    ],
                    yt_data,
                )
            elif "youtube" in errors:
                _pipeline_card(
                    "YouTube Search",
                    "error",
                    [f"`{errors['youtube'][:120]}`"],
                )
            else:
                _pipeline_card(
                    "YouTube Search",
                    "inactive",
                    ["Not needed — text intent detected."],
                )

        with arrow2:
            st.markdown(
                "<div style='text-align:center;font-size:22px;padding-top:28px'>→</div>",
                unsafe_allow_html=True,
            )

        with c3:
            if ts_data:
                _pipeline_card(
                    "Timestamp Picker",
                    "active",
                    [
                        f"Clip: `{fmt_ts(ts_data['start_seconds'])}` → `{fmt_ts(ts_data['end_seconds'])}`",
                        f"_{ts_data['reasoning']}_",
                    ],
                    ts_data,
                )
            elif "timestamp" in errors:
                _pipeline_card(
                    "Timestamp Picker",
                    "error",
                    [f"`{errors['timestamp'][:120]}`"],
                )
            else:
                _pipeline_card(
                    "Timestamp Picker",
                    "inactive",
                    ["Not needed — short-form or text intent."],
                )

        st.divider()
        st.caption(f"Total latency: **{result['timings_ms']['total']} ms**")


# ---------------------------------------------------------------------------
# How it works accordion
# ---------------------------------------------------------------------------


def render_how_it_works() -> None:
    with st.expander("📖  How this prototype works", expanded=False):

        st.markdown("### Overview")
        st.markdown(
            "This prototype simulates a proposed Gemini feature called **Visual Intent Detection**. "
            "Today, Gemini only embeds a YouTube video when you literally type the word *'video'* in your query. "
            "This feature adds three AI agents that collaborate to detect *when a video would be the better answer* — "
            "even when you never asked for one."
        )

        st.divider()

        st.markdown("### API connections")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Google Gemini API**")
            st.markdown(
                "- Model: `gemini-2.5-flash` \n"
                "- Used by: Intent Classifier, Timestamp Picker, text answer generator, and search query optimizer \n"
                "- Accessed via the `google-genai` Python SDK \n"
                "- Returns structured JSON using Gemini's native JSON mode with a typed schema"
            )
        with col2:
            st.markdown("**YouTube Data API v3**")
            st.markdown(
                "- Used by: YouTube Search agent \n"
                "- Calls the `search.list` endpoint to find relevant videos \n"
                "- Calls `videos.list` to fetch view counts, like counts, and exact duration \n"
                "- Free tier: 10,000 units/day (~100 searches)"
            )

        st.divider()

        st.markdown("### The three agents")

        st.markdown("#### 1 · Intent Classifier")
        st.markdown(
            "The first agent reads the user's query and decides whether video is the right medium. "
            "It uses a **few-shot prompt** — a set of 8 labeled examples built into the prompt — "
            "to teach Gemini the pattern before it sees the live query. "
            "It returns four fields:"
        )
        st.markdown(
            "| Field | What it means |\n"
            "|---|---|\n"
            "| `has_visual_intent` | True/False — should a video be shown at all? |\n"
            "| `format` | `short` (demo ≤ ~90 s), `long` (explainer, multi-step), or `none` (text only) |\n"
            "| `confidence` | 0–1 float. How certain the model is. Shown as a % in the pipeline card. |\n"
            "| `rationale` | One sentence explaining the decision. |\n"
        )
        st.markdown(
            "**What confidence % means:** a score of `0.95` means the model is 95% certain this query needs a video. "
            "Scores below ~0.70 indicate the query is ambiguous — the model is making a judgment call. "
            "This score does not come from a probability distribution over labels; "
            "it is Gemini's self-reported certainty from the structured output."
        )

        st.markdown("#### 2 · YouTube Search Agent")
        st.markdown(
            "Only fires when `has_visual_intent = True`. Before hitting YouTube, "
            "a quick Gemini call **rewrites the raw user query** into an optimized YouTube search string "
            "(e.g. *'how to fold a fitted sheet?'* → *'fold fitted sheet shorts'*). "
            "This avoids returning irrelevant viral videos that happen to match loose keywords."
        )
        st.markdown(
            "**Short vs. long video selection:**\n"
            "- `format = short` → searches `videoDuration=short` (under 4 min) and appends *'shorts'* to the query to surface YouTube Shorts. "
            "Genuine Shorts (≤ 90 s) get a **3× engagement score boost** so they beat similar-quality longer videos — "
            "but a tutorial with 10× more views still wins over a low-engagement Short.\n"
            "- `format = long` → searches `videoDuration=medium` and `long`, ranks by views + recency.\n\n"
            "Ranking always competes only within the **top 5 relevance results** returned by YouTube, "
            "so a viral but unrelated video ranked #8 can never beat the most relevant result."
        )

        st.markdown("#### 3 · Timestamp Picker")
        st.markdown(
            "Fires whenever the selected video is **longer than 60 seconds** — regardless of whether "
            "the intent was classified as short or long. "
            "It sends the YouTube URL directly to Gemini using a native video ingestion feature "
            "(`file_data` with a YouTube URI), capped at the first 10 minutes to control cost. "
            "Gemini watches the video and returns the exact `start_seconds` and `end_seconds` "
            "of the segment that best answers the query, plus a one-sentence reasoning. "
            "The YouTube embed is then loaded with `?start=X&end=Y` parameters so playback jumps "
            "directly to that moment and stops when the clip ends."
        )

        st.divider()

        st.markdown("### Decision tree — when does a video appear?")
        st.code(
            "User submits query\n"
            "│\n"
            "├─ Intent Classifier runs (always)\n"
            "│   ├─ has_visual_intent = False  →  TEXT ONLY in both columns\n"
            "│   └─ has_visual_intent = True\n"
            "│       ├─ YouTube Search runs\n"
            "│       │   ├─ No results found  →  TEXT ONLY + 'No relevant video found'\n"
            "│       │   └─ Video found\n"
            "│       │       ├─ Video ≤ 60 s  →  EMBED from 0:00 (genuine Short)\n"
            "│       │       └─ Video > 60 s  →  Timestamp Picker runs\n"
            "│       │           ├─ Valid timestamp returned  →  EMBED at start_seconds → end_seconds\n"
            "│       │           └─ Invalid / error           →  EMBED from 0:00 (safe fallback)\n"
            "│\n"
            "Control column: video only if the word 'video' appears in the query (today's behavior)",
            language="text",
        )

        st.divider()

        st.markdown("### Parallelism & latency")
        st.markdown(
            "To minimize wait time, the **text answer** and the **Intent Classifier** run in parallel "
            "using Python's `concurrent.futures.ThreadPoolExecutor`. "
            "The YouTube and Timestamp agents run sequentially after, since each depends on the previous result. "
            "Total latency is typically **4–12 seconds** depending on video length and API load. "
            "On transient Gemini overload (503 errors), each call automatically retries up to 3 times "
            "with exponential backoff (2 s → 4 s → 8 s) before surfacing an error."
        )


def main() -> None:
    st.set_page_config(
        page_title="Visual Intent Detection — Gemini Prototype",
        page_icon="🔍",
        layout="wide",
    )

    _init_state()
    render_sidebar()

    st.title("Visual Intent Detection")
    st.caption(
        "A prototype showing how Gemini could automatically detect when a query "
        "needs a video answer — and embed the right clip at the right moment."
    )
    st.divider()

    render_search_ui()

    # Validate and run
    if st.session_state.do_run:
        st.session_state.do_run = False
        query = st.session_state.query_input.strip()
        if not query:
            st.warning("Please enter a query.")
        elif len(query) > MAX_QUERY_LEN:
            st.error(f"Query too long ({len(query)} chars). Max is {MAX_QUERY_LEN}.")
        else:
            with st.spinner("Running agents…"):
                result = run_pipeline(query)
            st.session_state.last_result = result

            # Update history (newest first, cap at 5)
            intent = result.get("intent")
            had_video = result.get("youtube_result") is not None
            intent_label = (
                f"{intent.format} · {intent.confidence:.0%}" if intent else "error"
            )
            st.session_state.history = (
                [{"query": query, "intent": intent_label, "had_video": had_video}]
                + st.session_state.history
            )[:5]

    # Render last result if available
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.divider()
        left, right = st.columns(2, gap="large")
        with left:
            render_control_column(result["query"], result["text_answer"])
        with right:
            render_treatment_column(result)

        st.divider()
        render_agent_pipeline(result)
        render_how_it_works()


if __name__ == "__main__":
    main()
