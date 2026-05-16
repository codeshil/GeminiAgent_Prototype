"""Streamlit UI — Visual Intent Detection · Gemini VideoSense"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Secret loader (Streamlit Cloud compatible) ───────────────────────────────
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

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="VideoSense · Visual Intent Detection",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
    --g-blue: #4285F4;
    --g-blue-deep: #1a73e8;
    --g-text: #1f1f1f;
    --g-text-2: #5f6368;
    --g-text-3: #80868b;
    --g-surface: #ffffff;
    --g-surface-tint: #f0f4f9;
    --g-surface-soft: #f8fafd;
    --g-border: #e8eaed;
    --g-gradient: linear-gradient(120deg, #4285F4 0%, #9168C0 50%, #D96570 100%);
    --g-shadow: 0 1px 2px rgba(60,64,67,0.08), 0 4px 14px rgba(60,64,67,0.06);
    --g-shadow-sm: 0 1px 3px rgba(60,64,67,0.12);
}

/* ── Reset & base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--g-surface); }
::selection { background: rgba(66,133,244,0.18); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.75rem !important; padding-bottom: 2.5rem !important; max-width: 1180px; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #dadce0; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #bdc1c6; }

/* ══ Page header ══ */
.vs-header { display: flex; align-items: center; gap: 8px; margin-bottom: 0.15rem; }
.vs-logo { width: 26px; height: 26px; flex-shrink: 0; }
.vs-wordmark { font-size: 1.05rem; font-weight: 600; color: var(--g-text); letter-spacing: -0.2px; }
.vs-greeting {
    font-size: 2.1rem; font-weight: 500; letter-spacing: -0.8px; margin: 0.5rem 0 0 0;
    color: #1f1f1f;
    background: var(--g-gradient);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.vs-subtitle { font-size: 0.92rem; color: var(--g-text-2); margin: 0.35rem 0 1.5rem 0; line-height: 1.55; }
.chip-label { font-size: 0.78rem; color: var(--g-text-3); font-weight: 500; margin: 0.7rem 0 0.5rem 0; }

/* ══ Search bar ══ */
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border: 1px solid var(--g-border) !important;
    border-radius: 999px !important;
    padding: 0.7rem 1.25rem !important;
    background: var(--g-surface-tint) !important;
    transition: border-color 0.18s, box-shadow 0.18s, background 0.18s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--g-blue) !important;
    box-shadow: 0 0 0 4px rgba(66,133,244,0.12) !important;
    background: #ffffff !important;
}

/* ══ Buttons — example chips (secondary) & search (primary) ══ */
.stButton > button,
[data-testid="stFormSubmitButton"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 999px !important;
    transition: all 0.18s !important;
}
.stButton > button[kind="secondary"] {
    background: var(--g-surface-tint) !important;
    border: 1px solid transparent !important;
    color: #3c4043 !important;
    font-size: 0.84rem !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #e8f0fe !important;
    color: var(--g-blue-deep) !important;
    border: 1px solid transparent !important;
}
.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] button {
    background-color: #4285F4 !important;
    background-image: var(--g-gradient) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: var(--g-shadow-sm) !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.06);
    box-shadow: 0 3px 12px rgba(145,104,192,0.32) !important;
    color: #ffffff !important;
}

/* ══ Example chip row ══ */
.chip-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem; }

/* ══ Result columns ══ */
.col-card {
    background: var(--g-surface);
    border: 1px solid var(--g-border);
    border-radius: 24px;
    padding: 1.4rem 1.5rem;
    height: 100%;
    box-shadow: var(--g-shadow);
}
.col-card-treatment { border-top: 3px solid var(--g-blue); }
.col-card-control   { border-top: 3px solid #dadce0; }

.col-title {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.7px;
    text-transform: uppercase; color: var(--g-text-3);
    margin: 0 0 0.8rem 0;
    display: flex; align-items: center; gap: 7px;
}
.col-label-treatment { color: var(--g-blue); }
.col-label-control   { color: var(--g-text-3); }
.col-dot { width: 8px; height: 8px; border-radius: 999px; display: inline-block; flex-shrink: 0; }
.col-dot-treatment { background: var(--g-blue); }
.col-dot-control   { background: #bdc1c6; }

/* ══ Badges ══ */
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 11px; border-radius: 999px;
    font-size: 0.76rem; font-weight: 600; letter-spacing: 0.1px;
    font-family: 'DM Sans', sans-serif;
    white-space: nowrap;
}
.badge-blue   { background: #e8f0fe; color: #1967d2; }
.badge-green  { background: #e6f4ea; color: #137333; }
.badge-yellow { background: #fef7e0; color: #b06000; }
.badge-red    { background: #fce8e6; color: #c5221f; }
.badge-grey   { background: #f1f3f4; color: #5f6368; }
.badge-purple { background: #f3e8fd; color: #8430ce; }

/* ══ Verification strip ══ */
.verify-strip {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 11px 15px; border-radius: 14px;
    margin: 11px 0; font-size: 0.86rem;
    border: 1px solid transparent;
}
.verify-strong  { background: #e6f4ea; border-color: #ceead6; color: #137333; }
.verify-partial { background: #fef7e0; border-color: #feefc3; color: #b06000; }
.verify-poor    { background: #fce8e6; border-color: #fad2cf; color: #c5221f; }
.verify-icon    { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.verify-text b  { display: block; font-weight: 600; }

/* ══ Pipeline cards ══ */
.pipe-grid { display: grid; grid-template-columns: 1fr 28px 1fr 28px 1fr 28px 1fr; gap: 6px; align-items: start; margin: 0.5rem 0; }
.pipe-arrow { display: flex; align-items: center; justify-content: center; padding-top: 20px; color: #bdc1c6; font-size: 1.1rem; }
.pipe-card {
    background: var(--g-surface-soft); border: 1px solid var(--g-border);
    border-radius: 16px; padding: 13px 15px;
    font-size: 0.82rem;
}
.pipe-card-active   { border-top: 3px solid var(--g-blue); }
.pipe-card-inactive { opacity: 0.55; }
.pipe-card-error    { border-top: 3px solid #ea4335; background: #fce8e6; }
.pipe-card-title    { font-weight: 600; font-size: 0.84rem; color: var(--g-text); margin-bottom: 6px; }
.pipe-card-body     { color: var(--g-text-2); line-height: 1.55; }
.pipe-mono {
    font-family: 'DM Sans', sans-serif; font-weight: 600;
    font-size: 0.78rem; color: #3c4043;
    background: #e8eaed; padding: 1px 7px; border-radius: 6px;
}

/* ══ Banners ══ */
.err-banner, .warn-banner, .info-banner {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 13px 16px; border-radius: 14px;
    margin: 7px 0; font-size: 0.86rem;
    border: 1px solid transparent;
}
.err-banner  { border-color: #fad2cf; background: #fce8e6; color: #c5221f; }
.warn-banner { border-color: #feefc3; background: #fef7e0; color: #b06000; }
.info-banner { border-color: #d2e3fc; background: #e8f0fe; color: #1967d2; }
.banner-icon { font-size: 1rem; flex-shrink: 0; }
.banner-text b { font-weight: 600; display: block; }

/* ══ Video meta ══ */
.video-meta { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 8px 0; }
.stat-pill {
    font-size: 0.78rem; color: var(--g-text-2);
    background: var(--g-surface-tint); padding: 3px 10px;
    border-radius: 999px; font-family: 'DM Sans', sans-serif;
}

/* ══ Latency chip ══ */
.latency-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-family: 'DM Sans', sans-serif; font-size: 0.78rem; font-weight: 500;
    color: var(--g-text-2); background: var(--g-surface-tint);
    border: 1px solid var(--g-border); padding: 4px 12px;
    border-radius: 999px; margin-top: 8px;
}

/* ══ Sidebar — light Gemini surface ══ */
[data-testid="stSidebar"] { background: var(--g-surface-tint) !important; border-right: 1px solid var(--g-border); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li { color: var(--g-text-2) !important; font-size: 0.875rem !important; line-height: 1.6 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: var(--g-text) !important; }
[data-testid="stSidebar"] em { color: var(--g-text-3) !important; }
[data-testid="stSidebar"] hr { border-color: var(--g-border) !important; }

/* ══ History item ══ */
.hist-item {
    display: flex; align-items: center; gap: 8px;
    padding: 7px 11px; border-radius: 10px;
    background: #ffffff; border: 1px solid var(--g-border);
    margin-bottom: 5px; font-size: 0.8rem;
}
.hist-icon { flex-shrink: 0; }
.hist-query { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--g-text-2); }

/* ══ Expander & status ══ */
[data-testid="stExpander"] { border-radius: 16px !important; border: 1px solid var(--g-border) !important; box-shadow: var(--g-shadow-sm); }
[data-testid="stExpander"] summary { font-weight: 600 !important; }
[data-testid="stStatusWidget"], [data-testid="stStatus"] { border-radius: 16px !important; }

/* ══ Misc ══ */
hr { border-color: var(--g-border) !important; }
iframe { border-radius: 14px; }
.text-answer { font-size: 0.95rem; line-height: 1.7; color: #3c4043; padding: 0.25rem 0; }

/* ══ Motion ══ */
@keyframes vs-rise {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes vs-flow {
    0%   { background-position: 0% 0; }
    100% { background-position: 200% 0; }
}
@keyframes vs-pulse {
    0%, 100% { transform: scale(0.82); opacity: 0.7; }
    50%      { transform: scale(1.12); opacity: 1; }
}
.col-card  { animation: vs-rise 0.45s ease-out both; }
.pipe-card { animation: vs-rise 0.40s ease-out both; }

/* ══ Branded "thinking" shimmer ══ */
.vs-thinking {
    display: flex; align-items: center; gap: 11px;
    padding: 12px 16px; border-radius: 14px;
    background: var(--g-surface-tint); border: 1px solid var(--g-border);
    font-size: 0.9rem; color: var(--g-text-2); font-weight: 500;
    margin-bottom: 10px;
}
.vs-thinking-dot {
    width: 13px; height: 13px; flex-shrink: 0; border-radius: 999px;
    background: var(--g-gradient);
    animation: vs-pulse 1.2s ease-in-out infinite;
}
.vs-thinking-bar {
    flex: 1; height: 4px; border-radius: 999px;
    background: linear-gradient(90deg, #4285F4, #9168C0, #D96570, #9168C0, #4285F4);
    background-size: 200% 100%;
    animation: vs-flow 1.6s linear infinite;
}
@media (prefers-reduced-motion: reduce) {
    .col-card, .pipe-card, .vs-thinking-dot, .vs-thinking-bar { animation: none !important; }
    .vs-thinking-bar { background-position: 0 0; }
}

/* ══ Video carousel ══ */
.carousel-intro {
    font-size: 0.92rem; line-height: 1.5;
    color: var(--g-text-2);
    margin: 14px 0 10px 0;
}
.carousel-intro b {
    color: var(--g-text);
    font-weight: 600;
    display: block; margin-bottom: 2px;
}
.carousel-sub { color: var(--g-text-3); font-size: 0.86rem; }
.carousel-counter {
    text-align: center;
    font-size: 0.84rem; color: var(--g-text-2); font-weight: 500;
    padding-top: 10px;
}
.carousel-dots {
    display: inline-flex; gap: 6px;
    margin-left: 10px; vertical-align: middle;
}
.carousel-dot {
    width: 7px; height: 7px;
    border-radius: 999px;
    background: #dadce0;
    display: inline-block;
    transition: background 0.2s;
}
.carousel-dot-active { background: var(--g-blue); }
</style>
"""

# ── Constants ────────────────────────────────────────────────────────────────

MAX_QUERY_LEN = 500

EXAMPLES = [
    ("🛏️ Fold a fitted sheet", "how to fold a fitted sheet"),
    ("🔧 Clean car undercarriage", "how to clean the undercarriage of a car"),
    ("💬 What causes inflation", "what causes inflation"),
    ("🏋️ Deadlift form", "how to deadlift with proper form"),
    ("🪴 Repot a plant", "how to repot a houseplant"),
    ("📡 How GPS works", "how does GPS know where you are"),
]


# ── Session init ─────────────────────────────────────────────────────────────

def _init() -> None:
    defaults = {
        "query_input": "",
        "do_run": False,
        "last_result": None,
        "history": [],
        "current_video_idx": 0,    # which video the carousel is showing
        "session_log": [],         # full per-query agent trace (for professor review)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Formatters ───────────────────────────────────────────────────────────────

def fmt_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)

def fmt_ts(s: int) -> str:
    return f"{s // 60}:{s % 60:02d}"

def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'

def has_video_word(q: str) -> bool:
    return bool(re.search(r"\bvideos?\b", q, re.IGNORECASE))


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
            '<svg width="22" height="22" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
            '<defs><linearGradient id="geminiGradSb" x1="0%" y1="0%" x2="100%" y2="100%">'
            '<stop offset="0%" stop-color="#4285F4"/>'
            '<stop offset="50%" stop-color="#9168C0"/>'
            '<stop offset="100%" stop-color="#D96570"/>'
            '</linearGradient></defs>'
            '<path d="M12 0C12 6.627 6.627 12 0 12c6.627 0 12 5.373 12 12 '
            '0-6.627 5.373-12 12-12C17.373 12 12 6.627 12 0z" fill="url(#geminiGradSb)"/>'
            '</svg>'
            '<span style="font-size:1.3rem;font-weight:700;">VideoSense</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("*Visual Intent Detection*")
        st.divider()

        st.markdown("**The problem**")
        st.markdown(
            "Gemini only embeds a video when you type the word *'video'* — "
            "missing the far larger set of queries where a video is the better answer."
        )
        st.markdown("**The solution**")
        st.markdown(
            "Five AI agents collaborate:\n"
            "1. **Intent Classifier** — video needed? short or long?\n"
            "2. **YouTube Search** — find the best video\n"
            "3. **Timestamp Picker** — jump to the right moment\n"
            "4. **Verification Agent** — quality-gate the result ✨"
        )
        st.divider()
        st.markdown("**Reading the demo**")
        st.markdown(
            "**Left** = Gemini today (text-only unless you type 'video')  \n"
            "**Right** = New feature (video auto-detected from intent)"
        )

        history = st.session_state.get("history", [])
        if history:
            st.divider()
            st.markdown("**Recent queries**")
            for h in history:
                icon = "🎥" if h["had_video"] else "💬"
                st.markdown(
                    f'<div class="hist-item">'
                    f'<span class="hist-icon">{icon}</span>'
                    f'<span class="hist-query">{h["query"][:40]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # API key status
        st.divider()
        g_ok = bool(os.environ.get("GEMINI_API_KEY"))
        y_ok = bool(os.environ.get("YOUTUBE_API_KEY"))
        st.markdown("**API Status**")
        st.markdown(
            f"{'🟢' if g_ok else '🔴'} Gemini API  \n"
            f"{'🟢' if y_ok else '🔴'} YouTube API"
        )
        if not g_ok or not y_ok:
            st.warning("Add API keys in `.streamlit/secrets.toml` or as env vars.")


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH UI
# ══════════════════════════════════════════════════════════════════════════════

def render_search_ui() -> None:
    # A chip click queues its query here; apply it before the input widget is built
    # (writing to a widget's state key after instantiation raises StreamlitAPIException).
    if "pending_chip" in st.session_state:
        st.session_state.query_input = st.session_state.pop("pending_chip")

    # Header
    # Use the browser's timezone — Streamlit Cloud servers run UTC, so without this
    # the greeting would be wrong for non-UTC viewers (e.g. evening in LA -> "morning").
    try:
        tz = ZoneInfo(st.context.timezone)
    except Exception:
        tz = ZoneInfo("America/Los_Angeles")
    hour = datetime.now(tz).hour
    greeting = (
        "Good morning" if hour < 12
        else "Good afternoon" if hour < 18
        else "Good evening"
    )
    st.markdown(
        '<div class="vs-header">'
        '<svg class="vs-logo" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="geminiGrad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#4285F4"/>'
        '<stop offset="50%" stop-color="#9168C0"/>'
        '<stop offset="100%" stop-color="#D96570"/>'
        '</linearGradient></defs>'
        '<path d="M12 0C12 6.627 6.627 12 0 12c6.627 0 12 5.373 12 12 '
        '0-6.627 5.373-12 12-12C17.373 12 12 6.627 12 0z" fill="url(#geminiGrad)"/>'
        '</svg>'
        '<span class="vs-wordmark">Gemini VideoSense</span>'
        '</div>'
        f'<h1 class="vs-greeting">{greeting}, what should we try today?</h1>'
        '<p class="vs-subtitle">Visual Intent Detection Prototype · '
        'Auto-detects when your query needs a video — and finds the right moment.</p>',
        unsafe_allow_html=True,
    )

    with st.form("search_form", clear_on_submit=False):
        c1, c2 = st.columns([7, 1])
        with c1:
            st.text_input(
                "query",
                key="query_input",
                max_chars=MAX_QUERY_LEN,
                placeholder="Ask anything — e.g. 'how do I tie a Windsor knot'",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Search →", type="primary", use_container_width=True)
        if submitted:
            st.session_state.do_run = True

    # Example chips — below the search bar, Gemini-style
    st.markdown('<p class="chip-label">Try an example</p>', unsafe_allow_html=True)
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES))
    for col, (label, q) in zip(cols, EXAMPLES):
        with col:
            if st.button(label, use_container_width=True, key=f"chip_{q}"):
                st.session_state.pending_chip = q
                st.session_state.do_run = True
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION STRIP
# ══════════════════════════════════════════════════════════════════════════════

def render_verification(v) -> None:
    if v is None:
        st.markdown(
            '<div class="verify-partial">'
            '<span class="verify-icon">⚠️</span>'
            '<div class="verify-text"><b>Verification unavailable</b>Could not run quality check.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    if v.verdict == "strong_match":
        cls, icon = "verify-strong", "✅"
        label = "Verified match"
    elif v.verdict == "partial_match":
        cls, icon = "verify-partial", "⚠️"
        label = "Partial match"
    else:
        cls, icon = "verify-poor", "❌"
        label = "Poor match"

    conf_pct = f"{v.confidence:.0%}"
    st.markdown(
        f'<div class="verify-strip {cls}">'
        f'<span class="verify-icon">{icon}</span>'
        f'<div class="verify-text">'
        f'<b>{label} · {conf_pct} confidence</b>'
        f'{v.explanation}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ERROR BANNERS
# ══════════════════════════════════════════════════════════════════════════════

def render_error_banners(result: dict) -> None:
    """Top-level banners for *global* pipeline failures only.

    Per-video failures (timestamp, verification) are surfaced in each video's
    verification badge area + the Agent Pipeline expander, not here. And the
    "no videos found" outcome is shown inline in render_treatment as a warning —
    no top-level banner needed for that legitimate (non-error) case.
    """
    errors = result.get("errors", {})

    # Text answer failed
    if "text_answer" in errors or not result.get("text_answer", "").strip():
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>Text answer failed</b>'
            'The Gemini API may be overloaded. Wait a moment and try again. '
            f'<code>{str(errors.get("text_answer",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Intent classifier failed
    if "visual_intent" in errors:
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>Intent Classifier failed</b>'
            'Could not determine whether video is appropriate. '
            f'<code>{str(errors.get("visual_intent",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # YouTube API itself errored (network / key / quota — different from "no results")
    if "youtube" in errors:
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>YouTube Search failed</b>'
            'Check that your YOUTUBE_API_KEY is valid and the YouTube Data API v3 '
            f'is enabled in your Google Cloud project. '
            f'<code>{str(errors.get("youtube",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# RESULT COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def render_control(query: str, text_answer: str) -> None:
    st.markdown('<div class="col-card col-card-control">', unsafe_allow_html=True)
    st.markdown(
        '<p class="col-title col-label-control">'
        '<span class="col-dot col-dot-control"></span>CONTROL · Gemini Today</p>',
        unsafe_allow_html=True,
    )

    # Text answer — st.markdown parses markdown (headings, lists, bold) properly.
    # Don't wrap in raw HTML: Streamlit's markdown parser doesn't process markdown
    # syntax inside HTML blocks, so headers/lists would render as literal characters.
    if text_answer:
        st.markdown(text_answer)
    else:
        st.markdown(
            '<div class="info-banner"><span class="banner-icon">ℹ️</span>'
            '<div class="banner-text"><b>No text answer</b>API did not return a response.</div></div>',
            unsafe_allow_html=True,
        )

    # Video only if user typed "video"
    if has_video_word(query):
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-banner"><span class="banner-icon">🔍</span>'
            '<div class="banner-text"><b>Keyword detected</b>'
            "You typed 'video' — Gemini embeds a result.</div></div>",
            unsafe_allow_html=True,
        )
        try:
            yt = agents.find_youtube_video(query, "short")
            if yt:
                st.video(yt.video_url)
            else:
                st.markdown(
                    '<div class="warn-banner"><span class="banner-icon">⚠️</span>'
                    '<div class="banner-text"><b>No video found</b>Try a different query.</div></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.markdown(
                f'<div class="err-banner"><span class="banner-icon">❌</span>'
                f'<div class="banner-text"><b>YouTube lookup failed</b><code>{e}</code></div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-banner"><span class="banner-icon">💬</span>'
            '<div class="banner-text">No video — you didn\'t type the word \'video\'.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)


def _intent_badge_html(intent) -> str:
    if not intent or not intent.has_visual_intent:
        return badge("text only", "grey")
    if intent.format == "short":
        return badge("▶ short video", "green")
    return badge("▶ long video + timestamp", "blue")


def _render_carousel_controls(idx: int, total: int) -> None:
    """Prev/next arrows + 'Video X of Y' counter + dot indicators."""
    col_l, col_c, col_r = st.columns([1, 4, 1])
    with col_l:
        if st.button("◀", key="carousel_prev",
                     disabled=(idx == 0), use_container_width=True):
            st.session_state.current_video_idx = max(0, idx - 1)
            st.rerun()
    with col_c:
        dots = "".join(
            f'<span class="carousel-dot{" carousel-dot-active" if i == idx else ""}"></span>'
            for i in range(total)
        )
        st.markdown(
            f'<div class="carousel-counter">Video {idx + 1} of {total}'
            f'<span class="carousel-dots">{dots}</span></div>',
            unsafe_allow_html=True,
        )
    with col_r:
        if st.button("▶", key="carousel_next",
                     disabled=(idx == total - 1), use_container_width=True):
            st.session_state.current_video_idx = min(total - 1, idx + 1)
            st.rerun()


def _render_video_card(current: dict) -> None:
    """Per-video block: verification badge → correction notice → meta → title → embed.

    Reused by both the carousel (for the currently-selected video) and the
    single-video path (long-form strong match → no carousel UI).
    """
    yt = current["yt"]
    ts = current.get("ts")
    verif = current.get("verif")

    # Verification strip — the trust signal
    render_verification(verif)

    # Timestamp correction notice (only when verification overrode the picker)
    if current.get("timestamp_corrected"):
        orig = fmt_ts(current.get("timestamp_original_start", 0))
        new = fmt_ts(ts.start_seconds) if ts else "0:00"
        st.markdown(
            f'<div class="verify-strip verify-partial">'
            f'<span class="verify-icon">🔧</span>'
            f'<div class="verify-text">'
            f'<b>Timestamp corrected by verification agent</b>'
            f'Original start {orig} was not the relevant moment — adjusted to {new}.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # Video metadata row
    dur_label = f"{fmt_ts(yt.duration_secs)}" + (" · Short" if yt.is_short else "")
    likes_pill = f'<span class="stat-pill">👍 {fmt_n(yt.likes)}</span>' if yt.likes else ""
    st.markdown(
        f'<div class="video-meta">'
        f'<span class="stat-pill">👁 {fmt_n(yt.views)} views</span>'
        f'{likes_pill}'
        f'<span class="stat-pill">⏱ {dur_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Framing sentence + embed
    if ts:
        st.markdown(
            f"**{yt.title}** by *{yt.channel}*  \n"
            f"Playing {fmt_ts(ts.start_seconds)} → {fmt_ts(ts.end_seconds)} — {ts.reasoning}"
        )
        src = (
            f"https://www.youtube.com/embed/{yt.video_id}"
            f"?start={ts.start_seconds}&end={ts.end_seconds}&autoplay=0&rel=0"
        )
        st.iframe(src, height=330)
    else:
        st.markdown(f"**{yt.title}** by *{yt.channel}*")
        st.video(yt.video_url)


def _render_carousel_block(videos: list, intro_mode: str) -> None:
    """Render the carousel: intro + current video card + arrow controls.

    intro_mode:
      - 'video_first' → short-form, "Best answered with a video" framing
      - 'long_weak'   → long-form with weak top match, "alternates" framing
      - 'fallback'    → generic framing for other cases
    """
    total = len(videos)
    idx = st.session_state.get("current_video_idx", 0)
    idx = max(0, min(idx, total - 1))  # defensive clamp
    current = videos[idx]

    if intro_mode == "video_first":
        intro_main = "Best answered with a video"
        intro_sub = (
            f"Here are {total} picks — tap the arrows to browse."
            if total > 1 else "Here's the best pick."
        )
    elif intro_mode == "long_weak":
        intro_main = f"{total} videos to compare"
        intro_sub = "Top match wasn't a strong fit — browse alternates with the arrows."
    else:  # 'fallback'
        intro_main = f"{total} video{'s' if total > 1 else ''} that might help"
        intro_sub = (
            "Showing the best match — tap the arrows to browse the others."
            if total > 1 else "Showing the only match for this query."
        )

    st.markdown(
        f'<div class="carousel-intro">'
        f'<b>{intro_main}</b>'
        f'<span class="carousel-sub">{intro_sub}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _render_video_card(current)

    if total > 1:
        _render_carousel_controls(idx, total)


def render_treatment(result: dict) -> None:
    intent = result.get("intent")
    videos = result.get("videos", [])
    text_answer = result.get("text_answer", "")

    st.markdown('<div class="col-card col-card-treatment">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="col-title col-label-treatment">'
        f'<span class="col-dot col-dot-treatment"></span>TREATMENT · Visual Intent Detection '
        f'&nbsp;{_intent_badge_html(intent)}</p>',
        unsafe_allow_html=True,
    )

    # Intent failed
    if not intent:
        if text_answer:
            st.markdown(text_answer)
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>Intent classifier failed</b>'
            'Cannot determine whether to show a video. See pipeline below.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Text-only intent
    if not intent.has_visual_intent or intent.format == "none":
        if text_answer:
            st.markdown(text_answer)
        conf = f"{intent.confidence:.0%}"
        st.markdown(
            f'<div class="info-banner"><span class="banner-icon">💬</span>'
            f'<div class="banner-text"><b>Text is the right medium</b>'
            f'{intent.rationale} ({conf} confident)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # No videos found
    if not videos:
        if text_answer:
            st.markdown(text_answer)
        st.markdown(
            '<div class="warn-banner"><span class="banner-icon">⚠️</span>'
            '<div class="banner-text"><b>No relevant video found</b>'
            "Try rephrasing or adding 'how to'.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Smart placement (text ↔ video order) ────────────────────────────────
    # format == "short" (e.g. "tie a Windsor knot") → video is the answer; show
    # it FIRST so the user doesn't scroll past walls of text.
    # format == "long" (e.g. "how does GPS work") → text context first, video below.
    video_first = (intent.format == "short")

    # ── Smart exposure: how many videos do we expose? ──────────────────────
    # short        → always 3, browsing is natural for quick visual clips
    # long+strong  → just 1, AI is confident; alternatives would be noise
    # long+weak    → 3, surface alternatives so user can recover from a bad #1
    # (all 3 are still pre-computed in parallel — we just decide what to render)
    verdict_1 = (
        videos[0]["verif"].verdict
        if videos[0].get("verif") is not None else None
    )
    if intent.format == "short":
        videos_to_show = videos
        intro_mode = "video_first"
    elif verdict_1 == "strong_match":
        videos_to_show = [videos[0]]
        intro_mode = None  # signal: render single video, no carousel chrome
    else:
        videos_to_show = videos
        intro_mode = "long_weak"

    # Text answer ABOVE when text-first
    if not video_first and text_answer:
        st.markdown(text_answer)

    if intro_mode is None:
        # Confident long-form pick: just the video card, no intro, no controls.
        # The verification badge itself ("Verified match · 95% confidence") IS
        # the framing here.
        _render_video_card(videos_to_show[0])
    else:
        _render_carousel_block(videos_to_show, intro_mode=intro_mode)

    # Text answer BELOW when video-first
    if video_first and text_answer:
        st.markdown(
            '<hr style="margin:18px 0 12px 0;border:none;'
            'border-top:1px solid var(--g-border);">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:0.85rem;color:var(--g-text-3);font-weight:500;'
            'margin:0 0 6px 0;">More detail in text</p>',
            unsafe_allow_html=True,
        )
        st.markdown(text_answer)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _pcard(title: str, state: str, lines: list[str], detail: dict | None = None) -> None:
    icon = {"active": "✅", "inactive": "—", "error": "❌"}.get(state, "—")
    body_html = "".join(f"<div>{l}</div>" for l in lines)
    st.markdown(
        f'<div class="pipe-card pipe-card-{state}">'
        f'<div class="pipe-card-title">{icon} {title}</div>'
        f'<div class="pipe-card-body">{body_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if detail:
        with st.expander("JSON", expanded=False):
            st.json(detail)


def render_pipeline(result: dict) -> None:
    with st.expander("Agent pipeline", expanded=False):
        intent_d = result.get("visual_intent")
        errors_global = dict(result.get("errors", {}))
        videos = result.get("videos", [])
        total = len(videos)

        # Show diagnostic for the currently-displayed carousel video
        idx = st.session_state.get("current_video_idx", 0)
        idx = max(0, min(idx, total - 1)) if total else 0
        current = videos[idx] if total else None

        yt_d    = current["yt"].model_dump() if current and current.get("yt") else None
        ts_d    = current["ts"].model_dump() if current and current.get("ts") else None
        verif_d = current["verif"].model_dump() if current and current.get("verif") else None
        errors  = dict(errors_global)
        if current:
            errors.update(current.get("errors", {}))

        if total > 1:
            st.markdown(
                f'<p style="font-size:0.84rem;color:var(--g-text-2);margin:0 0 8px 0;">'
                f'Showing pipeline for <b>video {idx + 1} of {total}</b> — '
                f'switch via the carousel arrows above to inspect each one.'
                f'</p>',
                unsafe_allow_html=True,
            )

        # 4-card pipeline: intent → youtube → timestamp → verification
        c1, a1, c2, a2, c3, a3, c4 = st.columns([3, 0.4, 3, 0.4, 3, 0.4, 3])
        arrow = '<div class="pipe-arrow">→</div>'

        with c1:
            if intent_d:
                _pcard(
                    "Intent Classifier", "active",
                    [
                        f'Format: <span class="pipe-mono">{intent_d["format"].upper()}</span>',
                        f'Confidence: <span class="pipe-mono">{intent_d["confidence"]:.0%}</span>',
                        f'<em>{intent_d["rationale"]}</em>',
                    ],
                    intent_d,
                )
            else:
                _pcard("Intent Classifier", "error",
                       [f'<code>{str(errors.get("visual_intent",""))[:100]}</code>'])

        with a1:
            st.markdown(arrow, unsafe_allow_html=True)

        with c2:
            if yt_d:
                dur_s = yt_d.get("duration_secs", 0)
                _pcard(
                    "YouTube Search", "active",
                    [
                        f'<em>{yt_d["title"]}</em>',
                        f'<span class="pipe-mono">{yt_d["channel"]}</span>',
                        f'Duration: <span class="pipe-mono">{fmt_ts(dur_s)}</span>',
                        f'Query: <span class="pipe-mono">{yt_d.get("search_query_used","—")}</span>',
                        f'Views: <span class="pipe-mono">{fmt_n(yt_d["views"])}</span>',
                    ],
                    yt_d,
                )
            elif "youtube" in errors:
                _pcard("YouTube Search", "error",
                       [f'<code>{str(errors["youtube"])[:100]}</code>'])
            else:
                _pcard("YouTube Search", "inactive", ["Text intent — not needed."])

        with a2:
            st.markdown(arrow, unsafe_allow_html=True)

        with c3:
            if ts_d:
                _pcard(
                    "Timestamp Picker", "active",
                    [
                        f'Clip: <span class="pipe-mono">{fmt_ts(ts_d["start_seconds"])} → {fmt_ts(ts_d["end_seconds"])}</span>',
                        f'<em>{ts_d["reasoning"]}</em>',
                    ],
                    ts_d,
                )
            elif "timestamp" in errors:
                _pcard("Timestamp Picker", "error",
                       [f'<code>{str(errors["timestamp"])[:100]}</code>'])
            else:
                _pcard("Timestamp Picker", "inactive", ["Video ≤ 20s or text-only — not needed."])

        with a3:
            st.markdown(arrow, unsafe_allow_html=True)

        with c4:
            if verif_d:
                verdict_map = {
                    "strong_match":  ("active",   "✅ Strong match"),
                    "partial_match": ("inactive",  "⚠️ Partial match"),
                    "poor_match":    ("error",     "❌ Poor match"),
                }
                state, label = verdict_map.get(verif_d["verdict"], ("inactive", verif_d["verdict"]))
                ts_ok   = verif_d.get("timestamp_is_correct", True)
                ts_icon = "✅" if ts_ok else "🔧"
                ts_exp  = verif_d.get("timestamp_explanation", "—")
                corrected = verif_d.get("corrected_start_seconds")
                corrected_line = (
                    f'Corrected start: <span class="pipe-mono">{fmt_ts(corrected)}</span>'
                    if corrected is not None and not ts_ok else ""
                )
                _pcard(
                    "Verification Agent", state,
                    [
                        f'Verdict: <span class="pipe-mono">{label}</span>',
                        f'Confidence: <span class="pipe-mono">{verif_d["confidence"]:.0%}</span>',
                        f'<em>{verif_d["explanation"]}</em>',
                        f'{ts_icon} Timestamp: <em>{ts_exp}</em>',
                        corrected_line,
                    ],
                    verif_d,
                )
            elif "verification" in errors:
                _pcard("Verification Agent", "error",
                       [f'<code>{str(errors["verification"])[:100]}</code>'])
            else:
                _pcard("Verification Agent", "inactive", ["No video to verify."])

        st.markdown(
            f'<div class="latency-chip">⚡ Total latency: {result["timings_ms"]["total"]} ms</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SESSION LOG (for professor / grader review)
# ══════════════════════════════════════════════════════════════════════════════

def _append_session_log(query: str, trace: dict) -> None:
    """Build a structured log entry from a completed pipeline trace and append
    to st.session_state['session_log']. Runs AFTER run_pipeline returns — zero
    impact on pipeline latency. Captures: query, total time, intent decision,
    per-video metadata + verdict + timestamp, and any errors.
    """
    intent_obj = trace.get("intent")
    videos = trace.get("videos", [])

    # Format timestamp in the viewer's local timezone (falls back to LA)
    try:
        tz = ZoneInfo(st.context.timezone)
    except Exception:
        tz = ZoneInfo("America/Los_Angeles")

    entry = {
        "timestamp": datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "query": query,
        "total_ms": trace.get("timings_ms", {}).get("total", 0),
        "intent": {
            "format": intent_obj.format,
            "confidence": round(intent_obj.confidence, 3),
            "has_visual_intent": intent_obj.has_visual_intent,
            "rationale": intent_obj.rationale,
        } if intent_obj else None,
        "videos": [
            {
                "rank": i + 1,
                "title": v["yt"].title,
                "channel": v["yt"].channel,
                "views": v["yt"].views,
                "duration_secs": v["yt"].duration_secs,
                "video_url": v["yt"].video_url,
                "search_query_used": v["yt"].search_query_used,
                "verdict": v["verif"].verdict if v.get("verif") else None,
                "verdict_confidence": round(v["verif"].confidence, 3) if v.get("verif") else None,
                "verdict_explanation": v["verif"].explanation if v.get("verif") else None,
                "timestamp_picked": (
                    {"start": v["ts"].start_seconds, "end": v["ts"].end_seconds}
                    if v.get("ts") else None
                ),
                "timestamp_corrected": v.get("timestamp_corrected", False),
                "errors": list(v.get("errors", {}).keys()),
            }
            for i, v in enumerate(videos)
        ],
        "global_errors": list(trace.get("errors", {}).keys()),
    }
    st.session_state.setdefault("session_log", []).append(entry)


def _render_log_entry(entry: dict) -> None:
    intent = entry.get("intent")
    intent_str = (
        f"`{intent['format']}` ({intent['confidence']:.0%})"
        if intent else "intent failed"
    )
    videos = entry.get("videos", [])

    st.markdown(
        f"**`{entry['timestamp']}`** &nbsp;·&nbsp; *“{entry['query']}”*  \n"
        f"&nbsp;&nbsp;**{entry['total_ms']/1000:.1f}s** &nbsp;·&nbsp; "
        f"intent: {intent_str} &nbsp;·&nbsp; "
        f"{len(videos)} video{'s' if len(videos) != 1 else ''}"
    )

    if entry.get("global_errors"):
        st.markdown(
            f"&nbsp;&nbsp;❌ **Pipeline errors:** {', '.join(entry['global_errors'])}"
        )

    verdict_emoji = {
        "strong_match":  "✅",
        "partial_match": "⚠️",
        "poor_match":    "❌",
    }
    for v in videos:
        emoji = verdict_emoji.get(v.get("verdict"), "—")
        conf = v.get("verdict_confidence")
        conf_str = f" {conf:.0%}" if conf is not None else ""
        ts_str = ""
        if v.get("timestamp_picked"):
            tp = v["timestamp_picked"]
            ts_str = f" · clip {tp['start']}s→{tp['end']}s"
        if v.get("timestamp_corrected"):
            ts_str += " (corrected by verification)"
        err_str = f" · errors: {', '.join(v['errors'])}" if v.get("errors") else ""
        title = v["title"][:60] + ("…" if len(v["title"]) > 60 else "")
        st.markdown(
            f"&nbsp;&nbsp;{emoji} **#{v['rank']}** {title} "
            f"&nbsp;·&nbsp; *{v['channel']}*{conf_str}{ts_str}{err_str}"
        )


def render_session_log() -> None:
    """Bottom-of-page expander that shows every query run this session and how
    each agent performed. Designed for the professor / grader to inspect the
    multi-agent system end-to-end. Downloadable as JSON for permanent record."""
    log = st.session_state.get("session_log", [])
    if not log:
        return

    label = (
        f"📋 Session log — {len(log)} "
        f"{'query' if len(log) == 1 else 'queries'} so far"
    )
    with st.expander(label, expanded=False):
        st.caption(
            "Per-query trace of what each agent (Intent Classifier · YouTube "
            "Search · Timestamp Picker · Verification) decided. Most recent "
            "first. Persists for this browser session only — download as JSON "
            "for a permanent record."
        )

        log_json = json.dumps(log, indent=2, default=str)
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇ Download log (JSON)",
            data=log_json,
            file_name=f"videosense_log_{ts_slug}.json",
            mime="application/json",
            key="dl_session_log",
        )
        st.markdown("---")

        for entry in reversed(log):  # newest first
            _render_log_entry(entry)
            st.markdown("---")


def render_how_it_works() -> None:
    with st.expander("How this prototype works", expanded=False):
        st.markdown("### Five-agent pipeline")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Models & APIs**")
            st.markdown(
                "- **Gemini** (`gemini-3.1-flash-lite`) — powers the Text Answer, "
                "Intent Classifier, Timestamp Picker, Verification Agent, and the "
                "conditional LLM rewrite inside YouTube Search.\n"
                "- **YouTube Data API v3** — `search.list` + `videos.list` for "
                "candidate retrieval and video metadata."
            )
        with col2:
            st.markdown("**Parallelism**")
            st.markdown(
                "Text Answer + Intent Classifier run in parallel. YouTube Search "
                "returns the **top 3 candidates** ranked by view-count × short-form "
                "boost, then Timestamp Picker + Verification run for **all 3 videos "
                "in parallel** via `ThreadPoolExecutor`. This is ~3× the API cost "
                "but the same wall-clock latency as processing one — the user can "
                "then flip through the carousel locally without triggering new calls."
            )

        st.markdown("### Decision tree")
        st.code(
            "Query submitted\n"
            "├─ [parallel] Text Answer  +  Intent Classifier\n"
            "│\n"
            "├─ has_visual_intent = False  →  text answer only on both columns\n"
            "└─ has_visual_intent = True\n"
            "    └─ YouTube Search  (raw query for ≤10 words; LLM rewrite for >10)\n"
            "        ├─ No results  →  warning banner; no embed\n"
            "        └─ Top 3 videos found  →  parallel per-video pipeline:\n"
            "            ├─ duration ≤ 20s  →  embed from 0:00 (skip Timestamp Picker)\n"
            "            └─ duration > 20s  →  Timestamp Picker\n"
            "                                  ├─ Valid range  →  embed at start→end\n"
            "                                  └─ Invalid      →  embed from 0:00\n"
            "            └─ [always] Verification Agent\n"
            "                ├─ Video relevance  →  strong / partial / poor match badge\n"
            "                └─ Timestamp accuracy  →  can override Picker's start_seconds\n"
            "    └─ Carousel: arrow buttons flip between the 3 fully-analysed videos\n"
            "                 (instant — no new API calls; idx tracked in session_state)",
            language="text",
        )

        st.markdown("### Verification Agent")
        st.markdown(
            "After a video is found, a separate Gemini call **watches the actual "
            "video** (via `Part.from_uri` / `file_data` ingestion — the same "
            "capability the Timestamp Picker uses, not just title/metadata) and "
            "judges two things independently:\n\n"
            "1. **Video relevance** — does this video genuinely answer the query? "
            "Returns a verdict (`strong_match` / `partial_match` / `poor_match`) "
            "and confidence score, surfaced as a coloured strip above the embed.\n"
            "2. **Timestamp accuracy** — does the Picker's proposed clip actually "
            "point to the right moment? If not, the agent returns "
            "`corrected_start_seconds` and the pipeline overrides the Picker's "
            "choice with the verification agent's correction (visible in the UI "
            "as a 🔧 'Corrected start' notice).\n\n"
            "A `poor_match` verdict does **not** suppress the video — by design, "
            "it surfaces as a transparent trust signal so the user can judge "
            "for themselves rather than having results silently hidden."
        )

        st.markdown("### Search-query optimization")
        st.markdown(
            "For typical queries (**≤10 words**), the raw user query is sent to "
            "YouTube verbatim with a `shorts` / `tutorial` suffix appended — no "
            "LLM call, no risk of over-compression. Only **long, conversational "
            "queries (>10 words)** go through an LLM rewrite step, and even then "
            "with temperature `0.2` for obedience and a validation guard that "
            "falls back to the raw query if the rewrite degenerates to fewer "
            "than 3 words. This was hardened after an earlier failure where the "
            "optimizer compressed *'How do I tie a Windsor knot'* to *'Wind'*, "
            "which YouTube happily matched against the Naruto ending song."
        )

        st.markdown("### Error and fallback states")
        st.markdown(
            "| Failure | Fallback |\n"
            "|---|---|\n"
            "| Gemini overloaded (503) | Retries ×3 with exponential backoff (2s, 4s between attempts) |\n"
            "| Intent classifier fails | Treatment column shows text answer + error banner |\n"
            "| YouTube API error | Error banner with diagnosis; text answer still shown |\n"
            "| No video results | Warning banner; no embed |\n"
            "| Timestamp out of range | Plays from 0:00; warning shown in pipeline |\n"
            "| Verification fails | Warning strip in UI; does not block video display |"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _process_one_video(query: str, yt) -> dict:
    """Run Timestamp Picker + Verification for ONE candidate video.

    Returns a per-video dict containing the YouTube result, the (possibly
    verification-corrected) timestamp, the verification result, any errors,
    and bookkeeping fields. Safe to call in parallel — each call is independent
    and only touches its own dict.
    """
    video = {
        "yt": yt,
        "ts": None,
        "verif": None,
        "errors": {},
        "timestamp_corrected": False,
        "timestamp_original_start": 0,
    }

    # Timestamp Picker — only for videos >20s; shorter clips just play from 0:00.
    if yt.duration_secs > 20:
        try:
            ts = agents.find_timestamp(query, yt.video_url)
            if 0 <= ts.start_seconds < ts.end_seconds:
                video["ts"] = ts
            else:
                video["errors"]["timestamp"] = "invalid range — playing from start"
        except Exception as e:
            video["errors"]["timestamp"] = repr(e)

    # Verification — always runs once we have a video, regardless of timestamp.
    try:
        verif = agents.verify_video_match(query, yt, video["ts"])
        video["verif"] = verif

        # Apply timestamp correction if verification disagrees with Picker.
        if (
            not verif.timestamp_is_correct
            and verif.corrected_start_seconds is not None
            and verif.corrected_start_seconds >= 0
        ):
            ts = video["ts"]
            original_start = ts.start_seconds if ts else 0
            corrected = verif.corrected_start_seconds
            original_end = ts.end_seconds if ts else (corrected + 90)
            new_end = max(original_end, corrected + 30)
            video["ts"] = agents.TimestampPick(
                start_seconds=corrected,
                end_seconds=new_end,
                reasoning=verif.timestamp_explanation,
            )
            video["timestamp_corrected"] = True
            video["timestamp_original_start"] = original_start
    except Exception as e:
        video["errors"]["verification"] = repr(e)

    return video


def run_pipeline(query: str) -> dict:
    trace: dict = {"query": query, "errors": {}, "timings_ms": {}, "videos": []}
    t0 = time.perf_counter()

    shimmer = st.empty()
    shimmer.markdown(
        '<div class="vs-thinking">'
        '<span class="vs-thinking-dot"></span>'
        '<span>VideoSense agents working…</span>'
        '<span class="vs-thinking-bar"></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.status("Running VideoSense agents…", expanded=True) as status:

        # Step 1 — parallel: text answer + intent classifier
        st.write("🔍 Running Intent Classifier and generating text answer…")
        with ThreadPoolExecutor(max_workers=2) as pool:
            tf = pool.submit(agents.generate_text_answer, query)
            inf = pool.submit(agents.classify_visual_intent, query)
            try:
                text_answer = tf.result()
            except Exception as e:
                text_answer = ""
                trace["errors"]["text_answer"] = repr(e)
            try:
                intent = inf.result()
                trace["visual_intent"] = intent.model_dump()
            except Exception as e:
                intent = None
                trace["errors"]["visual_intent"] = repr(e)

        trace["text_answer"] = text_answer

        if intent:
            fmt_map = {"short": "short-form video", "long": "long-form video", "none": "text only"}
            st.write(
                f"✅ Intent: **{fmt_map.get(intent.format, intent.format)}** "
                f"· {intent.confidence:.0%} — _{intent.rationale}_"
            )
        else:
            st.write("❌ Intent Classifier failed — see pipeline for details")

        # Step 2 — YouTube Search: fetch top 3 candidates
        videos: list[dict] = []
        if intent and intent.has_visual_intent and intent.format in ("short", "long"):
            kind = "Shorts" if intent.format == "short" else "videos"
            st.write(f"🎬 Searching YouTube for the top 3 {kind}…")
            try:
                yts = agents.find_youtube_videos(query, intent.format, n=3)
                if yts:
                    titles = ", ".join(f"_{y.title[:35]}_" for y in yts[:3])
                    st.write(f"✅ Found {len(yts)} candidate{'s' if len(yts) > 1 else ''}: {titles}")
                else:
                    st.write("⚠️ No relevant videos found — try rephrasing")
            except Exception as e:
                trace["errors"]["youtube"] = repr(e)
                yts = []
                st.write("❌ YouTube Search failed — check pipeline for details")

            # Step 3 — parallel: Timestamp Picker + Verification for EACH video.
            # Wall-clock latency stays roughly the same as processing one video
            # (limited by the slowest), but the API cost scales 3×. The carousel
            # lets the user flip between fully-analysed alternatives instantly.
            if yts:
                st.write(
                    f"⏱️🔎 Analyzing all {len(yts)} videos in parallel "
                    "(Timestamp Picker + Verification per video)…"
                )
                with ThreadPoolExecutor(max_workers=len(yts)) as pool:
                    futures = [pool.submit(_process_one_video, query, yt) for yt in yts]
                    videos = [f.result() for f in futures]

                # Summarise per-video verdicts in the status log
                vmap = {"strong_match": "✅", "partial_match": "⚠️", "poor_match": "❌"}
                verdicts = [
                    vmap.get(v["verif"].verdict, "—") if v.get("verif") else "—"
                    for v in videos
                ]
                st.write(
                    "✅ Per-video verdicts: "
                    + "   ".join(f"#{i+1} {v}" for i, v in enumerate(verdicts))
                )
        else:
            if intent:
                st.write("💬 Text-only query — no video needed")

        trace["videos"] = videos
        trace["intent"] = intent
        trace["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)

        status.update(
            label=f"✓ Done in {trace['timings_ms']['total']} ms",
            state="complete",
            expanded=False,
        )

    shimmer.empty()
    return trace


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init()
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()
    render_search_ui()

    # Validate + run
    if st.session_state.do_run:
        st.session_state.do_run = False
        query = st.session_state.query_input.strip()
        if not query:
            st.warning("Please enter a query first.")
        elif len(query) > MAX_QUERY_LEN:
            st.error(f"Query too long ({len(query)} chars). Max {MAX_QUERY_LEN}.")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.error("❌ GEMINI_API_KEY is not set. Add it to `.streamlit/secrets.toml`.")
        elif not os.environ.get("YOUTUBE_API_KEY"):
            st.error("❌ YOUTUBE_API_KEY is not set. Add it to `.streamlit/secrets.toml`.")
        else:
            # Reset carousel position before running pipeline — the new query may
            # have fewer videos than the previous one, and we always want to start
            # on the highest-ranked match anyway.
            st.session_state.current_video_idx = 0
            result = run_pipeline(query)
            st.session_state.last_result = result
            # Sidebar history (lightweight summary, last 5 queries)
            st.session_state.history = (
                [{
                    "query": query,
                    "had_video": bool(result.get("videos")),
                    "intent": result["intent"].format if result.get("intent") else "error",
                }]
                + st.session_state.get("history", [])
            )[:5]
            # Full session log (every query, full per-agent trace — for grader review)
            _append_session_log(query, result)

    # Render last result
    if st.session_state.last_result:
        result = st.session_state.last_result
        st.divider()
        render_error_banners(result)
        left, right = st.columns(2, gap="large")
        with left:
            render_control(result["query"], result["text_answer"])
        with right:
            render_treatment(result)
        st.divider()
        render_pipeline(result)

    st.divider()
    render_session_log()
    render_how_it_works()

    st.markdown(
        '<p style="text-align:center;font-size:0.75rem;color:#94a3b8;margin-top:1rem;">'
        'Aashil Soni · Forum Sanjanwala &nbsp;·&nbsp; Visual Intent Detection Prototype'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
