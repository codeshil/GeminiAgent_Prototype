"""Streamlit UI — Visual Intent Detection · Gemini VideoSense"""
from __future__ import annotations

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
    errors = result.get("errors", {})
    intent = result.get("intent")
    yt = result.get("youtube_result")

    if "text_answer" in errors or not result.get("text_answer", "").strip():
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>Text answer failed</b>'
            'The Gemini API may be overloaded. Wait a moment and try again. '
            f'<code>{str(errors.get("text_answer",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if "visual_intent" in errors:
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>Intent Classifier failed</b>'
            'Could not determine whether video is appropriate. '
            f'<code>{str(errors.get("visual_intent",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if intent and intent.has_visual_intent and "youtube" in errors:
        st.markdown(
            '<div class="err-banner"><span class="banner-icon">❌</span>'
            '<div class="banner-text"><b>YouTube Search failed</b>'
            'Check that your YOUTUBE_API_KEY is valid and the YouTube Data API v3 '
            f'is enabled in your Google Cloud project. '
            f'<code>{str(errors.get("youtube",""))[:80]}</code>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    if intent and intent.has_visual_intent and not yt and "youtube" not in errors:
        st.markdown(
            '<div class="warn-banner"><span class="banner-icon">⚠️</span>'
            '<div class="banner-text"><b>No video found</b>'
            "Try rephrasing — add 'how to' or be more specific."
            '</div></div>',
            unsafe_allow_html=True,
        )

    if "timestamp" in errors:
        st.markdown(
            '<div class="warn-banner"><span class="banner-icon">⚠️</span>'
            '<div class="banner-text"><b>Timestamp Picker failed</b>'
            'Playing from the start instead. '
            f'<code>{str(errors.get("timestamp",""))[:80]}</code>'
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

    # Text answer
    if text_answer:
        st.markdown(f'<div class="text-answer">{text_answer}</div>', unsafe_allow_html=True)
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


def render_treatment(result: dict) -> None:
    intent = result.get("intent")
    yt     = result.get("youtube_result")
    ts     = result.get("timestamp_pick")
    verif  = result.get("verification")

    st.markdown('<div class="col-card col-card-treatment">', unsafe_allow_html=True)
    st.markdown(
        f'<p class="col-title col-label-treatment">'
        f'<span class="col-dot col-dot-treatment"></span>TREATMENT · Visual Intent Detection '
        f'&nbsp;{_intent_badge_html(intent)}</p>',
        unsafe_allow_html=True,
    )

    # Text answer
    if result.get("text_answer"):
        st.markdown(
            f'<div class="text-answer">{result["text_answer"]}</div>',
            unsafe_allow_html=True,
        )

    # Intent failed
    if not intent:
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
        conf = f"{intent.confidence:.0%}"
        st.markdown(
            f'<div class="info-banner"><span class="banner-icon">💬</span>'
            f'<div class="banner-text"><b>Text is the right medium</b>'
            f'{intent.rationale} ({conf} confident)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # No video found
    if not yt:
        st.markdown(
            '<div class="warn-banner"><span class="banner-icon">⚠️</span>'
            '<div class="banner-text"><b>No relevant video found</b>'
            "Try rephrasing or adding 'how to'.</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── Video found ──────────────────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)

    # Verification strip — video relevance
    render_verification(verif)

    # Timestamp correction notice (shown when verification overrode the picker)
    if result.get("timestamp_corrected"):
        orig = fmt_ts(result.get("timestamp_original_start", 0))
        new  = fmt_ts(ts.start_seconds) if ts else "0:00"
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

    # Framing sentence
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
        yt_d     = result.get("youtube")
        ts_d     = result.get("timestamp")
        verif_d  = result.get("verification_raw")
        errors   = result.get("errors", {})

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
                _pcard("Timestamp Picker", "inactive", ["Short-form or text — not needed."])

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

def render_how_it_works() -> None:
    with st.expander("How this prototype works", expanded=False):
        st.markdown("### Five-agent pipeline")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**APIs used**")
            st.markdown(
                "- **Gemini** (`gemini-2.5-flash`) — Intent Classifier, "
                "Timestamp Picker, Verification Agent, text answers, search query optimizer\n"
                "- **YouTube Data API v3** — search.list + videos.list for metadata"
            )
        with col2:
            st.markdown("**Parallelism**")
            st.markdown(
                "Text answer + Intent Classifier run in parallel (ThreadPoolExecutor). "
                "YouTube, Timestamp, and Verification run sequentially. "
                "Total latency: typically 4–14 s depending on video length."
            )

        st.markdown("### Decision tree")
        st.code(
            "Query submitted\n"
            "├─ [parallel] Text answer + Intent Classifier\n"
            "│\n"
            "├─ has_visual_intent = False → text only in both columns\n"
            "└─ has_visual_intent = True\n"
            "    └─ YouTube Search\n"
            "        ├─ No results → fallback: text + warning banner\n"
            "        └─ Video found\n"
            "            ├─ duration ≤ 60s → embed from 0:00\n"
            "            └─ duration > 60s → Timestamp Picker\n"
            "                ├─ Valid range → embed at start→end\n"
            "                └─ Invalid → embed from 0:00 (safe fallback)\n"
            "            └─ [always] Verification Agent → trust badge",
            language="text",
        )

        st.markdown("### Verification Agent (new)")
        st.markdown(
            "After a video is found, a separate Gemini call independently checks "
            "whether the title and metadata actually match the user's query. "
            "It returns a verdict (`strong_match`, `partial_match`, `poor_match`) "
            "and confidence score, surfaced as a coloured strip above the video embed. "
            "A `poor_match` verdict does not suppress the video — it acts as a "
            "transparent trust signal for the user."
        )

        st.markdown("### Error and fallback states")
        st.markdown(
            "| Failure | Fallback |\n"
            "|---|---|\n"
            "| Gemini overloaded (503) | Retries ×3 with exponential backoff (2s → 4s → 8s) |\n"
            "| Intent classifier fails | Treatment shows text only; error banner shown |\n"
            "| YouTube API error | Error banner with diagnosis; text answer still shown |\n"
            "| No video results | Warning banner; no embed |\n"
            "| Timestamp out of range | Plays from 0:00; warning shown in pipeline |\n"
            "| Verification fails | Warning strip in UI; does not block video display |"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(query: str) -> dict:
    trace: dict = {"query": query, "errors": {}, "timings_ms": {}}
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

        # Step 1 — parallel
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

        # Step 2 — YouTube
        yt = None
        ts = None
        verif = None

        if intent and intent.has_visual_intent and intent.format in ("short", "long"):
            kind = "Short" if intent.format == "short" else "video"
            st.write(f"🎬 Searching YouTube for the best {kind}…")
            try:
                yt = agents.find_youtube_video(query, intent.format)
                trace["youtube"] = yt.model_dump() if yt else None
                if yt:
                    st.write(f"✅ Found **{yt.title}** by {yt.channel} ({fmt_n(yt.views)} views)")
                else:
                    st.write("⚠️ No relevant video found — try rephrasing")
            except Exception as e:
                trace["errors"]["youtube"] = repr(e)
                st.write("❌ YouTube Search failed — check pipeline for details")

            # Step 3 — Timestamp (run for all videos over 20 s — even short-form
            # clips can have intros worth skipping; only true <20 s clips are exempt)
            if yt and yt.duration_secs > 20:
                st.write("⏱️ Reading video to find the exact relevant moment…")
                try:
                    ts = agents.find_timestamp(query, yt.video_url)
                    trace["timestamp"] = ts.model_dump()
                    if 0 <= ts.start_seconds < ts.end_seconds:
                        st.write(
                            f"✅ Timestamp: **{fmt_ts(ts.start_seconds)}** → "
                            f"**{fmt_ts(ts.end_seconds)}** — _{ts.reasoning}_"
                        )
                    else:
                        trace["errors"]["timestamp"] = "invalid range — playing from start"
                        ts = None
                        st.write("⚠️ Timestamp out of range — playing from start")
                except Exception as e:
                    trace["errors"]["timestamp"] = repr(e)
                    ts = None
                    st.write("⚠️ Timestamp Picker failed — playing from start")

            # Step 4 — Verification: watches the video at the proposed timestamp
            # and cross-checks both video relevance AND timestamp accuracy.
            # If it finds a better start second, that overrides the picker's answer.
            if yt:
                st.write("🔎 Verifying video and timestamp accuracy…")
                try:
                    verif = agents.verify_video_match(query, yt, ts)
                    trace["verification_raw"] = verif.model_dump()

                    verdict_labels = {
                        "strong_match": "✅ Strong match",
                        "partial_match": "⚠️ Partial match",
                        "poor_match": "❌ Poor match",
                    }
                    ts_icon = "✅" if verif.timestamp_is_correct else "🔧"
                    st.write(
                        f"{verdict_labels.get(verif.verdict, verif.verdict)} "
                        f"({verif.confidence:.0%}) — _{verif.explanation}_"
                    )
                    st.write(
                        f"{ts_icon} Timestamp check: _{verif.timestamp_explanation}_"
                    )

                    # Override timestamp with agent's correction if it disagrees
                    if (
                        not verif.timestamp_is_correct
                        and verif.corrected_start_seconds is not None
                        and verif.corrected_start_seconds >= 0
                    ):
                        original_start = ts.start_seconds if ts else 0
                        corrected = verif.corrected_start_seconds
                        # Preserve original end if we have one; otherwise add 90s window
                        original_end = ts.end_seconds if ts else (corrected + 90)
                        new_end = max(original_end, corrected + 30)
                        ts = agents.TimestampPick(
                            start_seconds=corrected,
                            end_seconds=new_end,
                            reasoning=verif.timestamp_explanation,
                        )
                        trace["timestamp"] = ts.model_dump()
                        trace["timestamp_corrected"] = True
                        trace["timestamp_original_start"] = original_start
                        st.write(
                            f"🔧 Corrected start: **{fmt_ts(original_start)}** → "
                            f"**{fmt_ts(corrected)}** based on verification"
                        )

                except Exception as e:
                    trace["errors"]["verification"] = repr(e)
                    verif = None
                    st.write("⚠️ Verification Agent failed — video shown with original timestamp")
        else:
            if intent:
                st.write("💬 Text-only query — no video needed")

        trace["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        trace.update({
            "youtube_result": yt,
            "timestamp_pick": ts,
            "intent": intent,
            "verification": verif,
        })
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
            result = run_pipeline(query)
            st.session_state.last_result = result
            # History
            st.session_state.history = (
                [{
                    "query": query,
                    "had_video": result.get("youtube_result") is not None,
                    "intent": result["intent"].format if result.get("intent") else "error",
                }]
                + st.session_state.history
            )[:5]

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
    render_how_it_works()

    st.markdown(
        '<p style="text-align:center;font-size:0.75rem;color:#94a3b8;margin-top:1rem;">'
        'Aashil Soni · Forum Sanjanwala &nbsp;·&nbsp; Visual Intent Detection Prototype'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
