# Gemini VideoSense — Visual Intent Detection prototype

A working Streamlit prototype of a proposed Gemini feature called **Visual
Intent Detection**: when a user's query is better answered by a video than by
text, the system embeds the right video at the right moment instead of
returning a wall of prose.

Built as the final assignment for **UCLA Anderson MGMT 275 (AI Product
Shipping), Spring 2026**. The prototype demonstrates a five-agent
orchestration that decides *whether* to show a video, *which* video, *where
in the video* the answer lives, and *how confident* the system is in that
answer — with the multi-agent reasoning visible to the grader.

## Live demo

The deployed URL is shared with the grader privately. Try one of the example
chips (*tie a Windsor knot*, *fold a fitted sheet*, *what causes inflation*)
or type your own. Open the **Agent pipeline** expander at the bottom of any
result for a per-agent trace, or **How this prototype works** for a Mermaid
flowchart of the full architecture.

## The five agents

| Agent | Model / API | Role |
| --- | --- | --- |
| **Text Answer** | `gemini-3.1-flash-lite` | Produces a full Gemini-style structured answer (headings, lists, "Pro Tip" callouts). Runs in parallel with the Intent Classifier. |
| **Intent Classifier** | `gemini-3.1-flash-lite` (JSON schema) | Decides if the query has visual intent, and if so whether the right format is `short` or `long`. Returns confidence + rationale. |
| **YouTube Search** | YouTube Data API v3 (+ optional `gemini-3.1-flash-lite` query rewrite) | Fetches the **top 3** candidate videos ranked by `views × short-form boost`. Only invokes the LLM rewrite for queries >10 words (with `temperature=0.2` and a validation guard). |
| **Timestamp Picker** | `gemini-3.1-flash-lite` ingesting the video via `Part.from_uri` + `VideoMetadata(end_offset="600s")` | For each candidate, picks the start/end seconds that best answer the query. Capped at the first 10 minutes. Runs in parallel for all 3 videos. |
| **Verification Agent** | `gemini-3.1-flash-lite` ingesting the video | For each candidate, independently *watches* the video and returns: (1) a relevance verdict (`strong_match` / `partial_match` / `poor_match`) + confidence, and (2) whether the proposed timestamp is correct (can supply a `corrected_start_seconds` that the pipeline applies). Runs in parallel for all 3 videos. |

**Parallelism map**

- *Step 1 (parallel):* Text Answer + Intent Classifier
- *Step 2 (sequential):* YouTube Search returns top 3 candidates
- *Step 3 (parallel × 3):* per-video pipeline = Timestamp Picker + Verification Agent
- *Step 4 (UI):* render decision based on `intent.format` + top video's verdict

Total wall-clock latency is dominated by Step 3 (~10–25 s); the 3-way fan-out
adds ~3× API cost but no additional wait time vs. processing one video.

## Key design decisions

1. **Three-video carousel with parallel pre-compute** (`agents.find_youtube_videos`, `app._render_carousel_block`)
   We fetch *and fully analyse* the top 3 candidates in parallel via
   `ThreadPoolExecutor`. The user flips between them with arrow buttons —
   no new API calls on switch, since all 3 are pre-computed.

2. **Smart placement** (`app.render_treatment`)
   - `format == "short"` (e.g. *tie a Windsor knot*) → video first, text below.
     Quick visual tasks shouldn't force the user to scroll past paragraphs.
   - `format == "long"` (e.g. *how does GPS work*) → text first, video below.
     Context helps frame complex topics.

3. **Smart exposure** (`app.render_treatment`)
   - `short` → always exposes 3-video carousel (browsing short clips is natural).
   - `long + strong_match` → shows just video #1 with no carousel UI. AI is
     confident; alternatives would be noise.
   - `long + partial/poor match` → re-exposes all 3 with *"top match wasn't a
     strong fit — browse alternates"* framing, so the user can recover from
     a bad #1.

4. **Tabbed comparison** (`app.main`)
   Default tab = solo VideoSense view (mimics a real consumer-Gemini surface).
   Second tab = side-by-side comparison with "Gemini today" (no Visual Intent
   Detection) for anyone curious about the before/after. The product feels
   like a real shipping feature, not a research harness.

5. **Query-optimizer guardrails** (`agents._optimize_query`)
   For queries ≤10 words, the raw query is sent verbatim with a `shorts` /
   `tutorial` suffix appended — no LLM call, no risk of over-compression.
   Only longer conversational queries go through an LLM rewrite, and even
   then at `temperature=0.2` with a validation guard that falls back to the
   raw query if the rewrite degenerates to fewer than 3 words. Hardened
   after a real failure where the optimizer compressed *"How do I tie a
   Windsor knot"* into *"Wind"* and YouTube served back the Naruto ending
   song.

6. **Verification ≠ censorship**
   A `poor_match` verdict does **not** suppress the video — it surfaces as a
   transparent trust signal (coloured strip above the embed) so the user can
   judge for themselves rather than having results silently hidden. The
   verdict *does* feed the exposure rule: a `strong_match` on a long-form
   query collapses the carousel to one video.

## Owner-only session log

The deployed app has a **hidden** per-query trace log for app owners to
review what users (including the grader) queried and how the agents behaved.

- Every completed pipeline appends a structured entry to a server-side
  `videosense_log.jsonl` (gitignored; never committed).
- The log expander is **invisible by default** — regular users see no extra UI.
- Access requires `?admin=<key>` URL parameter matching
  `st.secrets["ADMIN_KEY"]`.
- Owners download the full log as JSON from the admin view.
- File persists for the Streamlit Cloud container lifetime (resets on
  redeploy).

## File structure

```
visual-intent-prototype/
├── app.py                       # Streamlit UI + pipeline orchestrator + admin log
├── agents.py                    # All five agent functions (prompts inline)
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Template for local-dev API keys
├── .streamlit/
│   ├── config.toml              # Light theme, Google-blue accent
│   └── secrets.toml.example     # Includes ADMIN_KEY template
├── .gitignore                   # Excludes .env, secrets.toml, *.jsonl, .venv
└── README.md
```

Earlier versions split prompts into a separate `prompts.py`; they now live
inline in each agent function in `agents.py` for locality.

## 1. Install locally

Python 3.10+ recommended.

```bash
cd visual-intent-prototype
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Set up API keys

Two required:

- **Gemini API key** — https://aistudio.google.com/apikey
- **YouTube Data API v3 key** — https://console.cloud.google.com/apis/library/youtube.googleapis.com
  (create a project, enable the API, then create an API key under
  **Credentials**)

Optional but recommended for the admin log:

- **`ADMIN_KEY`** — any long, non-obvious string. Required to access the
  session-log admin view on the deployed app.

For local dev, copy `.env.example` to `.env`:

```bash
cp .env.example .env
# then edit .env so it reads:
#   GEMINI_API_KEY=AIza...
#   YOUTUBE_API_KEY=AIza...
```

`.env` is gitignored; your keys will never be committed.

## 3. Run locally

```bash
streamlit run app.py
```

Streamlit prints a local URL (default `http://localhost:8501`). Try a few
queries:

| Category | Try | Expected behavior |
| --- | --- | --- |
| Short visual | `how to tie a Windsor knot` | 3-video carousel of YouTube Shorts, video first, text below |
| Long visual | `how to replace a Samsung fridge water filter` | Single video if verification gives strong match; 3-video carousel otherwise, text first |
| Text-only | `what's the capital of France` | Text answer only, no video |
| Adversarial | `what is video compression` | Text-only despite the word "video" |
| Compare tab | (any query) | Click "Compare to Gemini today" tab → side-by-side with current Gemini behavior |

Open the **Agent pipeline** expander to inspect each agent's input/output.
Locally, append `?admin=anything` to the URL to unlock the session-log view
(the local fallback accepts any non-empty admin param when `ADMIN_KEY` isn't
configured).

## 4. Deploy to Streamlit Cloud

1. Push the repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo, branch (`main`), and entrypoint
   (`app.py`).
4. Click **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "AIza..."
   YOUTUBE_API_KEY = "AIza..."
   ADMIN_KEY = "your-long-random-string-here"
   ```

5. Click **Deploy**. The first build takes ~2–4 minutes while it installs
   `requirements.txt`.
6. Share the public URL with users. Bookmark the admin URL
   (`https://<your-app>.streamlit.app/?admin=<key>`) for yourself.

To rotate keys later: **Manage app → Settings → Secrets** — no redeploy
needed.

## Failure modes (handled in code)

| Failure | Fallback |
| --- | --- |
| Gemini overloaded (503) | Retries ×3 with exponential backoff (2s, 4s between attempts) |
| Intent classifier fails | Treatment view shows text answer + error banner |
| YouTube API error | Error banner with diagnosis; text answer still shown |
| No video results | Warning banner; no embed |
| Timestamp out of range | Plays from 0:00; warning surfaced in the pipeline expander |
| Verification fails | Warning strip in UI; does not block video display |
| Query > 500 chars | Rejected with a friendly error before any agent fires |

## Authors

- **Aashil Soni** — UCLA Anderson MBA, MGMT 275 (AI Product Shipping), Spring 2026
- **Forum Sanjanwala** — UCLA Anderson MBA, MGMT 275 (AI Product Shipping), Spring 2026
