# Visual Intent Detection — Gemini Prototype

A working Streamlit prototype of a proposed Gemini feature called **Visual
Intent Detection**: when a user's query would be better answered by a video
than by text, the system embeds the right video at the right moment instead
of returning a wall of text.

The app runs three agents and shows a side-by-side **Control vs. Treatment**
comparison so the multi-agent logic is visible to the grader.

## Architecture

| Agent | Model / API | Purpose |
| --- | --- | --- |
| `classify_visual_intent` | `gemini-2.5-flash` (JSON mode) | Decides if the query needs a video, and if so whether `short` or `long` form. |
| `find_youtube_video` | YouTube Data API v3 | Searches and ranks videos for the chosen format. |
| `find_timestamp` | `gemini-2.5-flash` ingesting the YouTube URL via `Part.from_uri` | For long videos, picks the start/end seconds that best answer the query. Capped at the first 10 minutes via `VideoMetadata`. |

The orchestrator runs the text answer and the visual-intent classifier in
parallel (`concurrent.futures.ThreadPoolExecutor`), then conditionally fires
the YouTube and timestamp agents.

## File structure

```
visual-intent-prototype/
├── app.py                       # Streamlit UI + orchestrator
├── agents.py                    # Three agent functions + helpers
├── prompts.py                   # Few-shot examples and prompt builders
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── secrets.toml.example
├── .gitignore
└── README.md
```

## 1. Install locally

You need Python 3.10+ and a way to create a virtual environment.

```bash
cd visual-intent-prototype
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Set up API keys

You need two keys:

- **Gemini API key** — https://aistudio.google.com/apikey
- **YouTube Data API v3 key** — https://console.cloud.google.com/apis/library/youtube.googleapis.com (create a project, enable the API, then create an API key under **Credentials**)

Copy the template and fill in your keys:

```bash
cp .env.example .env
# then edit .env so it reads:
#   GEMINI_API_KEY=AIza...
#   YOUTUBE_API_KEY=AIza...
```

`.env` is gitignored, so your keys will never be committed.

## 3. Run locally

```bash
streamlit run app.py
```

Streamlit prints a local URL (default `http://localhost:8501`). Open it and
try a few queries from each category:

| Category | Try | Expected behavior |
| --- | --- | --- |
| Short-intent | `how to tie a tie` | Treatment shows a YouTube Short demo. |
| Long-intent | `how to replace a Samsung fridge water filter` | Treatment embeds a longer video, jumping to the relevant segment. |
| Text-intent | `what's the capital of France` | Treatment is text-only. |
| Adversarial | `what is video compression` | Treatment is text-only despite the word "video". |
| Today's behavior | `tutorial video for python loops` | Control embeds a video (because of the word "video"); Treatment makes its own decision. |

Use the **Agent trace** expander at the bottom of the page to confirm that
each agent fired with the right inputs and outputs.

## 4. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Visual Intent Detection prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

Double-check that `.env` and `.streamlit/secrets.toml` did **not** get
committed (`.gitignore` covers them, but verify with `git status` before
pushing).

## 5. Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**, pick your repo, branch (`main`), and entrypoint (`app.py`).
3. Click **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "AIza..."
   YOUTUBE_API_KEY = "AIza..."
   ```

   (Same format as `.streamlit/secrets.toml.example`.)
4. Click **Deploy**. The first build takes ~2–3 minutes while it installs
   `requirements.txt`.
5. Once live, share the public URL with your grader.

If you ever rotate keys, edit them under **Manage app → Settings → Secrets**
on Streamlit Cloud — no redeploy is needed.

## Failure modes (handled in code)

- **YouTube returns nothing** → Treatment column shows the text answer plus
  a small "No relevant video found." note.
- **Timestamp agent fails or returns an invalid range** → Treatment renders
  the video from the start; the agent trace records the error.
- **Any API error** → caught and surfaced in the Agent trace expander; the
  app does not crash.
- **Query over 500 chars** → rejected with a friendly error before any agent
  is called.
