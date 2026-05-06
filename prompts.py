"""Prompts for the three Visual Intent Detection agents.

Kept in a single module so the prompt-engineering surface area is easy to
review during the assignment writeup.
"""

# ---------------------------------------------------------------------------
# Agent 1: Visual Intent classifier
# ---------------------------------------------------------------------------

VISUAL_INTENT_SYSTEM = """\
You classify user queries by whether the best answer is a video, and which
kind of video.

Definitions:
- has_visual_intent = true when the user would be better served by watching
  something than by reading text. False when text is the right medium.
- format = "short": procedural / demonstration query that can be answered in
  roughly 15-60 seconds (a YouTube Short or short clip).
- format = "long": explainer or multi-step query where the right answer lives
  inside a longer video at a specific moment.
- format = "none": text is the right medium; no video should be shown.
- confidence: float in [0, 1] reflecting how sure you are.
- rationale: ONE sentence explaining the call.

Rules:
- The literal word "video" in the query does NOT imply visual intent (e.g.
  "what is video compression" is a text question).
- Ambiguous shopping / brand queries default to format="none" unless they
  clearly ask "how to" do something.
- Return ONLY the JSON object that matches the response schema. No prose.
"""

VISUAL_INTENT_FEW_SHOTS = [
    # Clear short-intent
    {
        "query": "how to tie a tie",
        "output": {
            "has_visual_intent": True,
            "format": "short",
            "confidence": 0.97,
            "rationale": "Short physical demonstration is the canonical answer to this query.",
        },
    },
    {
        "query": "how to do a deadlift",
        "output": {
            "has_visual_intent": True,
            "format": "short",
            "confidence": 0.94,
            "rationale": "Form check is best shown in a brief demonstration video.",
        },
    },
    # Clear long-intent
    {
        "query": "how does photosynthesis work",
        "output": {
            "has_visual_intent": True,
            "format": "long",
            "confidence": 0.88,
            "rationale": "Multi-step explainer best served by jumping to the relevant segment of a longer video.",
        },
    },
    {
        "query": "how to replace a Samsung fridge water filter",
        "output": {
            "has_visual_intent": True,
            "format": "long",
            "confidence": 0.92,
            "rationale": "Multi-step procedure where the user benefits from a specific moment in a longer walkthrough.",
        },
    },
    # Clear text-intent
    {
        "query": "what's the capital of France",
        "output": {
            "has_visual_intent": False,
            "format": "none",
            "confidence": 0.99,
            "rationale": "One-word factual answer; video adds nothing.",
        },
    },
    {
        "query": "summarize the French Revolution",
        "output": {
            "has_visual_intent": False,
            "format": "none",
            "confidence": 0.9,
            "rationale": "Summarization request — text is the natural medium.",
        },
    },
    # Adversarial
    {
        "query": "what is video compression",
        "output": {
            "has_visual_intent": False,
            "format": "none",
            "confidence": 0.9,
            "rationale": "The word 'video' is the topic, not the desired medium; this is a definitional text question.",
        },
    },
    {
        "query": "tell me about On-Cloud shoes",
        "output": {
            "has_visual_intent": False,
            "format": "none",
            "confidence": 0.7,
            "rationale": "Ambiguous brand query without a 'how to' framing; default to text.",
        },
    },
]


def build_visual_intent_user_prompt(query: str) -> str:
    """Render the few-shot block + the live query into a single user message."""
    import json

    lines = ["Classify the following query.\n", "Examples:"]
    for ex in VISUAL_INTENT_FEW_SHOTS:
        lines.append(f'Query: "{ex["query"]}"')
        lines.append(f"Output: {json.dumps(ex['output'])}")
        lines.append("")
    lines.append(f'Query: "{query}"')
    lines.append("Output:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 3: Timestamp picker (Agent 2 hits the YouTube Data API directly,
# no LLM prompt needed)
# ---------------------------------------------------------------------------

TIMESTAMP_PROMPT_TEMPLATE = """\
Given the user's query: "{query}"

Identify the start and end timestamp (in seconds, integers) of the segment in
this video that best answers the query. Aim for a tight clip — typically 15 to
90 seconds long. If no segment in the first 10 minutes answers the query, set
both timestamps to 0 and explain why in `reasoning`.

Return ONLY a JSON object matching the response schema. No prose.
"""


def build_timestamp_prompt(query: str) -> str:
    return TIMESTAMP_PROMPT_TEMPLATE.format(query=query)


# ---------------------------------------------------------------------------
# Query optimizer — rewrites the raw user query into a tight YouTube search
# ---------------------------------------------------------------------------

SEARCH_QUERY_PROMPT = """\
Convert the user's question into an ideal YouTube search query: 3-8 words,
specific, no filler words, no punctuation. Think about what a human would
actually type into YouTube to find the most helpful {format} video.

User question: {query}

Return ONLY the search string. Nothing else.
"""


def build_search_query_prompt(query: str, format: str) -> str:
    label = "short tutorial or demonstration" if format == "short" else "detailed guide or explainer"
    return SEARCH_QUERY_PROMPT.format(query=query, format=label)


# ---------------------------------------------------------------------------
# Brief text answer (used in BOTH columns so the only difference is the
# video element)
# ---------------------------------------------------------------------------

TEXT_ANSWER_PROMPT = """\
Answer the following user query in a helpful, conversational tone, roughly
4-6 sentences — about the depth of a typical Gemini answer. Give the user the
actual steps, facts, or context they need; don't oversummarize. Do not mention
videos. Do not use markdown headers, bold, or bullet lists — write in flowing
prose.

Query: {query}
"""


def build_text_answer_prompt(query: str) -> str:
    return TEXT_ANSWER_PROMPT.format(query=query)
