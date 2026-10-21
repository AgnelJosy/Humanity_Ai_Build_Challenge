"""
Class Quality Analyzer - Humaniti AI Build Challenge
----------------------------------------------------
A Streamlit app that evaluates online class-recording transcripts against a
five-tier quality rubric (Excellent, Good, Average, Weak, Known Quality Issues)
using Google Gemini or OpenAI.

Run locally:  streamlit run app.py
"""

import json
import logging
import os
import re
import traceback
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Logging: errors are written to a file AND surfaced to the user as a popup.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    filename="app_errors.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("class_quality_analyzer")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CATEGORIES = ["Excellent", "Good", "Average", "Weak", "Known Quality Issues"]

# Score bands keep the numeric score consistent with the predicted category.
SCORE_BANDS = {
    "Excellent": (85, 100),
    "Good": (70, 84),
    "Average": (50, 69),
    "Weak": (30, 49),
    "Known Quality Issues": (1, 49),
}

BADGE_COLORS = {
    "Excellent": "#1B8A3C",
    "Good": "#2E86DE",
    "Average": "#E0A100",
    "Weak": "#E67E22",
    "Known Quality Issues": "#C0392B",
}

# Checklist criteria the model must assess (yes / no / unclear).
CHECKLIST_KEYS = {
    "self_intro_and_rapport": "Good introduction / rapport",
    "virtual_background_set": "Virtual background set (8x8)",
    "tutor_camera_on_face_visible": "Tutor camera on / face visible",
    "individual_questioning_by_name": "Individual questioning by name",
    "topic_videos_used": "Topic videos used",
    "homework_assigned": "Homework assigned",
    "proper_conclusion": "Proper conclusion",
    "new_joiner_intro_time": "Self-introduction time for new joiners",
    "energy_clarity_happy_tone": "Energetic, clear, happy communication",
}

MAX_TRANSCRIPT_CHARS = 120_000  # Truncation limit chosen for speed and cost.

# The rubric is embedded in the system prompt so the LLM is grounded in it.
SYSTEM_PROMPT = """You are a strict but fair quality auditor for online classes.
You evaluate a class-recording TRANSCRIPT against the rubric below and return JSON only.

RUBRIC
- Excellent: Good 1-to-1 interaction, virtual background set (8x8), individual
  questioning by name, topic videos used, and homework assigned.
- Good: Good intro/rapport, fixed background, homework given, but the interaction
  needs to be slightly more pleasant or balanced.
- Average: Background set, individual consideration during activities, but low
  energy, clarity issues, or needs more energetic communication.
- Weak: Background set, but lacks a proper intro/conclusion, is missing
  self-introduction time for new joiners, or the tutor camera is turned off.
- Known Quality Issues: Interaction is present but the virtual background is
  missing, the tutor's face is not visible, or there is a lack of happy communication.

IMPORTANT RULES
1. A transcript contains speech only. Visual criteria (virtual background, camera,
   face visibility) must be inferred ONLY from explicit spoken cues (e.g. "I can't
   see your face", "turn your camera on", "nice background"). If no cue exists,
   mark that criterion "unclear" - never invent evidence.
2. Every judgement must be backed by a short evidence quote (max 20 words) taken
   verbatim from the transcript.
3. Choose exactly ONE category from: Excellent, Good, Average, Weak, Known Quality Issues.
4. Score from 1-100 and stay inside these bands: Excellent 85-100, Good 70-84,
   Average 50-69, Weak 30-49, Known Quality Issues 1-49.
5. Feedback must be specific, constructive and addressed to the tutor.
6. Write "summary" as one concise rubric-record note (about 20-40 words),
   matching the examples' style: mention the tutor by name only if clearly
   identified in the transcript, summarize observed strengths and improvement
   points, and use plain natural language. Do not invent a session ID or date;
   those are not transcript content. Do not claim visual facts unless supported
   by explicit spoken evidence. Avoid a generic multi-sentence essay.

Return ONLY a JSON object with this exact schema:
{
  "category": "<one of the five categories>",
  "score": <integer 1-100>,
  "confidence": "<low|medium|high>",
    "summary": "<concise rubric-record note, about 20-40 words>",
  "checklist": {
    "self_intro_and_rapport": "<yes|no|unclear>",
    "virtual_background_set": "<yes|no|unclear>",
    "tutor_camera_on_face_visible": "<yes|no|unclear>",
    "individual_questioning_by_name": "<yes|no|unclear>",
    "topic_videos_used": "<yes|no|unclear>",
    "homework_assigned": "<yes|no|unclear>",
    "proper_conclusion": "<yes|no|unclear>",
    "new_joiner_intro_time": "<yes|no|unclear>",
    "energy_clarity_happy_tone": "<yes|no|unclear>"
  },
  "strengths": ["..."],
  "areas_for_improvement": ["..."],
  "tutor_feedback": ["<actionable step 1>", "<actionable step 2>"],
  "evidence": ["<verbatim quote>", "..."]
}"""

SAMPLE_TRANSCRIPT = """WEBVTT

00:00:01.000 --> 00:00:06.000
Tutor: Good evening everyone! Welcome to class. I'm so happy to see you all today.

00:00:07.000 --> 00:00:12.000
Tutor: Aarav, you are new here, would you like to introduce yourself to the class?

00:00:13.000 --> 00:00:18.000
Aarav: Hi, I'm Aarav, I love cricket and drawing.

00:00:19.000 --> 00:00:25.000
Tutor: Lovely, Aarav! Priya, can you tell me what the video about the water cycle showed?

00:00:26.000 --> 00:00:32.000
Priya: It showed evaporation and rain.

00:00:33.000 --> 00:00:40.000
Tutor: Excellent! Your background looks nice today, everyone can see me clearly. Homework: draw the water cycle.

00:00:41.000 --> 00:00:46.000
Tutor: Thank you all, great job today. See you tomorrow, bye!
"""


# --------------------------------------------------------------------------- #
# Error handling helpers
# --------------------------------------------------------------------------- #
def report_error(title: str, exc: Optional[BaseException] = None, message: str = "") -> None:
    """Log an error and queue it so it is shown as a popup message box.

    Args:
        title: Short heading for the popup.
        exc: Optional exception that triggered the error.
        message: Human-friendly explanation shown to the user.
    """
    try:
        details = (
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if exc
            else ""
        )
        logger.error("%s | %s | %s", title, message or (str(exc) if exc else ""), details)
        st.session_state["pending_error"] = {
            "title": title,
            "message": message or (str(exc) if exc else "Unknown error"),
        }
    except Exception:  # Never let the error reporter itself crash the app.
        logger.exception("Failed while reporting an error")


def _render_error_body(title: str, message: str) -> None:
    """Render the content of the error popup (shared by dialog and fallback)."""
    st.error(message)
    st.caption("Details were saved to app_errors.log for troubleshooting.")


if hasattr(st, "dialog"):

    @st.dialog("⚠️ Something went wrong")
    def error_dialog(title: str, message: str) -> None:
        """Show a modal popup (message box) containing the error."""
        st.markdown(f"**{title}**")
        _render_error_body(title, message)
        if st.button("Close", type="primary"):
            st.rerun()

else:  # Older Streamlit versions: fall back to toast + inline error.

    def error_dialog(title: str, message: str) -> None:
        """Fallback popup for Streamlit versions without st.dialog."""
        st.toast(f"{title}: {message}", icon="⚠️")
        st.error(f"**{title}** - {message}")


def show_pending_error() -> None:
    """Display (once) any error queued by report_error()."""
    try:
        err = st.session_state.pop("pending_error", None)
        if err:
            error_dialog(err["title"], err["message"])
    except Exception:
        logger.exception("Failed while displaying error popup")


# --------------------------------------------------------------------------- #
# Transcript handling
# --------------------------------------------------------------------------- #
def clean_vtt(raw_text: str) -> str:
    """Strip WEBVTT headers, cue numbers, timestamps and tags from a transcript.

    Plain .txt content passes through unchanged apart from whitespace cleanup.

    Args:
        raw_text: Raw text read from an uploaded file or the paste box.

    Returns:
        Cleaned transcript text containing only spoken lines.
    """
    try:
        lines: List[str] = []
        timestamp = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?[.,]\d{3}\s*-->")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line or line.upper().startswith(("WEBVTT", "NOTE", "KIND:", "LANGUAGE:")):
                continue
            if timestamp.match(line) or line.isdigit():
                continue
            lines.append(re.sub(r"<[^>]+>", "", line))  # remove <v Name> style tags
        return "\n".join(lines)
    except Exception as exc:
        report_error("Transcript cleaning failed", exc, "Could not parse the transcript format.")
        return raw_text


def read_uploaded_file(uploaded_file) -> str:
    """Read an uploaded .txt/.vtt file into a string.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Decoded file content, or an empty string on failure.
    """
    try:
        data = uploaded_file.read()
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unsupported file encoding.")
    except Exception as exc:
        report_error("File read error", exc, "The uploaded file could not be read.")
        return ""


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #
def get_api_key(provider: str, typed_key: str) -> str:
    """Resolve the API key from the sidebar, Streamlit secrets or environment.

    Args:
        provider: "Gemini" or "OpenAI".
        typed_key: Key entered in the sidebar (may be empty).

    Returns:
        The API key string, or an empty string if none was found.
    """
    if typed_key.strip():
        return typed_key.strip()
    secret_name = "GEMINI_API_KEY" if provider == "Gemini" else "OPENAI_API_KEY"
    try:
        if secret_name in st.secrets:
            return str(st.secrets[secret_name])
    except Exception:
        pass  # No secrets file configured - that's fine.
    return os.environ.get(secret_name, "")


def call_gemini(api_key: str, model_name: str, transcript: str) -> str:
    """Send a transcript using Gemini's Interactions REST API.

    Args:
        api_key: Gemini API key.
        model_name: Gemini model identifier.
        transcript: Cleaned transcript text.

    Returns:
        The model's raw text response.
    """
    endpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"
    response = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key},
        json={
            "model": model_name,
            "input": f"TRANSCRIPT:\n\"\"\"\n{transcript}\n\"\"\"",
            "system_instruction": SYSTEM_PROMPT,
            "generation_config": {"temperature": 0.2},
            "response_format": {"type": "text", "mime_type": "application/json"},
            # Transcripts may contain personal data; avoid retaining them remotely.
            "store": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    output_text = payload.get("output_text")
    if output_text:
        return output_text

    # Also support REST responses where output is represented as model steps.
    return "".join(
        item.get("text", "")
        for step in payload.get("steps", [])
        if step.get("type") == "model_output"
        for item in step.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )


def call_openai(api_key: str, model_name: str, transcript: str) -> str:
    """Send the transcript to OpenAI and return the raw JSON text.

    Args:
        api_key: OpenAI API key.
        model_name: OpenAI model identifier.
        transcript: Cleaned transcript text.

    Returns:
        The model's raw text response.
    """
    from openai import OpenAI  # Imported lazily: optional dependency.

    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TRANSCRIPT:\n\"\"\"\n{transcript}\n\"\"\""},
        ],
    )
    return completion.choices[0].message.content


def parse_model_json(raw: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from the model's raw output.

    Args:
        raw: Raw model text, possibly wrapped in code fences.

    Returns:
        Parsed dictionary.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw or "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return JSON.")
    return json.loads(cleaned[start : end + 1])


def validate_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise and sanity-check the model output.

    Ensures the category is valid, the score is an int inside the category's
    band, and list/checklist fields have the expected shape.

    Args:
        data: Parsed model JSON.

    Returns:
        A cleaned result dictionary safe to render.
    """
    category = str(data.get("category", "")).strip()
    # Case-insensitive match so "known quality issues" still works.
    category = next((c for c in CATEGORIES if c.lower() == category.lower()), None)
    if category is None:
        raise ValueError("Model returned an unrecognised category.")

    try:
        score = int(float(data.get("score", 0)))
    except (TypeError, ValueError):
        score = SCORE_BANDS[category][0]
    low, high = SCORE_BANDS[category]
    score = max(low, min(high, score))  # Clamp into the category's band.

    def as_list(value: Any) -> List[str]:
        """Coerce a value into a list of non-empty strings."""
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in (value or []) if str(v).strip()]

    checklist_in = data.get("checklist", {}) or {}
    checklist = {}
    for key in CHECKLIST_KEYS:
        val = str(checklist_in.get(key, "unclear")).lower().strip()
        checklist[key] = val if val in ("yes", "no", "unclear") else "unclear"

    confidence = str(data.get("confidence", "medium")).lower()
    return {
        "category": category,
        "score": score,
        "confidence": confidence if confidence in ("low", "medium", "high") else "medium",
        "summary": str(data.get("summary", "")).strip(),
        "checklist": checklist,
        "strengths": as_list(data.get("strengths")),
        "areas_for_improvement": as_list(data.get("areas_for_improvement")),
        "tutor_feedback": as_list(data.get("tutor_feedback")),
        "evidence": as_list(data.get("evidence")),
    }


def evaluate_transcript(provider: str, api_key: str, model_name: str, transcript: str) -> Optional[Dict[str, Any]]:
    """Run the full evaluation pipeline and return a validated result.

    Args:
        provider: "Gemini" or "OpenAI".
        api_key: API key for the chosen provider.
        model_name: Model identifier.
        transcript: Cleaned transcript text.

    Returns:
        Validated result dictionary, or None if anything failed (an error
        popup is queued in that case).
    """
    try:
        if len(transcript) > MAX_TRANSCRIPT_CHARS:
            transcript = transcript[:MAX_TRANSCRIPT_CHARS]
            st.toast("Transcript was very long and has been truncated.", icon="✂️")

        raw = (
            call_gemini(api_key, model_name, transcript)
            if provider == "Gemini"
            else call_openai(api_key, model_name, transcript)
        )
        return validate_result(parse_model_json(raw))
    except ImportError as exc:
        report_error(
            "Missing dependency",
            exc,
            "A required package is missing. Run: pip install -r requirements.txt",
        )
    except json.JSONDecodeError as exc:
        report_error("Invalid model response", exc, "The AI returned malformed JSON. Please try again.")
    except ValueError as exc:
        report_error("Unexpected model output", exc, str(exc))
    except Exception as exc:  # API/auth/network/quota errors land here.
        report_error(
            "AI request failed",
            exc,
            f"Could not complete the evaluation. Check your API key, model name and connection.\n\n{exc}",
        )
    return None


def clear_api_connection_status() -> None:
    """Clear the last connection result when the entered API key changes."""
    st.session_state.pop("api_connection_result", None)


def describe_api_error(exc: Exception, api_key: str = "") -> str:
    """Explain API failures using safe provider details without leaking secrets.

    Args:
        exc: The exception raised by the HTTP client or provider SDK.
        api_key: Optional credential to redact if a provider echoes it.

    Returns:
        A concise diagnostic suitable for display in the app.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", "unknown")
        try:
            payload = response.json()
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            detail = error.get("message", "") if isinstance(error, dict) else ""
        except (ValueError, AttributeError):
            detail = ""

        if not detail:
            detail = "The provider rejected the request."
        if api_key:
            detail = detail.replace(api_key, "[REDACTED]")
        return f"HTTP {status}: {detail}"

    if isinstance(exc, requests.exceptions.Timeout):
        return "The request timed out. Check the connection and try again."
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Could not reach the provider. Check your internet connection and try again."
    detail = str(exc) or type(exc).__name__
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    return f"{type(exc).__name__}: {detail}"


# --------------------------------------------------------------------------- #
# UI components
# --------------------------------------------------------------------------- #
def render_sidebar() -> Dict[str, str]:
    """Render the sidebar and return the provider configuration.

    Returns:
        Dict with keys: provider, api_key, model.
    """
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox("LLM provider", ["Gemini", "OpenAI"])
        typed_key = st.text_input(
            f"{provider} API key",
            type="password",
            help="Kept in memory for this session only. Never stored or logged.",
            key=f"api_key_{provider}",
            on_change=clear_api_connection_status,
        )
        default_model = "gemini-3.8-flash" if provider == "Gemini" else "gpt-4o-mini"
        model = st.text_input("Model name", value=default_model, key=f"model_{provider}")
        if st.button("Test API connection", key="test_api_connection"):
            api_key = get_api_key(provider, typed_key)
            if not api_key:
                st.session_state["api_connection_result"] = (provider, model.strip(), False, "Enter an API key first.")
            else:
                try:
                    response = (
                        call_gemini(api_key, model.strip(), "Reply with only: OK.")
                        if provider == "Gemini"
                        else call_openai(api_key, model.strip(), "Reply with only: OK.")
                    )
                    if not response or not response.strip():
                        raise ValueError("The provider returned an empty response.")
                    st.session_state["api_connection_result"] = (provider, model.strip(), True, "Connection successful.")
                except Exception as exc:
                    st.session_state["api_connection_result"] = (
                        provider,
                        model.strip(),
                        False,
                        f"Connection failed — {describe_api_error(exc, api_key)}",
                    )

        connection_result = st.session_state.get("api_connection_result")
        if connection_result and connection_result[:2] == (provider, model.strip()):
            if connection_result[2]:
                st.success(connection_result[3])
            else:
                st.error(connection_result[3])
        st.caption("Tip: you can also set GEMINI_API_KEY / OPENAI_API_KEY as environment variables or Streamlit secrets.")
        st.divider()
        st.markdown("**Rubric tiers**")
        for cat in CATEGORIES:
            st.markdown(
                f"<span style='color:{BADGE_COLORS[cat]}'>●</span> {cat}",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("🔒 Transcripts are sent to the selected AI provider. Do not upload data you are not permitted to share.")
    return {"provider": provider, "api_key": get_api_key(provider, typed_key), "model": model.strip()}


def render_results(result: Dict[str, Any]) -> None:
    """Render the evaluation dashboard.

    Args:
        result: Validated result dictionary from evaluate_transcript().
    """
    try:
        color = BADGE_COLORS[result["category"]]
        col_badge, col_score, col_conf = st.columns([2, 1, 1])
        with col_badge:
            st.markdown("**Predicted quality**")
            st.markdown(
                f"<div style='display:inline-block;padding:10px 22px;border-radius:999px;"
                f"background:{color};color:white;font-size:1.3rem;font-weight:700;'>"
                f"{result['category']}</div>",
                unsafe_allow_html=True,
            )
        with col_score:
            st.metric("Score", f"{result['score']} / 100")
        with col_conf:
            st.metric("Confidence", result["confidence"].capitalize())
        st.progress(result["score"] / 100)

        if result["summary"]:
            st.subheader("Rubric-style summary")
            st.markdown(f"**{result['category']}** — {result['summary']}")

        st.subheader("Rubric checklist")
        icons = {"yes": "✅", "no": "❌", "unclear": "❔"}
        cols = st.columns(3)
        for i, (key, label) in enumerate(CHECKLIST_KEYS.items()):
            cols[i % 3].markdown(f"{icons[result['checklist'][key]]} {label}")
        st.caption("❔ = cannot be determined from text alone (e.g. visual criteria with no spoken cue).")

        left, right = st.columns(2)
        with left:
            st.subheader("💪 Key strengths")
            for item in result["strengths"] or ["No specific strengths identified."]:
                st.markdown(f"- {item}")
        with right:
            st.subheader("🎯 Areas for improvement")
            for item in result["areas_for_improvement"] or ["No specific issues identified."]:
                st.markdown(f"- {item}")

        st.subheader("📝 Actionable tutor feedback")
        for i, item in enumerate(result["tutor_feedback"] or ["No feedback generated."], start=1):
            st.markdown(f"{i}. {item}")

        if result["evidence"]:
            with st.expander("Evidence quoted from the transcript"):
                for quote in result["evidence"]:
                    st.markdown(f"> {quote}")

        st.download_button(
            "⬇️ Download report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="class_quality_report.json",
            mime="application/json",
        )
    except Exception as exc:
        report_error("Display error", exc, "The results could not be displayed.")


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Configure the page, collect input, run the evaluation and show results."""
    st.set_page_config(page_title="Class Quality Analyzer", page_icon="🎓", layout="wide")
    st.title("🎓 Class Quality Analyzer")
    st.caption("AI-powered assessment of online class transcripts against the Humaniti quality rubric.")

    config = render_sidebar()

    tab_upload, tab_paste = st.tabs(["📂 Upload transcript", "✏️ Paste transcript"])
    transcript_text = ""
    with tab_upload:
        uploaded = st.file_uploader("Upload a .txt or .vtt transcript", type=["txt", "vtt"])
        if uploaded is not None:
            transcript_text = read_uploaded_file(uploaded)
    with tab_paste:
        use_sample = st.checkbox("Load sample transcript")
        pasted = st.text_area(
            "Paste transcript here",
            value=SAMPLE_TRANSCRIPT if use_sample else "",
            height=250,
        )
        if pasted.strip():
            transcript_text = pasted  # Pasted text takes priority when present.

    if st.button("🔍 Evaluate class quality", type="primary"):
        try:
            if not config["api_key"]:
                report_error("API key missing", message="Please enter your API key in the sidebar.")
            elif not transcript_text.strip():
                report_error("No transcript", message="Please upload or paste a transcript first.")
            else:
                cleaned = clean_vtt(transcript_text)
                if len(cleaned) < 50:
                    report_error("Transcript too short", message="The transcript is too short to evaluate reliably.")
                else:
                    with st.spinner("Analysing transcript..."):
                        st.session_state["result"] = evaluate_transcript(
                            config["provider"], config["api_key"], config["model"], cleaned
                        )
        except Exception as exc:
            report_error("Unexpected error", exc, "Something unexpected happened while starting the evaluation.")

    if st.session_state.get("result"):
        st.divider()
        render_results(st.session_state["result"])

    show_pending_error()  # Pop-up message box for any error raised above.


if __name__ == "__main__":
    main()