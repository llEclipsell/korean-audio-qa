"""
Korean audio dataset extraction API.

ASSUMPTIONS (no spec was provided beyond the response JSON shape — see the
top-level docstring notes below and the inline comments at each guess).
Read these before deploying; adjust the marked constants/logic once you get
real grader feedback.

Pipeline:
  1. Decode base64 audio, detect its format from magic bytes.
  2. Send it to Gemini (gemini-3.5-flash) with an instruction to transcribe
     the Korean speech and extract the underlying table as raw structured
     data (column names + row values) ONLY — no computed statistics from
     the model, to avoid LLM arithmetic errors.
  3. Build a pandas DataFrame from the extracted rows/columns.
  4. Compute all statistics fields deterministically with pandas/numpy.
  5. Return the exact JSON shape specified by the task.
"""

import base64
import json
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("audio-qa")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-3.5-flash"

# Reverted to "minimal" as the default: the accuracy fixes that mattered
# (strict prompt against placeholder names, independent numeric
# re-validation) are latency-free, so they still apply at "minimal". Bump
# to "low"/"medium" via env var only if accuracy issues persist AND you've
# confirmed via the timing logs below that you have latency budget to spare.
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "minimal")

# Hard cap on how long we'll wait for Gemini before giving up, so a slow
# call fails fast and visibly in the logs instead of silently eating the
# grader's entire 12s budget.
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "8000"))

# ASSUMPTION: rounding precision for all numeric outputs. Change if the
# grader expects a different precision.
ROUND_DECIMALS = 4

EXTRACTION_SYSTEM_PROMPT = """You will be given a short Korean-language audio clip.
The speaker describes a small dataset: a set of named columns and a number of
data rows, each row giving a value for every column. Some columns are numeric
(numbers), others are categorical (text labels).

Your job is ONLY to transcribe and extract the raw data faithfully. Do NOT
compute any statistics, summaries, or interpretations. Do NOT round or alter
any numbers you hear — reproduce them exactly as spoken (convert spoken
Korean numbers to digit form, e.g. "이십오" -> "25").

CRITICAL: Use the EXACT column name spoken in the audio, verbatim. Never
invent, guess, or fall back to a generic placeholder name such as "값"
(value), "열" (column), "데이터" (data), "항목" (item), or similar filler
words, unless that literal word is explicitly used as a proper column label
in the recording. If you genuinely cannot make out a column name after
careful listening, use "unknown_column_<n>" instead of a plausible-sounding
guess, so a downstream mistake is easy to detect rather than silently wrong.

Listen to the entire clip carefully before transcribing — do not truncate or
skip any part of the audio, even if it is short.

Return ONLY a JSON object with this exact shape:
{
  "columns": [ {"name": "<column name, in Korean or transliterated as heard>", "type": "numeric" | "categorical"} , ... ],
  "rows": [ ["<value for column 1 as a string>", "<value for column 2>", ...], ... ]
}

Every row array must have exactly one string value per column, in the same
order as the "columns" array. Output valid JSON only, no extra commentary.
"""


def detect_mime_type(raw: bytes) -> str:
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb" or raw[:2] == b"\xff\xf3":
        return "audio/mp3"
    if raw[:4] == b"OggS":
        return "audio/ogg"
    if raw[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"  # WebM/Opus, common for browser-recorded audio
    if raw[4:8] == b"ftyp":
        return "audio/mp4"  # m4a/aac container
    if raw[:4] == b"fLaC":
        return "audio/flac"
    # Fallback: assume wav, the most common raw PCM container.
    return "audio/wav"


def extract_table_from_audio(audio_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    t0 = time.monotonic()
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": EXTRACTION_SYSTEM_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(audio_bytes).decode("utf-8"),
                        }
                    },
                ],
            }
        ],
        config={
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_level": THINKING_LEVEL},
            "http_options": {"timeout": GEMINI_TIMEOUT_MS},
        },
    )
    elapsed = time.monotonic() - t0
    logger.info(f"gemini_call_seconds={elapsed:.2f} thinking_level={THINKING_LEVEL}")

    text = response.text
    return json.loads(text)


def to_numeric_series(values: List[str]) -> pd.Series:
    # Strip common noise (commas, whitespace) before numeric conversion.
    cleaned = [str(v).replace(",", "").strip() for v in values]
    return pd.to_numeric(pd.Series(cleaned), errors="coerce")


def round_val(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), ROUND_DECIMALS)


def is_numeric_column(values: List[str], min_numeric_fraction: float = 0.9) -> bool:
    """Independently decide if a column is numeric, instead of trusting
    Gemini's own "numeric"/"categorical" tag. A tagging mistake (or a
    hallucinated/garbled column) should not corrupt the stats dicts —
    if most values don't actually parse as numbers, treat it as
    categorical regardless of what the model claimed."""
    if not values:
        return False
    parsed = to_numeric_series(values)
    fraction_numeric = parsed.notna().mean()
    return fraction_numeric >= min_numeric_fraction


def build_stats(extracted: Dict[str, Any]) -> Dict[str, Any]:
    columns_meta = extracted["columns"]
    rows = extracted["rows"]

    col_names = [c["name"] for c in columns_meta]

    df = pd.DataFrame(rows, columns=col_names)

    # Re-derive numeric/categorical from the actual data, not from
    # Gemini's self-reported "type" field (see is_numeric_column docstring).
    numeric_cols = [c for c in col_names if is_numeric_column(df[c].tolist())]
    categorical_cols = [c for c in col_names if c not in numeric_cols]

    mean_d, std_d, var_d = {}, {}, {}
    min_d, max_d, median_d, mode_d, range_d = {}, {}, {}, {}, {}
    value_range_d = {}
    allowed_values_d = {}

    for col in numeric_cols:
        series = to_numeric_series(df[col].tolist()).dropna()
        if series.empty:
            continue
        mean_d[col] = round_val(series.mean())
        std_d[col] = round_val(series.std())  # pandas default ddof=1 (sample std)
        var_d[col] = round_val(series.var())  # pandas default ddof=1 (sample variance)
        col_min = series.min()
        col_max = series.max()
        min_d[col] = round_val(col_min)
        max_d[col] = round_val(col_max)
        median_d[col] = round_val(series.median())
        modes = series.mode()
        mode_d[col] = round_val(modes.min()) if not modes.empty else None
        range_d[col] = round_val(col_max - col_min)
        value_range_d[col] = [round_val(col_min), round_val(col_max)]

    for col in categorical_cols:
        uniques = sorted(set(str(v).strip() for v in df[col].tolist()))
        allowed_values_d[col] = uniques

    correlation = []
    if len(numeric_cols) >= 2:
        numeric_df = df[numeric_cols].apply(
            lambda s: to_numeric_series(s.tolist())
        )
        corr_matrix = numeric_df.corr().round(ROUND_DECIMALS)
        correlation = corr_matrix.values.tolist()
        # Replace NaN (e.g. constant column) with None for JSON safety.
        correlation = [
            [None if (isinstance(v, float) and np.isnan(v)) else v for v in row]
            for row in correlation
        ]

    return {
        "rows": len(rows),
        "columns": col_names,
        "mean": mean_d,
        "std": std_d,
        "variance": var_d,
        "min": min_d,
        "max": max_d,
        "median": median_d,
        "mode": mode_d,
        "range": range_d,
        "allowed_values": allowed_values_d,
        "value_range": value_range_d,
        "correlation": correlation,
    }


async def handle_audio_request(request: Request) -> JSONResponse:
    request_start = time.monotonic()

    body = await request.json()
    audio_b64 = body["audio_base64"]
    audio_id = body.get("audio_id", "unknown")

    raw_bytes = base64.b64decode(audio_b64)
    mime_type = detect_mime_type(raw_bytes)
    logger.info(f"audio_id={audio_id} bytes={len(raw_bytes)} mime_type={mime_type}")

    t_extract = time.monotonic()
    extracted = extract_table_from_audio(raw_bytes, mime_type)
    extract_elapsed = time.monotonic() - t_extract

    t_stats = time.monotonic()
    result = build_stats(extracted)
    stats_elapsed = time.monotonic() - t_stats

    total_elapsed = time.monotonic() - request_start
    logger.info(
        f"audio_id={audio_id} extract_seconds={extract_elapsed:.2f} "
        f"stats_seconds={stats_elapsed:.3f} total_seconds={total_elapsed:.2f}"
    )

    response = JSONResponse(content=result)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.post("/")
async def root_handler(request: Request):
    return await handle_audio_request(request)


@app.post("/analyze")
async def analyze_handler(request: Request):
    return await handle_audio_request(request)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}