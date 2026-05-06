"""
Evaluate 4 API-based models (ChatGPT, Claude, Gemini, Grok) on full_dataset.jsonl.

The dataset has two formats:
  - Lines with "annotation" key: each entry has an actionAnnotationList with
    start/end timestamps per question. The video is clipped to that segment.
  - Lines with "videoID" key: each entry has a QAs list; each QA may carry
    start/end timestamps. If present, the video is clipped to that segment.

Models NEVER receive the ground-truth answer or gemini_analysis.

Usage:
    python eval_4_models.py \
        --dataset full_dataset.jsonl \
        --output eval_results.jsonl \
        --chatgpt_model gpt-4o \
        --claude_model claude-sonnet-4-5 \
        --gemini_model gemini-2.5-pro \
        --grok_model grok-2-vision-1212
"""

import os
import sys
import json
import time
import hashlib
import base64
import mimetypes
import argparse
import tempfile
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Any

from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from anthropic import Anthropic
from tqdm import tqdm

# Gemini API errors we treat as retryable
_GEMINI_RETRYABLE_STATUS = {429, 500, 503}
CLIP_CACHE_DIR = "./eval_clips_cache"
PROMPT_PLACEHOLDER = "{{PROMPT}}"

# ---------------------------------------------------------------------------
# Socratic sequential-chunk constants  (easy to change in one place)
# ---------------------------------------------------------------------------
SOCRATIC_INTERVAL_S  = 120.0  # seconds per chunk for ChatGPT / Claude / Grok
GEMINI_INTERVAL_S    = 1800.0  # seconds per chunk for Gemini (longer — no frame limit)
SOCRATIC_FPS         = 0.5    # fps for frame downsampling per chunk
SOCRATIC_WIDTH       = 512    # pixel width  for frame downsampling
SOCRATIC_HEIGHT      = 384    # pixel height for frame downsampling
NON_SOCRATIC_FPS = 5.0

# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers (from original codebase)
# ---------------------------------------------------------------------------

def _run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr.strip()}")


def get_video_duration(video_path: str) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception:
        return None


def has_audio_stream(video_path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return "audio" in out.lower()
    except Exception:
        return False


def clip_video(video_path: str, start: float, end: float) -> str:
    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
    key = f"{video_path}|{start:.3f}|{end:.3f}"
    h = hashlib.md5(key.encode()).hexdigest()
    out_path = os.path.join(CLIP_CACHE_DIR, f"{h}.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    if end <= start:
        return video_path

    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
        "-i", video_path, "-c", "copy", out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return video_path
    return out_path


def get_standardized_video(video_path: str, width: int = SOCRATIC_WIDTH, height: int = SOCRATIC_HEIGHT) -> str:
    """Downsample resolution to 512x384 (preserving original FPS)."""
    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)
    key = f"{video_path}|std|{width}|{height}"
    h = hashlib.md5(key.encode()).hexdigest()
    out_path = os.path.join(CLIP_CACHE_DIR, f"{h}_std.mp4")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        out_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return video_path
    return out_path


def extract_frames(video_path: str, out_dir: str, fps: Optional[float] = None) -> List[str]:
    pattern = os.path.join(out_dir, "frame_%04d.jpg")

    cmd = ["ffmpeg", "-y", "-i", video_path]

    # If fps is provided, sample frames at that rate
    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]

    cmd += ["-q:v", "3", pattern]

    _run_cmd(cmd)

    frames = sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    return frames


def extract_audio_wav(video_path: str, wav_path: str) -> None:
    _run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", wav_path,
    ])


def b64_data_url(path: str) -> str:
    mt = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mt};base64,{data}"


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 60.0, exc: Optional[Exception] = None):
    ra = None
    if exc is not None:
        try:
            resp = getattr(exc, "response", None)
            if resp is not None:
                headers = getattr(resp, "headers", None) or {}
                ra_val = headers.get("Retry-After") or headers.get("retry-after")
                if ra_val:
                    ra = float(ra_val)
        except Exception:
            pass
    if ra is not None and ra > 0:
        time.sleep(min(cap, ra))
        return
    sleep_s = min(cap, base * (2 ** attempt)) * (0.7 + 0.6 * random.random())
    time.sleep(sleep_s)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in [
        "timeout", "rate limit", "rate_limit", "temporarily",
        "overloaded", "service", "529", "500", "502", "503",
    ])


def _api_call_with_retry(fn, max_retries: int = 6):
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1 or not _is_retryable(e):
                raise
            _backoff_sleep(attempt, exc=e)


# ---------------------------------------------------------------------------
# Gemini helpers  (google.genai SDK)
# ---------------------------------------------------------------------------

def _gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _is_gemini_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in _GEMINI_RETRYABLE_STATUS:
        return True
    return _is_retryable(exc)


def _gemini_upload(client: genai.Client, video_path: str, max_retries: int = 8):
    for attempt in range(max_retries):
        try:
            return client.files.upload(file=video_path)
        except Exception as e:
            if attempt == max_retries - 1 or not _is_gemini_retryable(e):
                raise
            _backoff_sleep(attempt, exc=e)


def _gemini_wait_active(client: genai.Client, video_file, max_wait_s: int = 600, poll_s: float = 3.0):
    start = time.time()
    while True:
        if time.time() - start > max_wait_s:
            raise TimeoutError(f"Gemini upload not ACTIVE within {max_wait_s}s: {video_file.name}")
        try:
            video_file = client.files.get(name=video_file.name)
        except Exception:
            time.sleep(poll_s)
            continue
        state = getattr(video_file, "state", None)
        state_name = state.name if hasattr(state, "name") else str(state)
        if state_name == "ACTIVE":
            return video_file
        if state_name in ("FAILED", "ERROR"):
            raise RuntimeError(f"Gemini upload failed (state={state_name})")
        time.sleep(poll_s)


def _gemini_generate(client: genai.Client, model_name: str, contents, temperature: float, max_tokens: int, max_retries: int = 8):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
        except Exception as e:
            if attempt == max_retries - 1 or not _is_gemini_retryable(e):
                raise
            _backoff_sleep(attempt, exc=e)


def _cleanup_gemini_files(client: genai.Client, keep_n: int = 5):
    try:
        files = list(client.files.list())
        if len(files) <= keep_n:
            return
        files_sorted = sorted(
            files,
            key=lambda f: getattr(f, "create_time", None) or datetime.min,
            reverse=True,
        )
        for f in files_sorted[keep_n:]:
            try:
                client.files.delete(name=getattr(f, "name", ""))
            except Exception:
                pass
            time.sleep(0.05)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-model query functions
# ---------------------------------------------------------------------------

def query_gemini(
    video_path: str,
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    upload_cache: Dict[str, Any],
    gemini_client: genai.Client,
) -> str:
    """Gemini receives the actual video file via the Files API upload."""
    video_file = upload_cache.get(video_path)
    if video_file is None:
        video_file = _gemini_upload(gemini_client, video_path)
        video_file = _gemini_wait_active(gemini_client, video_file)
        upload_cache[video_path] = video_file

    response = _gemini_generate(
        gemini_client, model_name, [prompt, video_file], temperature, max_tokens
    )

    try:
        text = response.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass

    out: List[str] = []
    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
            txt = getattr(part, "text", None)
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
    return "\n".join(out).strip()


def _frames_and_transcript(video_path: str, td: str):
    """Extract frames + audio transcript (shared by GPT / Claude / Grok).

    Returns (frames, audio_note) where audio_note is a ready-to-append string:
      - transcript text if speech was detected
      - "(no speech detected)" if audio track exists but is silent/non-verbal
      - "" if no audio track could be extracted at all
    """
    frames_dir = os.path.join(td, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    wav_path = os.path.join(td, "audio.wav")

    frames: List[str] = []
    try:
        frames = extract_frames(video_path, frames_dir, fps=NON_SOCRATIC_FPS)
    except Exception:
        pass

    audio_note = ""
    try:
        extract_audio_wav(video_path, wav_path)
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            client = OpenAI()
            with open(wav_path, "rb") as f:
                tr = _api_call_with_retry(
                    lambda: client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe", file=f,
                    )
                )
            transcript = getattr(tr, "text", "") or ""
            audio_note = transcript.strip() if transcript.strip() else "(no speech detected)"
    except Exception:
        pass

    return frames, audio_note


def query_chatgpt(
    video_path: str,
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    frames: Optional[List[str]] = None,
    audio_note: Optional[str] = None,
) -> str:
    """GPT receives sampled frames as base64 images + audio transcript.

    Uses chat.completions (not responses.create) because the Responses API
    does not support base64 data URIs for images — only public HTTPS URLs.
    """
    client = OpenAI()
    td_obj = None

    try:
        if frames is None or audio_note is None:
            td_obj = tempfile.TemporaryDirectory()
            frames, audio_note = _frames_and_transcript(video_path, td_obj.name)

        # Standard chat.completions vision format
        content_parts: List[dict] = []
        for fp in frames:
            try:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": b64_data_url(fp), "detail": "high"},
                })
            except Exception:
                continue

        prompt_text = prompt
        if audio_note:
            prompt_text += f"\n\nHere is the audio transcript of the video:\n{audio_note}"
        content_parts.append({"type": "text", "text": prompt_text})

        resp = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        )

        return (resp.choices[0].message.content or "").strip()
    finally:
        if td_obj is not None:
            td_obj.cleanup()


def query_claude(
    video_path: str,
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    frames: Optional[List[str]] = None,
    audio_note: Optional[str] = None,
) -> str:
    """Claude receives sampled frames as base64 images + audio transcript."""
    client = Anthropic()
    td_obj = None

    try:
        if frames is None or audio_note is None:
            td_obj = tempfile.TemporaryDirectory()
            frames, audio_note = _frames_and_transcript(video_path, td_obj.name)

        content_parts: List[dict] = []
        for fp in frames:
            try:
                with open(fp, "rb") as f:
                    image_data = base64.standard_b64encode(f.read()).decode()
                content_parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                })
            except Exception:
                continue

        prompt_text = prompt
        if audio_note:
            prompt_text += f"\n\nHere is the audio transcript of the video:\n{audio_note}"
        content_parts.append({"type": "text", "text": prompt_text})

        resp = _api_call_with_retry(
            lambda: client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content_parts}],
            )
        )

        text_parts: List[str] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()
    finally:
        if td_obj is not None:
            td_obj.cleanup()


def query_grok(
    video_path: str,
    prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    frames: Optional[List[str]] = None,
    audio_note: Optional[str] = None,
) -> str:
    """Grok uses the xAI chat.completions API with frames + transcript.

    xAI is OpenAI-compatible but uses chat.completions (not responses),
    and vision images use type="image_url" with a data-URI.
    Vision model: grok-2-vision-1212 (or similar).
    """
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    td_obj = None

    try:
        if frames is None or audio_note is None:
            td_obj = tempfile.TemporaryDirectory()
            frames, audio_note = _frames_and_transcript(video_path, td_obj.name)

        # xAI chat.completions vision format
        content_parts: List[dict] = []
        for fp in frames:
            try:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": b64_data_url(fp), "detail": "high"},
                })
            except Exception:
                continue

        prompt_text = prompt
        if audio_note:
            prompt_text += f"\n\nHere is the audio transcript of the video:\n{audio_note}"
        content_parts.append({"type": "text", "text": prompt_text})

        resp = _api_call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        return (resp.choices[0].message.content or "").strip()
    finally:
        if td_obj is not None:
            td_obj.cleanup()


# ---------------------------------------------------------------------------
# Socratic sequential-chunk helpers
# ---------------------------------------------------------------------------

def extract_frames_resized(
    video_path: str,
    out_dir: str,
    fps: float = SOCRATIC_FPS,
    width: int = SOCRATIC_WIDTH,
    height: int = SOCRATIC_HEIGHT,
) -> List[str]:
    """Extract frames at *fps* and scale to *width*x*height*."""
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    _run_cmd([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-q:v", "3", pattern,
    ])
    return sorted(
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )


def transcribe_with_timestamps(wav_path: str) -> str:
    """Transcribe *wav_path* with whisper-1 (verbose_json + segment timestamps).

    Returns a formatted string like::

        [0.0s – 12.4s]: Some spoken words.
        [12.4s – 20.1s]: More words.

    Returns an empty string if the file is missing, empty, or silent.
    """
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        return ""
    try:
        client = OpenAI()
        # Read bytes up-front so the file handle is not exhausted on retries
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        filename = os.path.basename(wav_path)
        result = _api_call_with_retry(
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=(filename, audio_bytes, "audio/wav"),
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        )
        segments = getattr(result, "segments", None) or []
        if segments:
            lines = []
            for seg in segments:
                t0 = getattr(seg, "start", 0.0)
                t1 = getattr(seg, "end", 0.0)
                txt = (getattr(seg, "text", "") or "").strip()
                if txt:
                    lines.append(f"[{t0:.1f}s \u2013 {t1:.1f}s]: {txt}")
            if lines:
                return "\n".join(lines)
        # Fallback: plain text (no timestamps)
        text = (getattr(result, "text", "") or "").strip()
        return text
    except Exception as e:
        tqdm.write(f"    [WARN] transcription error: {type(e).__name__}: {e}")
        return ""


# -- Per-model single-call helpers used inside run_socratic_sequential -------

def _socratic_call_chatgpt(
    frames: List[str],
    text_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = OpenAI()
    content_parts: List[dict] = []
    for fp in frames:
        try:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": b64_data_url(fp), "detail": "high"},
            })
        except Exception:
            continue
    content_parts.append({"type": "text", "text": text_prompt})
    resp = _api_call_with_retry(
        lambda: client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_parts}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    )
    return (resp.choices[0].message.content or "").strip()


def _socratic_call_claude(
    frames: List[str],
    text_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = Anthropic()
    content_parts: List[dict] = []
    for fp in frames:
        try:
            with open(fp, "rb") as fh:
                image_data = base64.standard_b64encode(fh.read()).decode()
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data,
                },
            })
        except Exception:
            continue
    content_parts.append({"type": "text", "text": text_prompt})
    resp = _api_call_with_retry(
        lambda: client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content_parts}],
        )
    )
    return "\n".join(
        block.text for block in resp.content if block.type == "text"
    ).strip()


def _socratic_call_grok(
    frames: List[str],
    text_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = OpenAI(
        api_key=os.environ.get("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )
    content_parts: List[dict] = []
    for fp in frames:
        try:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": b64_data_url(fp), "detail": "high"},
            })
        except Exception:
            continue
    content_parts.append({"type": "text", "text": text_prompt})
    resp = _api_call_with_retry(
        lambda: client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content_parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    return (resp.choices[0].message.content or "").strip()


_SOCRATIC_MODEL_CALLERS = {
    "gpt":    _socratic_call_chatgpt,
    "claude": _socratic_call_claude,
    "grok":   _socratic_call_grok,
}


def run_socratic_sequential(
    video_path: str,
    question: str,
    model_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    interval_s: float = SOCRATIC_INTERVAL_S,
    fps: float = SOCRATIC_FPS,
    width: int = SOCRATIC_WIDTH,
    height: int = SOCRATIC_HEIGHT,
    precomputed_transcripts: Optional[Dict[str, str]] = None,
    precomputed_frames: Optional[Dict[str, List[str]]] = None,
) -> tuple:  # (final_answer: str, transcripts: Dict[str, str])
    """Process *video_path* in sequential chunks for non-Gemini models.

    If *precomputed_transcripts* / *precomputed_frames* are provided, the
    per-chunk Whisper API call and ffmpeg frame extraction are skipped
    (resources shared across models, computed once before the thread pool).
    Returns (final_answer, transcripts) where transcripts maps
    chunk_label -> raw transcript text (for debugging).
    """
    caller = _SOCRATIC_MODEL_CALLERS.get(model_key)
    if caller is None:
        raise ValueError(f"No Socratic caller for model_key={model_key!r}")

    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        raise RuntimeError(f"Cannot determine duration for {video_path}")

    # Build chunk boundaries
    chunks: List[tuple] = []   # (start_s, end_s)
    t = 0.0
    while t < duration:
        chunks.append((t, min(t + interval_s, duration)))
        t += interval_s
    n_chunks = len(chunks)

    prev_cot = ""
    all_cots: List[str] = []
    all_transcripts: Dict[str, str] = {}  # chunk_label -> transcript

    for idx, (start_s, end_s) in enumerate(chunks):
        chunk_label = f"chunk {idx + 1}/{n_chunks}  [{_fmt_ts(start_s)} \u2013 {_fmt_ts(end_s)}]"
        tqdm.write(f"  [Socratic/{model_key}] {chunk_label}")

        seg_path = clip_video(video_path, start_s, end_s)

        with tempfile.TemporaryDirectory() as td:
            # 1. Extract frames — use pre-computed if available
            if precomputed_frames is not None and chunk_label in precomputed_frames:
                frames = precomputed_frames[chunk_label]
            else:
                frames_dir = os.path.join(td, "frames")
                os.makedirs(frames_dir, exist_ok=True)
                frames = []
                try:
                    frames = extract_frames_resized(seg_path, frames_dir, fps, width, height)
                except Exception as e:
                    tqdm.write(f"    [WARN] frame extraction failed: {e}")

            # 2. Transcribe — use pre-computed value if available
            if precomputed_transcripts is not None and chunk_label in precomputed_transcripts:
                transcript = precomputed_transcripts[chunk_label]
            else:
                wav_path = os.path.join(td, "audio.wav")
                transcript = ""
                try:
                    extract_audio_wav(seg_path, wav_path)
                    transcript = transcribe_with_timestamps(wav_path)
                except Exception as e:
                    tqdm.write(f"    [WARN] transcript failed: {e}")

            all_transcripts[chunk_label] = transcript or "(empty)"
            tqdm.write(f"    [transcript/{model_key}] {len(transcript)} chars")

            # 3. Build per-chunk prompt
            transcript_block = (
                f"Here is the audio transcript for this chunk:\n{transcript}"
                if transcript
                else "(no speech detected in this chunk)"
            )
            context_block = (
                f"Previous chain-of-thought reasoning from earlier chunks:\n{prev_cot}\n"
                if prev_cot
                else ""
            )
            chunk_prompt = (
                f"You are analyzing a video chunk by chunk.\n"
                f"This is {chunk_label}.\n\n"
                f"{transcript_block}\n\n"
                f"{context_block}"
                f"The frames below are sampled from this chunk at {fps} fps "
                f"({width}\u00d7{height} px).\n\n"
                f"Reason step by step about what you observe. "
                f"Do NOT give a final answer yet \u2014 just build your chain-of-thought.\n"
                f"Question (answer after all {n_chunks} chunks): {question}"
            )

            # 4. Call the model
            cot = caller(frames, chunk_prompt, model_name, temperature, max_tokens)

        prev_cot = cot
        all_cots.append(f"=== {chunk_label} ===\n{cot}")

    # Final synthesis pass (text only, no frames)
    cot_all = "\n\n".join(all_cots)
    
    # Aggregate all transcripts for the final synthesis
    agg_transcripts = []
    for label, txt in all_transcripts.items():
        if txt and txt != "(empty)":
            agg_transcripts.append(f"--- {label} ---\n{txt}")
    full_transcript_text = "\n\n".join(agg_transcripts)
    if full_transcript_text:
        transcript_context = f"Here is the full audio transcript of the video for reference:\n{full_transcript_text}\n\n"
    else:
        transcript_context = "No speech was detected in the video.\n\n"

    final_prompt = (
        f"You have now analyzed all {n_chunks} chunk(s) of the video.\n"
        f"Here is your accumulated chain-of-thought reasoning from each chunk:\n\n"
        f"{cot_all}\n\n"
        f"{transcript_context}"
        f"Now synthesize all of the above into a final response to the question:\n"
        f"{question}\n\n"
        f"Structure your response as:\n"
        f"Here's the reasoning process:\n"
        f"1. Visual Cue:\n"
        f"   - Timeframe:\n"
        f"   - Description:\n"
        f"   - Analysis:\n"
        f"2. Audio Cue:\n"
        f"   - Timeframe:\n"
        f"   - Description:\n"
        f"   - Analysis:\n\n"
        f"Final Answer:"
    )
    tqdm.write(f"  [Socratic/{model_key}] final synthesis pass")
    final_answer = caller([], final_prompt, model_name, temperature, max_tokens)
    return final_answer, all_transcripts


def _fmt_ts(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def run_socratic_gemini(
    video_path: str,
    question: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    gemini_client: genai.Client,
    upload_cache: Dict[str, Any],
    interval_s: float = GEMINI_INTERVAL_S,
    fps: float = SOCRATIC_FPS,
    width: int = SOCRATIC_WIDTH,
    height: int = SOCRATIC_HEIGHT,
) -> str:
    """Socratic sequential chunking for Gemini.

    Each chunk is clipped from the original video, then downsampled to
    *fps* / *width*x*height* (default 0.5 fps, 512×384) before being
    uploaded to Gemini via the Files API (audio preserved).
    Rolling CoT from each chunk is passed as text context into the next one.
    """
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        raise RuntimeError(f"Cannot determine duration for {video_path}")

    chunks: List[tuple] = []
    t = 0.0
    while t < duration:
        chunks.append((t, min(t + interval_s, duration)))
        t += interval_s
    n_chunks = len(chunks)

    prev_cot = ""
    all_cots: List[str] = []

    for idx, (start_s, end_s) in enumerate(chunks):
        chunk_label = f"chunk {idx + 1}/{n_chunks}  [{_fmt_ts(start_s)} \u2013 {_fmt_ts(end_s)}]"
        tqdm.write(f"  [Socratic/gemini] {chunk_label}")

        seg_path = clip_video(video_path, start_s, end_s)
        # resolution already standardized to 512x384 in main()

        context_block = (
            f"Previous chain-of-thought reasoning from earlier chunks:\n{prev_cot}\n\n"
            if prev_cot
            else ""
        )
        chunk_prompt = (
            f"You are analyzing a video chunk by chunk.\n"
            f"This is {chunk_label}.\n\n"
            f"{context_block}"
            f"Analyze the video clip carefully (audio + visuals).\n"
            f"Reason step by step about what you observe. "
            f"Do NOT give a final answer yet \u2014 just build your chain-of-thought.\n"
            f"Question (answer after all {n_chunks} chunks): {question}"
        )

        cot = query_gemini(
            seg_path, chunk_prompt, model_name, temperature, max_tokens,
            upload_cache, gemini_client,
        )
        prev_cot = cot
        all_cots.append(f"=== {chunk_label} ===\n{cot}")

    # Final synthesis pass — text only, no video
    cot_all = "\n\n".join(all_cots)
    final_prompt = (
        f"You have now analyzed all {n_chunks} chunk(s) of the video.\n"
        f"Here is your accumulated chain-of-thought reasoning from each chunk:\n\n"
        f"{cot_all}\n\n"
        f"Now synthesize all of the above into a final response to the question:\n"
        f"{question}\n\n"
        f"Structure your response as:\n"
        f"Here's the reasoning process:\n"
        f"1. Visual Cue:\n"
        f"   - Timeframe:\n"
        f"   - Description:\n"
        f"   - Analysis:\n"
        f"2. Audio Cue:\n"
        f"   - Timeframe:\n"
        f"   - Description:\n"
        f"   - Analysis:\n\n"
        f"Final Answer:"
    )
    tqdm.write("  [Socratic/gemini] final synthesis pass")
    # Synthesis is text-only: pass an empty placeholder video path so query_gemini
    # uses only the prompt. We call the generate API directly instead.
    response = _gemini_generate(
        gemini_client, model_name, [final_prompt], temperature, max_tokens
    )
    try:
        text = response.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    out: List[str] = []
    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(getattr(cand, "content", None), "parts", []) or []:
            txt = getattr(part, "text", None)
            if isinstance(txt, str) and txt.strip():
                out.append(txt.strip())
    return "\n".join(out).strip()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

MODEL_DISPATCH = {
    "gemini": query_gemini,
    "gpt": query_chatgpt,
    "claude": query_claude,
    "grok": query_grok,
}


def dispatch_model(
    model_key: str,
    model_name: str,
    video_path: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    gemini_upload_cache: Dict[str, Any],
    gemini_client: Optional[genai.Client] = None,
    # Socratic args
    use_socratic: bool = False,
    question: str = "",
    socratic_interval: float = SOCRATIC_INTERVAL_S,
    gemini_interval: float = GEMINI_INTERVAL_S,
    socratic_fps: float = SOCRATIC_FPS,
    socratic_width: int = SOCRATIC_WIDTH,
    socratic_height: int = SOCRATIC_HEIGHT,
    precomputed_transcripts: Optional[Dict[str, str]] = None,  # shared across models
    precomputed_frames: Optional[Dict[str, List[str]]] = None,  # shared across models
    non_socratic_frames: Optional[List[str]] = None,
    non_socratic_audio_note: Optional[str] = None,
) -> str:
    if model_key == "gemini":
        if use_socratic and question:
            # Socratic chunked processing for Gemini
            answer = run_socratic_gemini(
                video_path=video_path,
                question=question,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                gemini_client=gemini_client,
                upload_cache=gemini_upload_cache,
                interval_s=gemini_interval,
            )
            return answer, {}
        # Non-Socratic: uses the standardized video path directly
        return query_gemini(video_path, prompt, model_name, temperature, max_tokens, gemini_upload_cache, gemini_client), {}

    if use_socratic and question:
        return run_socratic_sequential(
            video_path=video_path,
            question=question,
            model_key=model_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            interval_s=socratic_interval,
            fps=socratic_fps,
            width=socratic_width,
            height=socratic_height,
            precomputed_transcripts=precomputed_transcripts,
            precomputed_frames=precomputed_frames,
        )

    # Standard (non-Socratic) path
    if model_key == "gpt":
        return query_chatgpt(video_path, prompt, model_name, temperature, max_tokens, non_socratic_frames, non_socratic_audio_note), {}
    elif model_key == "claude":
        return query_claude(video_path, prompt, model_name, temperature, max_tokens, non_socratic_frames, non_socratic_audio_note), {}
    elif model_key == "grok":
        return query_grok(video_path, prompt, model_name, temperature, max_tokens, non_socratic_frames, non_socratic_audio_note), {}
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def model_key_from_name(name: str) -> str:
    n = name.lower()
    if n.startswith("gemini"):
        return "gemini"
    if n.startswith("gpt"):
        return "gpt"
    if n.startswith("claude"):
        return "claude"
    if n.startswith("grok"):
        return "grok"
    raise ValueError(f"Cannot determine model key from name: {name!r}")


# ---------------------------------------------------------------------------
# Dataset parsing
# ---------------------------------------------------------------------------

def parse_dataset(dataset_path: str):
    """
    Yields dicts with keys:
        video_path, question, answer, start, end, line_num, format_type
    answer is kept for result recording only — never sent to models.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skipping malformed JSON at line {line_num}", file=sys.stderr)
                continue

            if "annotation" in obj:
                annotation = obj["annotation"]
                video_src = annotation.get("video", {}).get("src", "")
                if not video_src or not os.path.isfile(video_src):
                    print(f"[WARN] Line {line_num}: video not found: {video_src}", file=sys.stderr)
                    continue

                for action in annotation.get("actionAnnotationList", []):
                    if not isinstance(action, dict):
                        continue
                    question = action.get("question")
                    answer = action.get("answer")
                    if not question:
                        continue
                    start = _safe_float(action.get("start"))
                    end = _safe_float(action.get("end"))
                    yield {
                        "video_path": video_src,
                        "question": question.strip(),
                        "answer": (answer or "").strip(),
                        "start": start,
                        "end": end,
                        "line_num": line_num,
                        "format_type": "annotation",
                    }

            elif "videoID" in obj:
                video_path = obj["videoID"]
                if not video_path or not os.path.isfile(video_path):
                    print(f"[WARN] Line {line_num}: video not found: {video_path}", file=sys.stderr)
                    continue

                for qa in obj.get("QAs", []):
                    if not isinstance(qa, dict):
                        continue
                    question = qa.get("question")
                    answer = qa.get("answer")
                    if not question:
                        continue
                    start = _safe_float(qa.get("start"))
                    end   = _safe_float(qa.get("end"))
                    yield {
                        "video_path": video_path,
                        "question": question.strip(),
                        "answer": (answer or "").strip(),
                        "start": start,
                        "end": end,
                        "line_num": line_num,
                        "format_type": "videoID",
                    }


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_prompt(question: str,) -> str:
    """
    Build the final prompt. If the user provided a template with {{PROMPT}},
    substitute the question there. Otherwise use a default wrapper.
    """

    prompt = f"""
    From the given video or sequence of images, analyze deeply using the audio and visual cues, and produce detailed chain-of-thought reasoning.

    Question: {question}

    Please structure the reasoning as:

    Here's the reasoning process:

    1. Visual Cue:
    - Timeframe:
    - Description:
    - Analysis:

    2. Audio Cue:
    - Timeframe:
    - Description:
    - Analysis:

    Final Answer:
    """.strip()

    return prompt

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate 4 API models on a video QA dataset",
    )
    parser.add_argument("--dataset", required=True, help="Path to full_dataset.jsonl")
    parser.add_argument("--output", required=True, help="Output JSONL path for results")
    parser.add_argument("--chatgpt_model", default="gpt-5.2", help="ChatGPT model name (default: gpt-4o)")
    parser.add_argument("--claude_model", default="claude-sonnet-4-6", help="Claude model name (default: claude-sonnet-4-5)")
    parser.add_argument("--gemini_model", default="gemini-3-flash-preview", help="Gemini model name (default: gemini-2.5-pro)")
    parser.add_argument("--grok_model", default="grok-4-1-fast-reasoning", help="Grok vision model name (default: grok-2-vision-1212)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature (default: 0.3)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens (default: 1024)")
    parser.add_argument("--prompt_file", default="", help="Path to a text file with the prompt template. Use {{PROMPT}} as placeholder for the question.")
    parser.add_argument("--start_line", type=int, default=1, help="Start processing from this dataset line number (1-indexed, default: 1)")
    parser.add_argument("--end_line", type=int, default=-1, help="Stop processing after this dataset line number (-1 = all, default: -1)")
    parser.add_argument("--gemini_cleanup_interval", type=int, default=20, help="Clean up Gemini uploaded files every N questions (default: 20)")
    parser.add_argument("--shard", type=int, default=0, help="Shard index, 0-indexed (default: 0)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards / parallel workers (default: 1 = no sharding)")
    # --- Socratic sequential-chunk args ---
    parser.add_argument("--use_socratic", action="store_true",
                        help="Enable Socratic sequential-chunk processing for non-Gemini models")
    parser.add_argument("--socratic_interval", type=float, default=SOCRATIC_INTERVAL_S,
                        help=f"Chunk duration in seconds for ChatGPT/Claude/Grok (default: {SOCRATIC_INTERVAL_S})")
    parser.add_argument("--gemini_socratic_interval", type=float, default=GEMINI_INTERVAL_S,
                        help=f"Chunk duration in seconds for Gemini (default: {GEMINI_INTERVAL_S})")
    parser.add_argument("--socratic_fps", type=float, default=SOCRATIC_FPS,
                        help=f"FPS for frame extraction per chunk (default: {SOCRATIC_FPS})")
    parser.add_argument("--socratic_width", type=int, default=SOCRATIC_WIDTH,
                        help=f"Frame width for chunk downsampling (default: {SOCRATIC_WIDTH})")
    parser.add_argument("--socratic_height", type=int, default=SOCRATIC_HEIGHT,
                        help=f"Frame height for chunk downsampling (default: {SOCRATIC_HEIGHT})")
    parser.add_argument("--socratic_threshold", type=float, default=20.0,
                        help="Only apply Socratic if video duration >= N seconds (default: 0 = always)")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard < args.num_shards):
        raise ValueError(f"--shard must be in [0, num_shards). Got shard={args.shard}, num_shards={args.num_shards}")

    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    xai_key = os.environ.get("XAI_API_KEY")

    missing = []
    if not google_key:
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not anthropic_key:
        missing.append("ANTHROPIC_API_KEY")
    if not xai_key:
        missing.append("XAI_API_KEY")
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    gemini_client = genai.Client(api_key=google_key)

    models = {
        "chatgpt": {"name": args.chatgpt_model, "key": model_key_from_name(args.chatgpt_model)},
        "claude": {"name": args.claude_model, "key": model_key_from_name(args.claude_model)},
        "gemini": {"name": args.gemini_model, "key": model_key_from_name(args.gemini_model)},
        "grok": {"name": args.grok_model, "key": model_key_from_name(args.grok_model)},
    }

    user_prompt_template = ""
    if args.prompt_file and os.path.isfile(args.prompt_file):
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            user_prompt_template = f.read().strip()
        print(f"Loaded prompt template from {args.prompt_file}")

    # If sharding, append _shardN to the output filename so each worker writes
    # to its own file (avoids interleaved writes / race conditions).
    output_path = args.output
    if args.num_shards > 1:
        base, ext = os.path.splitext(args.output)
        output_path = f"{base}_shard{args.shard}{ext or '.jsonl'}"

    shard_info = f"shard {args.shard}/{args.num_shards}" if args.num_shards > 1 else "no sharding"
    print(f"Models: {json.dumps({k: v['name'] for k, v in models.items()}, indent=2)}")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {output_path}  ({shard_info})")
    print(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    print(f"Line range: {args.start_line} to {'end' if args.end_line == -1 else args.end_line}")
    print("-" * 60)

    already_done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    done_key = (rec.get("video_path", ""), rec.get("question", ""))
                    already_done.add(done_key)
                except Exception:
                    pass
        print(f"Resuming: {len(already_done)} entries already processed in {output_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(CLIP_CACHE_DIR, exist_ok=True)

    # Pre-collect all entries so tqdm knows the total up front.
    # Sharding is applied here: entry i (0-indexed) goes to shard (i % num_shards).
    print("Scanning dataset...", end=" ", flush=True)
    all_entries = [
        e for i, e in enumerate(parse_dataset(args.dataset))
        if args.start_line <= e["line_num"]
        and (args.end_line == -1 or e["line_num"] <= args.end_line)
        and (i % args.num_shards == args.shard)
    ]
    total_unsharded = sum(
        1 for e in parse_dataset(args.dataset)
        if args.start_line <= e["line_num"]
        and (args.end_line == -1 or e["line_num"] <= args.end_line)
    )
    print(
        f"{len(all_entries)} QA entries for this shard"
        + (f" ({args.shard + 1}/{args.num_shards}, {total_unsharded} total)" if args.num_shards > 1 else ".")
    )

    gemini_upload_cache: Dict[str, Any] = {}
    duration_cache: Dict[str, Optional[float]] = {}
    standard_video_cache: Dict[str, str] = {}
    socratic_precomp_cache: Dict[str, Dict[str, Any]] = {}  # video_path -> {transcripts: {}, frames: {}, frames_dir: str}

    total_processed = 0
    total_skipped = 0
    total_errors = 0

    bar = tqdm(
        all_entries,
        total=len(all_entries),
        unit="qa",
        dynamic_ncols=True,
        desc="Evaluating",
    )

    with open(output_path, "a", encoding="utf-8") as fout:
        for entry in bar:
            video_path = entry["video_path"]
            question = entry["question"]
            gt_answer = entry["answer"]
            start = entry["start"]
            end = entry["end"]
            duration = get_video_duration(video_path)

            resume_key = (video_path, question)
            if resume_key in already_done:
                total_skipped += 1
                bar.set_postfix(done=total_processed, skip=total_skipped, err=total_errors)
                continue

            video_name = os.path.basename(video_path)
            short_q = question[:50] + "…" if len(question) > 50 else question
            bar.set_description(f"[{video_name}] {short_q}")

            # Determine the video segment to send
            # raw_video_path: unmodified original or clip — used by Gemini (preprocessed
            # internally) and the Socratic chunker (which does its own resampling).
            # qa_video_path:  may be downsampled — used only for the legacy standard path
            #                 (non-Socratic ChatGPT/Claude/Grok).
            if start is not None and end is not None:
                clipped_path = clip_video(video_path, start, end)
            else:
                clipped_path = video_path


            prompt = build_prompt(question)

            # Determine whether to use Socratic for this entry's video
            # (threshold is 300s as per requirement)
            use_socratic_here = args.use_socratic
            if use_socratic_here:
                if start is not None and end is not None and end > start:
                    vid_dur = end - start
                else:
                    # includes start==end (or end<start): treat as "full video"
                    vid_dur = duration or 0.0

                if vid_dur < args.socratic_threshold:
                    use_socratic_here = False

            model_answers: Dict[str, str] = {}
            debug_transcripts: Dict[str, Any] = {}  # label -> {chunk_label: transcript}
            entry_errors = 0

            # ---------------------------------------------------------------
            # Pre-compute Socratic resources once per video/clip; shared globally
            # ---------------------------------------------------------------
            shared_transcripts: Optional[Dict[str, str]] = None
            shared_frames: Optional[Dict[str, List[str]]] = None
            
            if use_socratic_here:
                if clipped_path not in socratic_precomp_cache:
                    try:
                        dur_for_chunks = get_video_duration(clipped_path)
                        if dur_for_chunks and dur_for_chunks > 0:
                            precomp_chunks: List[tuple] = []
                            t = 0.0
                            while t < dur_for_chunks:
                                precomp_chunks.append((t, min(t + args.socratic_interval, dur_for_chunks)))
                                t += args.socratic_interval
                            n_total = len(precomp_chunks)

                            trans = {}
                            frms = {}
                            f_dir = tempfile.mkdtemp(prefix="socratic_frames_")

                            for cidx, (cs, ce) in enumerate(precomp_chunks):
                                lbl = f"chunk {cidx + 1}/{n_total}  [{_fmt_ts(cs)} \u2013 {_fmt_ts(ce)}]"
                                
                                # CRITICAL: Always clip from the original source for robustness
                                seg = clip_video(clipped_path, cs, ce)

                                chunk_frame_dir = os.path.join(f_dir, f"chunk_{cidx:04d}")
                                os.makedirs(chunk_frame_dir, exist_ok=True)
                                try:
                                    fr_list = extract_frames_resized(
                                        seg, chunk_frame_dir,
                                        args.socratic_fps, args.socratic_width, args.socratic_height
                                    )
                                    frms[lbl] = fr_list
                                    tqdm.write(f"  [precomp] {lbl}: {len(fr_list)} frames")
                                except Exception as e:
                                    frms[lbl] = []
                                    tqdm.write(f"  [precomp] {lbl}: frame extraction WARN: {e}")

                                with tempfile.TemporaryDirectory() as td:
                                    wav = os.path.join(td, "audio.wav")
                                    try:
                                        extract_audio_wav(seg, wav)
                                        tr = transcribe_with_timestamps(wav)
                                    except Exception as e:
                                        tqdm.write(f"  [precomp] {lbl}: transcript WARN: {e}")
                                        tr = ""
                                trans[lbl] = tr
                                tqdm.write(f"  [precomp] {lbl}: {len(tr)} transcript chars")

                            socratic_precomp_cache[clipped_path] = {
                                "transcripts": trans,
                                "frames": frms,
                                "frames_dir": f_dir
                            }
                    except Exception as e:
                        tqdm.write(f"  [precomp] global cache failed for {clipped_path}: {e}")

                # Retrieve from global cache
                cache_hit = socratic_precomp_cache.get(clipped_path)
                if cache_hit:
                    shared_transcripts = cache_hit["transcripts"]
                    shared_frames = cache_hit["frames"]

            # For non-Socratic path, we use the standardized video (512x384, original FPS)
            standard_video = None
            non_socratic_td = None
            non_socratic_frames = None
            non_socratic_audio_note = None

            if not use_socratic_here:
                if clipped_path not in standard_video_cache:
                    standard_video_cache[clipped_path] = get_standardized_video(clipped_path)
                standard_video = standard_video_cache[clipped_path]

                # Precompute frames and audio for GPT/Claude/Grok so they don't do it 3x
                non_socratic_td = tempfile.TemporaryDirectory()
                try:
                    non_socratic_frames, non_socratic_audio_note = _frames_and_transcript(
                        standard_video, non_socratic_td.name
                    )
                except Exception as e:
                    tqdm.write(f"  [WARN] non-socratic precompute failed: {e}")

            model_answers: Dict[str, str] = {}
            debug_transcripts: Dict[str, Any] = {}
            entry_errors = 0

            def _call_model(label: str, minfo: dict) -> tuple:
                answer, debug = dispatch_model(
                    model_key=minfo["key"],
                    model_name=minfo["name"],
                    video_path=standard_video or clipped_path,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    gemini_upload_cache=gemini_upload_cache,
                    gemini_client=gemini_client,
                    use_socratic=use_socratic_here,
                    question=question,
                    socratic_interval=args.socratic_interval,
                    gemini_interval=args.gemini_socratic_interval,
                    socratic_fps=args.socratic_fps,
                    socratic_width=args.socratic_width,
                    socratic_height=args.socratic_height,
                    precomputed_transcripts=shared_transcripts,
                    precomputed_frames=shared_frames,
                    non_socratic_frames=non_socratic_frames,
                    non_socratic_audio_note=non_socratic_audio_note,
                )
                return label, answer, debug

            try:
                with ThreadPoolExecutor(max_workers=len(models)) as pool:
                    futures = {
                        pool.submit(_call_model, label, minfo): label
                        for label, minfo in models.items()
                    }
                    bar.set_postfix(model="[parallel]", done=total_processed,
                                    skip=total_skipped, err=total_errors)
                    for fut in as_completed(futures):
                        label = futures[fut]
                        try:
                            lbl, answer, debug = fut.result()
                            model_answers[lbl] = answer
                            if debug:
                                debug_transcripts[lbl] = debug
                            tqdm.write(f"  [{lbl}] OK  ({len(answer)} chars)  \u2014 {video_name}")
                        except Exception as e:
                            model_answers[label] = f"ERROR: {type(e).__name__}: {e}"
                            tqdm.write(f"  [{label}] FAIL \u2014 {video_name}: {e}", file=sys.stderr)
                            entry_errors += 1
            except Exception as e:
                tqdm.write(f"  [pool] execution error: {e}")
            finally:
                if non_socratic_td is not None:
                    non_socratic_td.cleanup()

            total_errors += entry_errors

            record = {
                "video_path": video_path,
                "question": question,
                "ground_truth_answer": gt_answer,
                "format_type": entry["format_type"],
                "line_num": entry["line_num"],
                "clip_start": start,
                "clip_end": end,
                "video_sent": standard_video or clipped_path,
                "model_answers": model_answers,
                "debug_transcripts": debug_transcripts,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            total_processed += 1
            bar.set_postfix(done=total_processed, skip=total_skipped, err=total_errors)

            if total_processed % args.gemini_cleanup_interval == 0:
                _cleanup_gemini_files(gemini_client, keep_n=5)

    # Cleanup global Socratic frame directories at the very end
    for v_cache in socratic_precomp_cache.values():
        if "frames_dir" in v_cache and v_cache["frames_dir"]:
            import shutil
            shutil.rmtree(v_cache["frames_dir"], ignore_errors=True)

    bar.close()
    print("=" * 60)
    print(f"Done.  Processed: {total_processed}  |  Skipped: {total_skipped}  |  Model errors: {total_errors}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
