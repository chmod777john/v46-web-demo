#!/usr/bin/env python
# encoding: utf-8
"""
Standalone Gradio demo for MiniCPM-V 4.6 (instruct / thinking).

Uses the upstream-style HuggingFace transformers API:
    - AutoProcessor.apply_chat_template(..., tokenize=True, return_dict=True)
    - MiniCPMV4_6ForConditionalGeneration.generate(**inputs)

Supports:
    - Single / multiple images + text
    - Video (uses processor.video_processor.extract_frames under the hood)
    - Token-by-token streaming via TextIteratorStreamer
    - Enable/disable thinking mode, with <think>...</think> highlighting
"""

import argparse
import base64
import copy
import hashlib
import html
import json
import os
import re
import shutil
import threading
import time
import uuid

import gradio as gr
import modelscope_studio as mgr
from modelscope_studio.components.base import Application as MSApplication
from starlette.middleware import Middleware
import torch
from PIL import Image
from transformers import AutoProcessor, MiniCPMV4_6ForConditionalGeneration, TextIteratorStreamer

# ---------- Globals ----------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".wmv", ".webm", ".m4v"}
ERROR_MSG = "Error, please retry"
CLIENT_ID_HEADER = "x-v46-client-id"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_STORAGE_ROOT = "/data" if os.path.isdir("/data") else os.path.join(PROJECT_ROOT, "logs")
LOG_DIR = os.environ.get("V46_LOG_DIR", os.path.join(DEFAULT_STORAGE_ROOT, "logs"))
HTTP_LOG_FILE = os.environ.get(
    "V46_HTTP_LOG_FILE",
    os.path.join(LOG_DIR, "http_requests.jsonl"),
)
RAW_OUTPUT_LOG_FILE = os.environ.get(
    "V46_RAW_OUTPUT_LOG_FILE",
    os.path.join(LOG_DIR, "raw_model_outputs.jsonl"),
)
UPLOAD_LOG_DIR = os.environ.get(
    "V46_UPLOAD_LOG_DIR",
    os.path.join(DEFAULT_STORAGE_ROOT, "uploads"),
)
LOG_ALL_HTTP_REQUESTS = os.environ.get("V46_LOG_ALL_HTTP_REQUESTS", "0") == "1"
HTTP_LOG_LOCK = threading.Lock()
RAW_OUTPUT_LOG_LOCK = threading.Lock()
DEBUG_RESPONSES = os.environ.get("V46_DEBUG_RESPONSES", "0") == "1"

# MODELS / PROCESSORS are dicts keyed by "instruct" / "thinking". In
# single-model mode only one key exists.
MODELS: dict = {}
PROCESSORS: dict = {}
AVAILABLE_VARIANTS: list = []   # e.g. ["instruct", "thinking"] or ["instruct"] only
DEVICE = None
DTYPE = torch.bfloat16
DEFAULT_MODEL_NAME = "MiniCPM-V 4.6 1B"
DISABLE_TEXT_ONLY = False  # allow text-only chat
MODEL_LOAD_LOCK = threading.Lock()
LAZY_MODEL_CONFIG = {
    "instruct_path": None,
    "thinking_path": None,
    "device": "cuda",
}


# ---------- HTTP request logging ----------

CLIENT_ID_JS = r"""
() => {
  const key = "minicpm_v46_demo_client_id";
  const header = "x-v46-client-id";

  function newId() {
    if (window.crypto && window.crypto.randomUUID) {
      return "local-" + window.crypto.randomUUID();
    }
    const rand = Math.random().toString(36).slice(2);
    return "local-" + Date.now().toString(36) + "-" + rand;
  }

  let clientId = window.localStorage.getItem(key);
  if (!clientId) {
    clientId = newId();
    window.localStorage.setItem(key, clientId);
  }
  window.__minicpmV46ClientId = clientId;

  if (!window.__minicpmV46FetchPatched) {
    const originalFetch = window.fetch;
    window.fetch = function(input, init) {
      const nextInit = init ? Object.assign({}, init) : {};
      const headers = new Headers(
        nextInit.headers || (input instanceof Request ? input.headers : undefined)
      );
      headers.set(header, clientId);
      nextInit.headers = headers;
      return originalFetch.call(this, input, nextInit);
    };

    const originalOpen = XMLHttpRequest.prototype.open;
    const originalSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(method, url) {
      this.__minicpmV46Url = url;
      return originalOpen.apply(this, arguments);
    };
    XMLHttpRequest.prototype.send = function() {
      try {
        this.setRequestHeader(header, clientId);
      } catch (_) {}
      return originalSend.apply(this, arguments);
    };

    window.__minicpmV46FetchPatched = true;
  }
}
"""


def _headers_from_asgi(raw_headers) -> list[dict]:
    headers = []
    for raw_key, raw_value in raw_headers or []:
        headers.append({
            "name": raw_key.decode("latin-1", errors="replace"),
            "value": raw_value.decode("latin-1", errors="replace"),
        })
    return headers


def _header_value(headers: list[dict], name: str) -> str:
    name = name.lower()
    for header in headers:
        if header["name"].lower() == name:
            return header["value"]
    return ""


def _body_text(data: bytes, content_type: str) -> str | None:
    if not data:
        return ""
    lower_type = (content_type or "").lower()
    text_like = (
        lower_type.startswith("text/")
        or "json" in lower_type
        or "x-www-form-urlencoded" in lower_type
    )
    if not text_like:
        return None
    return data.decode("utf-8", errors="replace")


def _body_record(data: bytes, content_type: str) -> dict:
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest() if data else "",
        "base64": base64.b64encode(data).decode("ascii") if data else "",
        "text": _body_text(data, content_type),
    }


def _append_http_log(record: dict) -> None:
    os.makedirs(os.path.dirname(HTTP_LOG_FILE), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with HTTP_LOG_LOCK:
        with open(HTTP_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class HTTPRequestLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        started = time.time()
        request_id = uuid.uuid4().hex[:12]
        scope["v46_request_id"] = request_id
        status_code = None
        request_body = bytearray()
        response_headers = []
        response_body = bytearray()

        async def receive_wrapper():
            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                if chunk:
                    request_body.extend(chunk)
            return message

        async def send_wrapper(message):
            nonlocal status_code, response_headers
            if message.get("type") == "http.response.start":
                status_code = message.get("status")
                headers = list(message.get("headers", []))
                headers.append((b"x-v46-request-id", request_id.encode("ascii")))
                message["headers"] = headers
                response_headers = _headers_from_asgi(message.get("headers", []))
            elif message.get("type") == "http.response.body":
                chunk = message.get("body", b"") or b""
                if chunk:
                    response_body.extend(chunk)
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            request_headers = _headers_from_asgi(scope.get("headers", []))
            client = scope.get("client") or (None, None)
            request_content_type = _header_value(request_headers, "content-type")
            response_content_type = _header_value(response_headers, "content-type")
            record = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                "request_id": request_id,
                "client_id": _header_value(request_headers, CLIENT_ID_HEADER),
                "client_host": client[0],
                "client_port": client[1],
                "method": scope.get("method"),
                "path": scope.get("path"),
                "query_string": (scope.get("query_string") or b"").decode("latin-1", errors="replace"),
                "http_version": scope.get("http_version"),
                "request_headers": request_headers,
                "request_body": _body_record(bytes(request_body), request_content_type),
                "status_code": status_code,
                "response_headers": response_headers,
                "response_body": _body_record(bytes(response_body), response_content_type),
                "duration_ms": round((time.time() - started) * 1000, 2),
            }
            try:
                _append_http_log(record)
            except Exception as e:  # noqa: BLE001
                print(f"[http-log] failed to write request log: {e}", flush=True)


def http_request_logging_app_kwargs() -> dict:
    print(f"[model-call-log] writing model calls to {RAW_OUTPUT_LOG_FILE}", flush=True)
    if LOG_ALL_HTTP_REQUESTS:
        print(f"[http-log] writing all HTTP requests to {HTTP_LOG_FILE}", flush=True)
        return {"middleware": [Middleware(HTTPRequestLogMiddleware)]}
    print("[http-log] all-request logging disabled; set V46_LOG_ALL_HTTP_REQUESTS=1 to enable", flush=True)
    return {}


def _request_log_metadata(request: gr.Request | None) -> dict:
    if request is None:
        return {"request_id": "", "client_id": "", "client_host": "", "session_hash": ""}

    headers = dict(getattr(request, "headers", {}) or {})
    fastapi_request = getattr(request, "request", None)
    scope = getattr(fastapi_request, "scope", {}) or {}
    client = getattr(request, "client", None)
    return {
        "request_id": scope.get("v46_request_id", ""),
        "client_id": headers.get(CLIENT_ID_HEADER, ""),
        "client_host": getattr(client, "host", "") if client else "",
        "session_hash": getattr(request, "session_hash", "") or "",
    }


def _append_raw_output_log(record: dict) -> None:
    os.makedirs(os.path.dirname(RAW_OUTPUT_LOG_FILE), exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with RAW_OUTPUT_LOG_LOCK:
        with open(RAW_OUTPUT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _json_safe_for_log(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe_for_log(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_for_log(v) for v in value]
    if isinstance(value, Image.Image):
        return {"type": "PIL.Image", "mode": value.mode, "size": list(value.size)}
    if hasattr(value, "__fspath__"):
        return os.fspath(value)
    return repr(value)


def log_raw_model_output(request: gr.Request | None, **record) -> None:
    safe_record = {key: _json_safe_for_log(value) for key, value in record.items()}
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "call_id": uuid.uuid4().hex[:12],
        **_request_log_metadata(request),
        **safe_record,
    }
    try:
        _append_raw_output_log(payload)
    except Exception as e:  # noqa: BLE001
        print(f"[raw-output-log] failed to write raw output log: {e}", flush=True)


# ---------- Model loading ----------

def _load_one(path: str, device: str):
    print(f"[v46] Loading processor from: {path}")
    processor = AutoProcessor.from_pretrained(path)
    print(f"[v46] Loading model from: {path}")
    model = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        path,
        dtype=DTYPE,
        attn_implementation="sdpa",
    ).to(device).eval()
    print(f"[v46] -> on {model.device}, dtype={model.dtype}")
    return model, processor


def _register_available_variant(variant: str) -> None:
    if variant not in AVAILABLE_VARIANTS:
        AVAILABLE_VARIANTS.append(variant)


def load_models(instruct_path: str | None = None,
                thinking_path: str | None = None,
                device: str = "cuda") -> None:
    """Load instruct and/or thinking checkpoints. At least one must be provided."""
    global MODELS, PROCESSORS, AVAILABLE_VARIANTS, DEVICE
    DEVICE = device
    if not instruct_path and not thinking_path:
        raise ValueError("At least one of instruct_path / thinking_path must be set")

    if instruct_path:
        _register_available_variant("instruct")
    if thinking_path:
        _register_available_variant("thinking")

    if instruct_path and "instruct" not in MODELS:
        m, p = _load_one(instruct_path, device)
        MODELS["instruct"] = m
        PROCESSORS["instruct"] = p

    if thinking_path and "thinking" not in MODELS:
        m, p = _load_one(thinking_path, device)
        MODELS["thinking"] = m
        PROCESSORS["thinking"] = p

    print(f"[v46] Loaded variants: {AVAILABLE_VARIANTS}")


def configure_lazy_models(instruct_path: str | None = None,
                          thinking_path: str | None = None,
                          device: str = "cuda") -> None:
    """Register model paths without loading them until a request enters GPU scope."""
    global DEVICE
    if not instruct_path and not thinking_path:
        raise ValueError("At least one lazy model path must be set")
    LAZY_MODEL_CONFIG["instruct_path"] = instruct_path
    LAZY_MODEL_CONFIG["thinking_path"] = thinking_path
    LAZY_MODEL_CONFIG["device"] = device
    DEVICE = device
    if instruct_path:
        _register_available_variant("instruct")
    if thinking_path:
        _register_available_variant("thinking")
    print(f"[v46] Lazy model config: variants={AVAILABLE_VARIANTS}, device={device}", flush=True)


def ensure_models_loaded(variant: str | None = None) -> None:
    """Load the requested lazy model variant if it is not already resident."""
    if variant and variant in MODELS:
        return
    if not variant and MODELS:
        return

    with MODEL_LOAD_LOCK:
        if variant and variant in MODELS:
            return
        if not variant and MODELS:
            return

        instruct_path = LAZY_MODEL_CONFIG.get("instruct_path")
        thinking_path = LAZY_MODEL_CONFIG.get("thinking_path")
        device = LAZY_MODEL_CONFIG.get("device") or DEVICE or "cuda"
        if variant == "thinking":
            if not thinking_path:
                raise RuntimeError("Thinking model was requested but no thinking_path is configured")
            load_models(thinking_path=thinking_path, device=device)
            return
        if variant == "instruct":
            if not instruct_path:
                raise RuntimeError("Instruct model was requested but no instruct_path is configured")
            load_models(instruct_path=instruct_path, device=device)
            return
        load_models(instruct_path=instruct_path, thinking_path=thinking_path, device=device)


def pick_variant(use_thinking: bool) -> str:
    """Map the UI checkbox to an actual available variant."""
    if use_thinking and ("thinking" in MODELS or LAZY_MODEL_CONFIG.get("thinking_path")):
        return "thinking"
    if "instruct" in MODELS or LAZY_MODEL_CONFIG.get("instruct_path"):
        return "instruct"
    # Fallback: only one is loaded
    return AVAILABLE_VARIANTS[0]


# ---------- File helpers ----------

def _get_path(mm_file) -> str:
    """Try hard to get a local path out of a Gradio MultimodalInput file object."""
    if isinstance(mm_file, str):
        return mm_file
    for attr in ("path", "name", "orig_name", "url"):
        p = getattr(mm_file, attr, None)
        if isinstance(p, str) and p:
            return p
    fobj = getattr(mm_file, "file", None)
    if fobj is not None:
        for attr in ("path", "name", "orig_name"):
            p = getattr(fobj, attr, None)
            if isinstance(p, str) and p:
                return p
    return str(mm_file)


def _mm_type(mm_file) -> str | None:
    ext = os.path.splitext(_get_path(mm_file))[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _pil_load(path: str, max_side: int = 448 * 16) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h:
            nw = max_side
            nh = int(h * max_side / w)
        else:
            nh = max_side
            nw = int(w * max_side / h)
        img = img.resize((nw, nh), Image.BICUBIC)
    return img


# ---------- Message builder ----------

def build_messages(ctx: list[dict], user_question) -> tuple[list[dict], int, int]:
    """
    Convert the app ctx + the new user MultimodalInput into the v4.6-style
    `messages` list that can be fed to `processor.apply_chat_template`.

    Returns (messages, images_added, videos_added).
    """
    messages = []

    # History: ctx items already stored in v4.6 content format
    # ctx item: {"role": "user"/"assistant", "content": [{"type":"text","text":..} or {"type":"image","image":PIL} or {"type":"video","path":str}]}
    for item in ctx:
        messages.append({"role": item["role"], "content": copy.copy(item["content"])})

    # Current turn: interleave text and files by the [mm_media]N[/mm_media] markers
    files = user_question.files
    text = user_question.text or ""
    pattern = r"\[mm_media\]\d+\[/mm_media\]"
    parts = re.split(pattern, text)
    if len(parts) != len(files) + 1:
        # Fallback: user_question.text had no markers — just concat files then text
        parts = [""] + [""] * (len(files) - 1) + [text] if files else [text]

    new_content = []
    images_added = 0
    videos_added = 0

    first = parts[0].strip()
    if first:
        new_content.append({"type": "text", "text": first})

    for i, f in enumerate(files):
        t = _mm_type(f)
        path = _get_path(f)
        if t == "image":
            img = _pil_load(path)
            new_content.append({"type": "image", "image": img})
            images_added += 1
        elif t == "video":
            new_content.append({"type": "video", "path": path})
            videos_added += 1
        else:
            print(f"[v46] Skipping unknown file type: {path}")

        tail = parts[i + 1].strip()
        if tail:
            new_content.append({"type": "text", "text": tail})

    if not new_content:
        new_content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": new_content})
    return messages, images_added, videos_added


# ---------- Inference ----------

def _prepare_inputs(messages, enable_thinking: bool, variant: str,
                    max_frames: int | None = None):
    ensure_models_loaded(variant)
    model = MODELS[variant]
    processor = PROCESSORS[variant]
    # Official transformers expects processor kwargs under `processor_kwargs`,
    # and MiniCPM-V 4.6 names the video frame cap `max_num_frames`.
    tmpl_kwargs = {}
    if max_frames is not None:
        tmpl_kwargs["processor_kwargs"] = {
            "videos_kwargs": {"max_num_frames": int(max_frames)}
        }
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=enable_thinking,
        **tmpl_kwargs,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    out = {"input_ids": input_ids, "attention_mask": attention_mask}

    for key in ("pixel_values", "pixel_values_videos", "target_sizes", "target_sizes_videos"):
        value = inputs.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                out[key] = value.to(device=model.device, dtype=model.dtype)
            else:
                out[key] = value.to(model.device)
        else:
            out[key] = value
    return out


def _gen_params(sampling: bool, max_new_tokens: int, temperature: float, top_p: float, top_k: int):
    kw = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(sampling),
    }
    if sampling:
        kw.update({
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": 1.0,
        })
    else:
        kw.update({"num_beams": 1, "repetition_penalty": 1.0})
    return kw


def generate_stream(messages, enable_thinking: bool, variant: str, sampling: bool,
                    max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                    max_frames: int | None = None,
                    stop_control: dict | None = None):
    """Yield decoded text chunks (newly added characters) as the model generates."""
    ensure_models_loaded(variant)
    model = MODELS[variant]
    processor = PROCESSORS[variant]
    inputs = _prepare_inputs(messages, enable_thinking, variant, max_frames=max_frames)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = _gen_params(sampling, max_new_tokens, temperature, top_p, top_k)
    gen_kwargs["streamer"] = streamer

    def _worker():
        try:
            with torch.inference_mode():
                model.generate(**inputs, **gen_kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[v46] generate error: {e}")
            import traceback; traceback.print_exc()

    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    for chunk in streamer:
        if stop_control and stop_control.get("stop_streaming"):
            break
        if chunk:
            yield chunk
    th.join(timeout=1.0)


def generate_once(messages, enable_thinking: bool, variant: str, sampling: bool,
                  max_new_tokens: int, temperature: float, top_p: float, top_k: int,
                  max_frames: int | None = None) -> str:
    ensure_models_loaded(variant)
    model = MODELS[variant]
    processor = PROCESSORS[variant]
    inputs = _prepare_inputs(messages, enable_thinking, variant, max_frames=max_frames)
    gen_kwargs = _gen_params(sampling, max_new_tokens, temperature, top_p, top_k)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


# ---------- Response formatting (thinking highlight) ----------

def parse_thinking(full_text: str) -> tuple[str, str]:
    """Split `<think>...</think>` from the final answer."""
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, full_text, flags=re.DOTALL)
    thinking = "\n\n".join(m.strip() for m in matches) if matches else ""
    answer = re.sub(think_pattern, "", full_text, flags=re.DOTALL).strip()
    # Handle unterminated <think>...  (streaming mid-think)
    if not matches and "<think>" in full_text and "</think>" not in full_text:
        idx = full_text.rfind("<think>")
        thinking = full_text[idx + len("<think>"):].strip()
        answer = full_text[:idx].strip()
    return thinking, answer


def normalize_response_text(text: str) -> str:
    """
    UI rendering layer: convert literal \\n outside protected regions to real newlines.
    Protect fenced code blocks, inline code, LaTeX-like commands, and escaped \\n.
    """
    if not isinstance(text, str) or "\\" not in text:
        return text

    protected = {}
    counter = [0]

    def _protect(match):
        key = f"\x00P{counter[0]}\x00"
        counter[0] += 1
        protected[key] = match.group(0)
        return key

    result = text
    result = re.sub(r"```[\s\S]*?```", _protect, result)
    result = re.sub(r"`[^`]+`", _protect, result)
    result = re.sub(r"(?<!\\)\\r\\n", "\n", result)
    result = re.sub(r"(?<!\\)\\n(?![a-zA-Z])", "\n", result)
    result = re.sub(r"(?<!\\)\\r(?![a-zA-Z])", "\n", result)

    for key, value in protected.items():
        result = result.replace(key, value)
    return result


def format_response(text: str) -> str:
    """Markdown-format the (possibly partial) response.

    We deliberately stay in pure Markdown here – HTML wrappers cause the
    chatbot bubble to replace its whole innerHTML on every streaming tick,
    which makes the image thumbnails above the message flicker / jump.
    Thinking is rendered as a blockquote so the user can still distinguish it
    from the final answer.
    """
    text = normalize_response_text(text)
    thinking, answer = parse_thinking(text)
    if not thinking:
        return answer if answer else text
    quoted = "\n".join(f"> {line}" if line else ">"
                       for line in thinking.splitlines())
    return f"> **think**\n{quoted}\n\n{answer}"


# ---------- Gradio helpers ----------

def create_multimodal_input(upload_image_disabled=False, upload_video_disabled=False):
    """
    modelscope_studio 1.6.x only exposes a single `upload_button_props` instead
    of the old image/video split. We therefore pack everything into one button
    and enforce image/video/quantity limits on the backend side in `respond()`.
    """
    disable_upload = upload_image_disabled or upload_video_disabled
    return mgr.MultimodalInput(
        value={"files": [], "text": ""},
        upload_button_props={
            "label": "Upload",
            "interactive": not disable_upload,
            "file_count": "multiple",
            # One button that can accept both image and video MIME types.
            "file_types": ["image", "video"],
        },
        submit_button_props={"label": "Submit"},
    )


def check_file_counts(user_question):
    imgs, vids = 0, 0
    for f in user_question.files:
        t = _mm_type(f)
        if t == "image":
            imgs += 1
        elif t == "video":
            vids += 1
    return imgs, vids


# ---------- Core respond handlers ----------

def respond(user_question, chat_bot, app_cfg,
            params_form, thinking_mode, streaming_mode,
            max_new_tokens, temperature, top_p, top_k,
            max_frames):
    app_cfg.setdefault("session_id", uuid.uuid4().hex[:16])
    app_cfg["stop_streaming"] = False
    app_cfg["is_streaming"] = bool(streaming_mode)

    sampling = (params_form == "Sampling")
    if not sampling:
        streaming_mode = False

    # The "Thinking Mode" checkbox now does two things at once:
    #   - pick the thinking checkpoint if it's loaded (else fall back to instruct)
    #   - turn on enable_thinking in the chat template
    use_thinking = bool(thinking_mode)
    variant = pick_variant(use_thinking)
    enable_thinking = use_thinking and variant == "thinking"
    app_cfg["current_variant"] = variant
    print(f"[v46] respond variant={variant} enable_thinking={enable_thinking}")

    ctx = app_cfg.get("ctx", [])
    messages, new_imgs, new_vids = build_messages(ctx, user_question)

    cur_imgs = app_cfg.get("images_cnt", 0)
    cur_vids = app_cfg.get("videos_cnt", 0)
    # Outputs: (txt_message, chat_bot, app_cfg, stop_btn)
    #
    # Key invariant for NOT flickering images in the user bubble: during
    # streaming we must only ever *append* characters to chat_bot[-1][1].
    # modelscope_studio's Chatbot takes the append-only fast-path (no full
    # item re-render) as long as the new bot text is a strict prefix+append
    # of the previous one. That's why we stream the raw `full_text` here and
    # only run the pretty-printing `format_response(...)` once at the very
    # end – doing it per-tick rewrites the whole string structure.
    if new_vids + cur_vids > 1 or (new_vids + cur_vids == 1 and cur_imgs + new_imgs > 0):
        gr.Warning("Only supports single video and no mixing with images.")
        yield create_multimodal_input(True, True), chat_bot, app_cfg, gr.update(visible=False)
        return
    if DISABLE_TEXT_ONLY and (new_imgs + new_vids + cur_imgs + cur_vids) == 0:
        gr.Warning("Please chat with at least one image or video.")
        yield create_multimodal_input(False, False), chat_bot, app_cfg, gr.update(visible=False)
        return

    chat_bot.append((user_question, ""))
    upload_image_disabled = (cur_vids + new_vids) > 0
    upload_video_disabled = (cur_vids + new_vids) > 0 or (cur_imgs + new_imgs) > 0

    yield (create_multimodal_input(upload_image_disabled, upload_video_disabled),
           chat_bot, app_cfg, gr.update(visible=True))

    try:
        full_text = ""
        if streaming_mode:
            # Mirror the v4.5 demo exactly: reassign chat_bot[-1] with a
            # fresh tuple whose bot-side string is the monotonically growing
            # raw text, and yield on *every* chunk (no throttling).  In
            # modelscope_studio 1.6.1 with `flushing=False`, the Chatbot
            # does string-prefix diff and only appends new characters to
            # the existing bubble – which is precisely why the old demo
            # never flickered.  Any rewrite of the full string (e.g.
            # calling format_response in the loop, or throttled batch
            # updates that skip the prefix) breaks this diff and forces a
            # full re-render that re-downloads the attached image.
            for chunk in generate_stream(
                messages, enable_thinking=enable_thinking, variant=variant, sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames,
                stop_control=app_cfg,
            ):
                if app_cfg.get("stop_streaming"):
                    break
                full_text += chunk
                chat_bot[-1] = (user_question, full_text)
                yield gr.update(), chat_bot, app_cfg, gr.update()
        else:
            full_text = generate_once(
                messages, enable_thinking=enable_thinking, variant=variant, sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[v46] respond error: {e}")
        import traceback; traceback.print_exc()
        full_text = f"{ERROR_MSG}: {e}"

    # Final update – now we *can* format, because it's a one-shot change.
    _, answer_only = parse_thinking(full_text)
    chat_bot[-1] = (user_question, format_response(full_text))

    new_ctx = list(ctx)
    new_ctx.append({"role": "user", "content": messages[-1]["content"]})
    new_ctx.append({"role": "assistant", "content": [{"type": "text", "text": answer_only}]})
    app_cfg["ctx"] = new_ctx
    app_cfg["images_cnt"] = cur_imgs + new_imgs
    app_cfg["videos_cnt"] = cur_vids + new_vids
    app_cfg["is_streaming"] = False

    final_img_disabled = app_cfg["videos_cnt"] > 0
    final_vid_disabled = app_cfg["videos_cnt"] > 0 or app_cfg["images_cnt"] > 0
    yield (create_multimodal_input(final_img_disabled, final_vid_disabled),
           chat_bot, app_cfg, gr.update(visible=False))


def regenerate_clicked(user_question, image_input, user_message, assistant_message,
                       chat_bot, app_cfg,
                       params_form, thinking_mode, streaming_mode,
                       max_new_tokens, temperature, top_p, top_k, max_frames):
    """
    Regenerate the last assistant response. Dispatches to Chat or Few-Shot
    depending on app_cfg["chat_type"].
    Outputs: (txt_message, image_input, user_message, assistant_message,
              chat_bot, app_cfg, stop_btn)
    """
    if len(chat_bot) <= 1 or not chat_bot[-1][1]:
        gr.Warning("No question for regeneration.")
        yield user_question, image_input, user_message, assistant_message, \
              chat_bot, app_cfg, gr.update(visible=False)
        return

    chat_type = app_cfg.get("chat_type", "Chat")

    if chat_type == "Chat":
        last_question = chat_bot[-1][0]
        chat_bot = chat_bot[:-1]
        ctx = app_cfg.get("ctx", [])
        if len(ctx) >= 2:
            app_cfg["ctx"] = ctx[:-2]
        files_imgs, files_vids = check_file_counts(last_question)
        app_cfg["images_cnt"] = max(0, app_cfg.get("images_cnt", 0) - files_imgs)
        app_cfg["videos_cnt"] = max(0, app_cfg.get("videos_cnt", 0) - files_vids)

        for result in respond(last_question, chat_bot, app_cfg,
                              params_form, thinking_mode, streaming_mode,
                              max_new_tokens, temperature, top_p, top_k, max_frames):
            new_input, _cb, _cfg, _stop = result
            yield new_input, image_input, user_message, assistant_message, \
                  _cb, _cfg, _stop
    else:
        last_message = chat_bot[-1][0]
        last_image = None
        last_user = ""
        if hasattr(last_message, "text") and last_message.text:
            last_user = last_message.text
        if hasattr(last_message, "files") and last_message.files:
            last_image = _get_path(last_message.files[0])
        chat_bot = chat_bot[:-1]
        ctx = app_cfg.get("ctx", [])
        if len(ctx) >= 2:
            app_cfg["ctx"] = ctx[:-2]
        for result in fewshot_respond(last_image, last_user, chat_bot, app_cfg,
                                      params_form, thinking_mode, streaming_mode,
                                      max_new_tokens, temperature, top_p, top_k, max_frames):
            _img, _um, _am, _cb, _cfg, _stop = result
            yield user_question, _img, _um, _am, _cb, _cfg, _stop


def stop_clicked(app_cfg):
    app_cfg["stop_streaming"] = True
    app_cfg["is_streaming"] = False
    return app_cfg, gr.update(visible=False)


# ---------- Few-Shot helpers ----------

def fewshot_add_demonstration(_image, _user_message, _assistant_message,
                              _chat_bot, _app_cfg):
    """
    Add one (image, user_message, assistant_message) example to the context.
    The example is shown in the chatbot as a completed turn, and appended to
    `ctx` so it participates in the next generation as in-context demo.
    """
    if "session_id" not in _app_cfg:
        _app_cfg["session_id"] = uuid.uuid4().hex[:16]

    ctx = _app_cfg.setdefault("ctx", [])

    user_content = []
    message_item = []
    if _image is not None:
        img = _pil_load(_image)
        user_content.append({"type": "image", "image": img})
        _app_cfg["images_cnt"] = _app_cfg.get("images_cnt", 0) + 1
        if _user_message:
            user_content.append({"type": "text", "text": _user_message})
        ctx.append({"role": "user", "content": user_content})
        message_item.append(
            {"text": "[mm_media]1[/mm_media]" + (_user_message or ""),
             "files": [_image]}
        )
    else:
        if _user_message:
            user_content.append({"type": "text", "text": _user_message})
            ctx.append({"role": "user", "content": user_content})
            message_item.append({"text": _user_message, "files": []})
        else:
            message_item.append(None)

    if _assistant_message:
        ctx.append({"role": "assistant",
                    "content": [{"type": "text", "text": _assistant_message}]})
        message_item.append({"text": _assistant_message, "files": []})
    else:
        message_item.append(None)

    _chat_bot.append(message_item)
    return None, "", "", _chat_bot, _app_cfg


def fewshot_respond(_image, _user_message, _chat_bot, _app_cfg,
                    params_form, thinking_mode, streaming_mode,
                    max_new_tokens, temperature, top_p, top_k, max_frames):
    """
    Few-Shot generation: takes the in-context demos already stored in
    `_app_cfg["ctx"]`, appends a fresh user turn (image + question) and
    streams the model response.

    Outputs: (image_input, user_message, assistant_message, chat_bot,
              app_cfg, stop_btn)
    """
    _app_cfg.setdefault("session_id", uuid.uuid4().hex[:16])
    _app_cfg["stop_streaming"] = False
    _app_cfg["is_streaming"] = bool(streaming_mode)

    sampling = (params_form == "Sampling")
    if not sampling:
        streaming_mode = False

    use_thinking = bool(thinking_mode)
    variant = pick_variant(use_thinking)
    enable_thinking = use_thinking and variant == "thinking"
    _app_cfg["current_variant"] = variant

    if not _image and not (_user_message and _user_message.strip()):
        gr.Warning("Please provide an image and/or a question for Few-Shot generate.")
        yield _image, _user_message, "", _chat_bot, _app_cfg, gr.update(visible=False)
        return

    ctx = list(_app_cfg.get("ctx", []))
    user_content = []
    message_item = []
    if _image:
        img = _pil_load(_image)
        user_content.append({"type": "image", "image": img})
        message_item.append(
            {"text": "[mm_media]1[/mm_media]" + (_user_message or ""),
             "files": [_image]}
        )
    else:
        message_item.append({"text": _user_message or "", "files": []})
    if _user_message:
        user_content.append({"type": "text", "text": _user_message})

    messages = [{"role": it["role"], "content": copy.copy(it["content"])} for it in ctx]
    messages.append({"role": "user", "content": user_content})

    user_bubble = message_item[0]
    _chat_bot.append((user_bubble, ""))
    yield None, "", "", _chat_bot, _app_cfg, gr.update(visible=True)

    try:
        full_text = ""
        if streaming_mode:
            # Same prefix-append invariant as `respond()` – yield on every
            # chunk, write raw full_text into chat_bot[-1], so that the
            # modelscope_studio Chatbot (flushing=False) only diff-appends
            # new characters and never re-renders the user bubble image.
            for chunk in generate_stream(
                messages, enable_thinking=enable_thinking, variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, max_frames=max_frames,
                stop_control=_app_cfg,
            ):
                if _app_cfg.get("stop_streaming"):
                    break
                full_text += chunk
                _chat_bot[-1] = (user_bubble, full_text)
                yield gr.update(), gr.update(), gr.update(), _chat_bot, _app_cfg, gr.update()
        else:
            full_text = generate_once(
                messages, enable_thinking=enable_thinking, variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, max_frames=max_frames,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[v46] fewshot_respond error: {e}")
        import traceback; traceback.print_exc()
        full_text = f"{ERROR_MSG}: {e}"

    _, answer_only = parse_thinking(full_text)
    _chat_bot[-1] = (user_bubble, format_response(full_text))

    new_ctx = list(ctx)
    new_ctx.append({"role": "user", "content": user_content})
    new_ctx.append({"role": "assistant",
                    "content": [{"type": "text", "text": answer_only}]})
    _app_cfg["ctx"] = new_ctx
    if _image:
        _app_cfg["images_cnt"] = _app_cfg.get("images_cnt", 0) + 1
    _app_cfg["is_streaming"] = False

    yield None, "", "", _chat_bot, _app_cfg, gr.update(visible=False)


def select_chat_type(_tab, _app_cfg):
    """Remember which tab is currently active (Chat / Few Shot)."""
    _app_cfg["chat_type"] = _tab
    return _app_cfg


def flushed():
    """Re-enable the multimodal input after the chatbot finishes typing."""
    return gr.update(interactive=True)


def clear_all(txt_message, chat_bot, app_session):
    """Reset everything (Chat + Few-Shot inputs)."""
    if hasattr(txt_message, "files"):
        txt_message.files.clear()
    if hasattr(txt_message, "text"):
        txt_message.text = ""
    app_session["ctx"] = []
    app_session["images_cnt"] = 0
    app_session["videos_cnt"] = 0
    app_session["stop_streaming"] = False
    app_session["is_streaming"] = False
    app_session["session_id"] = uuid.uuid4().hex[:16]
    # outputs: (txt_message, chat_bot, app_session, image_input, user_msg, assistant_msg)
    return create_multimodal_input(), copy.deepcopy(INIT_CONV), app_session, None, "", ""


def update_streaming_mode_state(params_form):
    if params_form == "Beam Search":
        return gr.update(value=False, interactive=False, info="Beam Search does not support streaming output")
    return gr.update(value=True, interactive=True, info="Enable real-time streaming response")


def on_thinking_toggle(thinking_mode, chat_bot, app_session):
    """When the user toggles Thinking Mode, switch the active checkpoint
    and clear chat history to avoid mixing output styles."""
    use_thinking = bool(thinking_mode)
    new_variant = pick_variant(use_thinking)
    old_variant = app_session.get("current_variant")
    app_session["current_variant"] = new_variant

    only_one_loaded = len(MODELS) < 2
    no_real_switch = (old_variant == new_variant) or only_one_loaded
    # Has the user actually sent any message yet?
    has_history = bool(app_session.get("ctx"))

    if only_one_loaded and use_thinking and "thinking" not in MODELS:
        gr.Warning("Thinking checkpoint not loaded on this server, using instruct model.")
    elif only_one_loaded and not use_thinking and "instruct" not in MODELS:
        gr.Warning("Instruct checkpoint not loaded on this server, using thinking model.")

    if no_real_switch or not has_history:
        # Nothing to clear; just keep UI as-is. Must return 6 values to match
        # the output slots: (txt_message, chat_bot, app_session,
        #                    image_input, user_msg, assistant_msg)
        return gr.update(), gr.update(), app_session, \
               gr.update(), gr.update(), gr.update()

    gr.Info(f"Switched to '{new_variant}' model, history cleared.")
    app_session["ctx"] = []
    app_session["images_cnt"] = 0
    app_session["videos_cnt"] = 0
    app_session["stop_streaming"] = True
    app_session["is_streaming"] = False
    app_session["session_id"] = uuid.uuid4().hex[:16]
    # same output shape as clear_all to reuse the same output slots
    return create_multimodal_input(), copy.deepcopy(INIT_CONV), app_session, None, "", ""


# ---------- Native Gradio Chatbot helpers ----------

def native_file_path(file_obj) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        for key in ("path", "name", "orig_name", "url"):
            value = file_obj.get(key)
            if isinstance(value, str) and value:
                return value
    return _get_path(file_obj)


def _safe_upload_name(path: str) -> str:
    name = os.path.basename(path) or "upload.bin"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:120] or "upload.bin"


def _infer_upload_extension(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS:
        return ext

    try:
        with Image.open(path) as img:
            fmt = (img.format or "").lower()
        image_ext = {
            "jpeg": ".jpg",
            "jpg": ".jpg",
            "png": ".png",
            "gif": ".gif",
            "webp": ".webp",
            "bmp": ".bmp",
            "tiff": ".tiff",
        }.get(fmt)
        if image_ext:
            return image_ext
    except Exception:
        pass

    try:
        with open(path, "rb") as f:
            header = f.read(32)
        if len(header) >= 12 and header[4:8] == b"ftyp":
            return ".mp4"
        if header.startswith(b"\x1aE\xdf\xa3"):
            return ".webm"
    except Exception:
        pass

    return ""


def persist_uploaded_files(files: list[str], session_id: str) -> list[str]:
    """Copy Gradio temp uploads into the project log directory before inference."""
    if not files:
        return []

    session_dir = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id or "session").strip("._")
    dest_dir = os.path.join(UPLOAD_LOG_DIR, session_dir or "session")
    os.makedirs(dest_dir, exist_ok=True)

    persisted = []
    upload_root = os.path.abspath(UPLOAD_LOG_DIR)
    for src in files:
        src_path = os.path.abspath(src)
        if src_path.startswith(upload_root + os.sep):
            persisted.append(src_path)
            continue
        if not os.path.isfile(src_path):
            persisted.append(src)
            continue

        base = _safe_upload_name(src_path)
        inferred_ext = _infer_upload_extension(src_path)
        if inferred_ext and os.path.splitext(base)[1].lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            base = f"{base}{inferred_ext}"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        dest = os.path.join(dest_dir, f"{stamp}-{uuid.uuid4().hex[:8]}-{base}")
        shutil.copy2(src_path, dest)
        persisted.append(dest)
    return persisted


def native_normalize_input(user_input) -> tuple[str, list[str]]:
    if not user_input:
        return "", []
    if isinstance(user_input, dict):
        text = user_input.get("text") or ""
        files = user_input.get("files") or []
    else:
        text = getattr(user_input, "text", "") or ""
        files = getattr(user_input, "files", None) or []
    return text, [native_file_path(f) for f in files]


def native_file_kind(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    inferred_ext = _infer_upload_extension(path)
    if inferred_ext in IMAGE_EXTENSIONS:
        return "image"
    if inferred_ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def native_display_user_messages(text: str, files: list[str]) -> list[dict]:
    messages = []
    for file_path in files:
        kind = native_file_kind(file_path)
        if kind == "image":
            messages.append({"role": "user", "content": {"path": file_path}})
        elif kind == "video":
            url = "/gradio_api/file=" + file_path
            name = html.escape(os.path.basename(file_path))
            messages.append({
                "role": "user",
                "content": (
                    '<div class="native-video-bubble">'
                    f'<video controls preload="metadata" src="{url}"></video>'
                    f'<div class="native-video-name">🎬 {name}</div>'
                    '</div>'
                ),
            })
    if text.strip():
        messages.append({"role": "user", "content": text.strip()})
    elif not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def native_model_user_content(text: str, files: list[str]) -> tuple[list[dict], int, int]:
    content = []
    images = 0
    videos = 0
    for file_path in files:
        kind = native_file_kind(file_path)
        if kind == "image":
            content.append({"type": "image", "image": _pil_load(file_path)})
            images += 1
        elif kind == "video":
            content.append({"type": "video", "path": file_path})
            videos += 1
    if text.strip():
        content.append({"type": "text", "text": text.strip()})
    if not content:
        content.append({"type": "text", "text": text})
    return content, images, videos


def native_empty_input():
    return gr.MultimodalTextbox(value={"text": "", "files": []}, label="", show_label=False)


def native_capture_last_turn(app_cfg, source, display_start, display_count,
                             user_input, media_delta):
    app_cfg["native_last_turn"] = {
        "source": source,
        "display_start": display_start,
        "display_count": display_count,
        "user_input": user_input,
        "media_delta": media_delta,
    }


def native_remove_last_turn(chat_messages, app_cfg):
    last_turn = app_cfg.get("native_last_turn")
    if not last_turn:
        return None, chat_messages, app_cfg

    chat_messages = list(chat_messages or [])
    start = int(last_turn.get("display_start", len(chat_messages)))
    count = int(last_turn.get("display_count", 0))
    if count > 0:
        del chat_messages[start:start + count]

    ctx = list(app_cfg.get("ctx", []))
    if len(ctx) >= 2:
        app_cfg["ctx"] = ctx[:-2]

    media_delta = last_turn.get("media_delta", {}) or {}
    app_cfg["images_cnt"] = max(0, int(app_cfg.get("images_cnt", 0)) - int(media_delta.get("images", 0)))
    app_cfg["videos_cnt"] = max(0, int(app_cfg.get("videos_cnt", 0)) - int(media_delta.get("videos", 0)))
    app_cfg["native_last_turn"] = None
    return last_turn, chat_messages, app_cfg


def native_chat_respond(user_input, chat_messages, app_cfg,
                        params_form, thinking_mode, streaming_mode,
                        max_new_tokens, temperature, top_p, top_k, max_frames,
                        request: gr.Request):
    app_cfg.setdefault("session_id", uuid.uuid4().hex[:16])
    app_cfg["stop_streaming"] = False
    app_cfg["is_streaming"] = bool(streaming_mode)

    text, files = native_normalize_input(user_input)
    files = persist_uploaded_files(files, app_cfg.get("session_id", ""))
    user_content, new_imgs, new_vids = native_model_user_content(text, files)
    cur_imgs = app_cfg.get("images_cnt", 0)
    cur_vids = app_cfg.get("videos_cnt", 0)
    if new_vids + cur_vids > 1 or (new_vids + cur_vids == 1 and cur_imgs + new_imgs > 0):
        gr.Warning("Only supports single video and no mixing with images.")
        yield gr.update(), chat_messages, app_cfg, gr.update(visible=False)
        return

    chat_messages = list(chat_messages or [])
    display_start = len(chat_messages)
    chat_messages.extend(native_display_user_messages(text, files))
    assistant_index = len(chat_messages)
    chat_messages.append({"role": "assistant", "content": "⏳ Processing…"})
    yield native_empty_input(), chat_messages, app_cfg, gr.update(visible=True)

    ctx = app_cfg.get("ctx", [])
    messages = [{"role": item["role"], "content": copy.copy(item["content"])} for item in ctx]
    messages.append({"role": "user", "content": user_content})
    sampling = (params_form == "Sampling")
    if not sampling:
        streaming_mode = False
    use_thinking = bool(thinking_mode)
    variant = pick_variant(use_thinking)
    enable_thinking = use_thinking and variant == "thinking"
    app_cfg["current_variant"] = variant
    print(f"[native] respond variant={variant} enable_thinking={enable_thinking}", flush=True)

    try:
        full_text = ""
        if streaming_mode:
            for chunk in generate_stream(
                messages, enable_thinking=enable_thinking, variant=variant, sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames,
                stop_control=app_cfg,
            ):
                if app_cfg.get("stop_streaming"):
                    break
                full_text += chunk
                chat_messages[assistant_index]["content"] = normalize_response_text(full_text)
                yield gr.update(), chat_messages, app_cfg, gr.update()
        else:
            full_text = generate_once(
                messages, enable_thinking=enable_thinking, variant=variant, sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                max_frames=max_frames,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[native] respond error: {e}", flush=True)
        import traceback; traceback.print_exc()
        full_text = f"{ERROR_MSG}: {e}"

    _, answer_only = parse_thinking(full_text)
    rendered_text = format_response(full_text)
    log_raw_model_output(
        request,
        source="chat",
        session_id=app_cfg.get("session_id", ""),
        variant=variant,
        enable_thinking=enable_thinking,
        params_form=params_form,
        streaming_mode=bool(streaming_mode),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_frames=max_frames,
        user_text=text,
        user_files=files,
        model_messages=messages,
        context_turns=len(ctx),
        raw_full_text=full_text,
        answer_only=answer_only,
        rendered_text=rendered_text,
    )
    if DEBUG_RESPONSES:
        print(f"[native-debug] full_text repr (first 600 chars): {full_text[:600]!r}", flush=True)
    chat_messages[assistant_index]["content"] = rendered_text

    new_ctx = list(ctx)
    new_ctx.append({"role": "user", "content": user_content})
    new_ctx.append({"role": "assistant", "content": [{"type": "text", "text": answer_only}]})
    app_cfg["ctx"] = new_ctx
    app_cfg["images_cnt"] = cur_imgs + new_imgs
    app_cfg["videos_cnt"] = cur_vids + new_vids
    app_cfg["is_streaming"] = False
    native_capture_last_turn(
        app_cfg,
        "chat",
        display_start,
        len(chat_messages) - display_start,
        {"text": text, "files": files},
        {"images": new_imgs, "videos": new_vids},
    )

    yield native_empty_input(), chat_messages, app_cfg, gr.update(visible=False)


def native_fewshot_add_demonstration(_image, _user_message, _assistant_message,
                                     chat_messages, app_cfg):
    app_cfg.setdefault("session_id", uuid.uuid4().hex[:16])
    files = [_image] if _image else []
    files = persist_uploaded_files(files, app_cfg.get("session_id", ""))
    user_content, new_imgs, new_vids = native_model_user_content(_user_message or "", files)
    cur_imgs = app_cfg.get("images_cnt", 0)
    cur_vids = app_cfg.get("videos_cnt", 0)
    if not files and not (_user_message and _user_message.strip()):
        gr.Warning("Please provide an image and/or a user message.")
        return _image, _user_message, _assistant_message, chat_messages, app_cfg
    if new_vids + cur_vids > 1 or (new_vids + cur_vids == 1 and cur_imgs + new_imgs > 0):
        gr.Warning("Only supports single video and no mixing with images.")
        return _image, _user_message, _assistant_message, chat_messages, app_cfg

    chat_messages = list(chat_messages or [])
    ctx = list(app_cfg.get("ctx", []))
    chat_messages.extend(native_display_user_messages(_user_message or "", files))
    ctx.append({"role": "user", "content": user_content})

    if _assistant_message and _assistant_message.strip():
        chat_messages.append({"role": "assistant", "content": format_response(_assistant_message.strip())})
        ctx.append({"role": "assistant", "content": [{"type": "text", "text": _assistant_message.strip()}]})

    app_cfg["ctx"] = ctx
    app_cfg["images_cnt"] = cur_imgs + new_imgs
    app_cfg["videos_cnt"] = cur_vids + new_vids
    app_cfg["native_last_turn"] = None
    return None, "", "", chat_messages, app_cfg


def native_fewshot_respond(_image, _user_message, _chat_messages, _app_cfg,
                           params_form, thinking_mode, streaming_mode,
                           max_new_tokens, temperature, top_p, top_k, max_frames,
                           request: gr.Request):
    _app_cfg.setdefault("session_id", uuid.uuid4().hex[:16])
    _app_cfg["stop_streaming"] = False
    _app_cfg["is_streaming"] = bool(streaming_mode)

    if not _image and not (_user_message and _user_message.strip()):
        gr.Warning("Please provide an image and/or a question for Few-Shot generate.")
        yield _image, _user_message, "", _chat_messages, _app_cfg, gr.update(visible=False)
        return

    files = [_image] if _image else []
    files = persist_uploaded_files(files, _app_cfg.get("session_id", ""))
    user_content, new_imgs, new_vids = native_model_user_content(_user_message or "", files)
    cur_imgs = _app_cfg.get("images_cnt", 0)
    cur_vids = _app_cfg.get("videos_cnt", 0)
    if new_vids + cur_vids > 1 or (new_vids + cur_vids == 1 and cur_imgs + new_imgs > 0):
        gr.Warning("Only supports single video and no mixing with images.")
        yield _image, _user_message, "", _chat_messages, _app_cfg, gr.update(visible=False)
        return

    _chat_messages = list(_chat_messages or [])
    display_start = len(_chat_messages)
    _chat_messages.extend(native_display_user_messages(_user_message or "", files))
    assistant_index = len(_chat_messages)
    _chat_messages.append({"role": "assistant", "content": "⏳ Processing…"})
    yield None, "", "", _chat_messages, _app_cfg, gr.update(visible=True)

    ctx = list(_app_cfg.get("ctx", []))
    messages = [{"role": item["role"], "content": copy.copy(item["content"])} for item in ctx]
    messages.append({"role": "user", "content": user_content})
    sampling = (params_form == "Sampling")
    if not sampling:
        streaming_mode = False
    use_thinking = bool(thinking_mode)
    variant = pick_variant(use_thinking)
    enable_thinking = use_thinking and variant == "thinking"
    _app_cfg["current_variant"] = variant
    print(f"[native] fewshot variant={variant} enable_thinking={enable_thinking}", flush=True)

    try:
        full_text = ""
        if streaming_mode:
            for chunk in generate_stream(
                messages, enable_thinking=enable_thinking, variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, max_frames=max_frames,
                stop_control=_app_cfg,
            ):
                if _app_cfg.get("stop_streaming"):
                    break
                full_text += chunk
                _chat_messages[assistant_index]["content"] = normalize_response_text(full_text)
                yield gr.update(), gr.update(), gr.update(), _chat_messages, _app_cfg, gr.update()
        else:
            full_text = generate_once(
                messages, enable_thinking=enable_thinking, variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, top_k=top_k, max_frames=max_frames,
            )
    except Exception as e:  # noqa: BLE001
        print(f"[native] fewshot_respond error: {e}", flush=True)
        import traceback; traceback.print_exc()
        full_text = f"{ERROR_MSG}: {e}"

    _, answer_only = parse_thinking(full_text)
    rendered_text = format_response(full_text)
    log_raw_model_output(
        request,
        source="fewshot",
        session_id=_app_cfg.get("session_id", ""),
        variant=variant,
        enable_thinking=enable_thinking,
        params_form=params_form,
        streaming_mode=bool(streaming_mode),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_frames=max_frames,
        user_text=_user_message or "",
        user_files=files,
        model_messages=messages,
        context_turns=len(ctx),
        raw_full_text=full_text,
        answer_only=answer_only,
        rendered_text=rendered_text,
    )
    if DEBUG_RESPONSES:
        print(f"[native-debug] fewshot full_text repr (first 600 chars): {full_text[:600]!r}", flush=True)
    _chat_messages[assistant_index]["content"] = rendered_text

    new_ctx = list(ctx)
    new_ctx.append({"role": "user", "content": user_content})
    new_ctx.append({"role": "assistant", "content": [{"type": "text", "text": answer_only}]})
    _app_cfg["ctx"] = new_ctx
    _app_cfg["images_cnt"] = cur_imgs + new_imgs
    _app_cfg["videos_cnt"] = cur_vids + new_vids
    _app_cfg["is_streaming"] = False
    native_capture_last_turn(
        _app_cfg,
        "fewshot",
        display_start,
        len(_chat_messages) - display_start,
        {"image": _image, "user_message": _user_message or ""},
        {"images": new_imgs, "videos": new_vids},
    )
    yield None, "", "", _chat_messages, _app_cfg, gr.update(visible=False)


def native_regenerate_clicked(chat_messages, app_cfg,
                              params_form, thinking_mode, streaming_mode,
                              max_new_tokens, temperature, top_p, top_k, max_frames,
                              request: gr.Request):
    last_turn, chat_messages, app_cfg = native_remove_last_turn(chat_messages, app_cfg)
    if not last_turn:
        gr.Warning("No question for regeneration.")
        yield gr.update(), chat_messages, app_cfg, gr.update(visible=False)
        return

    if last_turn.get("source") == "fewshot":
        user_input = last_turn.get("user_input", {})
        for result in native_fewshot_respond(
            user_input.get("image"), user_input.get("user_message", ""),
            chat_messages, app_cfg,
            params_form, thinking_mode, streaming_mode,
            max_new_tokens, temperature, top_p, top_k, max_frames,
            request,
        ):
            _img, _user, _assistant, _chat, _cfg, _stop = result
            yield gr.update(), _chat, _cfg, _stop
    else:
        user_input = last_turn.get("user_input", {"text": "", "files": []})
        for result in native_chat_respond(
            user_input, chat_messages, app_cfg,
            params_form, thinking_mode, streaming_mode,
            max_new_tokens, temperature, top_p, top_k, max_frames,
            request,
        ):
            yield result


def native_clear_all(txt_message, chat_messages, app_session):
    app_session["ctx"] = []
    app_session["images_cnt"] = 0
    app_session["videos_cnt"] = 0
    app_session["stop_streaming"] = False
    app_session["is_streaming"] = False
    app_session["session_id"] = uuid.uuid4().hex[:16]
    app_session["native_last_turn"] = None
    return native_empty_input(), [], app_session, None, "", ""


def native_on_thinking_toggle(thinking_mode, chat_messages, app_session):
    target_variant = pick_variant(bool(thinking_mode))
    if target_variant != app_session.get("current_variant"):
        gr.Info(f"Switched to '{target_variant}' model, history cleared.")
    app_session["current_variant"] = target_variant
    return native_clear_all(None, chat_messages, app_session)


# ---------- UI ----------

INIT_CONV = [
    [None, {"text": format_response("You can talk to me now"), "flushing": False}],
]

CSS = """
video { height: auto !important; }
.response-container { margin: 0; }
.thinking-section {
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border: 1px solid #d1d9ff;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 8px;
    box-shadow: 0 2px 8px rgba(67, 90, 235, 0.1);
}
.thinking-header {
    font-weight: 600;
    color: #4c5aa3;
    font-size: 14px;
    margin-bottom: 8px;
}
.thinking-content {
    color: #5a6ba8;
    font-size: 13px;
    line-height: 1.4;
    font-style: italic;
    background: rgba(255, 255, 255, 0.6);
    padding: 10px 12px;
    border-radius: 8px;
    border-left: 3px solid #4c5aa3;
    white-space: pre-wrap;
}
.formal-section {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.formal-header {
    font-weight: 600;
    color: #28a745;
    font-size: 14px;
    margin-bottom: 8px;
}
.formal-content {
    color: #333;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
}
.thinking-chatbot .message .content p { margin: 0 !important; }
.thinking-chatbot .message .content { margin: 0; }
#native-chatbot img,
#native-chatbot video {
    max-height: 360px !important;
    max-width: min(100%, 720px) !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 10px;
}
#native-chatbot .message-wrap,
#native-chatbot .message { overflow: visible; }
"""


def build_ui(model_display_name: str, default_thinking: bool):
    variants_str = " + ".join(AVAILABLE_VARIANTS) if AVAILABLE_VARIANTS else "n/a"
    thinking_help = (
        "Switches to the thinking checkpoint and turns on chain-of-thought generation. "
        "Toggling this will clear chat history."
        if len(MODELS) >= 2 else
        "Thinking mode is available when the thinking checkpoint is loaded."
    )

    with gr.Blocks(css=CSS, title=model_display_name, js=CLIENT_ID_JS) as demo:
        with MSApplication():
            with gr.Tab(model_display_name):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown(
                            f"## {model_display_name}\n\n"
                            f"- Loaded variants: **{variants_str}**\n"
                            "- Chat with single / multiple images\n"
                            "- Chat with a video\n"
                            "- Few-shot in-context examples\n"
                            "- Text-only chat\n"
                        )
                        params_form = gr.Radio(
                            choices=["Beam Search", "Sampling"], value="Sampling",
                            interactive=True, label="Decode Type",
                        )
                        thinking_mode = gr.Checkbox(
                            value=default_thinking, interactive=True,
                            label="Thinking Mode (switch to thinking model)",
                            info=thinking_help,
                        )
                        streaming_mode = gr.Checkbox(
                            value=True, interactive=True,
                            label="Enable Streaming Mode",
                        )
                        max_new_tokens = gr.Slider(
                            minimum=64, maximum=16384, value=2048, step=64,
                            label="Max New Tokens",
                        )
                        temperature = gr.Slider(
                            minimum=0.01, maximum=2.0, value=0.7, step=0.01,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.05, maximum=1.0, value=1.0, step=0.05,
                            label="Top-p",
                        )
                        top_k = gr.Slider(
                            minimum=0, maximum=200, value=0, step=1,
                            label="Top-k",
                        )
                        max_frames = gr.Slider(
                            minimum=8, maximum=256, value=64, step=8,
                            label="Max Frames (video sampling)",
                            info="Max frames to sample from a video. "
                                 "Higher = more temporal detail but slower.",
                        )
                        regenerate_btn = gr.Button("Regenerate")
                        clear_btn = gr.Button("Clear History")
                        stop_btn = gr.Button("Stop", visible=False)

                    with gr.Column(scale=3, min_width=500):
                        session_id = uuid.uuid4().hex[:16]
                        initial_variant = pick_variant(default_thinking)
                        app_session = gr.State({
                            "ctx": [],
                            "images_cnt": 0,
                            "videos_cnt": 0,
                            "stop_streaming": False,
                            "is_streaming": False,
                            "session_id": session_id,
                            "current_variant": initial_variant,
                            "chat_type": "Chat",
                        })
                        chat_bot = gr.Chatbot(
                            type="messages",
                            label=f"Chat with {model_display_name}",
                            value=[{"role": "assistant", "content": "You can talk to me now"}],
                            height=600,
                            render_markdown=True,
                            line_breaks=True,
                            bubble_full_width=False,
                            elem_id="native-chatbot",
                        )

                        with gr.Tab("Chat") as chat_tab:
                            txt_message = gr.MultimodalTextbox(
                                value={"text": "", "files": []},
                                file_count="multiple",
                                file_types=["image", "video"],
                                label="",
                                show_label=False,
                                placeholder="Upload image/video and ask a question...",
                                submit_btn=True,
                            )
                            chat_tab_label = gr.Textbox(
                                value="Chat", interactive=False, visible=False,
                            )
                            txt_message.submit(
                                native_chat_respond,
                                [txt_message, chat_bot, app_session,
                                 params_form, thinking_mode, streaming_mode,
                                 max_new_tokens, temperature, top_p, top_k, max_frames],
                                [txt_message, chat_bot, app_session, stop_btn],
                            )

                        with gr.Tab("Few Shot") as fewshot_tab:
                            fewshot_tab_label = gr.Textbox(
                                value="Few Shot", interactive=False, visible=False,
                            )
                            with gr.Row():
                                with gr.Column(scale=1):
                                    image_input = gr.Image(
                                        type="filepath", sources=["upload"],
                                        label="Example Image",
                                    )
                                with gr.Column(scale=3):
                                    user_message = gr.Textbox(
                                        label="User",
                                        placeholder="e.g. What animal is in this image?",
                                    )
                                    assistant_message = gr.Textbox(
                                        label="Assistant",
                                        placeholder="Leave empty when asking, fill for a demo.",
                                    )
                                    with gr.Row():
                                        add_demo_btn = gr.Button("Add Example")
                                        generate_btn = gr.Button("Generate", variant="primary")

                            add_demo_btn.click(
                                native_fewshot_add_demonstration,
                                [image_input, user_message, assistant_message,
                                 chat_bot, app_session],
                                [image_input, user_message, assistant_message,
                                 chat_bot, app_session],
                            )
                            generate_btn.click(
                                native_fewshot_respond,
                                [image_input, user_message, chat_bot, app_session,
                                 params_form, thinking_mode, streaming_mode,
                                 max_new_tokens, temperature, top_p, top_k, max_frames],
                                [image_input, user_message, assistant_message,
                                 chat_bot, app_session, stop_btn],
                            )

                        # Tab switch events: remember current tab + clear state
                        chat_tab.select(
                            select_chat_type,
                            [chat_tab_label, app_session],
                            [app_session],
                        )
                        chat_tab.select(
                            native_clear_all,
                            [txt_message, chat_bot, app_session],
                            [txt_message, chat_bot, app_session,
                             image_input, user_message, assistant_message],
                        )
                        fewshot_tab.select(
                            select_chat_type,
                            [fewshot_tab_label, app_session],
                            [app_session],
                        )
                        fewshot_tab.select(
                            native_clear_all,
                            [txt_message, chat_bot, app_session],
                            [txt_message, chat_bot, app_session,
                             image_input, user_message, assistant_message],
                        )

                        params_form.change(
                            update_streaming_mode_state,
                            inputs=[params_form],
                            outputs=[streaming_mode],
                        )
                        thinking_mode.change(
                            native_on_thinking_toggle,
                            inputs=[thinking_mode, chat_bot, app_session],
                            outputs=[txt_message, chat_bot, app_session,
                                     image_input, user_message, assistant_message],
                        )
                        regenerate_btn.click(
                            native_regenerate_clicked,
                            [chat_bot, app_session,
                             params_form, thinking_mode, streaming_mode,
                             max_new_tokens, temperature, top_p, top_k, max_frames],
                            [txt_message, chat_bot, app_session, stop_btn],
                        )
                        clear_btn.click(
                            native_clear_all,
                            [txt_message, chat_bot, app_session],
                            [txt_message, chat_bot, app_session,
                             image_input, user_message, assistant_message],
                        )
                        stop_btn.click(
                            stop_clicked,
                            [app_session],
                            [app_session, stop_btn],
                        )

            with gr.Tab("How to use"):
                with gr.Column():
                    with gr.Row():
                        gr.Image(
                            value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/m_bear2.gif",
                            label="1. Chat with single or multiple images",
                            interactive=False, width=400, elem_classes="example",
                        )
                        gr.Image(
                            value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/video2.gif",
                            label="2. Chat with video",
                            interactive=False, width=400, elem_classes="example",
                        )
                        gr.Image(
                            value="http://thunlp.oss-cn-qingdao.aliyuncs.com/multi_modal/never_delete/fshot.gif",
                            label="3. Few shot",
                            interactive=False, width=400, elem_classes="example",
                        )
    return demo


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for MiniCPM-V 4.6 (instruct + thinking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # New (preferred) — load both checkpoints so the Thinking Mode toggle
    # actually switches the active model.
    parser.add_argument("--instruct_path", type=str, default=None,
                        help="Path to the instruct checkpoint")
    parser.add_argument("--thinking_path", type=str, default=None,
                        help="Path to the thinking checkpoint")
    # Backward compatible single-model launch.
    parser.add_argument("--model_path", type=str, default=None,
                        help="[legacy] single model path; if set, only one model "
                             "is loaded and the Thinking toggle just flips "
                             "enable_thinking in the chat template")
    parser.add_argument("--legacy_variant", type=str, default="instruct",
                        choices=["instruct", "thinking"],
                        help="[legacy] which variant the --model_path checkpoint is")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Display name in the UI")
    parser.add_argument("--default_thinking", action="store_true",
                        help="Set the Thinking Mode checkbox to True by default")
    parser.add_argument("--device", type=str, default="cuda",
                        help="torch device (both models live on the same device)")
    args = parser.parse_args()

    # Resolve what to load
    if args.instruct_path or args.thinking_path:
        load_models(
            instruct_path=args.instruct_path,
            thinking_path=args.thinking_path,
            device=args.device,
        )
    elif args.model_path:
        print(f"[v46] Legacy single-model mode: variant={args.legacy_variant}")
        kwargs = {args.legacy_variant + "_path": args.model_path}
        load_models(device=args.device, **kwargs)
    else:
        parser.error("must provide at least one of --instruct_path / --thinking_path / --model_path")

    demo = build_ui(DEFAULT_MODEL_NAME, default_thinking=args.default_thinking)
    demo.queue(api_open=False).launch(
        share=False,
        show_api=False,
        server_port=args.port,
        server_name="0.0.0.0",
        allowed_paths=[UPLOAD_LOG_DIR],
        app_kwargs=http_request_logging_app_kwargs(),
    )


if __name__ == "__main__":
    main()
