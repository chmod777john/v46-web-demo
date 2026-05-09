#!/usr/bin/env python
# encoding: utf-8
"""Native Gradio Chatbot prototype for MiniCPM-V 4.6.

Purpose: test方案B for image-bubble flicker.

Key difference from app.py:
- uses gr.Chatbot(type="messages")
- appends user media messages once
- streams only by mutating the last assistant message content

This should keep user-side image DOM nodes stable during assistant streaming.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

import app as core


IMAGE_EXTENSIONS = core.IMAGE_EXTENSIONS
VIDEO_EXTENSIONS = core.VIDEO_EXTENSIONS


def _file_path(file_obj: Any) -> str:
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        for key in ("path", "name", "orig_name", "url"):
            v = file_obj.get(key)
            if isinstance(v, str) and v:
                return v
    for attr in ("path", "name", "orig_name", "url"):
        v = getattr(file_obj, attr, None)
        if isinstance(v, str) and v:
            return v
    return str(file_obj)


def _file_kind(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _normalize_user_input(user_input: dict | None) -> tuple[str, list[str]]:
    if not user_input:
        return "", []
    text = user_input.get("text") or ""
    files = [_file_path(f) for f in (user_input.get("files") or [])]
    return text, files


def _display_user_messages(text: str, files: list[str]) -> list[dict]:
    messages: list[dict] = []
    # Put media first, then text. Each media item is a separate user message,
    # and will never be touched again while assistant text streams.
    for path in files:
        kind = _file_kind(path)
        if kind == "image":
            messages.append({"role": "user", "content": {"path": path}})
        elif kind == "video":
            # Native gr.Chatbot supports FileData-like dicts; for video this is
            # enough to show a stable file/video bubble.
            messages.append({"role": "user", "content": {"path": path}})
    if text.strip():
        messages.append({"role": "user", "content": text.strip()})
    elif not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def _model_user_content(text: str, files: list[str]) -> tuple[list[dict], int, int]:
    content: list[dict] = []
    images = 0
    videos = 0
    for path in files:
        kind = _file_kind(path)
        if kind == "image":
            content.append({"type": "image", "image": core._pil_load(path)})
            images += 1
        elif kind == "video":
            content.append({"type": "video", "path": path})
            videos += 1
    if text.strip():
        content.append({"type": "text", "text": text.strip()})
    if not content:
        content.append({"type": "text", "text": text})
    return content, images, videos


def _pick_variant(use_thinking: bool) -> tuple[str, bool]:
    variant = core.pick_variant(use_thinking)
    return variant, bool(use_thinking and variant == "thinking")


def respond(
    user_input,
    chat_messages: list[dict],
    model_ctx: list[dict],
    media_counts: dict,
    decode_type: str,
    thinking_mode: bool,
    streaming_mode: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_frames: int,
):
    text, files = _normalize_user_input(user_input)
    user_content, new_imgs, new_vids = _model_user_content(text, files)

    cur_imgs = int(media_counts.get("images", 0))
    cur_vids = int(media_counts.get("videos", 0))
    if new_vids + cur_vids > 1 or (new_vids + cur_vids == 1 and cur_imgs + new_imgs > 0):
        gr.Warning("Only supports single video and no mixing with images.")
        yield gr.update(), chat_messages, model_ctx, media_counts
        return

    chat_messages = list(chat_messages or [])
    model_ctx = list(model_ctx or [])

    # Add user-side media/text messages once. We never mutate these messages
    # later, so image nodes should remain stable while assistant streams.
    chat_messages.extend(_display_user_messages(text, files))
    assistant_index = len(chat_messages)
    chat_messages.append({"role": "assistant", "content": ""})
    yield gr.MultimodalTextbox(value={"text": "", "files": []}), chat_messages, model_ctx, media_counts

    messages = [{"role": item["role"], "content": item["content"]} for item in model_ctx]
    messages.append({"role": "user", "content": user_content})

    sampling = decode_type == "Sampling"
    if not sampling:
        streaming_mode = False
    variant, enable_thinking = _pick_variant(thinking_mode)
    print(f"[native] respond variant={variant} enable_thinking={enable_thinking}", flush=True)

    try:
        full_text = ""
        if streaming_mode:
            for chunk in core.generate_stream(
                messages,
                enable_thinking=enable_thinking,
                variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_frames=max_frames,
                stop_control=None,
            ):
                full_text += chunk
                # Critical: only mutate assistant message. User image messages
                # remain byte-for-byte untouched in the list.
                chat_messages[assistant_index]["content"] = full_text
                yield gr.update(), chat_messages, model_ctx, media_counts
        else:
            full_text = core.generate_once(
                messages,
                enable_thinking=enable_thinking,
                variant=variant,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_frames=max_frames,
            )
            chat_messages[assistant_index]["content"] = full_text
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        full_text = f"Error, please retry: {e}"
        chat_messages[assistant_index]["content"] = full_text

    _, answer_only = core.parse_thinking(full_text)
    print(f"[native-debug] full_text repr (first 600 chars): {full_text[:600]!r}", flush=True)
    chat_messages[assistant_index]["content"] = core.format_response(full_text)
    model_ctx.append({"role": "user", "content": user_content})
    model_ctx.append({"role": "assistant", "content": [{"type": "text", "text": answer_only}]})
    media_counts = {"images": cur_imgs + new_imgs, "videos": cur_vids + new_vids}
    yield gr.update(), chat_messages, model_ctx, media_counts


def clear_all():
    return [], [], {"images": 0, "videos": 0}, gr.MultimodalTextbox(value={"text": "", "files": []})


def build_ui(model_name: str, default_thinking: bool):
    with gr.Blocks(title=model_name) as demo:
        gr.Markdown(f"## {model_name} — native Gradio Chatbot spike")
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                decode_type = gr.Radio(["Beam Search", "Sampling"], value="Sampling", label="Decode Type")
                thinking_mode = gr.Checkbox(value=default_thinking, label="Thinking Mode")
                streaming_mode = gr.Checkbox(value=True, label="Streaming")
                max_new_tokens = gr.Slider(64, 16384, value=2048, step=64, label="Max New Tokens")
                temperature = gr.Slider(0.01, 2.0, value=0.7, step=0.01, label="Temperature")
                top_p = gr.Slider(0.05, 1.0, value=1.0, step=0.05, label="Top-p")
                top_k = gr.Slider(0, 200, value=0, step=1, label="Top-k")
                max_frames = gr.Slider(8, 256, value=64, step=8, label="Max Frames")
                clear_btn = gr.Button("Clear")
            with gr.Column(scale=3):
                chat = gr.Chatbot(type="messages", height=620, render_markdown=True, line_breaks=True)
                txt = gr.MultimodalTextbox(
                    file_count="multiple",
                    file_types=["image", "video"],
                    placeholder="Upload image/video and ask a question...",
                    submit_btn=True,
                )
        model_ctx = gr.State([])
        media_counts = gr.State({"images": 0, "videos": 0})
        txt.submit(
            respond,
            [txt, chat, model_ctx, media_counts, decode_type, thinking_mode, streaming_mode,
             max_new_tokens, temperature, top_p, top_k, max_frames],
            [txt, chat, model_ctx, media_counts],
        )
        clear_btn.click(clear_all, outputs=[chat, model_ctx, media_counts, txt])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instruct_path", type=str, default=None)
    p.add_argument("--thinking_path", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--legacy_variant", type=str, default="instruct", choices=["instruct", "thinking"])
    p.add_argument("--port", type=int, default=8895)
    p.add_argument("--model_name", type=str, default="MiniCPM-V 4.6 Native Chatbot")
    p.add_argument("--default_thinking", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    if args.instruct_path or args.thinking_path:
        core.load_models(args.instruct_path, args.thinking_path, args.device)
    elif args.model_path:
        core.load_models(device=args.device, **{args.legacy_variant + "_path": args.model_path})
    else:
        p.error("must provide at least one model path")

    build_ui(args.model_name, args.default_thinking).queue(api_open=False).launch(
        share=False,
        show_api=False,
        server_port=args.port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    main()
