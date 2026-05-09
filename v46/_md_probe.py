#!/usr/bin/env python
"""Local-only Markdown render probe for the v46 demo's Chatbot.

Reuses the EXACT same `mgr.Chatbot` configuration as `v46/app.py`, but:
- doesn't load any model
- doesn't accept any input
- pre-populates the chat with a markdown sample so we can eyeball whether
  the chatbot is actually parsing markdown / line breaks / code fences.

Run on localhost only:
    .venv/v46/bin/python v46/_md_probe.py --port 8899
Then `curl -s http://127.0.0.1:8899/` from the same machine, or open in
your local browser via SSH port-forwarding:
    ssh -L 8899:127.0.0.1:8899 <this host>
"""

import argparse
import copy

import gradio as gr
import modelscope_studio as mgr
from modelscope_studio.components.base import Application as MSApplication


PROBE_LINES = [
    "# H1 标题",
    "## H2 标题",
    "### H3 标题",
    "",
    "**bold text** and *italic text* and ~~strike~~",
    "",
    "inline `code()` plus block:",
    "",
    "```python",
    "def hello():",
    '    print("world")',
    "```",
    "",
    "- list item A",
    "- list item B",
    "- list item C",
    "",
    "1. ordered one",
    "2. ordered two",
    "",
    "> blockquote line 1",
    "> blockquote line 2",
    "",
    "[OpenBMB](https://openbmb.cn)",
    "",
    "| col A | col B |",
    "| ----- | ----- |",
    "| a     | b     |",
    "",
    "paragraph 1",
    "",
    "paragraph 2 (blank-line separated)",
    "",
    "line X (single newline below)",
    "line Y",
]
PROBE = "\n".join(PROBE_LINES)


def build():
    css = """
    .response-container { margin: 0; }
    .thinking-chatbot ::-webkit-scrollbar { width: 6px; }
    """
    with gr.Blocks(css=css, title="MD probe") as demo:
        with MSApplication():
            gr.Markdown("### chatbot bubble (mgr.Chatbot, render_markdown=True)")
            chat = mgr.Chatbot(
                value=[
                    [None, {"text": PROBE, "flushing": False}],
                ],
                height=560,
                flushing=False,
                bubble_full_width=False,
                elem_classes="thinking-chatbot",
            )
            gr.Markdown("---")
            gr.Markdown("### control: gr.Markdown (should always render)")
            gr.Markdown(PROBE)
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()
    demo = build()
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
