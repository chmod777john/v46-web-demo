# MiniCPM-V 4.6 Gradio Demo

Single-process Gradio demo for **MiniCPM-V 4.6**. One process can load BOTH
the `instruct` and `thinking` checkpoints at once; the "Thinking Mode" toggle
in the UI then switches the active model (and flips `enable_thinking` in the
chat template) on the fly.

Unlike the v4.5 demo (which uses a FastAPI `server` + Gradio `client` split
and the `model.chat(...)` custom API), v4.6 upstreams to the standard
HuggingFace transformers API:

```python
from transformers import AutoProcessor, MiniCPMV4_6ForConditionalGeneration

processor = AutoProcessor.from_pretrained(path)
model     = MiniCPMV4_6ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)

inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=..., do_sample=...)
text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
```

Streaming is implemented via `transformers.TextIteratorStreamer` in a
background thread so the Gradio UI can yield chunks token-by-token.

## Directory layout

```
gradio/v46/
├── app.py          # single-process Gradio app (loads 1 or 2 checkpoints)
├── start.sh        # tmux launcher (dual / instruct / thinking variants, LB registration)
├── requirements.txt
└── README.md
```

## Models

| Variant   | Path                                                                    |
| --------- | ----------------------------------------------------------------------- |
| instruct  | `/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-instruct`     |
| thinking  | `/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-thinking`     |

Note: `config.json` for both was patched to include `image_token_id: 248056`
(the id of the `<|image_pad|>` special token) — without this the model raises
`AttributeError: 'bool' object has no attribute 'sum'` inside
`get_placeholder_mask`.

## Environment

A dedicated `v46` conda env was cloned from `omni` and then:

```bash
conda create --clone omni -n v46 --offline
conda activate v46

PYTHONNOUSERSITE=1 pip install -e /cache/caitianchi/code/v50/code/new-model-addition-MiniCPM-V-4.6 --no-deps
PYTHONNOUSERSITE=1 pip install -U "huggingface_hub>=1.0" "tokenizers>=0.22.0,<=0.23.0" "regex>=2025.10.22" "mistral_common>=1.11.0"
```

`PYTHONNOUSERSITE=1` is required on this host because
`/cache/caitianchi/.local/lib/python3.10/site-packages/` otherwise shadows the
conda-env `transformers` and `huggingface_hub` with incompatible versions.

A small patch was also applied to
`src/transformers/models/minicpmv4_6/configuration_minicpmv4_6.py`:

```python
# BEFORE:
merge_kernel_size: list[int] = [2, 2]
# AFTER:
from dataclasses import field
merge_kernel_size: list[int] = field(default_factory=lambda: [2, 2])
```

(Mutable-list defaults are not allowed in Python 3.10+ `@dataclass`.)

## How the Thinking toggle works

```
  ┌─────────────────────────────────────────────┐
  │  one app.py process  (same CUDA device)     │
  │                                             │
  │   MODELS = {                                │
  │     "instruct":  <MiniCPMV4_6…instruct>     │
  │     "thinking":  <MiniCPMV4_6…thinking>     │
  │   }                                         │
  └─────────────────────────────────────────────┘

  Checkbox OFF  →  variant="instruct",  enable_thinking=False
  Checkbox ON   →  variant="thinking",  enable_thinking=True
```

When the checkbox is flipped, the chat history is automatically cleared and a
toast `Switched to 'thinking' model, history cleared.` is shown. History is
cleared because the two checkpoints produce stylistically different outputs
(`<think>…</think>` vs plain answer) and mixing them in one conversation tends
to confuse the model on later turns.

GPU memory: each checkpoint is ≈16 GB in bfloat16, so the dual-model process
needs ≈32 GB. Recommended on 80 GB A100/H100. On smaller cards, launch with
`--variant instruct` or `--variant thinking` to load only one checkpoint.

## Launch

### A. Quick start — one dual-model instance on a single GPU

```bash
cd /cache/caitianchi/code/MiniCPM-o-demo-web/gradio/v46

# dual-model on GPU 7, port 8890, no load balancer
bash start.sh -n 1 --gpu-start 7 --port-base 8890 --no-lb
```

Browse to `http://<host>:8890`, flip the "Thinking Mode" checkbox to switch
checkpoints.

### B. Production — multiple dual-model instances behind a load balancer

Architecture:

```
                  ┌──────────────────────────────────────────┐
      user ─────▶ │  load_balancer :8121  (ip_hash + SSE)    │
                  └──┬───────────────┬──────────────┬────────┘
                     │               │              │
                     ▼               ▼              ▼
              127.0.0.1:8890  127.0.0.1:8891  127.0.0.1:8892     ← v4.6 app.py (dual)
              (GPU 7)         (GPU 6)         (GPU 5)
                 │               │              │
                 └─── each process holds BOTH instruct + thinking ───
```

Because every backend serves BOTH variants, there's **only one LB pool**, and
ip_hash session stickiness keeps each user on one backend (so their checkbox
state stays consistent).

#### 1) Start the load balancer

```bash
cd ../load_balancer
python load_balancer.py --port 8121 --strategy ip_hash
```

#### 2) Start the Gradio instances (auto-registers to LB)

```bash
cd ../v46

# 4 dual-model instances on GPU 7,6,5,4 → LB :8121
bash start.sh -n 4 \
    --gpu-start 7 --port-base 8890 \
    --lb-host 127.0.0.1 --lb-port 8121
```

End users access: `http://<host>:8121`.

#### 3) Status / stop

```bash
bash start.sh --status
bash start.sh --stop        # also unregisters from LB
```

### C. Single-variant cluster (less GPU memory per process)

If your cards can't fit ~32 GB, deploy the two variants separately. In this
mode the "Thinking Mode" toggle only flips `enable_thinking` (it cannot
switch models because only one is loaded), and you need two LB ports so
users don't accidentally land on the other model.

```bash
bash start.sh -n 4 --variant instruct  --gpu-start 7 --port-base 8890 --lb-port 8121
bash start.sh -n 4 --variant thinking  --gpu-start 3 --port-base 8900 --lb-port 8122
```

### D. Run app.py manually (no tmux, no LB)

```bash
conda activate v46

# Dual-model (checkbox switches models)
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 python app.py \
    --instruct_path=/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-instruct \
    --thinking_path=/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-thinking \
    --port=8890

# Single-model (legacy)
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=7 python app.py \
    --instruct_path=/cache/caitianchi/code/v50/ckpt/minicpm-v-4_6-0420-rlaif-instruct \
    --port=8890
```

### Multi-machine deployment

Same as v4.5. Run the LB on the primary machine, then on each worker machine:

```bash
bash start.sh -n K --lb-host <primary-ip> --lb-port 8121 --local-ip <this-ip>
```

## UI features

- **Upload**: single or multiple images, or one video per turn. No mixing
  video+image (enforced on the backend).
- **Decode Type**: Beam Search (deterministic, `num_beams=1`) or Sampling.
- **Thinking Mode**: switches the active checkpoint when two are loaded, or
  flips `enable_thinking` when only one is loaded. Toggling this clears the
  chat history to avoid mixing output styles.
- **Enable Streaming Mode**: token-by-token updates via
  `TextIteratorStreamer`. Beam Search disables streaming automatically.
- **Sliders**: `max_new_tokens`, `temperature`, `top_p`, `top_k`.
- **Regenerate / Clear History / Stop** buttons.
- `<think>…</think>` segments are rendered in a distinct blue card above the
  answer so you can see the model's reasoning in real time.

## Known differences from v4.5 demo

| Aspect                | v4.5                                                  | v4.6                                                              |
| --------------------- | ----------------------------------------------------- | ----------------------------------------------------------------- |
| Architecture          | FastAPI server + Gradio client + LB                   | Single-process Gradio app (optionally behind LB)                  |
| Models per process    | 1                                                     | **1 or 2** (dual instruct+thinking)                               |
| Model loading         | `AutoModel.from_pretrained(trust_remote_code=True)`   | `MiniCPMV4_6ForConditionalGeneration.from_pretrained(...)`        |
| Inference             | `model.chat(msgs, tokenizer, processor, ...)`         | `model.generate(**processor.apply_chat_template(...))`            |
| Video encoding        | Client pre-extracts frames → base64 per-frame POST   | Processor extracts frames internally from a local path            |
| Streaming             | Custom `chat(stream=True)` → SSE over HTTP            | `TextIteratorStreamer` in a thread → direct Gradio yield         |
| Thinking mode         | `enable_thinking` to `model.chat`                     | `enable_thinking` to `apply_chat_template` + model switch         |
