# MiniCPM-V 4.6 Gradio Demo - Deployment Bundle

Single-instance, dual-model (instruct + thinking) Gradio demo for
MiniCPM-V 4.6. The transformers fork and the model checkpoints are NOT
bundled - you supply them and point the launcher at them.

## Bundle layout

```
v46-deploy/
├── DEPLOY.md          this file
├── run_single.sh      one-command launcher (edit paths inside)
└── v46/
    ├── app.py         UI + dual-model loading + streaming
    ├── start.sh       richer launcher (multi-GPU, tmux, etc.) - optional
    ├── README.md      legacy docs
    └── requirements.txt
```

## External dependencies (provide separately)

1. **Custom transformers fork** - `new-model-addition-MiniCPM-V-4.6`,
   installed editable:

   ```bash
   pip install -e /path/to/new-model-addition-MiniCPM-V-4.6 --no-deps
   ```

   Two patches MUST be present in this fork (already applied to your
   working copy at `/cache/caitianchi/code/v50/code/new-model-addition-MiniCPM-V-4.6`).
   If you ever re-pull from upstream, re-apply them or the demo will not
   work:

   - `src/transformers/models/minicpmv4_6/configuration_minicpmv4_6.py` -
     `merge_kernel_size` must be defined with
     `field(default_factory=lambda: [2, 2])`, otherwise the module fails
     to import (`mutable default <class 'list'> ... is not allowed`).

   - `src/transformers/models/minicpmv4_6/modeling_minicpmv4_6.py` - in
     `MiniCPMV4_6PreTrainedModel._init_weights`, the
     `MiniCPMV4_6ViTWindowAttentionMerger` branch must be a no-op
     (`return`) instead of calling `module._init_weights()`. The custom
     init uses `.data.zero_() / .data.normal_()` which bypasses HF's
     "skip already-loaded params" fast-init protection and clobbers the
     loaded pretrained weights of the visual merger - the model would
     then be unable to describe images even though loading reports no
     missing/unexpected keys.

2. **Two checkpoints** - `minicpm-v-4_6-0420-rlaif-instruct` and
   `minicpm-v-4_6-0420-rlaif-thinking`. Both must already be patched:

   - `config.json` contains `"image_token_id": 248056`
   - `model.safetensors` keys have been renamed:
     `model.vpm.* -> model.vision_tower.*`,
     `model.vit_merger.* -> model.vision_tower.vit_merger.*`,
     `model.merger.mlp.0.mlp.0.* -> model.merger.mlp.0.linear_1.*`,
     `model.merger.mlp.0.mlp.2.* -> model.merger.mlp.0.linear_2.*`.

   The checkpoints under `/cache/caitianchi/code/v50/ckpt/` are already
   patched (originals are kept as `model.safetensors.bak`), so you can
   point the launcher at them as-is.

3. **Conda env** - we built `v46` by cloning `omni`. From scratch:

   ```bash
   conda create -n v46 python=3.10 -y
   conda activate v46
   # install your CUDA-matching torch first, then:
   pip install -r v46/requirements.txt
   pip install -e /path/to/new-model-addition-MiniCPM-V-4.6 --no-deps
   ```

   Pinned/required versions are already in `v46/requirements.txt`:
   `gradio>=5.0,<6`, `modelscope_studio==1.6.1`,
   `huggingface_hub>=1.0`, `tokenizers>=0.22.0,<=0.23.0`,
   `mistral_common>=1.11.0`, `accelerate>=1.1.0`,
   `Pillow>=10.0`, `decord>=0.6.0`.

## Run it

Easiest path - edit the variables at the top of `run_single.sh`, then:

```bash
bash run_single.sh
```

Or call `app.py` directly:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python -u v46/app.py \
  --port 8890 \
  --instruct_path /path/to/minicpm-v-4_6-0420-rlaif-instruct \
  --thinking_path /path/to/minicpm-v-4_6-0420-rlaif-thinking
```

Then open `http://<host>:8890/` in your browser.

`PYTHONNOUSERSITE=1` is important when the host has a polluted
`~/.local/lib/...` site-packages that may shadow the editable
transformers fork.

## CLI flags (`app.py`)

| flag                 | default      | description                                  |
|----------------------|--------------|----------------------------------------------|
| `--instruct_path`    | -            | path to `...rlaif-instruct`                  |
| `--thinking_path`    | -            | path to `...rlaif-thinking`                  |
| `--model_path`       | -            | legacy single-model mode                     |
| `--legacy_variant`   | `instruct`   | which variant `--model_path` represents      |
| `--port`             | `8890`       | HTTP port                                    |
| `--model_name`       | -            | display name in the page title               |
| `--default_thinking` | `False`      | start with the thinking model selected       |
| `--device`           | `cuda`       | torch device for both models                 |

You must pass at least one of `--instruct_path / --thinking_path / --model_path`.

## Troubleshooting

- **Model produces nonsense for images** -> the modeling-fork patch was
  lost, or the checkpoint was not key-renamed. Quick check:

  ```python
  m = MiniCPMV4_6ForConditionalGeneration.from_pretrained(ckpt_dir, ...)
  w = m.model.vision_tower.vit_merger.linear_1.weight.float()
  print(w.std())   # must be ~0.02 (pretrained), NOT ~0.5 (random)
  ```

- **`mutable default <class 'list'>` import error** -> the
  `configuration_minicpmv4_6.py` `field(default_factory=...)` patch was
  lost.

- **`AttributeError: 'bool' object has no attribute 'sum'`** ->
  `image_token_id` missing from the checkpoint's `config.json`.

- **UI flickers during streaming** -> something rewrote the prefix-append
  invariant in `respond()`. Each chunk MUST yield with
  `chat_bot[-1] = (user_q, full_text)` (a fresh tuple, raw text, no
  throttling), and `mgr.Chatbot(..., flushing=False, ...)`. Don't add
  throttling, don't switch to `flushing=True`, don't keep the old tuple
  and mutate the second element.
