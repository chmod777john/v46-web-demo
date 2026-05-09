# MiniCPM-V 4.6 Demo Runbook

实际部署的现状记录。`DEPLOY.md` 是 upstream 通用说明，本文是当前两台机器的具体配置，停服/重启/排错请先看本文。

## TL;DR

| 区域 | 公网入口                    | 内部进程             | 显卡              | 模型                 |
|------|------------------------------|----------------------|-------------------|----------------------|
| 国内 | http://82.157.64.212:9443/   | 8891 (本机) ← frp    | A100 80G GPU 1    | Instruct (定版)      |
| 海外 | http://34.125.240.119:8000/  | 8000 (海外机)        | NVIDIA L4 23G     | Instruct + Thinking  |

HF Static Space landing：

```
https://huggingface.co/spaces/openbmb/MiniCPM-V-4_6-Demo
源文件: /user/weihongliang/v46-hf-space/{index.html,README.md}
```

## 1. 国内部署（A100，CN）

### 1.1 路径

```
代码          /user/weihongliang/v46-deploy
venv          /user/weihongliang/v46-deploy/.venv/v46          (Python 3.10, torch 2.8.0+cu128)
权重 (instruct) /user/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6
权重 (thinking) /user/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6-Thinking
frpc 二进制   /user/weihongliang/frp_0.65.0_linux_amd64/frpc
frpc 配置     /user/weihongliang/frp_0.65.0_linux_amd64/frpc_v46_8891_9443.toml
frpc 日志     /user/weihongliang/frp_0.65.0_linux_amd64/frpc_v46_8891_9443.log
```

> Thinking 权重已下载解压在本地，但当前服务只加载 Instruct（同卡上有其他用户进程，
> 不留充足显存给双模型）。要切双模型时，把启动命令加上 `--thinking_path ...`。

### 1.2 启动 / 停止 / 查看

启动（前台 / 长跑用 tmux 或 nohup 包一层都行）：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 \
  /user/weihongliang/v46-deploy/.venv/v46/bin/python -u \
  /user/weihongliang/v46-deploy/v46/app.py \
  --port 8891 \
  --instruct_path /user/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6 \
  --model_name "MiniCPM-V 4.6 Instruct Final"
```

停止：

```bash
pkill -f '/user/weihongliang/v46-deploy/v46/app.py --port 8891'
```

状态：

```bash
pgrep -af '/user/weihongliang/v46-deploy/v46/app.py'
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
curl -sI http://127.0.0.1:8891/ | head -1
```

### 1.3 frp 反向代理 (8891 → 82.157.64.212:9443)

配置：

```toml
serverAddr = "82.157.64.212"
serverPort = 7000
auth.method = "token"
auth.token  = "modelbest-frp-token"

[[proxies]]
name      = "minicpmv46-instruct-8891-9443"
type      = "tcp"
localIP   = "127.0.0.1"
localPort = 8891
remotePort= 9443
```

启动：

```bash
cd /user/weihongliang/frp_0.65.0_linux_amd64
nohup ./frpc -c frpc_v46_8891_9443.toml > frpc_v46_8891_9443.log 2>&1 &
```

停止：

```bash
pkill -f 'frpc -c frpc_v46_8891_9443.toml'
```

公网验证：

```bash
curl -sI http://82.157.64.212:9443/ | head -1     # 应 200
```

## 2. 海外部署（GCP L4，US）

机器：`weihongliang@34.125.240.119` (Ubuntu 24.04, NVIDIA L4 23G)

公网防火墙只放开 22 (SSH) 和 8000，因此 demo 直接绑 8000，不走 frp。

### 2.1 路径

```
代码          ~/v46-deploy
venv          ~/v46-deploy/.venv/v46                              (Python 3.12, torch 2.8.0+cu128)
权重 (instruct) ~/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6
权重 (thinking) ~/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6-Thinking
日志          ~/v46-deploy/app.log
HF token      ~/.hf-token  (chmod 600，仅推 HF Space 用)
```

### 2.2 启动 / 停止 / 查看

启动（双模型）：

```bash
ssh weihongliang@34.125.240.119 \
  'cd ~/v46-deploy && setsid bash -c "(.venv/v46/bin/python -u v46/app.py \
     --port 8000 \
     --instruct_path /home/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6 \
     --thinking_path /home/weihongliang/models/minicpmv46-final/checkpoints/MiniCPM-V-4.6-Thinking \
     --model_name \"MiniCPM-V 4.6 (Overseas)\" \
     > ~/v46-deploy/app.log 2>&1) < /dev/null > /dev/null 2>&1 &"'
```

停止 / 状态：

```bash
ssh weihongliang@34.125.240.119 'pkill -f "/home/weihongliang/v46-deploy/v46/app.py"'
ssh weihongliang@34.125.240.119 'pgrep -af "v46/app.py"; tail -30 ~/v46-deploy/app.log'
```

公网验证：

```bash
curl -sI http://34.125.240.119:8000/ | head -1
```

> SSH 注意：`pkill -f` 模式不要写得太宽（比如只写 `pkill -f port_probe.py`），
> 否则会把 SSH wrapper 进程也匹配进去，导致 SSH 自杀（exit 255）。

## 3. 代码补丁（不要回退）

为了对齐官方 `transformers 5.8.0`，对 `v46/app.py` 做了三处必要修改：

1. **视频帧参数搬到 `processor_kwargs`，并改名 `max_num_frames`**
   ```python
   tmpl_kwargs = {}
   if max_frames is not None:
       tmpl_kwargs["processor_kwargs"] = {
           "videos_kwargs": {"max_num_frames": int(max_frames)}
       }
   inputs = processor.apply_chat_template(..., **tmpl_kwargs)
   ```
   旧版 demo 写的是 `videos_kwargs={"max_frames": ...}`，会触发
   `merged_typed_dict.__init__() got an unexpected keyword argument 'max_frames'`。

2. **`pixel_values` 保留官方 tensor 输出，并兼容视频字段**
   ```python
   for key in ("pixel_values", "pixel_values_videos",
               "target_sizes", "target_sizes_videos"):
       value = inputs.get(key)
       if value is None: continue
       if isinstance(value, torch.Tensor):
           if torch.is_floating_point(value):
               out[key] = value.to(device=model.device, dtype=model.dtype)
           else:
               out[key] = value.to(model.device)
       else:
           out[key] = value
   ```
   旧版会把 `pixel_values` 转成嵌套 list，新版 modeling 期望 tensor，
   不改会报 `'list' object has no attribute 'shape'`。

3. **生成参数对齐定版 `generation_config.json`**（孙先确认）
   - `_gen_params()` 里 `repetition_penalty` 从 `1.05` 改成 `1.0`
   - UI 默认 `top_p` 从 `0.8` 改成 `1.0`
   - UI 默认 `top_k` 从 `100` 改成 `0`

## 4. 依赖清单（与上游 `requirements.txt` 的差异）

补 / 收紧：

```
torch==2.8.0+cu128 torchaudio==2.8.0 torchvision==0.23.0   # 必须装 torchvision，否则视频 processor 起不来
av==17.0.1                                                  # PyAV，给 torchvision/transformers video pipeline 用
transformers>=5.8.0                                         # 5.2.0 还没合 MiniCPM-V 4.6
gradio>=5.0,<6                                              # 锁版本避免 resolver 回溯
modelscope_studio==1.6.1                                    # 同上
```

安装顺序（避免共享盘小文件写入卡死）：

```bash
.venv/v46/bin/pip install --no-compile --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
.venv/v46/bin/pip install --no-compile --index-url https://pypi.org/simple \
    "transformers" "gradio>=5.0,<6" "modelscope_studio==1.6.1" \
    "Pillow>=10.0" "decord>=0.6.0" "mistral_common>=1.11.0" \
    "accelerate>=1.1.0" "av"
```

> 国内机器有公司 mirror `https://mirror.corp.modelbest.co/python/pypi/simple/`，
> 但部分 wheel（`pip-26.x`、`nvidia-cusparselt-cu12`、`transformers-5.8.0`）
> 经过 `/file/` 转发会偶发 502；遇到时直接走 PyPI 即可。

## 5. HF Static Space landing

入口：

```
https://huggingface.co/spaces/openbmb/MiniCPM-V-4_6-Demo
渲染:  https://openbmb-minicpm-v-4-6-demo.static.hf.space/
```

源文件在仓库**外面**：

```
/user/weihongliang/v46-hf-space/index.html
/user/weihongliang/v46-hf-space/README.md  (frontmatter sdk: static)
```

要改文案/链接：编辑 `index.html` 后用海外机器（国内不通 HF）推送：

```bash
rsync -av /user/weihongliang/v46-hf-space/ weihongliang@34.125.240.119:~/v46-hf-space/
ssh weihongliang@34.125.240.119 \
  'HF_TOKEN=$(cat ~/.hf-token | tr -d "[:space:]") \
   /home/weihongliang/v46-deploy/.venv/v46/bin/python - <<PY
from huggingface_hub import HfApi; import os
HfApi(token=os.environ["HF_TOKEN"]).upload_folder(
    folder_path="/home/weihongliang/v46-hf-space",
    repo_id="openbmb/MiniCPM-V-4_6-Demo",
    repo_type="space",
    commit_message="update landing",
)
PY'
```

## 6. 常见排错

- **页面打开返回 "Error, please retry: ..."**
  → 看进程日志（国内是 demo 进程的 stdout，海外是 `~/v46-deploy/app.log`），通常是 transformers API 兼容问题，往第 3 节看是不是哪个补丁被回退了。

- **页面打开是 "抱歉，我无法处理这个请求。"**
  → 这是模型生成的拒答，**不是后端报错**。日志里没有 traceback 即可确认。

- **视频上传后没有进度条**
  → modelscope_studio MultimodalInput 自身行为；上传完→处理中→流式返回前没有可视进度。属于正常。

- **GPU 显存只升不降**
  → PyTorch CUDA allocator 缓存行为，不是泄漏。Gradio 默认串行执行，不同用户的会话之间互不累积；同一个用户长对话时上下文长度上涨会让单轮峰值变高，点 UI 上的 Clear History 即可释放。

- **`Use% 100%` 但 `Avail` 还有几 T**
  → `/user/weihongliang` 的 GPFS 整体高水位，跟你这点目录无关；客户端没装 `mmlsquota`，要看大头得从存储侧查。

- **海外机器 SSH 命令输出空 / exit 255**
  → 多半是 `pkill -f` 的模式过宽匹配了 SSH wrapper。要么换更精确的模式，要么把 `pkill` 拆到独立调用。

## 7. 部署过程回顾（首次怎么做出来的）

> 这部分是"换一台机器从零再做一遍"会用到的步骤，已经按踩过的坑顺序整理。
> 命令大都已经在前面 1-5 节出现过，本节侧重 **顺序和动机**。

### 7.1 国内 A100

1. 解压 deploy 包到 `/user/weihongliang/v46-deploy`：
   ```bash
   cd /user/weihongliang/autoshow_omni
   tar -xzf v46-deploy.tar.gz -C ..
   ```
   `DEPLOY.md`、`run_single.sh`、`v46/{app.py,start.sh,requirements.txt,README.md}` 都在里面。
2. 一开始拿了一份 instruct 临时权重做兼容性验证（路径 `/user/weihongliang/models/minicpmv46-instruct/...`），确认 `transformers 5.8.0` 已经合入 v4.6 后跳过 fork 这步。
3. 在项目目录建 venv：`/user/weihongliang/v46-deploy/.venv/v46`。Python 3.10。
   - `python3 -m venv ... ` 在共享盘上 ensurepip 一度卡住进入 D 状态，等 IO 自然恢复。
   - 公司 mirror `https://mirror.corp.modelbest.co/python/pypi/simple/` 多数包能用，但
     遇到 502 的时候直接走 PyPI 或 PyTorch cu128 源就好（见 4 节）。
4. 安装依赖（顺序见 4 节）。装完 torch 后必须再装 `torchvision`、`av`，不然
   v4.6 video processor 起不来。
5. 第一次跑视频请求 → 报 `max_frames` 不接受 → 应用补丁 1。
6. 再跑视频请求 → 报 `'list' object has no attribute 'shape'` → 应用补丁 2。
7. 拿到 `孙先` 给的定版权重链接，下载到 `~/models/minicpmv46-final/`，解压
   去掉 tar 里 `backup/user/sunxian/` 前缀（`tar -xf ... --strip-components=3`）。
8. 切换 demo 加载到定版 instruct，按定版 `generation_config.json` 同步
   `_gen_params()` 默认值（补丁 3）。
9. 配 frpc 把 `127.0.0.1:8891` 反代到 `82.157.64.212:9443`（见 1.3 节）。
10. 公网 `curl -sI http://82.157.64.212:9443/` 返回 200 → ✅。

### 7.2 海外 GCP L4

1. SSH 自检 + 探测公网端口：
   - 在远端跑一个多端口监听小脚本（参考 `~/port_probe.py`，结束后 `pkill` 掉）。
   - 国内 `curl http://34.125.240.119:<port>/` 测试。结果只有 `8000` 是通的，
     `80/443/8080/8443/9443` 都被 GCP firewall 拦。
2. 远端环境：Ubuntu 24.04，没有 ensurepip，需要 `sudo apt-get install -y python3-venv python3-pip`。
3. `rsync -av --exclude '.venv' /user/weihongliang/v46-deploy/ weihongliang@34.125.240.119:~/v46-deploy/`，
   把代码（含已合入的三处补丁）同步过去。
4. 在远端建 venv：`python3 -m venv ~/v46-deploy/.venv/v46`（Python 3.12）。
5. 装依赖（见 4 节）。海外机器直连 PyPI 和 PyTorch 源都很快，不需要 mirror。
6. 后台并行下载两个定版 tar 到 `~/models/minicpmv46-final/`（每个约 2.5G，
   ~5MB/s 大概 8-10 分钟）。然后 `tar -xf ... --strip-components=3` 解压。
7. L4 23GB 显存够同时加载 instruct + thinking，所以海外用双模型启动（见 2.2 节）。
8. 公网 `curl -sI http://34.125.240.119:8000/` 返回 200 → ✅。

> SSH 注意（再强调一次）：`pkill -f port_probe.py` 在远端会同时匹配到执行
> `pkill` 那条 SSH wrapper bash 命令本身（因为它的 `argv` 里也有 `port_probe.py`），
> 直接把 SSH session 杀掉，于是本机看到 exit 255 / 输出空白。改成
> `pkill -f "python3.*port_probe\.py$"` 之类的精确模式即可。

## 8. HF Space 相关操作记录

### 8.1 token & 身份

- 本地 token 路径：`/user/weihongliang/autoshow_omni/hf.token`（37 字节，`hf_X...`）。
- `HfApi.whoami()` 返回：用户 `userisuser`，包含 `openbmb` 组织成员关系。
- 国内机器到 `huggingface.co` 直连超时（curl exit 28），所有写操作必须**从海外
  机器执行**。token 用 `scp` 复制到 `weihongliang@34.125.240.119:~/.hf-token`，
  `chmod 600`。

### 8.2 创建 v46 landing Space

源文件：

```
/user/weihongliang/v46-hf-space/
  ├── index.html   暗色主题、两个入口卡片，target="_blank" 跳转两个 demo
  └── README.md    HF Space frontmatter (sdk: static)
```

部署（在海外机器）：

```python
from huggingface_hub import HfApi
import os
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(
    repo_id="openbmb/MiniCPM-V-4_6-Demo",
    repo_type="space",
    space_sdk="static",
    private=False,
    exist_ok=True,
)
api.upload_folder(
    folder_path="/home/weihongliang/v46-hf-space",
    repo_id="openbmb/MiniCPM-V-4_6-Demo",
    repo_type="space",
    commit_message="add static landing for MiniCPM-V 4.6 demos",
)
```

最终入口：

```
https://huggingface.co/spaces/openbmb/MiniCPM-V-4_6-Demo
渲染 host: https://openbmb-minicpm-v-4-6-demo.static.hf.space/
```

> Static Space 的 host 是 `<owner>-<repo>.static.hf.space`，Gradio Space 是
> `<owner>-<repo>.hf.space`。两个域名规则不一样，curl 调试时别搞混。

### 8.3 修改 `openbmb/MiniCPM-o-4_5-Demo` 的嵌入页

`userisuser` token 在 `openbmb` 组里有写权限（已验证，commit 成功）。

操作：

1. 拉 `app.py`：
   ```bash
   curl -sL https://huggingface.co/spaces/openbmb/MiniCPM-o-4_5-Demo/raw/main/app.py -o /tmp/o45_app.py
   ```
2. 替换 4 处 iframe URL：
   ```bash
   sed -i 's|https://openbmb.github.io/MiniCPM-o-Demo/|https://minicpmo45.modelbest.cn/|g' /tmp/o45_app.py
   ```
3. 推回去：
   ```python
   api.upload_file(
       path_or_fileobj="/tmp/o45_app.py",
       path_in_repo="app.py",
       repo_id="openbmb/MiniCPM-o-4_5-Demo",
       repo_type="space",
       commit_message="switch embedded demo URL to minicpmo45.modelbest.cn",
   )
   ```
4. 等约 30-60 秒，Gradio Space 自动重建，状态 `RUNNING` 即生效。

### 8.4 为什么两个 demo URL 不能 iframe 进 v46 landing

- HF Space 走 HTTPS，国内/海外两个 demo 都是 HTTP。浏览器 mixed-content 会拦截。
- 所以 `index.html` 里用 `target="_blank"` 新标签页跳转，避开 mixed content。
- 如果未来想 iframe 嵌入：要么给两个 demo 配 HTTPS（国内已经过 frp 走 9443，
  但还是 HTTP；海外要在 GCP 防火墙开 443 + Caddy/Nginx + Let's Encrypt），
  要么模仿 o45 那种"Gradio Space 内部嵌一个域名"的玩法，但目标必须是 HTTPS。

## 9. 已知遗留事项

- 国内 8891 当前是单 Instruct，要双模型需另外腾出 ~30G 显存或换到一张更空的卡。
- 海外 8000 是裸 HTTP；如果要 HTTPS 需要在 GCP 防火墙上开 443 并配证书。
- HF landing Space 当前在 `userisuser` 用户名下，是否转到 `openbmb` 组织待定。
- HF token 文件 `~/.hf-token` 在海外机器上明文保存（`chmod 600`），用完可以删除。
