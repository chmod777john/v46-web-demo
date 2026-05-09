#!/usr/bin/env bash
# ==============================================================================
# Launch N Gradio instances of MiniCPM-V 4.6 in tmux, each loading BOTH the
# instruct and thinking checkpoints in the same process. The "Thinking Mode"
# toggle in the UI then just switches the active model. Instances optionally
# auto-register to a single load_balancer pool (ip_hash + SSE).
#
# Defaults (can be overridden by env or CLI):
#   - Num      : 2 instances
#   - GPUs     : descending from GPU_START (default 7)  -> GPU 7, 6, ...
#   - Port base: PORT_BASE (default 8890)               -> 8890, 8891, ...
#   - LB host  : 127.0.0.1  (set LB_HOST="" or --no-lb to skip)
#   - LB port  : 8121
#
# Typical workflow:
#
#   # 1) start the load balancer (once)
#   cd ../load_balancer
#   python load_balancer.py --port 8121 --strategy ip_hash
#
#   # 2) start 4 dual-model instances on GPU 7,6,5,4, register to LB
#   bash start.sh -n 4
#
#   # stop everything
#   bash start.sh --stop
#
# Single-model modes (legacy, lighter GPU mem):
#   bash start.sh --variant instruct -n 4
#   bash start.sh --variant thinking -n 4
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PY="${SCRIPT_DIR}/app.py"

# -------- Paths --------
CKPT_ROOT="${CKPT_ROOT:-/cache/caitianchi/code/v50/ckpt}"
INSTRUCT_PATH="${INSTRUCT_PATH:-${CKPT_ROOT}/minicpm-v-4_6-0420-rlaif-instruct}"
THINKING_PATH="${THINKING_PATH:-${CKPT_ROOT}/minicpm-v-4_6-0420-rlaif-thinking}"

# -------- Env --------
CONDA_PATH="${CONDA_PATH:-/cache/caitianchi/install/miniconda3}"
CONDA_ENV="${CONDA_ENV:-v46}"

# -------- Defaults (override via CLI) --------
NUM_INSTANCES="${NUM_INSTANCES:-2}"
VARIANT="${VARIANT:-dual}"             # dual | instruct | thinking
GPU_START="${GPU_START:-7}"            # legacy (fallback when --gpus not given)
GPUS="${GPUS:-}"                       # explicit comma-list, e.g. "0,1,2,3"
PORT_BASE="${PORT_BASE:-8890}"
SESSION_PREFIX="${SESSION_PREFIX:-v46}"

LB_HOST="${LB_HOST:-127.0.0.1}"
LB_PORT="${LB_PORT:-8121}"
LOCAL_IP="${LOCAL_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
LOCAL_IP="${LOCAL_IP:-127.0.0.1}"

# -------- Flags --------
DO_STOP=false
FORCE_RESTART=false
DO_STATUS=false
NO_LB=false

print_info() { echo "[INFO] $*"; }
print_ok()   { echo "[OK]   $*"; }
print_warn() { echo "[WARN] $*" >&2; }
print_err()  { echo "[ERR]  $*" >&2; }

usage() {
    sed -n '/^# ====/,/^# ====/p' "$0" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
    -n, --num N             number of instances (default: ${NUM_INSTANCES})
    --variant {dual|instruct|thinking}
                            dual    = load BOTH checkpoints in each process
                                      (needs ~32GB / GPU, recommended 80G cards)
                            instruct/thinking = load only one (needs ~16GB / GPU)
                            (default: ${VARIANT})
    --gpus LIST             comma-separated GPU ids, instances distributed round-robin
                            e.g. --gpus 0,1,2,3 -n 8 gives 2 instances per GPU
    --gpu-start N           (fallback) first instance uses GPU N, decrementing
                            (default: ${GPU_START}, only used if --gpus not set)
    --port-base N           first instance uses port N, incrementing (default: ${PORT_BASE})
    --session-prefix STR    tmux session prefix (default: ${SESSION_PREFIX})
    --lb-host HOST          load balancer host (default: ${LB_HOST})
    --lb-port PORT          load balancer port (default: ${LB_PORT})
    --local-ip IP           IP to register to the LB (default: auto-detect)
    --no-lb                 don't register to any load balancer
    -f, --force             kill existing tmux session if it exists
    --stop                  stop all ${SESSION_PREFIX}- tmux sessions (also unregisters)
    --status                show tmux sessions + LB status
    -h, --help              show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num)            NUM_INSTANCES="$2"; shift 2 ;;
        --variant)           VARIANT="$2"; shift 2 ;;
        --gpus)              GPUS="$2"; shift 2 ;;
        --gpu-start)         GPU_START="$2"; shift 2 ;;
        --port-base)         PORT_BASE="$2"; shift 2 ;;
        --session-prefix)    SESSION_PREFIX="$2"; shift 2 ;;
        --lb-host)           LB_HOST="$2"; shift 2 ;;
        --lb-port)           LB_PORT="$2"; shift 2 ;;
        --local-ip)          LOCAL_IP="$2"; shift 2 ;;
        --no-lb)             NO_LB=true; shift ;;
        -f|--force)          FORCE_RESTART=true; shift ;;
        --stop)              DO_STOP=true; shift ;;
        --status)            DO_STATUS=true; shift ;;
        -h|--help)           usage; exit 0 ;;
        *)                   print_err "Unknown argument: $1"; usage; exit 1 ;;
    esac
done

[[ -z "$LB_HOST" ]] && NO_LB=true

session_exists() { tmux has-session -t "$1" 2>/dev/null; }

kill_session() {
    local name="$1"
    if session_exists "$name"; then
        tmux kill-session -t "$name"
        print_info "killed session: $name"
    fi
}

lb_register() {
    local host="$1" port="$2" name="$3"
    $NO_LB && return 0
    local url="http://${LB_HOST}:${LB_PORT}/_lb/register"
    local data="{\"host\": \"${host}\", \"port\": ${port}, \"name\": \"${name}\"}"
    for i in 1 2 3; do
        local resp
        resp=$(curl -s -X POST "$url" -H "Content-Type: application/json" -d "$data" 2>/dev/null || echo "failed")
        if [[ "$resp" != "failed" ]] && [[ "$resp" == *"status"* ]]; then
            print_ok "registered ${name} (${host}:${port}) -> ${LB_HOST}:${LB_PORT}"
            return 0
        fi
        print_info "register retry ($i/3)..."
        sleep 2
    done
    print_warn "register failed, LB not reachable at ${LB_HOST}:${LB_PORT}"
    return 1
}

lb_unregister() {
    local host="$1" port="$2"
    $NO_LB && return 0
    curl -s -X POST "http://${LB_HOST}:${LB_PORT}/_lb/unregister" \
        -H "Content-Type: application/json" \
        -d "{\"host\": \"${host}\", \"port\": ${port}}" >/dev/null 2>&1 || true
}

start_one_gradio() {
    # $1=session  $2=gpu  $3=port  $4=app_args
    local name="$1" gpu="$2" port="$3" app_args="$4"

    if session_exists "$name"; then
        if $FORCE_RESTART; then
            print_info "restarting $name..."
            kill_session "$name"
            sleep 1
        else
            print_info "session $name already exists, skip (use --force to restart)"
            return 0
        fi
    fi

    # Write a tiny launcher script and have tmux run it. This avoids the
    # triple-quoting nightmare of embedding the whole conda+python command
    # in tmux's argv.
    local log_dir="${SCRIPT_DIR}/logs"
    mkdir -p "$log_dir"
    local launcher="${log_dir}/launch-${name}.sh"
    local logfile="${log_dir}/${name}.log"

    cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -eu
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${gpu}
echo "[launcher] start $(date) gpu=${gpu} port=${port}"
exec python -u app.py ${app_args} --port=${port}
EOF
    chmod +x "$launcher"

    # keep the tmux window alive on exit so we can see the traceback
    tmux new-session -d -s "$name" "bash '$launcher' 2>&1 | tee '$logfile'; echo '---EXITED---'; sleep 60"
    print_ok "started session=${name} gpu=${gpu} port=${port}  log=${logfile}"
}

# -------- Stop --------
if $DO_STOP; then
    echo "=== stopping all sessions matching '${SESSION_PREFIX}-' ==="
    sessions=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | grep "^${SESSION_PREFIX}-" || true)
    if [[ -z "$sessions" ]]; then
        print_info "no matching sessions"
        exit 0
    fi
    for s in $sessions; do
        port_part="${s##*-}"
        if [[ "$port_part" =~ ^[0-9]+$ ]]; then
            lb_unregister "$LOCAL_IP" "$port_part"
        fi
        kill_session "$s"
    done
    exit 0
fi

# -------- Status --------
if $DO_STATUS; then
    echo "=== tmux sessions ==="
    tmux list-sessions 2>/dev/null | grep "^${SESSION_PREFIX}-" || echo "  (none)"
    echo ""
    if ! $NO_LB; then
        echo "=== LB status (${LB_HOST}:${LB_PORT}) ==="
        curl -s "http://${LB_HOST}:${LB_PORT}/_lb/status" 2>/dev/null | python3 -m json.tool 2>/dev/null \
            || echo "  (LB not reachable)"
    fi
    exit 0
fi

# -------- Resolve variant -> app.py args --------
case "$VARIANT" in
    dual)
        APP_ARGS="--instruct_path='${INSTRUCT_PATH}' --thinking_path='${THINKING_PATH}' --model_name='MiniCPM-V 4.6'"
        VARIANT_TAG="dual"
        ;;
    instruct)
        APP_ARGS="--instruct_path='${INSTRUCT_PATH}' --model_name='MiniCPM-V 4.6 (instruct)'"
        VARIANT_TAG="instruct"
        ;;
    thinking)
        APP_ARGS="--thinking_path='${THINKING_PATH}' --default_thinking --model_name='MiniCPM-V 4.6 (thinking)'"
        VARIANT_TAG="thinking"
        ;;
    *)
        print_err "unknown --variant: $VARIANT (expected: dual | instruct | thinking)"
        exit 1
        ;;
esac

# -------- Main launch --------
# -------- Build GPU list --------
if [[ -n "$GPUS" ]]; then
    IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
else
    GPU_ARRAY=()
    for ((i=0; i<NUM_INSTANCES; i++)); do
        GPU_ARRAY+=("$((GPU_START - i))")
    done
fi

echo "=============================================="
echo " MiniCPM-V 4.6 Gradio Cluster"
echo "=============================================="
echo " variant       : ${VARIANT}"
echo " num instances : ${NUM_INSTANCES}"
echo " gpu pool      : ${GPU_ARRAY[*]}  (round-robin)"
echo " port base     : ${PORT_BASE} (ascending)"
[[ "$VARIANT" != "thinking" ]] && echo " instruct ckpt : ${INSTRUCT_PATH}"
[[ "$VARIANT" != "instruct" ]] && echo " thinking ckpt : ${THINKING_PATH}"
echo " local ip      : ${LOCAL_IP}"
if $NO_LB; then
    echo " load balancer : DISABLED"
else
    echo " load balancer : ${LB_HOST}:${LB_PORT}"
fi
echo "=============================================="
echo ""

for ((i=0; i<NUM_INSTANCES; i++)); do
    gpu="${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}"
    port=$((PORT_BASE + i))

    if (( gpu < 0 )); then
        print_warn "GPU id negative for instance #${i}, skipping"
        continue
    fi

    session="${SESSION_PREFIX}-${VARIANT_TAG}-${port}"
    reg_name="GPU${gpu}@${LOCAL_IP}"

    echo "--- instance #${i} ---"

    if $FORCE_RESTART && ! $NO_LB; then
        lb_unregister "$LOCAL_IP" "$port"
    fi

    start_one_gradio "$session" "$gpu" "$port" "$APP_ARGS"

    if ! $NO_LB; then
        sleep 2
        lb_register "$LOCAL_IP" "$port" "$reg_name" || true
    fi
    echo ""
done

echo "=============================================="
echo " done. useful commands:"
echo "   tmux list-sessions | grep ${SESSION_PREFIX}-"
echo "   tmux attach -t ${SESSION_PREFIX}-${VARIANT_TAG}-${PORT_BASE}"
echo "   bash $0 --status"
echo "   bash $0 --stop"
if ! $NO_LB; then
    echo ""
    echo " entrypoint for users:  http://${LB_HOST}:${LB_PORT}"
fi
echo "=============================================="
