#!/bin/bash
# Shared helpers for managing a SplatSim sim process from bash wrappers.
#
# This file is meant to be `source`'d, not executed directly. It defines
# four functions (prefixed `splat_*`) used by callers that need to spin up
# their own sim process instead of requiring a pre-launched one on
# --env_external_port:
#   splat_port_in_use <port>            → echo pid (or empty) bound to port
#   splat_wait_for_port <port> <max_s>  → block until port comes up; rc=0/1
#   splat_start_sim                     → launch SplatSim in background
#   splat_stop_sim                      → SIGTERM (then SIGKILL) the sim
#
# Mirrors the orchestrator's own sim lifecycle in
# my_scripts/dagger_orchestrate.sh:1774-1911 (~80 lines). Callers set the
# following globals before invoking splat_start_sim:
#   SPLATSIM_ROOT          absolute path to SplatSim repo (default: ~/code/SplatSim)
#   ENV_EXTERNAL_PORT      TCP port for the sim to bind (default: 6001)
#   ENV_EXTERNAL_HOST      hostname (default: 127.0.0.1)
#   SPLATSIM_ROBOT         --robot arg (default: sim_ur_pybullet_small_engine_new_interactive)
#   SPLATSIM_ROBOT_NAME    --robot_name arg (default: robot_iphone_w_engine_new)
#   EVAL_BENCHMARK_REPO_ID --eval_benchmark_repo_id (REQUIRED — no default)
#   HEADLESS               "true"/"false" — if true, adds --headless (p.DIRECT)
#   LEROBOT_ROOT           where to write the sim log under outputs/dagger/
#   DRY_RUN                "true"/"false" — if true, prints commands
#
# Sets these globals so callers can introspect / clean up:
#   MANAGED_SIM_PID        pid of the running sim (or "DRYRUN" in dry-run,
#                          empty if not running)
#   MANAGED_SIM_LOG        absolute path to the sim's log file
#
# Typical caller pattern:
#   source "$SCRIPT_DIR/lib_splatsim_manage.sh"
#   ENV_EXTERNAL_PORT=6001
#   EVAL_BENCHMARK_REPO_ID=JennyWWW/eval_splatsim_approach_lever_benchmark_1000
#   HEADLESS=true
#   trap splat_stop_sim EXIT
#   splat_start_sim
#   ... run lerobot-eval against --env.external_port=$ENV_EXTERNAL_PORT ...
#   # trap handles cleanup; explicit splat_stop_sim works too

# Guard against double-sourcing.
if [[ -n "${_LIB_SPLATSIM_MANAGE_LOADED:-}" ]]; then
    return 0
fi
_LIB_SPLATSIM_MANAGE_LOADED=1

# Defaults — callers can override before splat_start_sim runs.
: "${SPLATSIM_ROOT:=$HOME/code/SplatSim}"
: "${ENV_EXTERNAL_PORT:=6001}"
: "${ENV_EXTERNAL_HOST:=127.0.0.1}"
: "${SPLATSIM_ROBOT:=sim_ur_pybullet_small_engine_new_interactive}"
: "${SPLATSIM_ROBOT_NAME:=robot_iphone_w_engine_new}"
: "${HEADLESS:=false}"
: "${DRY_RUN:=false}"
# Where to write the per-launch sim log. Callers should set this to a path
# INSIDE their training/output dir (e.g., <train_dir>/dagger/) so the log
# is co-located with the rest of that run's artifacts and gets wiped
# automatically when the lineage is cleaned up via dagger_cleanup_lineage.
# Default: $LEROBOT_ROOT/outputs/dagger/ — the legacy shared location
# (where logs accumulate orphaned because nothing else lives there to
# anchor cleanup). New code should override; only the default exists for
# back-compat with callers that haven't been updated yet.
: "${MANAGED_SIM_LOG_DIR:=}"
MANAGED_SIM_PID=""
MANAGED_SIM_LOG=""


splat_port_in_use() {
    # echoes pid bound to TCP port $1 (first one found), or empty. Tolerant
    # of lsof's nonzero-on-no-match exit code under `set -e + pipefail`.
    local out
    out="$(lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null || true)"
    printf '%s\n' "${out%%$'\n'*}"
}


splat_wait_for_port() {
    # Args: $1=port, $2=max_wait_seconds. rc=0 once a TCP connect succeeds,
    # rc=1 on timeout.
    local port="$1" max_wait="$2"
    local i
    for ((i=1; i<=max_wait; i++)); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3<&-; exec 3>&-
            return 0
        fi
        sleep 1
    done
    return 1
}


splat_start_sim() {
    if [[ -z "${EVAL_BENCHMARK_REPO_ID:-}" ]]; then
        echo "ERROR (lib_splatsim_manage): EVAL_BENCHMARK_REPO_ID must be set before splat_start_sim" >&2
        exit 1
    fi
    # Optional benchmark subset string ("3,7,10,23,25") — when non-empty,
    # passed to launch_nodes.py via --eval_benchmark_subset so SplatSim's
    # _eval_benchmark_subset matches the caller's intent. Empty = use
    # launch_nodes.py default (full subset). tyro (launch_nodes.py's
    # parser) expects each List[int] element as its own argv slot, so we
    # split on commas before appending.
    local _subset_arg=()
    if [[ -n "${EVAL_BENCHMARK_SUBSET:-}" ]]; then
        local _ebs_arr=()
        IFS=',' read -ra _ebs_arr <<< "$EVAL_BENCHMARK_SUBSET"
        _subset_arg=( --eval_benchmark_subset "${_ebs_arr[@]}" )
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        [[ "$MANAGED_SIM_PID" == "DRYRUN" ]] && return 0
        local _hl=""
        [[ "$HEADLESS" == "true" ]] && _hl=" --headless"
        local _subset_str=""
        [[ -n "${EVAL_BENCHMARK_SUBSET:-}" ]] && _subset_str=" --eval_benchmark_subset $EVAL_BENCHMARK_SUBSET"
        echo "[DRY-RUN] would start SplatSim on port $ENV_EXTERNAL_PORT:"
        echo "[DRY-RUN]   cwd: $SPLATSIM_ROOT"
        echo "[DRY-RUN]   cmd: python scripts/launch_nodes.py --robot $SPLATSIM_ROBOT --robot_port $ENV_EXTERNAL_PORT --hostname $ENV_EXTERNAL_HOST --robot_name $SPLATSIM_ROBOT_NAME --eval_benchmark_repo_id $EVAL_BENCHMARK_REPO_ID$_subset_str$_hl"
        MANAGED_SIM_PID="DRYRUN"
        return 0
    fi
    # Already running under our management?
    if [[ -n "$MANAGED_SIM_PID" ]] && kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        return 0
    fi
    # Port already in use by someone else?
    local existing
    existing="$(splat_port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -n "$existing" ]]; then
        echo "ERROR: port $ENV_EXTERNAL_PORT already in use by pid $existing." >&2
        echo "  Either kill it, change --env_external_port, or change ENV_EXTERNAL_PORT and re-run." >&2
        exit 1
    fi
    : "${LEROBOT_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    # Resolve log directory. Caller-provided dir wins (preferred — lives
    # inside the lineage's training tree). Falls back to the legacy shared
    # location for callers that haven't been updated.
    local _log_dir
    if [[ -n "$MANAGED_SIM_LOG_DIR" ]]; then
        _log_dir="$MANAGED_SIM_LOG_DIR"
    else
        _log_dir="$LEROBOT_ROOT/outputs/dagger"
    fi
    mkdir -p "$_log_dir"
    MANAGED_SIM_LOG="$_log_dir/splatsim_$(date +%Y%m%d_%H%M%S).log"
    local launch_cmd=(
        python scripts/launch_nodes.py
        --robot              "$SPLATSIM_ROBOT"
        --robot_port         "$ENV_EXTERNAL_PORT"
        --hostname           "$ENV_EXTERNAL_HOST"
        --robot_name         "$SPLATSIM_ROBOT_NAME"
        --eval_benchmark_repo_id "$EVAL_BENCHMARK_REPO_ID"
    )
    [[ "${#_subset_arg[@]}" -gt 0 ]] && launch_cmd+=( "${_subset_arg[@]}" )
    [[ "$HEADLESS" == "true" ]] && launch_cmd+=( --headless )
    echo "Starting SplatSim:"
    echo "  cwd:     $SPLATSIM_ROOT"
    echo "  cmd:     ${launch_cmd[*]}"
    echo "  log:     $MANAGED_SIM_LOG"
    # Launch sim in background. setsid puts it in its own session so it survives
    # SIGINT to our shell (we kill it explicitly in splat_stop_sim / trap).
    (
        cd "$SPLATSIM_ROOT" || exit 1
        setsid "${launch_cmd[@]}" </dev/null >"$MANAGED_SIM_LOG" 2>&1
    ) &
    local SIM_LAUNCH_BG_PID=$!
    echo "  launcher subshell pid: $SIM_LAUNCH_BG_PID (sim will fork under setsid)"
    echo -n "  waiting for port $ENV_EXTERNAL_PORT to come up "
    if splat_wait_for_port "$ENV_EXTERNAL_PORT" 300; then
        echo "ready."
    else
        echo
        echo "ERROR: SplatSim did not come up within 300s. Last 30 log lines:" >&2
        tail -30 "$MANAGED_SIM_LOG" >&2 || true
        exit 1
    fi
    MANAGED_SIM_PID="$(splat_port_in_use "$ENV_EXTERNAL_PORT")"
    if [[ -z "$MANAGED_SIM_PID" ]]; then
        echo "WARNING: port is up but pid lookup via lsof returned empty." >&2
    else
        echo "  pid:     $MANAGED_SIM_PID"
    fi
    # Give the sim a beat to finish internal init beyond just port-bind.
    sleep 3
}


splat_stop_sim() {
    if [[ "$MANAGED_SIM_PID" == "DRYRUN" ]]; then
        echo "[DRY-RUN] would stop SplatSim on port $ENV_EXTERNAL_PORT"
        MANAGED_SIM_PID=""
        return 0
    fi
    [[ -z "$MANAGED_SIM_PID" ]] && return 0
    if ! kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        MANAGED_SIM_PID=""
        return 0
    fi
    echo "Stopping SplatSim (pid=$MANAGED_SIM_PID)..."
    local pgid
    pgid="$(ps -o pgid= -p "$MANAGED_SIM_PID" 2>/dev/null | tr -d ' ')"
    if [[ -n "$pgid" ]]; then
        kill -TERM -"$pgid" 2>/dev/null || true
    else
        kill -TERM "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    local i
    for ((i=1; i<=30; i++)); do
        kill -0 "$MANAGED_SIM_PID" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$MANAGED_SIM_PID" 2>/dev/null; then
        echo "  sim did not exit on SIGTERM; sending SIGKILL."
        [[ -n "$pgid" ]] && kill -KILL -"$pgid" 2>/dev/null || kill -KILL "$MANAGED_SIM_PID" 2>/dev/null || true
    fi
    for ((i=1; i<=15; i++)); do
        [[ -z "$(splat_port_in_use "$ENV_EXTERNAL_PORT")" ]] && break
        sleep 1
    done
    MANAGED_SIM_PID=""
    echo "  stopped."
}
