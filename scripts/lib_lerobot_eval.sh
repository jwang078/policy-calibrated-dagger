#!/bin/bash
# Shared helpers for invoking `lerobot-eval` from bash wrappers.
#
# This file is meant to be `source`'d, not executed directly. It defines
# functions (prefixed `lle_*`) used by:
#   * my_scripts/run_all_evals.sh           — eval every checkpoint in a folder
#   * my_scripts/dagger_reeval_lineage.sh   — re-eval a DAgger lineage's rounds
#                                             at a new --env.episode_length
# Keeping these in one place avoids drift between the two callers and
# means future eval-invocation scripts can compose them.
#
# Functions:
#   lle_camera_names_from_train_config <ckpt_dir>       → JSON list string
#   lle_image_resize_modes_from_train_config <ckpt_dir> → JSON list string
#   lle_rename_map_from_train_config <ckpt_dir>         → JSON object string
#   lle_n_action_steps_from_train_config <ckpt_dir>     → int or empty
#   lle_build_eval_cmd ...                              → echoes a lerobot-eval
#                                                          command on stdout
#
# All helpers tolerate missing files / fields by returning sensible defaults
# matching what run_all_evals.sh used to hard-code in-place.

# Guard against double-sourcing (harmless but noisy).
if [[ -n "${_LIB_LEROBOT_EVAL_LOADED:-}" ]]; then
    return 0
fi
_LIB_LEROBOT_EVAL_LOADED=1


# Search a checkpoint dir for its train_config.json. Returns 0 + prints the
# path on success; returns 1 on failure (caller decides how to recover).
#
# Accepts either a `pretrained_model` dir (where train_config.json lives
# directly) or its parent (where pretrained_model/train_config.json lives).
_lle_resolve_train_config() {
    local ckpt_dir="$1"
    if [[ -f "$ckpt_dir/train_config.json" ]]; then
        echo "$ckpt_dir/train_config.json"
        return 0
    fi
    if [[ -f "$ckpt_dir/pretrained_model/train_config.json" ]]; then
        echo "$ckpt_dir/pretrained_model/train_config.json"
        return 0
    fi
    return 1
}


# Extract camera names from a checkpoint's train_config.json. Prefers
# `env.camera_names`; falls back to parsing them out of
# `policy.input_features` keys like `observation.images.<NAME>`. Returns
# `["base_rgb"]` if neither source is populated.
lle_camera_names_from_train_config() {
    local ckpt_dir="$1"
    local config_file
    if ! config_file=$(_lle_resolve_train_config "$ckpt_dir"); then
        echo '["base_rgb"]'
        return 0
    fi
    python3 -c "
import json, re
config = json.load(open('$config_file'))
env = config.get('env')
if env and env.get('camera_names'):
    print(json.dumps(env['camera_names']))
else:
    input_features = config.get('policy', {}).get('input_features', {})
    names = []
    for key in input_features:
        m = re.match(r'observation\.images\.(\w+)', key)
        if m: names.append(m.group(1))
    print(json.dumps(sorted(names)) if names else '[\"base_rgb\"]')
" 2>/dev/null || echo '["base_rgb"]'
}


# Extract image resize modes (as JSON list string). Supports both the new
# field name (`env.image_resize_modes`, a list) and the legacy
# `env.image_resize_mode` (a single string). Default `["letterbox"]`.
lle_image_resize_modes_from_train_config() {
    local ckpt_dir="$1"
    local config_file
    if ! config_file=$(_lle_resolve_train_config "$ckpt_dir"); then
        echo '["letterbox"]'
        return 0
    fi
    python3 -c "
import json
config = json.load(open('$config_file'))
env = config.get('env', {})
if env.get('image_resize_modes'):
    print(json.dumps(env['image_resize_modes']))
elif env.get('image_resize_mode'):
    print(json.dumps([env['image_resize_mode']]))
else:
    print('[\"letterbox\"]')
" 2>/dev/null || echo '["letterbox"]'
}


# Extract the rename_map (used to remap raw env observation keys to the
# policy's input feature keys). Default `{}`.
lle_rename_map_from_train_config() {
    local ckpt_dir="$1"
    local config_file
    if ! config_file=$(_lle_resolve_train_config "$ckpt_dir"); then
        echo '{}'
        return 0
    fi
    python3 -c "
import json
config = json.load(open('$config_file'))
print(json.dumps(config.get('rename_map', {})))
" 2>/dev/null || echo '{}'
}


# Build a `lerobot-eval` command string and echo it to stdout. Caller is
# responsible for printing / executing / logging the command.
#
# Usage:
#   lle_build_eval_cmd \
#       --policy_path=PATH                  # REQUIRED — checkpoint pretrained_model dir
#       --output_dir=DIR                    # REQUIRED — eval_info.json + videos land here
#       --eval_benchmark_repo_id=REPO       # REQUIRED — eval benchmark dataset
#       --task=TASK                         # default upright_small_engine_new
#       --camera_names=JSON                 # default auto-derived from train_config
#       --image_resize_modes=JSON           # default auto-derived
#       --rename_map=JSON                   # default auto-derived
#       --fps=N                             # default 30
#       --n_episodes=N                      # default 5
#       --seed=N                            # default 0
#       --episode_length=N                  # OPTIONAL — adds --env.episode_length
#       --eval_benchmark_subset=JSON        # OPTIONAL — adds --env.eval_benchmark_subset
#       --extra_args="STR"                  # OPTIONAL — verbatim appended (e.g. --policy.last_mile_config.enabled=true)
#
# Defaults match run_all_evals.sh's longstanding invocation so existing
# behavior round-trips identically when callers pass equivalent inputs.
lle_build_eval_cmd() {
    local policy_path="" output_dir="" eval_benchmark_repo_id=""
    local task="upright_small_engine_new"
    local camera_names="" image_resize_modes="" rename_map=""
    local fps=30 n_episodes=5 seed=0
    local episode_length="" eval_benchmark_subset="" extra_args=""

    for arg in "$@"; do
        case "$arg" in
            --policy_path=*)             policy_path="${arg#*=}" ;;
            --output_dir=*)              output_dir="${arg#*=}" ;;
            --eval_benchmark_repo_id=*)  eval_benchmark_repo_id="${arg#*=}" ;;
            --task=*)                    task="${arg#*=}" ;;
            --camera_names=*)            camera_names="${arg#*=}" ;;
            --image_resize_modes=*)      image_resize_modes="${arg#*=}" ;;
            --rename_map=*)              rename_map="${arg#*=}" ;;
            --fps=*)                     fps="${arg#*=}" ;;
            --n_episodes=*)              n_episodes="${arg#*=}" ;;
            --seed=*)                    seed="${arg#*=}" ;;
            --episode_length=*)          episode_length="${arg#*=}" ;;
            --eval_benchmark_subset=*)   eval_benchmark_subset="${arg#*=}" ;;
            --extra_args=*)              extra_args="${arg#*=}" ;;
            *) echo "lle_build_eval_cmd: unknown arg $arg" >&2; return 1 ;;
        esac
    done

    for required in policy_path output_dir eval_benchmark_repo_id; do
        if [[ -z "${!required}" ]]; then
            echo "lle_build_eval_cmd: --${required}=... is required" >&2
            return 1
        fi
    done

    # Auto-derive whatever wasn't passed explicitly.
    [[ -z "$camera_names" ]]       && camera_names=$(lle_camera_names_from_train_config "$policy_path")
    [[ -z "$image_resize_modes" ]] && image_resize_modes=$(lle_image_resize_modes_from_train_config "$policy_path")
    [[ -z "$rename_map" ]]         && rename_map=$(lle_rename_map_from_train_config "$policy_path")

    # Single-quote JSON fields so embedded spaces / brackets survive the eval.
    local cmd
    cmd="lerobot-eval \\
        --env.type=splatsim \\
        --env.task=$task \\
        --env.camera_names='$camera_names' \\
        --env.image_resize_modes='$image_resize_modes' \\
        --env.fps=$fps \\
        --env.eval_benchmark_repo_id=$eval_benchmark_repo_id \\
        --policy.path=$policy_path \\
        --eval.n_episodes=$n_episodes \\
        --output_dir=$output_dir \\
        --eval.batch_size=1 \\
        --eval.use_async_envs=false \\
        --seed=$seed \\
        --rename_map='$rename_map'"

    if [[ -n "$episode_length" ]]; then
        cmd="$cmd \\
        --env.episode_length=$episode_length"
    fi
    if [[ -n "$eval_benchmark_subset" ]]; then
        cmd="$cmd \\
        --env.eval_benchmark_subset='$eval_benchmark_subset'"
    fi
    if [[ -n "$extra_args" ]]; then
        cmd="$cmd \\
        $extra_args"
    fi

    echo "$cmd"
}
