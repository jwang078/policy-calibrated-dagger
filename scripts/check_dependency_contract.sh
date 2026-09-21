#!/bin/bash
# Run after updating lerobot (the fork) or SplatSim: is pcdagger still compatible?
#   bash scripts/check_dependency_contract.sh
# 1. tests/test_dependency_contract.py — every dependency name pcdagger imports + the fork hooks' signatures
# 2. how far pcdagger's train / eval scripts have drifted from the fork's current lerobot_train / lerobot_eval
#    (they started as copies; when upstream changes those scripts, port the change here)
# 3. the fork's own divergence from huggingface/lerobot (scripts/sync_upstream.sh --check)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../pcdagger/paths.sh"
set -u
cd "$PCDAGGER_ROOT"
echo "== 1. dependency contract"
"$PCDAGGER_PY" -m pytest -q tests/test_dependency_contract.py || exit 1
echo "== 2. drift of pcdagger/train.py and pcdagger/dagger/eval.py from the fork's scripts (lines changed)"
for pair in "pcdagger/train.py:$LEROBOT_ROOT/src/lerobot/scripts/lerobot_train.py" \
            "pcdagger/dagger/eval.py:$LEROBOT_ROOT/src/lerobot/scripts/lerobot_eval.py"; do
    ours=${pair%%:*}; theirs=${pair#*:}
    printf '  %-26s %s\n' "$ours" "$(diff "$theirs" "$ours" | grep -c '^[<>]') lines differ (fork at $(git -C "$LEROBOT_ROOT" rev-parse --short HEAD))"
done
echo "== 3. fork vs huggingface/lerobot"
(cd "$LEROBOT_ROOT" && bash "$PCDAGGER_ROOT/scripts/sync_upstream.sh" --check) 2>&1 | sed 's/^/  /'
