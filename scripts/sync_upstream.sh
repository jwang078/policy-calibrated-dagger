#!/usr/bin/env bash
# Rebase this fork onto upstream/main, with the guardrails learned the hard way.
#
#   ./my_scripts/sync_upstream.sh            # rebase main onto upstream/main
#   ./my_scripts/sync_upstream.sh --check    # report divergence only, change nothing
#
# WHY THIS EXISTS
#   A plain `git pull --rebase upstream main` replays EVERY fork commit. With ~80
#   commits and ~30 files that upstream also touches, the same file conflicts over
#   and over. Two things make that bearable, and both are set up here:
#     * rerere       — records each conflict resolution and REPLAYS it automatically,
#                      so a file that conflicts in 10 replayed commits is resolved once.
#     * sync often   — the conflict count scales with how far you have drifted.
#                      Monthly is minutes; 6 months is a day.
#
# HARD-WON NOTES
#   * Nightly torch/upstream wheels get PRUNED from the index. If you pin one, the
#     ONLY way back is a conda env clone. Snapshot before large dependency moves.
#   * Upstream renames modules (lerobot.types -> lerobot.lerobot_types) and factors
#     functions out of train() . Files that auto-merge WITHOUT a conflict can still
#     break: always run the post-rebase verification below, not just `git status`.
#   * TrainPipelineConfig field renames are absorbed by
#     `_migrate_legacy_renamed_fields` in src/lerobot/configs/train.py — add an entry
#     there instead of hand-editing saved train_config.json files.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

CHECK_ONLY=false
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=true

git remote get-url upstream >/dev/null 2>&1 || {
    echo "ERROR: no 'upstream' remote. git remote add upstream git@github.com:huggingface/lerobot.git" >&2; exit 1; }

echo "== enabling rerere (replays repeated conflict resolutions) =="
git config rerere.enabled true
git config rerere.autoupdate true

echo "== fetching upstream =="
git fetch upstream --quiet

BRANCH=$(git rev-parse --abbrev-ref HEAD)
read -r BEHIND AHEAD < <(git rev-list --left-right --count "upstream/main...$BRANCH" | awk '{print $1, $2}')
echo "  $BRANCH is $AHEAD commit(s) ahead, $BEHIND behind upstream/main"

BASE=$(git merge-base upstream/main "$BRANCH")
OVERLAP=$(comm -12 \
    <(git diff --name-only "$BASE" "$BRANCH" | sort -u) \
    <(git diff --name-only "$BASE" upstream/main | sort -u) | wc -l)
echo "  files touched by BOTH sides (the conflict surface): $OVERLAP"

if [[ "$CHECK_ONLY" == true ]]; then
    comm -12 <(git diff --name-only "$BASE" "$BRANCH" | sort -u) \
             <(git diff --name-only "$BASE" upstream/main | sort -u) | sed 's/^/    /'
    exit 0
fi

[[ -z "$(git status --porcelain)" ]] || { echo "ERROR: working tree dirty; commit or stash first." >&2; exit 1; }

SNAP="backup/${BRANCH//\//-}-$(date +%Y%m%d-%H%M%S)"
git branch "$SNAP"
echo "== snapshot: $SNAP  (git reset --hard $SNAP to undo everything) =="

echo "== rebasing =="
if git rebase upstream/main; then
    echo "== rebase clean =="
else
    cat <<'MSG'

Conflicts. Resolve, `git add`, then `git rebase --continue` (rerere will
auto-apply any conflict you have already resolved once).

Resolution heuristics that held up across this repo's history:
  * orthogonal additions (imports, new functions, extra validations) -> KEEP BOTH
  * upstream refactored a block you also edited -> take UPSTREAM's structure and
    re-apply your feature on top (e.g. pass a hook/param) rather than reverting it
  * your side re-adds a block that already exists below the conflict -> take
    UPSTREAM's (empty) side; duplicating it is the most common trap here
MSG
    exit 1
fi

cat <<'MSG'

== VERIFY (auto-merged files can still be broken) ==
MSG
git grep -n '^<<<<<<< \|^>>>>>>> ' -- 'src/**' 'my_scripts/**' && { echo "  conflict markers left!"; exit 1; } || echo "  no conflict markers"
python -c "
import lerobot, lerobot.scripts.lerobot_train, lerobot.scripts.lerobot_eval
import lerobot.datasets.factory, lerobot.policies.factory, lerobot.envs.factory
print('  imports OK')"

# Dependency drift. A sync moves you onto upstream code that may REQUIRE newer
# deps than your env has, and nothing about the merge reveals it — you find out
# when a specific call fails at runtime. (Real example: upstream moved to
# draccus>=0.11.6, whose `encode(obj, declared_type)` takes a second argument;
# on the pinned-but-stale 0.10.0 every run trained fine and then died at the
# FIRST CHECKPOINT with "encode() takes 1 positional argument but 2 were given".)
python - <<'PYEOF'
import tomllib, importlib.metadata as md
from packaging.requirements import Requirement
from packaging.version import Version
bad = []
for line in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]:
    try:
        r = Requirement(line)
    except Exception:
        continue
    if r.marker and not r.marker.evaluate():
        continue
    try:
        v = md.version(r.name)
    except md.PackageNotFoundError:
        bad.append(f"{r.name}: NOT INSTALLED (need {r.specifier})")
        continue
    if r.specifier and not r.specifier.contains(Version(v), prereleases=True):
        bad.append(f"{r.name}: have {v}, need {r.specifier}")
print("  deps OK" if not bad else "  DEPENDENCY DRIFT (pip install -U the following):")
for b in bad:
    print("   ", b)
PYEOF
pre-commit run --all-files || echo "  (pre-commit reported issues — fix before pushing)"

cat <<MSG

Next:
  git push --force-with-lease origin $BRANCH
  (force is expected: rebase rewrites history. --force-with-lease refuses if
   origin moved since your last fetch.)
MSG
