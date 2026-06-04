#!/usr/bin/env python3
"""Canonical source of truth for DAgger artifact naming.

Owns BOTH directions:
  * forward (config → name) — what the orchestrator uses to derive every
    dataset / training-dir name from a run's configuration.
  * inverse (name → config) — what downstream viz / cleanup scripts use to
    parse a name back into its constituent parts.

All DAgger Python scripts import from this module. The bash orchestrator
(`dagger_orchestrate.sh`) calls into the CLI shim at the bottom of this file
so its forward-mapping helpers stay in sync without duplicating regex /
string-concatenation logic in two languages.

CLI shim usage from bash:
    SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
    _py_dagger_name() { python3 "$SCRIPT_DIR/dagger_naming.py" "$@"; }
    int_short_for_round() {
        _py_dagger_name int_short \
            --prefix="$SOURCE_INT_SHORT_PREFIX" \
            --infix="$ACTION_INFIX" \
            --round="$1"
    }
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# ────────────────────────────────────────────────────────────────────────────
# Forward functions: config → name.
# Mirrors the bash helpers in dagger_orchestrate.sh (~lines 1148-1167 for
# the per-round helpers, ~lines 787-796 and ~877-885 for the base-name
# derivation helpers).
# ────────────────────────────────────────────────────────────────────────────


def blend_tag_for_ratio(ratio: float) -> str:
    """Convert a forward-flow ratio (float in [0, 1]) → 3-digit zero-padded
    percent string. e.g. 0.9 → "090", 0.1 → "010", 0.95 → "095".

    Mirrors `_blend_tag_for_ratio` in dagger_orchestrate.sh:1165.
    """
    return f"{int(round(float(ratio) * 100)):03d}"


def ratio_for_blend_tag(tag: str | int) -> float:
    """Inverse of `blend_tag_for_ratio`: "090" → 0.9, "010" → 0.1, "095" → 0.95.

    Used by cleanup / introspection scripts that need to round-trip a blend
    dataset name back to its ratio. Accepts the 3-digit zero-padded string
    OR the integer percent value.
    """
    return round(int(tag) / 100, 2)


def int_short(prefix: str, infix: str, round: int) -> str:
    """Intervention dataset short name for a given round.
    Mirrors `int_short_for_round` in dagger_orchestrate.sh:1148.
    """
    return f"{prefix}_{infix}_dag{round}"


def int_repo(hf_user: str, prefix: str, infix: str, round: int) -> str:
    """Full intervention dataset repo id (`<hf_user>/<int_short>`).
    Mirrors `int_repo_for_round` in dagger_orchestrate.sh:1149.
    """
    return f"{hf_user}/{int_short(prefix, infix, round)}"


def blend_short(prefix: str, infix: str, round: int, ratio: float) -> str:
    """Blend dataset short name: `<int_short>_blend<NNN>`.
    Mirrors `blend_short_for_round` in dagger_orchestrate.sh:1166.
    """
    return f"{int_short(prefix, infix, round)}_blend{blend_tag_for_ratio(ratio)}"


def blend_repo(hf_user: str, prefix: str, infix: str, round: int, ratio: float) -> str:
    """Full blend dataset repo id.
    Mirrors `blend_repo_for_round` in dagger_orchestrate.sh:1167.
    """
    return f"{hf_user}/{blend_short(prefix, infix, round, ratio)}"


def nocoll_short(prefix: str, infix: str, round: int, ratio: float) -> str:
    """Collision-filtered blend dataset short name: `<blend_short>_nocoll`.

    Produced by `filter_blend_collisions.py` when
    `--filter_blend_collisions` is set on the orchestrator. Replays the
    blend dataset through a headless simulator and drops episodes that
    touch obstacles; the surviving (and possibly-trimmed) episodes are
    written under this name and used as the merge source instead of the
    raw `_blend<NNN>` dataset.
    """
    return f"{blend_short(prefix, infix, round, ratio)}_nocoll"


def nocoll_repo(hf_user: str, prefix: str, infix: str, round: int, ratio: float) -> str:
    """Full collision-filtered blend dataset repo id."""
    return f"{hf_user}/{nocoll_short(prefix, infix, round, ratio)}"


def merged_short(base_dataset_short: str, infix: str, round: int) -> str:
    """Merged training-dataset short: `<base_dataset_short>_<infix>_dag<N>_m`.
    Mirrors `merged_short_for_round` in dagger_orchestrate.sh:1156.

    Note: this uses BASE_DATASET_SHORT (the NEW lineage's prefix), not
    SOURCE_INT_SHORT_PREFIX. The merged dataset is THIS run's artifact;
    only the intervention + blend datasets reuse source's prefix in rerun mode.
    """
    return f"{base_dataset_short}_{infix}_dag{round}_m"


def merged_repo(hf_user: str, base_dataset_short: str, infix: str, round: int) -> str:
    """Mirrors `merged_repo_for_round` in dagger_orchestrate.sh:1157."""
    return f"{hf_user}/{merged_short(base_dataset_short, infix, round)}"


def alias_short(prefix: str, infix: str, round: int, model: str, action_format: str) -> str:
    """Alias dataset short: `<int_short>_<model><action_format>00`.
    Mirrors `alias_short_for_round` in dagger_orchestrate.sh:1150.
    """
    return f"{int_short(prefix, infix, round)}_{model}{action_format}00"


def alias_repo(hf_user: str, prefix: str, infix: str, round: int, model: str, action_format: str) -> str:
    """Mirrors `alias_repo_for_round` in dagger_orchestrate.sh:1151."""
    return f"{hf_user}/{alias_short(prefix, infix, round, model, action_format)}"


def int_cache_path(lerobot_cache: str | Path, hf_user: str, prefix: str, infix: str, round: int) -> Path:
    """On-disk cache path for an intervention dataset."""
    return Path(lerobot_cache) / int_repo(hf_user, prefix, infix, round)


def blend_cache_path(
    lerobot_cache: str | Path, hf_user: str, prefix: str, infix: str, round: int, ratio: float
) -> Path:
    """On-disk cache path for a blend dataset."""
    return Path(lerobot_cache) / blend_repo(hf_user, prefix, infix, round, ratio)


def format_blends_tag(blends: list[float]) -> str:
    """Format a list of blend ratios into the BLENDS_TAG used in derived names.

    Sorted descending (so [0.8, 0.9] and [0.9, 0.8] produce the same tag),
    each ratio rendered via `blend_tag_for_ratio`, joined with `_`, prefixed
    by `b`. Empty list → empty string (back-compat: no tag at all).

    Mirrors the inline tag-derivation block in dagger_orchestrate.sh:770-781.

    Examples:
        format_blends_tag([])           == ""
        format_blends_tag([0.9])        == "b090"
        format_blends_tag([0.9, 0.8])   == "b090_080"
        format_blends_tag([0.7, 0.1])   == "b070_010"
    """
    if not blends:
        return ""
    sorted_desc = sorted((float(b) for b in blends), reverse=True)
    parts = [blend_tag_for_ratio(b) for b in sorted_desc]
    return "b" + "_".join(parts)


def derive_base_dataset_short(
    stem: str, run_tag: str, model_tag: str, method_tag: str, blends_tag: str
) -> str:
    """Combine BASE_DATASET_STEM + tags → BASE_DATASET_SHORT.

    Mirrors `_derive_base_dataset_short_for_tags` in dagger_orchestrate.sh:787-796.
    Empty tags are skipped (no leading/trailing underscores added).

    Caller is responsible for passing the right tags; in rerun mode the
    orchestrator calls this twice — once with the current command's tags,
    once with source's tags — to compute both BASE_DATASET_SHORT and
    SOURCE_INT_SHORT_PREFIX.
    """
    out = stem
    if run_tag:
        out = f"{out}_{run_tag}"
    if model_tag:
        out = f"{out}_{model_tag}"
    if method_tag:
        out = f"{out}_{method_tag}"
    if blends_tag:
        out = f"{out}_{blends_tag}"
    return out


def derive_base_policy_name(stem: str, run_tag: str, model_tag: str, method_tag: str, blends_tag: str) -> str:
    """Combine BASE_POLICY_STEM + tags → BASE_POLICY_NAME.

    Mirrors `_derive_base_policy_name_for_tags` in dagger_orchestrate.sh:877-885.

    Note: bash version IGNORES model_tag (the model prefix on the policy
    stem already disambiguates pi05_/diffusion_/act_). We accept the arg for
    a uniform signature with derive_base_dataset_short but never include it.
    """
    del model_tag  # intentionally unused; see docstring
    out = stem
    if run_tag:
        out = f"{out}_{run_tag}"
    if method_tag:
        out = f"{out}_{method_tag}"
    if blends_tag:
        out = f"{out}_{blends_tag}"
    return out


# ────────────────────────────────────────────────────────────────────────────
# Inverse functions: name → config.
# Two distinct parse domains:
#   * dataset short names    e.g. "lever_grip0_d5jvm_diff_r_dag7_blend010"
#   * training dir basenames e.g. "diffusion_..._d5jvm_ft_dag5_oversample3"
# Each gets its own dataclass + parser.
# ────────────────────────────────────────────────────────────────────────────

# Dataset short name pattern: <prefix>_<infix>_dag<N>[_blend<NNN>|_m]?
# Examples this matches:
#   lever_grip0_d5jvm_diff_r_dag7              → intervention
#   lever_grip0_d5jvm_diff_r_dag7_blend010     → blend (pct=10)
#   lever_grip0_d5jvm_diff_r_dag7_m            → merged
#   lever_grip0_rerun_v1_diff_b070_010_r_dag3  → intervention (rerun naming)
# Anything not matching is treated as kind="base".
# Suffix grammar: `_blend<NNN>` may itself be followed by `_nocoll` (collision-
# filtered variant); merged datasets end in `_m`. Anything else is intervention.
_DATASET_SHORT_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<infix>[ra])_dag(?P<round>\d+)"
    r"(?:_(?P<suffix>blend\d{3}(?:_nocoll)?|m))?$"
)

DatasetKind = Literal["base", "intervention", "blend", "merged"]


@dataclass(frozen=True)
class ParsedDatasetName:
    """Result of `parse_dataset_short`. Fields not relevant to a given `kind`
    are None (e.g. blend_pct is None for kind != "blend")."""

    kind: DatasetKind
    name: str  # original input (for round-tripping / logging)
    prefix: str | None  # everything before _<infix>_dag<N>; None for base
    infix: str | None  # "r" or "a"; None for base
    round: int | None  # None for base
    blend_pct: int | None  # only for kind="blend"
    is_nocoll: bool = False  # True for collision-filtered blend variants (kind="blend")


def parse_dataset_short(name: str) -> ParsedDatasetName:
    """Parse a LeRobot dataset short name into its DAgger-naming components.

    Strips any trailing slash (so callers can pass `Path(...).name` from a
    dir path interchangeably). The input must be the dataset *short* name
    (no `<hf_user>/` prefix); callers operating on full repo ids should
    pre-strip the user prefix.

    Anything not matching the `<prefix>_<infix>_dag<N>...` pattern is
    treated as `kind="base"` — i.e. the script assumes the user passed
    either a base dataset OR a name from outside DAgger naming. The PCA
    script's caller is responsible for using --round explicitly in the
    "base" case (no round number is available from the name itself).
    """
    short = name.rstrip("/")
    m = _DATASET_SHORT_RE.match(short)
    if not m:
        return ParsedDatasetName(kind="base", name=short, prefix=None, infix=None, round=None, blend_pct=None)
    suffix = m.group("suffix")
    kind: DatasetKind
    blend_pct: int | None = None
    is_nocoll = False
    if suffix is None:
        kind = "intervention"
    elif suffix == "m":
        kind = "merged"
    elif suffix.startswith("blend"):
        kind = "blend"
        if suffix.endswith("_nocoll"):
            is_nocoll = True
            blend_pct = int(suffix[len("blend") : -len("_nocoll")])
        else:
            blend_pct = int(suffix[len("blend") :])
    else:  # pragma: no cover  (regex alternation excludes anything else)
        raise AssertionError(f"unexpected suffix: {suffix!r}")
    return ParsedDatasetName(
        kind=kind,
        name=short,
        prefix=m.group("prefix"),
        infix=m.group("infix"),
        round=int(m.group("round")),
        blend_pct=blend_pct,
        is_nocoll=is_nocoll,
    )


# Training dir suffix pattern: matches `_ft_dag5`, `_dag10`, or
# `_ft_dag5_<retrain_suffix>` at the end of a training-dir basename.
# Group 1: round number. Group 2 (optional): retrain suffix without leading _.
# Migrated as-is from dagger_plot.py:40 (ROUND_SUFFIX_RE) for byte-identical
# behavior. Kept as a module-level constant because scan_round() (and
# similar callers) consume it directly.
ROUND_SUFFIX_RE = re.compile(r"(?:_ft)?_dag(\d+)(?:_([^/]+))?$")


def _argv_get_flag(argv: list[str], name: str, default: str | None = None) -> str | None:
    """Pull `--<name>=<value>` from an argv list (used to introspect a sidecar's
    `orchestrator_invocation.argv`). Returns the value or `default` if absent."""
    pfx = f"--{name}="
    for a in argv:
        if a.startswith(pfx):
            return a[len(pfx) :]
    return default


def load_sidecar(path: str | Path) -> dict:
    """Read a `dagger/config.json` sidecar produced by dagger_orchestrate.sh."""
    return json.loads(Path(path).read_text())


def find_sidecar_by_prefix(training_root: str | Path, prefix: str) -> Path | None:
    """Scan `<training_root>/*/dagger/config.json` for the first sidecar whose
    `naming.base_dataset_short == prefix` OR `rerun_mode.source_int_short_prefix
    == prefix`. Returns the sidecar Path or None.

    Multiple sidecars per lineage are normal (one per round); they all share
    the same source pointers and the same `naming.base_dataset_short`, so
    returning the first match is sufficient for base-repo resolution.

    In rerun mode, two different reruns can share the same
    `source_int_short_prefix` — that's fine here, because their base_repo
    pointers (via `config.initial_policy_path`) are identical.
    """
    root = Path(training_root)
    if not root.is_dir():
        return None
    for sidecar in sorted(root.glob("*/dagger/config.json")):
        try:
            sc = load_sidecar(sidecar)
        except (OSError, json.JSONDecodeError):
            continue
        naming = sc.get("naming") or {}
        if naming.get("base_dataset_short") == prefix:
            return sidecar
        rerun = sc.get("rerun_mode") or {}
        if rerun.get("source_int_short_prefix") == prefix:
            return sidecar
    return None


def load_initial_policy_train_config(sidecar: dict) -> dict | None:
    """Walk the sidecar's `config.initial_policy_path` to find a loaded
    `train_config.json` dict (or None if not findable).

    Matches the orchestrator's `resolve_latest_checkpoint` candidate order:
        1. checkpoints/last/pretrained_model/train_config.json
        2. checkpoints/<highest-numbered>/pretrained_model/train_config.json
        3. pretrained_model/train_config.json
        4. train_config.json (in base dir)

    Re-anchors relative `initial_policy_path` against the lerobot root
    inferred from the sidecar's absolute `training_output_dir`.
    """
    config = sidecar.get("config") or {}
    initial_policy = config.get("initial_policy_path")
    if not initial_policy:
        return None
    roots: list[Path] = [Path.cwd()]
    train_out = sidecar.get("training_output_dir")
    if train_out:
        for ancestor in Path(train_out).parents:
            if ancestor.name == "outputs":
                roots.insert(0, ancestor.parent)
                break
    ip = Path(initial_policy)
    bases: list[Path] = [ip] if ip.is_absolute() else []
    for root in roots:
        bases.append(root / initial_policy)
    for base in bases:
        checkpoint_dir = base / "checkpoints"
        numbered: list[Path] = []
        if checkpoint_dir.is_dir():
            for entry in checkpoint_dir.iterdir():
                if entry.is_dir() and entry.name.isdigit():
                    numbered.append(entry)
            numbered.sort(key=lambda p: int(p.name), reverse=True)
        candidates: list[Path] = [
            checkpoint_dir / "last" / "pretrained_model" / "train_config.json",
        ]
        for n in numbered:
            candidates.append(n / "pretrained_model" / "train_config.json")
        candidates.append(base / "pretrained_model" / "train_config.json")
        candidates.append(base / "train_config.json")
        for cand in candidates:
            if cand.is_file():
                try:
                    return json.loads(cand.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
    return None


def resolve_base_repo(
    sidecar: dict | None,
    *,
    explicit_override: str | None = None,
    hf_user: str = "JennyWWW",
) -> tuple[str | None, str]:
    """Resolve the base dataset repo id (`<hf_user>/<short>`) for a DAgger
    lineage. Returns `(resolved_repo_id, source_tag)`.

    Resolution order (HIGHEST priority first):
      1. `explicit_override`            → source = "override"
      2. sidecar["naming"]["base_repo"] → source = "sidecar.naming"
         (added in step 6 of the naming-refactor plan; old sidecars lack it)
      3. sidecar["config"]["initial_policy_path"] → read its
         checkpoints/last/pretrained_model/train_config.json,
         pull dataset.repo_id          → source = "train_config"
      4. f"{hf_user}/{argv['--base_short']}" parsed from
         sidecar["orchestrator_invocation"]["argv"]
                                       → source = "argv.base_short"
      5. None                          → source = "unresolved"

    Mirrors the orchestrator's own 3-step fallback in
    dagger_orchestrate.sh:671-706, with `explicit_override` moved to
    HIGHEST priority per the plan.
    """
    if explicit_override:
        return explicit_override, "override"
    if sidecar is None:
        return None, "unresolved"

    naming = sidecar.get("naming") or {}
    sidecar_base = naming.get("base_repo")
    if sidecar_base:
        return sidecar_base, "sidecar.naming"

    tc = load_initial_policy_train_config(sidecar)
    if tc:
        repo = (tc.get("dataset") or {}).get("repo_id")
        if repo:
            return repo, "train_config"

    invocation = sidecar.get("orchestrator_invocation") or {}
    argv = invocation.get("argv") or []
    base_short = _argv_get_flag(argv, "base_short")
    if base_short:
        return f"{hf_user}/{base_short}", "argv.base_short"

    return None, "unresolved"


def enumerate_blend_paths_on_disk(
    lerobot_cache: str | Path,
    hf_user: str,
    prefix: str,
    infix: str,
    round: int,
) -> list[tuple[int, Path]]:
    """Glob `<cache>/<hf_user>/<prefix>_<infix>_dag<round>_blend*` and parse
    each percent value out of the trailing `_blend<NNN>` suffix.

    Returns `[(pct, path), ...]` sorted by pct ascending. Empty list if no
    blends exist on disk for this round."""
    int_dir = int_cache_path(lerobot_cache, hf_user, prefix, infix, round)
    parent = int_dir.parent
    if not parent.is_dir():
        return []
    matches = sorted(parent.glob(f"{int_dir.name}_blend*"))
    out: list[tuple[int, Path]] = []
    pct_re = re.compile(r"_blend(\d{3})$")
    for m in matches:
        mm = pct_re.search(m.name)
        if mm and m.is_dir():
            out.append((int(mm.group(1)), m))
    out.sort(key=lambda t: t[0])
    return out


def lineage_of(dir_name: str, model: str) -> str | None:
    """Map a training-dir basename to its lineage key, or None if not a
    DAgger round dir.

    Migrated as-is from dagger_plot.py:146.

    Examples:
        lineage_of("diffusion_..._d5jvm_ft_dag5", "diffusion")
            == "..._d5jvm"
        lineage_of("diffusion_..._d5jvm", "diffusion")
            == None  (no _dag suffix)
        lineage_of("pi05_..._ft_dag5_oversample3", "pi05")
            == "..."  (retrain-suffix dropped from the lineage key)
    """
    prefix = f"{model}_"
    if not dir_name.startswith(prefix):
        return None
    rest = dir_name[len(prefix) :]
    m = ROUND_SUFFIX_RE.search(rest)
    if not m:
        return None
    return rest[: m.start()]


# ────────────────────────────────────────────────────────────────────────────
# CLI shim — invoked from bash via `python3 dagger_naming.py <subcommand> ...`.
# Each subcommand prints exactly one line (the result) to stdout.
# ────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dagger_naming",
        description="Canonical DAgger artifact naming. See module docstring.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Per-round helpers --------------------------------------------------------
    sp = sub.add_parser("blend_tag", help="ratio → 3-digit zero-padded percent")
    sp.add_argument("--ratio", required=True, type=float)

    sp = sub.add_parser("blend_ratio", help="3-digit percent tag → ratio (inverse of blend_tag)")
    sp.add_argument("--tag", required=True, help="3-digit zero-padded percent, e.g. '090'")

    sp = sub.add_parser("int_short", help="intervention dataset short")
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)

    sp = sub.add_parser("int_repo", help="intervention dataset repo id")
    sp.add_argument("--hf_user", required=True)
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)

    sp = sub.add_parser("blend_short", help="blend dataset short")
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--ratio", required=True, type=float)

    sp = sub.add_parser("blend_repo", help="blend dataset repo id")
    sp.add_argument("--hf_user", required=True)
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--ratio", required=True, type=float)

    sp = sub.add_parser("nocoll_short", help="collision-filtered blend dataset short")
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--ratio", required=True, type=float)

    sp = sub.add_parser("nocoll_repo", help="collision-filtered blend dataset repo id")
    sp.add_argument("--hf_user", required=True)
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--ratio", required=True, type=float)

    sp = sub.add_parser("merged_short", help="merged dataset short")
    sp.add_argument("--base_dataset_short", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)

    sp = sub.add_parser("merged_repo", help="merged dataset repo id")
    sp.add_argument("--hf_user", required=True)
    sp.add_argument("--base_dataset_short", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)

    sp = sub.add_parser("alias_short", help="alias dataset short")
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--model", required=True)
    sp.add_argument("--action_format", required=True)

    sp = sub.add_parser("alias_repo", help="alias dataset repo id")
    sp.add_argument("--hf_user", required=True)
    sp.add_argument("--prefix", required=True)
    sp.add_argument("--infix", required=True)
    sp.add_argument("--round", required=True, type=int)
    sp.add_argument("--model", required=True)
    sp.add_argument("--action_format", required=True)

    # Base-name derivation -----------------------------------------------------
    sp = sub.add_parser("format_blends_tag", help="ratios → BLENDS_TAG")
    sp.add_argument("--blends", required=True, help="space-separated ratios; empty string → empty tag")

    sp = sub.add_parser("derive_base_dataset_short", help="stem + tags → BASE_DATASET_SHORT")
    sp.add_argument("--stem", required=True)
    sp.add_argument("--run_tag", default="")
    sp.add_argument("--model_tag", default="")
    sp.add_argument("--method_tag", default="")
    sp.add_argument("--blends_tag", default="")

    sp = sub.add_parser("derive_base_policy_name", help="stem + tags → BASE_POLICY_NAME")
    sp.add_argument("--stem", required=True)
    sp.add_argument("--run_tag", default="")
    sp.add_argument("--model_tag", default="")  # accepted but ignored, see derive_base_policy_name docstring
    sp.add_argument("--method_tag", default="")
    sp.add_argument("--blends_tag", default="")

    # Inverse helpers ----------------------------------------------------------
    sp = sub.add_parser(
        "parse",
        help="parse a dataset short name → JSON {kind, prefix, infix, round, blend_pct}",
    )
    sp.add_argument("name", help="dataset short name (no <hf_user>/ prefix)")

    sp = sub.add_parser("lineage_of", help="training-dir basename → lineage key (empty if not a dag dir)")
    sp.add_argument("--model", required=True)
    sp.add_argument("dir_name", help="training dir basename (no path components)")

    return p


def _cli_main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv)
    cmd = args.cmd

    if cmd == "blend_tag":
        print(blend_tag_for_ratio(args.ratio))
    elif cmd == "blend_ratio":
        print(ratio_for_blend_tag(args.tag))
    elif cmd == "int_short":
        print(int_short(args.prefix, args.infix, args.round))
    elif cmd == "int_repo":
        print(int_repo(args.hf_user, args.prefix, args.infix, args.round))
    elif cmd == "blend_short":
        print(blend_short(args.prefix, args.infix, args.round, args.ratio))
    elif cmd == "blend_repo":
        print(blend_repo(args.hf_user, args.prefix, args.infix, args.round, args.ratio))
    elif cmd == "nocoll_short":
        print(nocoll_short(args.prefix, args.infix, args.round, args.ratio))
    elif cmd == "nocoll_repo":
        print(nocoll_repo(args.hf_user, args.prefix, args.infix, args.round, args.ratio))
    elif cmd == "merged_short":
        print(merged_short(args.base_dataset_short, args.infix, args.round))
    elif cmd == "merged_repo":
        print(merged_repo(args.hf_user, args.base_dataset_short, args.infix, args.round))
    elif cmd == "alias_short":
        print(alias_short(args.prefix, args.infix, args.round, args.model, args.action_format))
    elif cmd == "alias_repo":
        print(alias_repo(args.hf_user, args.prefix, args.infix, args.round, args.model, args.action_format))
    elif cmd == "format_blends_tag":
        blends_str = args.blends.strip()
        blends = [float(x) for x in blends_str.split()] if blends_str else []
        print(format_blends_tag(blends))
    elif cmd == "derive_base_dataset_short":
        print(
            derive_base_dataset_short(
                args.stem, args.run_tag, args.model_tag, args.method_tag, args.blends_tag
            )
        )
    elif cmd == "derive_base_policy_name":
        print(
            derive_base_policy_name(args.stem, args.run_tag, args.model_tag, args.method_tag, args.blends_tag)
        )
    elif cmd == "parse":
        parsed = parse_dataset_short(args.name)
        print(json.dumps(asdict(parsed)))
    elif cmd == "lineage_of":
        result = lineage_of(args.dir_name, args.model)
        # Bash convention: empty stdout = "no match". Caller checks `[[ -n "$out" ]]`.
        print(result if result is not None else "")
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv[1:]))
