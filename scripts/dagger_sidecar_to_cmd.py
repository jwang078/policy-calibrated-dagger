#!/usr/bin/env python3
"""Print the runnable command stored in a dagger sidecar config.json.

The orchestrator writes its own argv into every round's
outputs/training/<...>/dagger/config.json (keys: sweep_invocation,
orchestrator_invocation). This turns that record back into a shell command
you can rerun, share, or wrap in the repeat harness:

    python my_scripts/dagger_sidecar_to_cmd.py <.../dagger/config.json>
    python my_scripts/dagger_sidecar_to_cmd.py <config.json> --repeats=20
    python my_scripts/dagger_sidecar_to_cmd.py <config.json> --which=orchestrator

    # old sidecar (no repeat_invocation) from rep K of a repeat study —
    # undo the rep-K tag/seed rewrites to recover the study command:
    python my_scripts/dagger_sidecar_to_cmd.py <config.json> --repeat_index=7 --repeats=20

    # straight to a file:
    python my_scripts/dagger_sidecar_to_cmd.py <config.json> --repeats=20 \
        > outputs/dagger/my_cmd.sh

Every argv entry is shlex-quoted, so the multi-word --*_extra_args strings
(including the JSON-bearing --round0_extra_args) survive copy-paste intact.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys

REPEAT_WRAPPER = "my_scripts/dagger_orchestrate_repeat.sh"
ORCH_WRAPPER = "my_scripts/dagger_orchestrate.sh"


def undo_rep_rewrites(argv: list[str], k: int) -> list[str]:
    """Undo dagger_orchestrate_repeat.sh's rep-``k`` rewrites on a sweep argv.

    Inverse of build_rep_args (rep k >= 2): --run_tag/--rerun_blends_from lose
    the ``k`` suffix, the appended intervention ``--seed=k-1`` is dropped, and
    blend ``--sample_seed`` shifts back down by k-1. Every inversion VALIDATES
    the expected rep-k signature and exits with a diagnostic on mismatch —
    silently "fixing" an argv that was never rep-rewritten would corrupt it.
    """
    if k == 1:
        return list(argv)  # rep 1 runs the study args verbatim
    suf = str(k)
    off = k - 1
    out = []
    for a in argv:
        if a.startswith("--run_tag="):
            v = a.split("=", 1)[1]
            if not (v.endswith(suf) and len(v) > len(suf)):
                sys.exit(f"error: --repeat_index={k} but --run_tag={v} does not end in '{suf}'")
            out.append(f"--run_tag={v[: -len(suf)]}")
        elif a.startswith("--rerun_blends_from="):
            v = a.split("=", 1)[1]
            tag, sep, rest = v.partition(":")
            if not (tag.endswith(suf) and len(tag) > len(suf)):
                sys.exit(
                    f"error: --repeat_index={k} but --rerun_blends_from tag '{tag}' does not end in '{suf}'"
                )
            out.append(f"--rerun_blends_from={tag[: -len(suf)]}{sep}{rest}")
        elif a.startswith("--intervention_extra_args="):
            v = a.split("=", 1)[1]
            appended = f" --seed={off}"
            if not v.endswith(appended):
                sys.exit(
                    f"error: --repeat_index={k} expects intervention_extra_args to end with "
                    f"'{appended.strip()}' (the harness appends it last); got tail: '...{v[-30:]}'"
                )
            out.append(f"--intervention_extra_args={v[: -len(appended)]}")
        elif a.startswith("--blend_extra_args="):
            v = a.split("=", 1)[1]
            m = re.search(r"--sample_seed[= ](-?\d+)", v)
            if m and int(m.group(1)) >= 0:
                seed = int(m.group(1))
                if seed < off:
                    sys.exit(
                        f"error: --repeat_index={k} needs blend --sample_seed >= {off} to "
                        f"shift back; got {seed}"
                    )
                sep_ch = m.group(0)[len("--sample_seed")]
                v = v.replace(m.group(0), f"--sample_seed{sep_ch}{seed - off}", 1)
            out.append(f"--blend_extra_args={v}")
        else:
            out.append(a)
    return out


def main() -> int:
    """Parse args, load the sidecar, and print the reconstructed command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a .../dagger/config.json sidecar")
    parser.add_argument(
        "--which",
        choices=["auto", "repeat", "sweep", "orchestrator"],
        default="auto",
        help="which recorded invocation to print. auto (default) prefers "
        "repeat_invocation — the PRISTINE study command — when the sidecar has "
        "one; sweep/orchestrator argvs from a repeat study are the rep's "
        "REWRITTEN args (tags suffixed, seeds shifted).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="set/override the repetition count: rewrites --repeats=N in a "
        "repeat_invocation, or wraps a sweep invocation in "
        "dagger_orchestrate_repeat.sh --repeats=N",
    )
    parser.add_argument(
        "--repeat_index",
        type=int,
        default=None,
        metavar="K",
        help="declare that this sidecar is from repetition K of a repeat study "
        "(what newer sidecars record as repeat_invocation.repeat_index — this "
        "flag supplies it for sidecars that predate that recording). The rep-K "
        "rewrites in the sweep argv are undone (the K suffix stripped from "
        "--run_tag/--rerun_blends_from, the appended intervention --seed=K-1 "
        "dropped, blend --sample_seed shifted back down by K-1), yielding the "
        "pristine study command. Combine with --repeats=N for the full repeat "
        "command.",
    )
    parser.add_argument(
        "--oneline",
        action="store_true",
        help="print as a single line instead of one flag per line",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        sidecar = json.load(f)

    which = args.which
    if which == "auto":
        which = "repeat" if (sidecar.get("repeat_invocation") or {}).get("argv") else "sweep"

    if args.repeat_index is not None:
        if args.repeat_index < 1:
            sys.exit("error: --repeat_index must be >= 1")
        if args.which == "auto":
            which = "sweep"
        elif which != "sweep":
            sys.exit("error: --repeat_index operates on the sweep argv; drop --which or use --which=sweep")
        if (sidecar.get("repeat_invocation") or {}).get("argv"):
            sys.exit(
                "error: this sidecar already records repeat_invocation (the pristine study "
                "command, including its repeat_index) — use --which=repeat instead of "
                "--repeat_index"
            )

    key = f"{which}_invocation"
    inv = sidecar.get(key)
    if not inv or not inv.get("argv"):
        present = [
            k
            for k in ("repeat_invocation", "sweep_invocation", "orchestrator_invocation")
            if (sidecar.get(k) or {}).get("argv")
        ]
        hint = ""
        if which == "repeat" and "sweep_invocation" in (present or []):
            run_tag = (sidecar.get("naming") or {}).get("run_tag", "")
            m = re.search(r"^(.*?)(\d+)$", run_tag)
            guess = (
                f" — run_tag '{run_tag}' ends in digits; if this is rep {m.group(2)} of base "
                f"tag '{m.group(1)}', rerun with --repeat_index={m.group(2)}"
                if m
                else ""
            )
            hint = (
                "\nThis sidecar predates repeat_invocation recording. For a rep-K sidecar of "
                "a repeat study, --repeat_index=K reconstructs the pristine study command "
                "from the sweep argv" + guess
            )
        sys.exit(f"error: no {key} in {args.config} (present: {present or 'none'}){hint}")

    if args.repeats is not None and args.repeats < 1:
        sys.exit("error: --repeats must be >= 1")

    argv = list(inv["argv"])
    warnings = []
    if which == "repeat":
        # argv already carries --repeats=N; --repeats overrides it in place.
        if args.repeats is not None:
            argv = [a for a in argv if not a.startswith("--repeats=")]
            argv.insert(0, f"--repeats={args.repeats}")
        cmd = ["bash", inv.get("wrapper") or REPEAT_WRAPPER, *argv]
    else:
        if args.repeat_index is not None:
            argv = undo_rep_rewrites(argv, args.repeat_index)
            warnings.append(
                f"rep-{args.repeat_index} rewrites undone (tags de-suffixed, seeds "
                "shifted back): this is the reconstructed pristine study command."
            )
        rep_idx = (sidecar.get("repeat_invocation") or {}).get("repeat_index")
        if sidecar.get("repeat_invocation"):
            warnings.append(
                f"WARNING: this sidecar is from repetition {rep_idx} of a repeat study — the "
                f"{which} argv below is the rep-REWRITTEN form (tags suffixed, seeds "
                "shifted). Use --which=repeat for the pristine study command."
            )
        elif "repeat_invocation" not in sidecar and args.repeat_index is None:
            warnings.append(
                "NOTE: sidecar predates repeat_invocation recording. If this lineage was a "
                "repetition of a repeat study (run_tag = base tag + rep index), the argv "
                "below carries that rep's rewritten tags/seeds (intervention --seed=k-1 "
                "appended, blend --sample_seed shifted by k-1) — de-rewrite by hand."
            )
        if args.repeats is not None:
            if which != "sweep":
                sys.exit("error: --repeats wraps the SWEEP invocation; drop --which=orchestrator")
            cmd = ["bash", REPEAT_WRAPPER, f"--repeats={args.repeats}", *argv]
        else:
            cmd = ["bash", inv.get("wrapper") or ORCH_WRAPPER, *argv]

    quoted = [shlex.quote(a) for a in cmd]
    if not args.oneline:
        # Comment header so a redirected file is self-describing.
        for w in warnings:
            print(f"# {w}")
        print(f"# {which} invocation from: {args.config}")
        print(f"# round: {sidecar.get('round')}  recorded: {sidecar.get('timestamp')}")
        print("# run from the lerobot repo root")
        head = " ".join(quoted[:2])
        print(head + " \\")
        for q in quoted[2:-1]:
            print(f"  {q} \\")
        print(f"  {quoted[-1]}")
    else:
        for w in warnings:
            print(f"# {w}", file=sys.stderr)
        print(" ".join(quoted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
