#!/usr/bin/env python
"""Plot + measure RECORDED blend datasets against their source DAgger lineage.

Two outputs from one invocation:

1. **Per-episode plots** — same style as ``visualize_shared_autonomy_sim.py``
   (joint-angle grid + interactive 3D EE trajectory, one plasma-colored line
   per blend ratio, source demo in green) but sourced from datasets already
   produced by ``augment_dataset_with_blending.py`` instead of live rollouts.
   No sim server, no policy inference; ``--policy_path`` is only used for
   pybullet FK (EE plot) and is optional.

2. **Dataset-wide speed table** — base vs intervention vs every blend sibling:

   * ``|dq|/tick``  — mean per-tick commanded joint delta over the arm joints
                      (the crawl metric; demos ≈ 0.007 rad/tick on planar).
   * ``p95``        — 95th percentile of the same (burstiness check).
   * ``togoal%``    — fraction of episodes whose FINAL env_state end-effector
                      is within ``--goal_thresh`` of the goal (planar env_state
                      layout by default: ee dims 6,7 / goal dims 0,1).
   * ``tointerv%``  — blends/nocolls only: fraction of episodes whose final EE
                      is within ``--goal_thresh`` of where their SOURCE
                      intervention episode ended (did the replay at least get
                      to where the demo got?). Base/intervention rows show —.

   Trailing hold frames (exact zero-delta tails from success holds /
   min-length padding) are trimmed first so padded episodes don't dilute the
   speeds. This is the acceptance test for blend-code changes — the numbers
   behind the "blend rollouts crawl and truncate mid-task" pathology
   documented in CLAUDE.md's DAgger-blended-data-source section.

Given ANY dataset of a DAgger round — a blend (``..._dag1_blend010``), a
``_nocoll`` variant, or the raw intervention (``..._dag6``) — the script:

* parses the name via ``dagger_naming.parse_dataset_short``;
* resolves the round's raw intervention dataset (the "guidance"/demo),
  auto-discovers every sibling ``_blend<NNN>`` on disk, and resolves the base
  dataset from the lineage's dagger sidecar;
* for the plots, maps ``--episode_index`` (an episode of the GIVEN dataset) to
  the shared ``source_episode_idx`` and picks each sibling's episode for that
  same source episode.

Examples:
    # Plots for source episode 0 + the round-1 speed table
    python my_scripts/visualize_blend_dataset.py \\
        --dataset_repo_id JennyWWW/planar_8_03dag_smooth_diff_r_dag1_blend010 \\
        --episode_index 0 \\
        --policy_path outputs/training/diffusion_planar_3joint_8_delta_stateng_03dag_smooth_ft_dag9/checkpoints/165000/pretrained_model \\
        --no_show

    # Speed table only (no plots), across every round on disk
    python my_scripts/visualize_blend_dataset.py \\
        --dataset_repo_id JennyWWW/planar_8_03dag_smooth_diff_r_dag1 \\
        --speed_only --all_rounds

    # Collision-filtered variants instead of the raw blends
    python my_scripts/visualize_blend_dataset.py \\
        --dataset_repo_id JennyWWW/planar_8_03dag_smooth_diff_r_dag1_blend030 \\
        --episode_index 0 --variant nocoll --no_show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before any pyplot import (see sibling scripts)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
for p in (str(_HERE), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dagger_naming import (  # type: ignore[import-not-found]  # noqa: E402
    enumerate_blend_paths_on_disk,
    find_sidecar_by_prefix,
    int_cache_path,
    int_short,
    load_sidecar,
    nocoll_short,
    parse_dataset_short,
    resolve_base_repo,
)
from lib_dataset_episode_io import (  # type: ignore[import-not-found]  # noqa: E402
    find_parquet_files,
    load_episodes_meta,
)
from lib_sa_plotting import (  # type: ignore[import-not-found]  # noqa: E402
    plot_ee_trajectories_3d,
    plot_joint_angles,
)

from lerobot.utils.lerobot_dataset_utils import resolve_dataset_dir  # noqa: E402

# ── episode loading ───────────────────────────────────────────────────────────


def load_full_episode(data_dir: Path, episode_index: int) -> pd.DataFrame:
    """All rows of one episode (actions + states only — no video decode),
    sorted by frame_index. Thin variant of ``lib_dataset_episode_io.
    load_episode_frames`` without the fixed-window requirement.
    """
    dfs = [
        df
        for f in find_parquet_files(data_dir)
        if len(df := pd.read_parquet(f, filters=[("episode_index", "==", episode_index)])) > 0
    ]
    if not dfs:
        raise ValueError(f"Episode {episode_index} not found in {data_dir}")
    return pd.concat(dfs, ignore_index=True).sort_values("frame_index").reset_index(drop=True)


def episode_actions(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.asarray(a, dtype=np.float32) for a in df["action"]])


def find_episode_for_source(meta: pd.DataFrame, source_episode_idx: int) -> int | None:
    """Episode index in a blend dataset whose provenance row points at
    ``source_episode_idx`` (written per-episode by the blend script).
    """
    if meta.empty or "source_episode_idx" not in meta.columns:
        return None
    rows = meta.loc[meta["source_episode_idx"] == source_episode_idx, "episode_index"]
    return int(rows.iloc[0]) if len(rows) else None


def mean_tick_delta(actions: np.ndarray, num_dofs: int) -> float:
    """Mean |Δq| per tick over the arm joints — the crawl-pathology metric."""
    if actions.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(actions[:, :num_dofs], axis=0)).mean())


# ── dataset-wide speed metrics ────────────────────────────────────────────────


def trim_hold_tail(arr: np.ndarray) -> np.ndarray:
    """Drop the trailing run of frames identical to the last one (success-hold
    / min-length padding writes exact copies, so deltas there are exactly 0).
    """
    n = arr.shape[0]
    while n > 1 and np.array_equal(arr[n - 1], arr[n - 2]):
        n -= 1
    return arr[:n]


def measure_dataset(
    data_dir: Path,
    num_dofs: int | None,
    ee_dims: tuple[int, int],
    goal_dims: tuple[int, int],
    goal_thresh: float,
    interv_final_ee: dict[int, np.ndarray] | None = None,
    source_map: dict[int, int] | None = None,
) -> dict:
    """Speed + goal-reach stats for one dataset (parquet-only, no video decode).

    ``interv_final_ee`` (source_episode_idx → final EE of the intervention
    episode) together with ``source_map`` (this dataset's episode_index →
    source_episode_idx) enables the ``tointerv`` metric: fraction of episodes
    whose final EE lands within ``goal_thresh`` of where their SOURCE
    intervention episode ended.
    """
    cols = ["episode_index", "frame_index", "action"]
    has_env = None
    frames_list = []
    for f in find_parquet_files(data_dir):
        if has_env is None:
            import pyarrow.parquet as pq

            has_env = "observation.environment_state" in pq.read_schema(f).names
            if has_env:
                cols.append("observation.environment_state")
        frames_list.append(pd.read_parquet(f, columns=cols))
    df = pd.concat(frames_list, ignore_index=True)

    deltas, togoal, tointerv, n_frames = [], [], [], 0
    final_ee_by_ep: dict[int, np.ndarray] = {}
    for ep, g in df.groupby("episode_index"):
        g = g.sort_values("frame_index")
        actions = trim_hold_tail(np.stack([np.asarray(a, dtype=np.float64) for a in g["action"]]))
        nd = num_dofs if num_dofs is not None else actions.shape[1] - 1
        n_frames += actions.shape[0]
        if actions.shape[0] >= 2:
            deltas.append(np.abs(np.diff(actions[:, :nd], axis=0)))
        if has_env:
            es = np.asarray(g["observation.environment_state"].iloc[-1], dtype=np.float64)
            if max(*ee_dims, *goal_dims) < es.shape[0]:
                ee, goal = es[list(ee_dims)], es[list(goal_dims)]
                final_ee_by_ep[int(ep)] = ee
                togoal.append(float(np.linalg.norm(ee - goal)) <= goal_thresh)
                if interv_final_ee is not None and source_map is not None:
                    src_ee = interv_final_ee.get(source_map.get(int(ep), -1))
                    if src_ee is not None:
                        tointerv.append(float(np.linalg.norm(ee - src_ee)) <= goal_thresh)
    d = np.concatenate(deltas) if deltas else np.zeros((0, 1))
    return {
        "eps": int(df["episode_index"].nunique()),
        "frames": n_frames,
        "dq": float(d.mean()) if d.size else float("nan"),
        "p95": float(np.percentile(d, 95)) if d.size else float("nan"),
        "togoal": (100.0 * np.mean(togoal)) if togoal else None,
        "tointerv": (100.0 * np.mean(tointerv)) if tointerv else None,
        "final_ee_by_ep": final_ee_by_ep,
    }


def load_source_map(data_dir: Path) -> dict[int, int] | None:
    """episode_index → source_episode_idx from a blend dataset's episode meta."""
    meta = load_episodes_meta(data_dir)
    if meta.empty or "source_episode_idx" not in meta.columns:
        return None
    return {
        int(m["episode_index"]): int(m["source_episode_idx"])
        for _, m in meta.iterrows()
        if pd.notna(m.get("source_episode_idx"))
    }


def _print_speed_row(label: str, m: dict) -> None:
    def pct(v):
        return f"{v:6.1f}%" if v is not None else "     — "

    print(
        f"  {label:<68} {m['eps']:>5} {m['frames']:>8} {m['dq']:>10.5f} {m['p95']:>10.5f} "
        f"{pct(m['togoal']):>8} {pct(m['tointerv']):>10}"
    )


def print_speed_table(args: argparse.Namespace, parsed, hf_user: str, lerobot_cache: Path) -> None:
    """Base vs intervention vs blend speed/coverage table (one block per round)."""
    prefix, infix = parsed.prefix, parsed.infix
    if args.rounds:
        rounds = sorted(args.rounds)
    elif args.all_rounds:
        rounds = sorted(
            p.round
            for d in (lerobot_cache / hf_user).glob(f"{prefix}_{infix}_dag*")
            if (p := parse_dataset_short(d.name)).kind == "intervention" and p.round is not None
        )
    else:
        rounds = [parsed.round]

    # Base dataset via the lineage's sidecar (same resolution the PCA script uses).
    sidecar_path = find_sidecar_by_prefix(args.training_root, prefix)
    sidecar = load_sidecar(sidecar_path) if sidecar_path else None
    base_repo, base_src = resolve_base_repo(sidecar, explicit_override=args.base_repo_id, hf_user=hf_user)

    measure = lambda data_dir, **kw: measure_dataset(  # noqa: E731
        data_dir,
        args.num_dofs,
        tuple(args.env_state_ee_dims),
        tuple(args.env_state_goal_dims),
        args.goal_thresh,
        **kw,
    )

    width = 127
    print(
        f"\n{'dataset':<70} {'eps':>5} {'frames':>8} {'|dq|/tick':>10} {'p95':>10} {'togoal%':>8} "
        f"{'tointerv%':>10}"
        f"\n{'-' * width}"
    )
    if base_repo:
        _print_speed_row(
            f"{base_repo.rpartition('/')[2]} [base, via {base_src}]",
            measure(resolve_dataset_dir(base_repo, args.dataset_dir)),
        )
    else:
        print("  (base dataset unresolved — pass --base_repo_id for the reference row)")

    for r in rounds:
        int_dir = int_cache_path(lerobot_cache, hf_user, prefix, infix, r)
        print(f"{'-' * width}")
        if (int_dir / "data").is_dir():
            m_int = measure(int_dir / "data")
            _print_speed_row(f"{int_dir.name} [intervention]", m_int)
            interv_final_ee = m_int["final_ee_by_ep"]
        else:
            print(f"  {int_dir.name}: not on disk")
            continue

        def measure_blend(ds_root: Path, **kw):
            # tointerv%: compare each blend episode's final EE against its
            # SOURCE intervention episode's final EE (provenance metadata).
            data_dir = ds_root / "data"
            return measure(
                data_dir, interv_final_ee=interv_final_ee, source_map=load_source_map(data_dir), **kw
            )

        for pct, blend_path in enumerate_blend_paths_on_disk(lerobot_cache, hf_user, prefix, infix, r):
            if args.variant in ("raw", "both"):
                _print_speed_row(f"{blend_path.name} [blend {pct / 100:.2f}]", measure_blend(blend_path))
            if args.variant in ("nocoll", "both"):
                nc = blend_path.parent / nocoll_short(prefix, infix, r, pct / 100.0)
                if (nc / "data").is_dir():
                    _print_speed_row(f"{nc.name} [nocoll {pct / 100:.2f}]", measure_blend(nc))
    print()


# ── FK (optional, needs --policy_path) ────────────────────────────────────────


def try_load_fk_wrapper(policy_path: str | None, robot_name: str | None, num_dofs: int, device: str):
    """Load the SA wrapper purely for its pybullet FK client (EE plots).

    Reuses ``lib_sa_policy_loading.load_wrapped_policy`` — heavier than a bare
    URDF load, but guarantees FK identical to every other SA script. Returns
    None (→ joint-angle plot only) when no policy_path is given.
    """
    if policy_path is None:
        return None
    from lib_sa_policy_loading import load_wrapped_policy  # type: ignore[import-not-found]

    train_cfg_path = Path(policy_path) / "train_config.json"
    env_json: dict = {}
    if train_cfg_path.is_file():
        try:
            env_json = json.loads(train_cfg_path.read_text()).get("env") or {}
        except (json.JSONDecodeError, OSError):
            pass
    robot_name = robot_name or env_json.get("robot_name") or "robot_iphone_w_engine_curtain"
    print(f"Loading policy for FK only (robot_name={robot_name}) …")
    wrapper, _ = load_wrapped_policy(
        policy_path=policy_path, robot_name=robot_name, num_dofs=num_dofs, device=device
    )
    return wrapper


# ── per-episode plots ─────────────────────────────────────────────────────────


def run_episode_plots(args: argparse.Namespace, parsed, hf_user: str, given_data_dir: Path) -> None:
    """The visualize part: one figure pair for one source episode."""
    round_ = parsed.round
    intervention_short = int_short(parsed.prefix, parsed.infix, round_)
    int_data_dir = resolve_dataset_dir(f"{hf_user}/{intervention_short}", args.dataset_dir)
    lerobot_cache = given_data_dir.parent.parent.parent

    # ── source-episode resolution ─────────────────────────────────────────────
    if parsed.kind == "blend":
        meta = load_episodes_meta(given_data_dir)
        row = meta.loc[meta["episode_index"] == args.episode_index] if not meta.empty else pd.DataFrame()
        if row.empty or "source_episode_idx" not in row.columns:
            raise SystemExit(
                f"Episode {args.episode_index} of {parsed.name} has no source_episode_idx metadata."
            )
        source_ep = int(row["source_episode_idx"].iloc[0])
        print(f"Episode {args.episode_index} of {parsed.name} → source intervention episode {source_ep}")
    else:
        source_ep = args.episode_index

    # ── guidance = the raw intervention episode ───────────────────────────────
    int_df = load_full_episode(int_data_dir, source_ep)
    guidance_actions = episode_actions(int_df)
    action_dim = guidance_actions.shape[1]
    num_dofs = args.num_dofs if args.num_dofs is not None else max(1, action_dim - 1)

    # ── discover sibling blend datasets and pull the matching episode ─────────
    blends = enumerate_blend_paths_on_disk(lerobot_cache, hf_user, parsed.prefix, parsed.infix, round_)
    if args.ratios is not None:
        wanted = {int(round(r * 100)) for r in args.ratios}
        blends = [(pct, p) for pct, p in blends if pct in wanted]
    if not blends:
        raise SystemExit(f"No _blend* siblings of {intervention_short} found on disk (or all filtered out).")

    series_by_ratio: dict[float, np.ndarray] = {}
    diags: list[str] = []
    for pct, blend_path in blends:
        short = blend_path.name
        if args.variant == "nocoll":
            nc_short = nocoll_short(parsed.prefix, parsed.infix, round_, pct / 100.0)
            nc_path = blend_path.parent / nc_short
            if not nc_path.is_dir():
                print(f"  [skip] {nc_short}: no _nocoll sibling on disk")
                continue
            short, blend_path = nc_short, nc_path
        data_dir = blend_path / "data" if (blend_path / "data").is_dir() else blend_path
        ep = find_episode_for_source(load_episodes_meta(data_dir), source_ep)
        if ep is None:
            print(f"  [skip] {short}: no episode for source_episode_idx={source_ep} (filtered/not recorded)")
            continue
        actions = episode_actions(load_full_episode(data_dir, ep))
        series_by_ratio[pct / 100.0] = actions
        diags.append(
            f"  ratio={pct / 100.0:.2f} ({short} ep{ep}): {actions.shape[0]} frames, "
            f"mean |dq|/tick={mean_tick_delta(actions, num_dofs):.5f}"
        )

    if not series_by_ratio:
        raise SystemExit(f"No sibling episode found for source episode {source_ep} in any blend dataset.")

    print(
        f"Guidance ({intervention_short} ep{source_ep}): {guidance_actions.shape[0]} frames, "
        f"mean |dq|/tick={mean_tick_delta(guidance_actions, num_dofs):.5f}"
    )
    print("Recorded blend series:")
    for line in diags:
        print(line)

    # ── output naming (mirrors the sim visualizer's style) ────────────────────
    if args.output_dir is None:
        variant_tag = "_nocoll" if args.variant == "nocoll" else ""
        output_dir = Path("outputs/viz") / f"blend_dataset_{intervention_short}_srcep{source_ep}{variant_tag}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    np.savez(
        output_dir / "blend_dataset_data.npz",
        guidance_actions_raw=guidance_actions,
        **{f"ratio_{r:.2f}": a for r, a in series_by_ratio.items()},
    )

    joint_names = [f"joint_{i + 1}" for i in range(min(num_dofs, action_dim))]
    if action_dim > num_dofs:
        joint_names.append("gripper")

    print("Plotting joint angles …")
    plot_joint_angles(
        action_chunks_by_ratio=series_by_ratio,
        joint_names=joint_names,
        episode_index=source_ep,
        frame_index=0,
        obs_states_raw=None,
        guidance_actions_raw=guidance_actions,
        output_path=output_dir / "joint_angles.png",
        no_show=args.no_show,
    )

    # ── EE plot (only with FK) ────────────────────────────────────────────────
    wrapper = try_load_fk_wrapper(args.policy_path, args.robot_name, num_dofs, args.device)
    if wrapper is None:
        print("No --policy_path given — skipping the FK-based EE trajectory plot.")
        return
    from lib_ee_kinematics import compute_ee_from_states  # type: ignore[import-not-found]

    print("Computing EE trajectories via pybullet FK …")
    guidance_ee = compute_ee_from_states(wrapper, guidance_actions)
    ee_by_ratio = {r: compute_ee_from_states(wrapper, a) for r, a in series_by_ratio.items()}
    for r in sorted(ee_by_ratio):
        gap = float(np.linalg.norm(ee_by_ratio[r][-1] - guidance_ee[-1]))
        print(f"  ratio={r:.2f}: final EE is {gap:.4f} m from the demo's endpoint")

    plot_ee_trajectories_3d(
        ee_trajectories_by_ratio=ee_by_ratio,
        episode_index=source_ep,
        frame_index=0,
        guidance_ee_positions=guidance_ee,
        output_path=output_dir / "ee_trajectory.html",
        no_show=args.no_show,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset_repo_id",
        required=True,
        help="Any dataset of the round: blend, _nocoll blend, or raw intervention "
        "(e.g. JennyWWW/planar_8_03dag_smooth_diff_r_dag1_blend010).",
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        default=0,
        help="Episode index IN THE GIVEN DATASET (mapped to the shared source episode via metadata).",
    )
    parser.add_argument(
        "--policy_path",
        default=None,
        help="Checkpoint dir — used ONLY for pybullet FK (EE trajectory plot). Omit for joint plot only.",
    )
    parser.add_argument(
        "--variant",
        choices=["raw", "nocoll", "both"],
        default="both",
        help="Which blend siblings to use. Plots: 'nocoll' switches to the filtered datasets, anything "
        "else plots the raw blends. Speed table: 'both' (default) lists raw + nocoll rows.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        default=None,
        help="Restrict the PLOTS to these blend ratios (e.g. 0.1 0.3). Default: every sibling on disk.",
    )
    parser.add_argument("--robot_name", default=None, help="FK robot override (default: from checkpoint).")
    parser.add_argument(
        "--num_dofs",
        type=int,
        default=None,
        help="Arm DOF count (default: action_dim - 1, i.e. all-but-gripper).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset_dir", default=None, help="Dataset cache root override.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    # ── speed-table options ───────────────────────────────────────────────────
    parser.add_argument(
        "--speed_only", action="store_true", help="Skip the per-episode plots; print only the speed table."
    )
    parser.add_argument("--no_speed_table", action="store_true", help="Skip the speed table; plots only.")
    parser.add_argument(
        "--all_rounds", action="store_true", help="Speed table for every round found on disk."
    )
    parser.add_argument(
        "--rounds", type=int, nargs="*", default=None, help="Explicit speed-table round list."
    )
    parser.add_argument(
        "--base_repo_id", default=None, help="Base demo dataset (default: resolved from the lineage sidecar)."
    )
    parser.add_argument(
        "--training_root", default="outputs/training", help="Where to look for dagger sidecars."
    )
    parser.add_argument(
        "--goal_thresh", type=float, default=0.05, help="togoal%%/tointerv%% distance threshold (m)."
    )
    parser.add_argument(
        "--env_state_ee_dims",
        type=int,
        nargs=2,
        default=(6, 7),
        help="env_state dims of the EE (planar default).",
    )
    parser.add_argument(
        "--env_state_goal_dims",
        type=int,
        nargs=2,
        default=(0, 1),
        help="env_state dims of the goal/block (planar default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed_only and args.no_speed_table:
        raise SystemExit("--speed_only and --no_speed_table are mutually exclusive.")

    hf_user, _, given_short = args.dataset_repo_id.rpartition("/")
    if not hf_user:
        raise SystemExit(f"--dataset_repo_id must be '<hf_user>/<short>', got {args.dataset_repo_id!r}")
    parsed = parse_dataset_short(given_short)
    if parsed.kind not in ("blend", "intervention"):
        raise SystemExit(
            f"{given_short!r} parses as kind={parsed.kind!r}; pass a blend or raw intervention dataset "
            f"of a DAgger round (…_<r|a>_dag<N>[_blend<NNN>[_nocoll]])."
        )
    assert parsed.prefix is not None and parsed.infix is not None and parsed.round is not None
    print(
        f"Parsed {given_short!r}: kind={parsed.kind}, round={parsed.round}, "
        f"intervention={int_short(parsed.prefix, parsed.infix, parsed.round)}"
    )
    given_data_dir = resolve_dataset_dir(args.dataset_repo_id, args.dataset_dir)
    lerobot_cache = given_data_dir.parent.parent.parent  # <cache>/<user>/<short>/data

    if not args.speed_only:
        run_episode_plots(args, parsed, hf_user, given_data_dir)
    if not args.no_speed_table:
        print_speed_table(args, parsed, hf_user, lerobot_cache)
    print("Done.")


if __name__ == "__main__":
    main()
