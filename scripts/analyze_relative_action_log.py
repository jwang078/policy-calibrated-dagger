#!/usr/bin/env python3
"""Analyze a CSV produced by LEROBOT_REL_ACTION_DEBUG_LOG.

Looks for state-anchor desync at chunk boundaries: at the moment the cache
is updated, the cached anchor should equal the state the model is seeing.
At every postprocessor call, the same anchor should be in use until the next
cache update.

Usage:
    python my_scripts/analyze_relative_action_log.py path/to/log.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd


def _parse_vec(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).strip()
    if not s or s == "nan":
        return None
    s = s.strip("[]")
    if not s:
        return None
    return np.array([float(x) for x in s.split(";")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to debug CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for col in ["state", "anchor", "rel_action", "abs_action"]:
        df[col] = df[col].apply(_parse_vec)

    pre = df[df["event"] == "preproc"].reset_index(drop=True)
    post = df[df["event"] == "postproc"].reset_index(drop=True)
    print(f"Total rows: {len(df)}  ({len(pre)} preproc, {len(post)} postproc)")

    # 1. At cache-update events, verify state == anchor (sanity check; should be exact).
    cache_updates = pre[pre["did_cache_update"].astype(str).str.lower() == "true"].reset_index(drop=True)
    print(f"\n=== Cache update events: {len(cache_updates)} ===")
    if len(cache_updates) > 0:
        diffs = []
        for _, row in cache_updates.iterrows():
            if row["state"] is not None and row["anchor"] is not None:
                diffs.append(np.abs(row["state"] - row["anchor"]).max())
        print(f"  max |state - anchor| at cache-update: {max(diffs):.2e}  (should be 0)")

    # 2. Mid-chunk preprocessor calls: how does the live state diverge from the cached anchor?
    midchunk = pre[pre["did_cache_update"].astype(str).str.lower() == "false"].reset_index(drop=True)
    print(f"\n=== Mid-chunk preprocessor calls: {len(midchunk)} ===")
    if len(midchunk) > 0:
        diffs = []
        max_diffs = []
        for _, row in midchunk.iterrows():
            if row["state"] is not None and row["anchor"] is not None:
                d = row["state"] - row["anchor"]
                diffs.append(np.abs(d).mean())
                max_diffs.append(np.abs(d).max())
        diffs = np.array(diffs)
        max_diffs = np.array(max_diffs)
        print(f"  mean |state - anchor| (per-call avg over dims): {diffs.mean():.4f} rad")
        print(f"  max  |state - anchor| (per-call worst dim):     {max_diffs.max():.4f} rad")
        print("    → this is how stale the anchor gets within a chunk; tells you the per-chunk drift bound")

    # 3. Per-chunk segment analysis: anchor at chunk start vs state at end of that chunk.
    #    A chunk = run of postproc calls between two cache updates.
    print("\n=== Per-chunk drift (anchor vs state at end of chunk) ===")
    seg_drifts = []
    cur_anchor = None
    last_state_in_chunk = None
    for _, row in pre.iterrows():
        if str(row["did_cache_update"]).lower() == "true":
            if cur_anchor is not None and last_state_in_chunk is not None:
                seg_drifts.append(np.abs(last_state_in_chunk - cur_anchor).max())
            cur_anchor = row["anchor"]
            last_state_in_chunk = row["state"]
        else:
            if row["state"] is not None:
                last_state_in_chunk = row["state"]
    if cur_anchor is not None and last_state_in_chunk is not None:
        seg_drifts.append(np.abs(last_state_in_chunk - cur_anchor).max())
    if seg_drifts:
        sd = np.array(seg_drifts)
        print(f"  chunks observed: {len(sd)}")
        print(f"  end-of-chunk |state - anchor| (worst dim): mean={sd.mean():.4f}  max={sd.max():.4f}  rad")
        print("    → 5cm at 0.5m moment arm ~= 0.1 rad. Compare.")

    # 4. Mid-chunk anchor consistency: every postproc within a chunk must use the SAME anchor.
    #    If the anchor changes mid-chunk, then deltas k=1..n-1 are denormalized against
    #    a stale or refreshed anchor → wrong absolute actions.
    print("\n=== Mid-chunk postproc anchor consistency ===")
    chunk_anchor_inconsistency = []  # max |anchor_k - anchor_0| within each chunk
    chunk_anchor_vs_preproc = []  # max |postproc_anchor - latest_preproc_anchor|
    cur_chunk_anchors = []
    latest_cache_anchor = None
    for _, row in df.iterrows():
        if row["event"] == "preproc":
            if str(row["did_cache_update"]).lower() == "true":
                # New chunk starting — flush previous chunk's anchors
                if cur_chunk_anchors:
                    a0 = cur_chunk_anchors[0]
                    diffs = [np.abs(a - a0).max() for a in cur_chunk_anchors[1:]]
                    if diffs:
                        chunk_anchor_inconsistency.append(max(diffs))
                cur_chunk_anchors = []
                latest_cache_anchor = row["anchor"]
        elif row["event"] == "postproc" and row["anchor"] is not None:
            cur_chunk_anchors.append(row["anchor"])
            if latest_cache_anchor is not None:
                chunk_anchor_vs_preproc.append(np.abs(row["anchor"] - latest_cache_anchor).max())
    # Flush final chunk
    if cur_chunk_anchors:
        a0 = cur_chunk_anchors[0]
        diffs = [np.abs(a - a0).max() for a in cur_chunk_anchors[1:]]
        if diffs:
            chunk_anchor_inconsistency.append(max(diffs))

    if chunk_anchor_inconsistency:
        ai = np.array(chunk_anchor_inconsistency)
        print("  per-chunk: max |anchor_k - anchor_0| within a chunk")
        print(f"    mean: {ai.mean():.2e}  max: {ai.max():.2e}  rad")
        print("    → must be 0; nonzero = anchor changed mid-chunk (bug)")
    if chunk_anchor_vs_preproc:
        av = np.array(chunk_anchor_vs_preproc)
        print("  postproc anchor vs latest cache-update anchor:")
        print(f"    mean: {av.mean():.2e}  max: {av.max():.2e}  rad")
        print("    → must be 0; nonzero = postproc using a different anchor than what was cached")

    # 5. Sanity: how many postproc calls per chunk?
    print("\n=== Chunk size sanity ===")
    chunk_lens = []
    cur = 0
    for _, row in df.iterrows():
        if row["event"] == "preproc" and str(row["did_cache_update"]).lower() == "true":
            if cur > 0:
                chunk_lens.append(cur)
            cur = 0
        elif row["event"] == "postproc":
            cur += 1
    if cur > 0:
        chunk_lens.append(cur)
    if chunk_lens:
        print(
            f"  postproc calls per chunk: min={min(chunk_lens)}  median={int(np.median(chunk_lens))}  max={max(chunk_lens)}"
        )
        print("    → should match your --policy.n_action_steps")


if __name__ == "__main__":
    sys.exit(main())
