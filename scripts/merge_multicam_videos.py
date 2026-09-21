#!/usr/bin/env python3
"""Merge same-fps multicam videos into a stacked grid, time-aligned.

Layout: each --row is a horizontal strip of videos (scaled to a common
height); rows narrower than the widest row are centered on black; rows are
stacked vertically. Videos that start late (per the chosen alignment) get
black lead-in frames and videos that end early get black tail frames, so
every stream spans the same total frame range.

Alignment modes (--align):
  none      all videos start together at their first frame.
  timecode  use the embedded MP4 timecode track (e.g. GoPro TCD stream).
            WARNING: GoPro Labs QR clock sync is only good to ~1-2 s and has
            model/firmware-dependent latency (a Hero 12 and Hero 9 synced
            from the same QR were measured ~0.76 s apart; two Hero 9s were
            ~0.19 s apart) — treat as a coarse first pass.
  audio     cross-correlate the audio tracks (ground truth to ~a frame when
            all cameras heard the same scene). Requires numpy + scipy.

Example (two on top, one centered on the bottom):
  python3 merge_multicam_videos.py --align audio \
      --row GX019799.MP4 GX010081.MP4 --row GX019804.MP4 -o merged.mp4
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def ffprobe_video(path):
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration:stream_tags=timecode",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    fps = num / den
    if "nb_frames" in s:
        nb = int(s["nb_frames"])
    else:
        nb = round(float(s["duration"]) * fps)
    return {
        "path": path,
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": fps,
        "nb_frames": nb,
        "timecode": s.get("tags", {}).get("timecode"),
    }


def timecode_to_frames(tc, fps):
    """HH:MM:SS:FF (or ';' drop-frame separator) -> absolute frame count.

    Relative offsets between clips shot minutes apart make drop-frame
    compensation negligible, so it is ignored.
    """
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        sys.exit(f"unparsable timecode {tc!r}")
    hh, mm, ss, ff = map(int, parts)
    return (hh * 3600 + mm * 60 + ss) * round(fps) + ff


def timecode_offsets(videos):
    for v in videos:
        if not v["timecode"]:
            sys.exit(f"{v['path']} has no timecode tag; use --align audio/none")
    starts = [timecode_to_frames(v["timecode"], v["fps"]) for v in videos]
    return [s - min(starts) for s in starts]


def audio_offsets(videos, sr):
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import fftconvolve

    def extract(path, wav):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                str(sr),
                "-f",
                "wav",
                wav,
            ],
            check=True,
        )
        _, x = wavfile.read(wav)
        x = x.astype(np.float64)
        return x - x.mean()

    with tempfile.TemporaryDirectory() as td:
        sigs = [extract(v["path"], os.path.join(td, f"{i}.wav")) for i, v in enumerate(videos)]

    ref = sigs[0]
    offsets_sec = [0.0]
    for i, sig in enumerate(sigs[1:], start=1):
        # c[lag] = sum_j sig[j]*ref[j-lag]; peak at L means sig(t) = ref(t-L),
        # i.e. this clip started (-L) seconds after the reference clip.
        corr = fftconvolve(sig, ref[::-1], mode="full")
        lags = np.arange(-len(ref) + 1, len(sig))
        peak = int(np.argmax(corr))
        d = 0.0
        if 0 < peak < len(corr) - 1:
            y0, y1, y2 = corr[peak - 1], corr[peak], corr[peak + 1]
            d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        strength = corr[peak] / (np.linalg.norm(ref) * np.linalg.norm(sigs[i]))
        offsets_sec.append(-(lags[peak] + d) / sr)
        print(
            f"  {videos[i]['path']} vs {videos[0]['path']}: "
            f"started {offsets_sec[-1]:+.4f}s (peak corr {strength:.3f})"
        )
        if strength < 0.1:
            print("  WARNING: weak correlation peak — alignment may be bogus")

    fps = videos[0]["fps"]
    frames = [round(o * fps) for o in offsets_sec]
    return [f - min(frames) for f in frames]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--row",
        action="append",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="videos for one horizontal row (repeatable)",
    )
    ap.add_argument("--align", choices=["none", "timecode", "audio"], default="none")
    ap.add_argument("--height", type=int, default=720, help="per-video scaled height in px (default 720)")
    ap.add_argument("--audio_sr", type=int, default=16000, help="resample rate for audio cross-correlation")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="fast")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    rows = args.row
    paths = [p for row in rows for p in row]
    videos = [ffprobe_video(p) for p in paths]

    fps = videos[0]["fps"]
    for v in videos[1:]:
        if abs(v["fps"] - fps) > 1e-6:
            sys.exit(f"fps mismatch: {v['path']} is {v['fps']:.3f}, {videos[0]['path']} is {fps:.3f}")

    if args.align == "timecode":
        start_pads = timecode_offsets(videos)
    elif args.align == "audio":
        start_pads = audio_offsets(videos, args.audio_sr)
    else:
        start_pads = [0] * len(videos)

    total = max(sp + v["nb_frames"] for sp, v in zip(start_pads, videos))
    for v, sp in zip(videos, start_pads):
        print(f"{v['path']}: lead-in {sp} frames, tail {total - sp - v['nb_frames']} frames")

    H = args.height
    widths = [2 * round(v["width"] * H / v["height"] / 2) for v in videos]
    row_slices, i = [], 0
    for row in rows:
        row_slices.append(list(range(i, i + len(row))))
        i += len(row)
    row_widths = [sum(widths[i] for i in sl) for sl in row_slices]
    max_w = max(row_widths)

    parts = []
    for i, (v, sp) in enumerate(zip(videos, start_pads)):
        ep = total - sp - v["nb_frames"]
        parts.append(
            f"[{i}:v]scale={widths[i]}:{H},setsar=1,"
            f"tpad=start={sp}:stop={ep}:start_mode=add:stop_mode=add:color=black[s{i}]"
        )
    row_labels = []
    for j, sl in enumerate(row_slices):
        label = f"s{sl[0]}"
        if len(sl) > 1:
            parts.append("".join(f"[s{i}]" for i in sl) + f"hstack=inputs={len(sl)}[h{j}]")
            label = f"h{j}"
        if row_widths[j] < max_w:
            parts.append(f"[{label}]pad={max_w}:{H}:(ow-iw)/2:0:black[r{j}]")
            label = f"r{j}"
        row_labels.append(label)
    if len(row_labels) > 1:
        parts.append("".join(f"[{lb}]" for lb in row_labels) + f"vstack=inputs={len(row_labels)}[v]")
        out_label = "v"
    else:
        out_label = row_labels[0]

    cmd = ["ffmpeg", "-y"]
    for v in videos:
        cmd += ["-i", v["path"]]
    cmd += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        f"[{out_label}]",
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-preset",
        args.preset,
        "-an",
        args.output,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {args.output}: {max_w}x{H * len(rows)}, {total} frames ({total / fps:.2f}s @ {fps:.2f}fps)")


if __name__ == "__main__":
    main()
