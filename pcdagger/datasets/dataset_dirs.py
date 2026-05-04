from pathlib import Path

from lerobot.utils.constants import HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE


def resolve_dataset_dir(repo_id: str, explicit_dir: str | None = None) -> Path:
    """Resolve the dataset 'data/' directory on disk.

    Priority:
      1. --dataset_dir flag (used as-is).
      2. Flat layout: $HF_LEROBOT_HOME/{repo_id}/data (created by lerobot-record etc.).
      3. Snapshot cache: $HF_LEROBOT_HUB_CACHE/datasets--{org}--{name}/snapshots/<sha>/data
         (created by lerobot-train via huggingface_hub.snapshot_download). Uses
         refs/main when present, otherwise the most recently modified snapshot.
    """
    if explicit_dir:
        return Path(explicit_dir)

    flat = HF_LEROBOT_HOME / repo_id / "data"
    if flat.exists():
        return flat

    cache_dir = HF_LEROBOT_HUB_CACHE / f"datasets--{repo_id.replace('/', '--')}"
    refs_main = cache_dir / "refs" / "main"
    if refs_main.exists():
        sha = refs_main.read_text().strip()
        candidate = cache_dir / "snapshots" / sha / "data"
        if candidate.exists():
            return candidate

    snapshots_dir = cache_dir / "snapshots"
    if snapshots_dir.exists():
        for snap in sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if (snap / "data").exists():
                return snap / "data"

    raise FileNotFoundError(
        f"Could not find dataset {repo_id!r} on disk. Tried:\n"
        f"  - {flat}\n"
        f"  - {snapshots_dir}/<sha>/data\n"
        "Pass --dataset_dir explicitly, or download the dataset first."
    )


def make_default_rename_map(camera_names: list[str], image_resize_mode: str) -> dict[str, str]:
    """Build the default {f'observation.images.{cam}_{mode}': f'observation.images.{cam}'} map.

    SplatSim datasets store image columns under the resize-mode-suffixed key (e.g.
    ``observation.images.base_rgb_letterbox``). Policies are typically trained against the
    bare name (``observation.images.base_rgb``). This helper produces the rename_map that
    bridges the two.
    """
    return {
        f"observation.images.{cam}_{image_resize_mode}": f"observation.images.{cam}" for cam in camera_names
    }
