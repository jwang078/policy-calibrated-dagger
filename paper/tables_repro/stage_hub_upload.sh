#!/bin/bash
# Stage the simulator data for the Hugging Face Hub: the two SplatSim scene tarballs the SplatSim README
# tells people to download (today from Google Drive). Nothing is uploaded; the tree under $HUB_STAGE is
# hard links / copies to review, and the upload command is printed at the end.
#
#   bash stage_hub_upload.sh
#
# The tarballs are the ones the README documents: each unpacks to data/stages/<scene>/ and carries only
# the scan bytes (splat/ and sfm/); the scene's stage.yaml and segmentation labels are tracked in git.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../pcdagger/paths.sh"
HUB_STAGE=${HUB_STAGE:-$HOME/paper_archive/hub_upload}
SCENES_DIR=${SCENES_DIR:-$HOME/data}          # where the tarballs were built
rm -rf "$HUB_STAGE"; mkdir -p "$HUB_STAGE/splatsim-scenes"
for t in robot_iphone_w_engine_curtain vine_scene; do
    src="$SCENES_DIR/$t.tar.gz"; [ -f "$src" ] || { echo "missing $src"; exit 1; }
    ln "$src" "$HUB_STAGE/splatsim-scenes/$t.tar.gz" 2>/dev/null || cp "$src" "$HUB_STAGE/splatsim-scenes/$t.tar.gz"
    # sanity: the tarball unpacks to its own folder and holds the splat the stage.yaml points at
    listing=$(tar tzf "$src")   # (not a pipe into grep -q: SIGPIPE + pipefail would fail a good tarball)
    grep -q "^$t/splat/point_cloud/" <<<"$listing" || { echo "$t.tar.gz has no splat/point_cloud/"; exit 1; }
done
cat > "$HUB_STAGE/splatsim-scenes/README.md" <<'CARD'
---
license: mit
pretty_name: SplatSim scenes
---
# SplatSim scenes

The two example scenes of [SplatSim](https://github.com/jwang078/SplatSim), one tarball each, named after the
`data/stages/<scene>/` folder it unpacks to. Each carries the scan bytes only (`splat/`, the Gaussian-splatting
output, and `sfm/`, the COLMAP reconstruction); the scene's `stage.yaml` and segmentation labels are in the
SplatSim repo.

- `robot_iphone_w_engine_curtain.tar.gz` (390 MB) — the UR5 with its wrist camera, scanned in the engine scene.
  The lever task of *Policy-Calibrated DAgger* (ICRA 2027) renders from this scan.
- `vine_scene.tar.gz` (1.0 GB) — the grape-vine prop scanned in the highbay.

```bash
hf download JennyWWW/splatsim-scenes --repo-type dataset --local-dir /tmp/splatsim-scenes
tar xzf /tmp/splatsim-scenes/robot_iphone_w_engine_curtain.tar.gz -C data/stages
tar xzf /tmp/splatsim-scenes/vine_scene.tar.gz -C data/stages
```
CARD
echo "== staged under $HUB_STAGE:"; ls -la "$HUB_STAGE/splatsim-scenes"
echo
echo "== upload command (after review):"
echo "  hf upload --repo-type dataset JennyWWW/splatsim-scenes $HUB_STAGE/splatsim-scenes ."
