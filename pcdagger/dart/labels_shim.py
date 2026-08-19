"""Compatibility shim — the DART label core moved into lerobot proper.

The implementation lives in :mod:`lerobot.datasets.dart_relabel` so the
train-time wrapper can be used by the dataset factory
(``--dataset.dart_relabel=true``) without importing my_scripts. This module
re-exports the full API for the recorder (augment_dataset_with_blending) and
the visualizers, which historically import from here.
"""

from lerobot.datasets.dart_relabel import (  # noqa: F401
    DartChunkDataset,
    DemoGeometry,
    _interp_rows,
    chunk_labels,
    demo_geometry,
    load_source_geometries,
    maybe_wrap_dart,
    per_frame_labels,
    project_states,
)
