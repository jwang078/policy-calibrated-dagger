"""Policy-Calibrated DAgger.

Layout (see MIGRATION_PLAN.md for the move from the lerobot fork):

    pcdagger.dart      calibrated noise injection: DART relabelling, sigma / W measurement, schedules
    pcdagger.blend     partial denoising as a learned interpolator: the shared-autonomy wrapper,
                       guidance sources, blend rollouts and dataset augmentation
    pcdagger.dagger    the interactive loop: intervention controller, recording, lineage naming
    pcdagger.datasets  multi-source weighted datasets and relative-action stats
    pcdagger.lerobot_glue  the only module that touches lerobot's factories and config classes
    pcdagger.viz       plotting helpers shared by the scripts and the paper figures
    pcdagger.paths     where the outputs, datasets and sibling checkouts live (env-var overridable)

`pcdagger` depends on `lerobot` and, optionally, `splatsim`; nothing depends on `pcdagger`.
"""

__version__ = "0.1.0"
