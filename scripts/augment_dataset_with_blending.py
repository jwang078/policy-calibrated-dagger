#!/usr/bin/env python
"""CLI for pcdagger.blend.augment (dataset augmentation via closed-loop blended-policy rollouts).
Kept as a script because the DAgger orchestrator and the tables kit invoke it by path; see the module for the flags."""

from pcdagger.blend.augment import main

if __name__ == "__main__":
    main()
