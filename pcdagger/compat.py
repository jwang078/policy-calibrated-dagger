"""The one place pcdagger reaches for lerobot internals that are not part of its public surface.

Everything else pcdagger imports from lerobot is a public name (configs, factories, processors,
constants). The handful of private helpers below are re-exported here with fallbacks, so a rename
upstream is fixed in this file instead of in every caller. ``tests/test_dependency_contract.py``
checks the rest of the surface (every lerobot / splatsim name pcdagger imports, plus the fork's
hook signatures) and is the first thing to run after updating either dependency.
"""

from __future__ import annotations

from lerobot.utils.import_utils import is_package_available

try:  # private in lerobot.policies.factory; pcdagger re-attaches the relative/absolute steps of a loaded policy
    from lerobot.policies.factory import _reconnect_relative_absolute_steps as reconnect_relative_absolute_steps
except ImportError as e:  # pragma: no cover - only reached on an incompatible lerobot
    raise ImportError(
        "pcdagger needs lerobot.policies.factory._reconnect_relative_absolute_steps (jwang078/lerobot fork); "
        "see pcdagger/compat.py"
    ) from e


def peft_available() -> bool:
    """Whether `peft` is importable (lerobot keeps this as the private module constant `_peft_available`)."""
    return is_package_available("peft")


__all__ = ["peft_available", "reconnect_relative_absolute_steps"]
