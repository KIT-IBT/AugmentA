from dataclasses import dataclass, field
from typing import Optional

from Atrial_LDRBM.LDRBM.Fiber_RA.ra_fiber_config import RAFiberConfig


@dataclass
class AugmentAConfig:
    """Top-level configuration aggregating all sub-configs."""
    ra: RAFiberConfig = field(default_factory=RAFiberConfig)
    # la: LAFiberConfig = field(default_factory=LAFiberConfig)  # TODO: add later


_instance: Optional[AugmentAConfig] = None


def get_config() -> AugmentAConfig:
    global _instance
    if _instance is None:
        _instance = AugmentAConfig()
    return _instance