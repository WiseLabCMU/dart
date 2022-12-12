"""Main library."""

# Haiku (0.0.9) modules currently don't work with jaxtyped (0.2.8).
from dart import fields

from jaxtyping import install_import_hook
with install_import_hook("foo", ("beartype", "beartype")):
    from dart.spatial import interpolate
    from dart.sensor import VirtualRadar, RadarPose
    from dart.pose import make_pose
    from dart.sensor_column import TrainingColumn
    from dart.dart import DART, ModelState
    from dart import dataset

__all__ = [
    "DART", "ModelState",
    "VirtualRadar", "RadarPose", "TrainingColumn",
    "make_pose", "interpolate", "dataset", "fields"
]
