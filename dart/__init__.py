"""Main library."""

# from jaxtyping import install_import_hook
# with install_import_hook("dart", ("beartype", "beartype")):
from dart.sensor import VirtualRadar
from dart.dart import DART
from dart import dataset
from dart import types
from dart.script import script_train
from dart.camera import VirtualCamera, VirtualCameraImage
from dart.result import DartResult

# Haiku (0.0.9) modules currently don't work with jaxtyped (0.2.8).
from dart import fields

__all__ = [
    "DART", "VirtualRadar",
    "dataset", "types", "fields", "script_train",
    "VirtualCamera", "VirtualCameraImage", "DartResult"
]
