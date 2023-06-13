"""Main library."""

from jaxtyping import install_import_hook
with install_import_hook("dart", ("beartype", "beartype")):
    from dart import dataset
    from dart import types
    from dart import fields
    from dart.dart import DART
    from dart.sensor import VirtualRadar
    from dart.camera import VirtualCamera, VirtualCameraImage
    from dart.result import DartResult
    from dart.script import script_train

__all__ = [
    "dataset", "types", "fields",
    "DART", "VirtualRadar",
    "VirtualCamera", "VirtualCameraImage",
    "DartResult",
    "script_train"
]
