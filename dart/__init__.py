r"""DART implementation.

::
     ____________       ______  _____
          ______ \  /\ (_____ \(_____)
        _____   \ \/  \ ____)  ) |
         ___ |__/ / /\ (____  (  |_
      ___________/ /  \_|   |_|\___)
      Doppler Aided Radar Tomography

.
"""

from jaxtyping import install_import_hook
with install_import_hook("dart", ("beartype", "beartype")):
    from dart import dataset
    from dart import types
    from dart import fields
    from dart import utils
    from dart import jaxcolors
    from dart import pose
    from dart import metrics
    from dart.dart import DART
    from dart.sensor import VirtualRadar
    from dart.camera import VirtualCamera, VirtualCameraImage
    from dart.result import DartResult
    from dart.script import script_train

__all__ = [
    "dataset", "types", "fields", "utils", "metrics", "jaxcolors",
    "DART", "VirtualRadar",
    "VirtualCamera", "VirtualCameraImage",
    "DartResult",
    "script_train", "pose"
]
