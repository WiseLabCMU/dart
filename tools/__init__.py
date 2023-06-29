"""DART scripts."""

from . import evaluate
from . import map
from . import examples
from . import simulate
from . import video
from . import slice
from . import preprocess
from . import ssim
from . import gt_map
from . import compare


commands = {
    "simulate": simulate,
    "evaluate": evaluate,
    "video": video,
    "slice": slice,
    "examples": examples,
    "map": map,
    "preprocess": preprocess,
    "ssim": ssim,
    "gt_map": gt_map,
    "compare": compare,
}
