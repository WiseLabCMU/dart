"""DART scripts."""

from . import evaluate
from . import map
from . import simulate
from . import video
from . import slice
from . import metrics, compare, dataset
from . import psnr


commands = {
    "simulate": simulate,
    "evaluate": evaluate,
    "video": video,
    "slice": slice,
    "map": map,
    "metrics": metrics,
    "compare": compare,
    "dataset": dataset,
    "psnr": psnr
}
