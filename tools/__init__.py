"""DART scripts."""

from . import evaluate
from . import map
from . import examples
from . import simulate
from . import video
from . import slice
from . import preprocess


commands = {
    "simulate": simulate,
    "evaluate": evaluate,
    "video": video,
    "slice": slice,
    "examples": examples,
    "map": map,
    "preprocess": preprocess
}
