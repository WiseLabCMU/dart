"""DART scripts."""

from . import evaluate
from . import map
from . import examples
from . import simulate
from . import video


commands = {
    "simulate": simulate,
    "evaluate": evaluate,
    "video": video,
    "examples": examples,
    "map": map
}
