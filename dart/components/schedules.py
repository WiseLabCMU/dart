"""Hyperparameter schedules.

Hyperparameter schedules should create a callable which takes the epoch and
step indices as arguments, and outputs a hyperparameter dict.
"""

from dart.types import HyperparameterSchedule


def constant(value: float = 0.0) -> HyperparameterSchedule:
    """Constant schedule."""
    def schedule_func(epoch, step):
        return value

    return schedule_func


def linear(
    start: float = 0.0, end: float = 1.0, steps: int = 1000, warmup: int = 0
) -> HyperparameterSchedule:
    """Schedule by optimization step using a linear schedule."""
    def schedule_func(epoch, step):
        if step == -1:
            return end
        elif step < warmup:
            return start
        else:
            return start + min(step - warmup, steps) / steps * (end - start)

    return schedule_func


def linear_piecewise(
    values: list[float] = [0, 1], steps: list[int] = [0]
) -> HyperparameterSchedule:
    """Piecewise linear function."""
    def schedule_func(epoch, step):
        if step == -1:
            return values[-1]
        prev = values[0]
        for duration, val in zip(steps, values[1:]):
            if step < duration:
                return (step / duration) * (val - prev) + prev
            else:
                step -= duration
                prev = val
        else:
            return values[-1]

    return schedule_func


def enum_epoch(schedule: list[float] = [0.0]) -> HyperparameterSchedule:
    """Schedule by epoch using an enumerated list."""
    def schedule_func(epoch, step):
        return schedule[min(epoch, len(schedule) - 1)]

    return schedule_func
