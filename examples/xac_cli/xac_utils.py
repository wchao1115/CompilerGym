import random
from typing import Iterable, List, TypeVar

T = TypeVar('T')

def round_robin_iter(items: List[T]) -> Iterable[T]:
    """
    Returns an infinite round-robin iterable over the input list.
    """
    while True:
        for item in items:
            yield item

def sequential_iter(items: List[T]) -> Iterable[T]:
    """
    Returns a finite iterable that yields the items in the input list sequentially, once each.
    """
    for item in items:
        yield item

class MovingExponentialAverage:
    """Simple class to calculate exponential moving averages."""

    def __init__(self, smoothing_factor):
        self._smoothing_factor = smoothing_factor
        self._value = None

    def next(self, entry):
        assert entry is not None
        if self._value is None:
            self._value = entry
        else:
            self._value = (
                entry * (1 - self._smoothing_factor) + self._value * self._smoothing_factor
            )
        return self._value
    
    @property
    def value(self):
        return self._value
