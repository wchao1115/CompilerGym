import random
from typing import Iterable, List, TypeVar
import torch

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

def or_mask_across_episodes(flat_mask: torch.Tensor, episode_length: int, num_actions: int, current_step: int) -> torch.Tensor:
    """
    Given a flattened mask tensor of shape (episode_length * num_actions,),
    returns a mask of shape (num_actions,) where each entry is True if that action
    was masked in any previous step (from 0 to current_step-1).
    """
    if current_step == 0:
        return torch.zeros(num_actions, dtype=flat_mask.dtype, device=flat_mask.device)
    
    # Reshape to (episode_length, num_actions)
    mask_2d = flat_mask.view(episode_length, num_actions)
    # OR across all previous steps for each action
    or_mask = mask_2d[:current_step, :].any(dim=0)
    return or_mask
