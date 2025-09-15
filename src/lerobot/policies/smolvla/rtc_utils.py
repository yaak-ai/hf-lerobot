from typing import TypeAlias

import torch
from typing_extensions import Literal


def temporal_ensabling(pred_actions, action, step, delta):
    batch_size = pred_actions.shape[0]
    history = pred_actions[: step * batch_size, ...]

    horizon = action.shape[1]
    for window in range(delta, horizon, delta):
        if window < history.shape[0]:
            history[window, ...] = action[window - delta, ...]


def get_prefix_weights(start: int, end: int, total: int, schedule: str) -> torch.Tensor:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """  # noqa: DOC201, DOC501
    start = min(start, end)
    if schedule == "ones":
        w = torch.ones(total)
    elif schedule == "zeros":
        w = (torch.arange(total) < start).float()
    elif schedule in {"linear", "exp"}:
        w = torch.clamp((start - 1 - torch.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (torch.exp(torch.ones(1)) - 1)
    else:
        msg = f"Invalid schedule: {schedule}"
        raise ValueError(msg)
    return torch.where(torch.arange(total) >= end, torch.tensor(0.0), w)


type PrefixAttentionSchedule = Literal["linear", "exp", "ones", "zeros"]
