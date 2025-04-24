import math
import numpy as np
from functools import partial
from torch.optim.lr_scheduler import LambdaLR


def _get_linear_warmup_cosine_decay_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    **kwargs, # for compatibility with other schedules
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_plateau_steps:
        return 1.0
    progress = float(current_step - num_warmup_steps - num_plateau_steps) / float(
        max(1, num_training_steps - num_warmup_steps - num_plateau_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def get_linear_warmup_cosine_decay(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    last_epoch: int = -1,
    **kwargs, # for compatibility with other schedules
):
    """
    Create a schedule with a learning rate that follows a cosine curve after a warmup period.
    Optionally includes a plateau phase at maximum learning rate.
    """
    lr_lambda = partial(
        _get_linear_warmup_cosine_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_plateau_steps=num_plateau_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_smooth_warmup_cosine_decay_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    **kwargs, # for compatibility with other schedules
):
    if current_step < num_warmup_steps:
        return 0.5 * (1.0 - math.cos(math.pi * current_step / num_warmup_steps))
    if current_step < num_warmup_steps + num_plateau_steps:
        return 1.0
    progress = float(current_step - num_warmup_steps - num_plateau_steps) / float(
        max(1, num_training_steps - num_warmup_steps - num_plateau_steps)
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def get_smooth_warmup_cosine_decay(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    last_epoch: int = -1,
    **kwargs, # for compatibility with other schedules
):
    """
    Create a schedule with a smooth (cosine) warmup, optional plateau, and cosine decay.
    """
    lr_lambda = partial(
        _get_smooth_warmup_cosine_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_plateau_steps=num_plateau_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_linear_warmup_bumpy_decay_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    num_cycles: int = 3,
    oscillation_amplitude: float = 0.1,
    **kwargs, # for compatibility with other schedules
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_plateau_steps:
        return 1.0

    # Sine wave decay with linear envelope
    decay_steps = num_training_steps - num_warmup_steps - num_plateau_steps
    progress = float(current_step - num_warmup_steps - num_plateau_steps) / float(max(1, decay_steps))

    # Calculate sine wave with linear decay envelope and phase shift
    phase_shift = math.pi / 2  # Shift by 90 degrees to start with cosine-like decrease
    sine_value = math.sin(2 * math.pi * num_cycles * progress + phase_shift)
    linear_envelope = 1.0 - progress  # Linear decay from 1 to 0
    oscillation = (1 - oscillation_amplitude) + oscillation_amplitude * sine_value  # Oscillate between 0.8 - oscillation_amplitude and 0.8 + oscillation_amplitude
    return max(0.0, linear_envelope * oscillation)

def get_linear_warmup_bumpy_decay(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_plateau_steps: int = 0,
    num_cycles: int = 3,
    oscillation_amplitude: float = 0.1,
    last_epoch: int = -1,
    **kwargs, # for compatibility with other schedules
):
    """
    Create a schedule with smooth warmup, plateau, and sine wave decay with decreasing amplitude.
    """
    lr_lambda = partial(
        _get_linear_warmup_bumpy_decay_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_plateau_steps=num_plateau_steps,
        num_cycles=num_cycles,
        oscillation_amplitude=oscillation_amplitude,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def build_scheduler(optimizer, name, scheduler_config):
    if name == 'linear_cosine':
        return get_linear_warmup_cosine_decay(optimizer, **scheduler_config)
    elif name == 'cosine_cosine':
        return get_smooth_warmup_cosine_decay(optimizer, **scheduler_config)
    elif name == 'linear_sine':
        return get_linear_warmup_bumpy_decay(optimizer, **scheduler_config)
    else:
        raise ValueError(f"Scheduler {name} not found")