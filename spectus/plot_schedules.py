import matplotlib.pyplot as plt
import numpy as np
from .schedule import (
    _get_linear_warmup_cosine_decay_lr_lambda,
    _get_smooth_warmup_cosine_decay_lr_lambda,
    _get_linear_warmup_bumpy_decay_lr_lambda,
)

def plot_schedule(
    lr_lambda,
    num_training_steps=5000,
    num_warmup_steps=1000,
    num_plateau_steps=1000,
    title="Learning Rate Schedule",
    ax=None,
    size=(10, 6),
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # Generate steps and learning rates
    steps = np.arange(num_training_steps)
    lrs = np.array([
        lr_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_plateau_steps=num_plateau_steps,
            **kwargs,
        )
        for step in steps
    ])

    # Plot the schedule
    ax.plot(steps, lrs, 'b-', linewidth=2)

    # Color the different phases
    if num_warmup_steps > 0:
        ax.axvspan(0, num_warmup_steps, color='green', alpha=0.1, label='Warmup')
    if num_plateau_steps > 0:
        ax.axvspan(
            num_warmup_steps,
            num_warmup_steps + num_plateau_steps,
            color='yellow',
            alpha=0.1,
            label='Plateau'
        )
    if num_training_steps > num_warmup_steps + num_plateau_steps:
        ax.axvspan(
            num_warmup_steps + num_plateau_steps,
            num_training_steps,
            color='red',
            alpha=0.1,
            label='Decay'
        )

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate Multiplier')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax

def plot_all_schedules(num_training_steps=5000, num_warmup_steps=1000, num_plateau_steps=500, size=(12, 20), **kwargs):
    fig, axes = plt.subplots(3, 1, figsize=size)

    # Cosine with linear warmup
    plot_schedule(
        _get_linear_warmup_cosine_decay_lr_lambda,
        num_training_steps,
        num_warmup_steps,
        num_plateau_steps,
        title="Cosine Schedule with Linear Warmup",
        ax=axes[0],
        **kwargs
    )

    # Smooth warmup cosine decay
    plot_schedule(
        _get_smooth_warmup_cosine_decay_lr_lambda,
        num_training_steps,
        num_warmup_steps,
        num_plateau_steps,
        title="Cosine Schedule with Smooth Warmup",
        ax=axes[1],
        **kwargs
    )

    # Sine warmup decay
    plot_schedule(
        _get_linear_warmup_bumpy_decay_lr_lambda,
        num_training_steps,
        num_warmup_steps,
        num_plateau_steps,
        title="Sine Schedule with Smooth Warmup and Decaying Oscillations",
        ax=axes[2],
        **kwargs
    )

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage
    fig = plot_all_schedules()
    plt.show()