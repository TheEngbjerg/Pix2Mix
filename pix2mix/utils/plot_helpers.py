import matplotlib.pyplot as plt
import numpy as np

def create_pref_target_spectrogram(prediction: np.ndarray, target: np.ndarray, save_path: str = None, title_prefix: str = ""):
    """Plot prediction and target spectrograms side by side.

    Args:
        prediction: 2D array with prediction spectrogram values.
        target: 2D array with target spectrogram values.
        save_path: Optional path to save the figure. If None, the figure will be shown.
        title_prefix: Optional prefix for plot titles.

    Returns:
        The created matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im_target = axes[0].imshow(target, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title(f"{title_prefix} Target Spectrogram")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")
    fig.colorbar(im_target, ax=axes[0], format="%.2f")

    im_pred = axes[1].imshow(prediction, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_title(f"{title_prefix} Prediction Spectrogram")
    axes[1].set_xlabel("Time")
    fig.colorbar(im_pred, ax=axes[1], format="%.2f")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

    return fig