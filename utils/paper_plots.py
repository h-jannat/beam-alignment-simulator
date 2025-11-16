"""
Plotting helpers that reproduce the figures from
"Deep Learning Based Adaptive Joint mmWave Beam Alignment".

Each helper mirrors one figure from the paper. The functions focus on two
goals:
1. Provide a consistent look-and-feel with the publication.
2. Keep the API flexible so researchers can plug in their own simulation data.

Usage example:
    from utils import paper_plots
    fig = paper_plots.plot_nn_variant_vs_snr(
        snr_db=np.linspace(-10, 20, 7),
        beamforming_gain={
            "C1": np.random.uniform(0, 5, 7),
            "C2": np.random.uniform(2, 8, 7),
            "C3": np.random.uniform(4, 10, 7),
            "Exhaustive search": np.random.uniform(-2, 3, 7),
        },
        satisfaction_prob={
            "C1": np.random.uniform(0.2, 0.8, 7),
            "C2": np.random.uniform(0.4, 0.9, 7),
            "C3": np.random.uniform(0.6, 1.0, 7),
            "Exhaustive search": np.random.uniform(0.1, 0.5, 7),
        },
    )
    fig.savefig("figure4.png", dpi=300, bbox_inches="tight")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

ColorMap = Mapping[str, str]


# -----------------------------------------------------------------------------
# Helper dataclasses
# -----------------------------------------------------------------------------
@dataclass
class Series2D:
    """Container for a 2D curve."""

    label: str
    x: Sequence[float]
    y: Sequence[float]
    style: str = "-"

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)


# -----------------------------------------------------------------------------
# Diagram-style figures (Fig. 1 and Fig. 2)
# -----------------------------------------------------------------------------
def _init_diagram(fig_size: Tuple[float, float]) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    return fig, ax


def plot_system_model_diagram() -> plt.Figure:
    """
    Recreates Figure 1 from the paper: the joint BS/UE beam-alignment system.
    """

    fig, ax = _init_diagram((9, 4))

    # BS block
    bs_box = patches.FancyBboxPatch(
        (0.5, 1.5),
        2.5,
        2.5,
        boxstyle="round,pad=0.2",
        linewidth=1.4,
        edgecolor="navy",
        facecolor="#D9E3F0",
    )
    ax.add_patch(bs_box)
    ax.text(1.75, 3.2, "BS Controller", ha="center", va="center", fontsize=12, color="navy")
    ax.text(1.75, 2.4, r"$f_t \in \mathbb{C}^{N_{TX}}$", ha="center")

    # UE block
    ue_box = patches.FancyBboxPatch(
        (6.8, 1.5),
        2.5,
        2.5,
        boxstyle="round,pad=0.2",
        linewidth=1.4,
        edgecolor="darkgreen",
        facecolor="#DEF2D6",
    )
    ax.add_patch(ue_box)
    ax.text(8.05, 3.2, "UE Controller", ha="center", va="center", fontsize=12, color="darkgreen")
    ax.text(8.05, 2.4, r"$w_t \in \mathbb{C}^{N_{RX}}$", ha="center")

    # Channel depiction
    ax.arrow(3.2, 2.75, 3.2, 0, width=0.02, head_width=0.2, length_includes_head=True, color="black")
    ax.text(5.3, 3.0, r"$\mathbf{H} = \sum_{\ell=1}^{L} \alpha_{\ell} \mathbf{a}_{RX}(\phi_{\ell}) \mathbf{a}_{TX}^H(\phi_{\ell})$", ha="center")

    # Feedback
    ax.arrow(6.4, 1.75, -3.2, 0, width=0.02, head_width=0.2, length_includes_head=True, color="crimson")
    ax.text(5.3, 1.4, r"$m_{FB}$", color="crimson", ha="center")

    # Observations
    ax.text(3.2, 2.4, r"$\mathbf{y}_t = \mathbf{w}_t^H \mathbf{H} \mathbf{f}_t + \mathbf{w}_t^H \mathbf{n}$", ha="center", fontsize=11)

    ax.text(1.75, 1.7, "TX array", ha="center", fontsize=10)
    ax.text(8.05, 1.7, "RX array", ha="center", fontsize=10)

    ax.set_title("Fig. 1 – Joint mmWave Beam Alignment System", fontsize=13, pad=12)
    fig.tight_layout()
    return fig


def plot_unrolled_algorithm_diagram(T: int = 8) -> plt.Figure:
    """
    Recreates Figure 2: timeline of the unrolled joint BA algorithm.

    Parameters
    ----------
    T : int
        Number of sensing steps displayed in the diagram.
    """

    fig, ax = _init_diagram((10, 4))
    ax.set_ylim(0, 6)
    ax.set_xlim(0, T + 2)
    ax.set_title("Fig. 2 – Unrolled Joint BA Algorithm", fontsize=13, pad=12)

    for t in range(T + 1):
        x = t + 0.5
        color = "#F7D7C4" if t < T else "#FFD966"
        label = r"$\mathbf{f}_t$ (codebook)" if t < T else r"$\mathbf{f}_T(m_{FB})$"
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.3, 3.5),
                0.6,
                0.8,
                boxstyle="round,pad=0.1",
                linewidth=1,
                edgecolor="sienna",
                facecolor=color,
            )
        )
        ax.text(x, 3.9, label, rotation=90, ha="center", va="center", fontsize=9)

        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.3, 1.2),
                0.6,
                0.8,
                boxstyle="round,pad=0.1",
                linewidth=1,
                edgecolor="darkgreen",
                facecolor="#CDEACE",
            )
        )
        ax.text(x, 1.6, r"$\mathbf{w}_t$", rotation=90, ha="center", va="center", fontsize=9)

        ax.annotate(
            "",
            xy=(x, 3.4),
            xytext=(x, 2.2),
            arrowprops=dict(arrowstyle="->", color="black"),
        )
        ax.text(x, 2.7, r"$\mathbf{y}_t$", rotation=90, ha="center", va="center", fontsize=8)

    ax.text(0.7, 5.1, "BS codebook sweep (trainable)", color="sienna")
    ax.text(0.7, 0.7, "UE adaptive combiner (trainable)", color="darkgreen")
    ax.text(T + 0.8, 3.0, r"$m_{FB}$", color="crimson")
    ax.axvline(T + 0.5, color="crimson", linestyle="--", linewidth=1)
    ax.text(T + 0.55, 4.6, "Final beam selection", color="crimson", rotation=90)
    ax.annotate(
        "",
        xy=(T + 0.5, 4.3),
        xytext=(T + 1.1, 4.3),
        arrowprops=dict(arrowstyle="->", color="crimson"),
    )
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figure 3: learned beam patterns
# -----------------------------------------------------------------------------
def plot_learned_beampatterns(
    angles_rad: Sequence[float],
    heatmap: np.ndarray,
    time_labels: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Recreates Figure 3: heatmap visualization of beampattern magnitudes.

    Parameters
    ----------
    angles_rad : Sequence[float]
        Grid of azimuth/elevation angles (in radians) on the horizontal axis.
    heatmap : np.ndarray, shape (T, len(angles_rad))
        Magnitude of the learned beams per sensing step.
    time_labels : optional
        Custom labels for each sensing step row.
    """

    heatmap = np.asarray(heatmap, dtype=float)
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array of shape (T, len(angles))")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(
        heatmap,
        aspect="auto",
        extent=[angles_rad[0], angles_rad[-1], 0, heatmap.shape[0]],
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel(r"Angle $\theta$ [rad]")
    ax.set_ylabel("Sensing step t")
    ax.set_title("Fig. 3 – Learned Beampatterns", pad=12)
    if time_labels is not None and len(time_labels) == heatmap.shape[0]:
        ax.set_yticks(np.arange(heatmap.shape[0]) + 0.5)
        ax.set_yticklabels(time_labels)
    fig.colorbar(im, ax=ax, label="Normalized magnitude")
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Figures 4–7: metric curves
# -----------------------------------------------------------------------------
def _plot_metric_curves(
    ax: plt.Axes,
    x: Sequence[float],
    curves: Mapping[str, Sequence[float]],
    styles: Optional[ColorMap] = None,
    linestyle: str = "-",
    ylabel: str = "",
) -> None:
    x_arr = np.asarray(x, dtype=float)
    for label, y in curves.items():
        y_arr = np.asarray(y, dtype=float)
        if y_arr.shape != x_arr.shape:
            raise ValueError(f"Series '{label}' does not align with x-axis length {x_arr.size}")
        color = None if styles is None else styles.get(label)
        ax.plot(x_arr, y_arr, linestyle=linestyle, marker="o", label=label, color=color)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", linewidth=0.7)


def plot_nn_variant_vs_snr(
    snr_db: Sequence[float],
    beamforming_gain: Mapping[str, Sequence[float]],
    satisfaction_prob: Mapping[str, Sequence[float]],
    styles: Optional[ColorMap] = None,
) -> plt.Figure:
    """
    Recreates Figure 4: comparison of controller variants versus SNR.
    """

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    _plot_metric_curves(axes[0], snr_db, beamforming_gain, styles, "-", "Beamforming gain [dB]")
    axes[0].set_title("Fig. 4 – Impact of NN variants")
    axes[1].set_title("")
    _plot_metric_curves(axes[1], snr_db, satisfaction_prob, styles, "--", "Satisfaction probability")
    axes[1].set_xlabel("Per-antenna SNR [dB]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best")
    axes[1].legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig


def plot_sensing_steps_vs_performance(
    T_values: Sequence[int],
    beamforming_gain: Mapping[str, Sequence[float]],
    satisfaction_prob: Mapping[str, Sequence[float]],
    styles: Optional[ColorMap] = None,
) -> plt.Figure:
    """
    Recreates Figure 5: influence of sensing steps T for a fixed BS codebook.
    """

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    _plot_metric_curves(axes[0], T_values, beamforming_gain, styles, "-", "Beamforming gain [dB]")
    axes[0].set_title("Fig. 5 – Impact of sensing steps T")
    _plot_metric_curves(axes[1], T_values, satisfaction_prob, styles, "--", "Satisfaction probability")
    axes[1].set_xlabel("Number of sensing steps T")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best")
    axes[1].legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig


def plot_scheme_comparison_vs_snr(
    snr_db: Sequence[float],
    beamforming_gain: Mapping[str, Sequence[float]],
    satisfaction_prob: Mapping[str, Sequence[float]],
    styles: Optional[ColorMap] = None,
) -> plt.Figure:
    """
    Recreates Figure 6: comparison of several BA schemes versus SNR.
    """

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    _plot_metric_curves(axes[0], snr_db, beamforming_gain, styles, "-", "Beamforming gain [dB]")
    axes[0].set_title("Fig. 6 – BA scheme comparison")
    _plot_metric_curves(axes[1], snr_db, satisfaction_prob, styles, "--", "Satisfaction probability")
    axes[1].set_xlabel("Per-antenna SNR [dB]")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best")
    axes[1].legend(handles, labels, loc="best")
    fig.tight_layout()
    return fig


def plot_beamforming_gain_vs_steps(
    T_values: Sequence[int],
    gain_curves: Mapping[str, Sequence[float]],
    styles: Optional[ColorMap] = None,
) -> plt.Figure:
    """
    Recreates Figure 7: beamforming gain versus number of sensing steps T.
    """

    fig, ax = plt.subplots(figsize=(6, 4))
    _plot_metric_curves(ax, T_values, gain_curves, styles, "-", "Beamforming gain [dB]")
    ax.set_xlabel("Number of sensing steps T")
    ax.set_title("Fig. 7 – Beamforming gain vs sensing steps")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


__all__ = [
    "plot_system_model_diagram",
    "plot_unrolled_algorithm_diagram",
    "plot_learned_beampatterns",
    "plot_nn_variant_vs_snr",
    "plot_sensing_steps_vs_performance",
    "plot_scheme_comparison_vs_snr",
    "plot_beamforming_gain_vs_steps",
]
