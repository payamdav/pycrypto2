"""Volume profile tools: raw-price-space KDE construction and POC peak analysis.
"""

from packages.kde_tools.kernels import make_kernel
from packages.volume_profile.histogram import weighted_histogram
from packages.volume_profile.kde import compute_kde
from packages.volume_profile.peaks import (
    point_of_control,
    top_kde_peaks,
    kde_peaks_above_below,
    recursive_poc,
)

__all__ = [
    "make_kernel",
    "weighted_histogram",
    "compute_kde",
    "point_of_control",
    "top_kde_peaks",
    "kde_peaks_above_below",
    "recursive_poc",
]
