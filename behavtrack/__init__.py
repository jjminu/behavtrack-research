"""behavtrack: webcam-based rodent behavior tracking and analysis.

Approach ① (classic computer vision, background subtraction) works out of the box.
Approach ② (pose estimation with DeepLabCut / SLEAP, behavior classifiers) plugs into the
same per-frame track format, so every analysis below works for both.
"""

__version__ = "0.2.0"
