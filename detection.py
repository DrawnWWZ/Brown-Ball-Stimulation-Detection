"""Detect compact, single-scale beads in an 8-bit microscope image."""
import math

import cv2
import numpy as np


def detect(gray, diameter=4.0, polarity="dark", snr=5.0):
    """Return single-bead candidates and rejected silhouettes in image coordinates.

    diameter is the approximate full width at half contrast of an isolated bead
    in the recorded pixels. snr sets the minimum peak response in units of the
    spatial noise estimated after filtering. diameter_px is the equivalent
    diameter of the measured half-maximum footprint (after filtering).
    """
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("Expected a nonempty 2D uint8 image")
    if not np.isfinite(diameter) or diameter < 2:
        raise ValueError("Bead diameter must be finite and at least 2 pixels")
    if not np.isfinite(snr) or snr <= 0 or polarity not in ("bright", "dark"):
        raise ValueError("Expected positive finite SNR and bright or dark polarity")

    signal = gray.astype(np.float32)
    fine = cv2.GaussianBlur(signal, (0, 0), 0.20 * diameter)
    background = cv2.GaussianBlur(signal, (0, 0), 1.8 * diameter)
    response = background - fine if polarity == "dark" else fine - background
    median = np.median(response)
    noise = max(0.5, float(np.median(np.abs(response - median))) * 1.4826)
    # Require strong peaks but grow their footprints down to a lower contour.
    peak_threshold = snr * noise
    mask = (response >= 0.50 * peak_threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    radius = max(1, int(round(0.55 * diameter)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    peaks = (response == cv2.dilate(response, kernel)) & (response >= peak_threshold)
    peak_components = np.unique(labels[peaks])
    detections = []
    for label in peak_components:
        x0, y0, width, height, _ = stats[label]
        box = np.s_[y0:y0 + height, x0:x0 + width]
        footprint = labels[box] == label
        local = response[box]
        peak = float(local[footprint].max())
        half = footprint & (local >= 0.5 * peak)
        yy, xx = np.nonzero(half)
        half_width = 2 * math.sqrt(len(xx) / math.pi)
        # Tiny fluctuations and isolated hot pixels usually occupy too little area.
        if half_width < 0.55 * diameter:
            continue
        # Area alone can admit a touching pair. Measure elongation of its core
        # and count distinct significant peaks inside the same lower contour.
        centered = np.stack((xx - xx.mean(), yy - yy.mean()))
        eigenvalues = np.linalg.eigvalsh(centered @ centered.T / len(xx) + np.eye(2) / 12)
        aspect = math.sqrt(eigenvalues[1] / eigenvalues[0])
        significant = footprint & peaks[box] & (local >= 0.45 * peak)
        peak_count = cv2.connectedComponents(significant.astype(np.uint8), connectivity=8)[0] - 1

        # A floating half-height center is less biased by noise than the brightest
        # pixel. Use the wider, above-quarter-height support for smoother positions.
        weights = np.where(footprint, np.maximum(local - max(0.25 * peak, 0.5 * peak_threshold), 0), 0)
        mass = weights.sum(dtype=np.float64)
        x = x0 + weights.sum(axis=0) @ np.arange(width) / mass
        y = y0 + weights.sum(axis=1) @ np.arange(height) / mass
        if x0 == 0 or y0 == 0 or x0 + width == gray.shape[1] or y0 + height == gray.shape[0]:
            status = "edge"
        elif half_width > 1.55 * diameter or aspect > 1.65 or peak_count > 1:
            status = "aggregate"
        else:
            status = "candidate"
        detections.append(dict(x=float(x), y=float(y), diameter_px=half_width,
                               status=status, aspect=aspect, peak_snr=peak / noise))
    return detections
