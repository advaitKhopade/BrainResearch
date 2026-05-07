from __future__ import annotations

import numpy as np


def spectral_slope_1d(signal: np.ndarray, tr: float = 1.0) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 16:
        return float("nan")

    x = x - np.mean(x)
    n = x.size

    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=tr)

    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]

    if len(freqs) < 8:
        return float("nan")

    lo = max(1, int(0.05 * len(freqs)))
    hi = max(lo + 2, int(0.80 * len(freqs)))

    f = freqs[lo:hi]
    p = power[lo:hi]

    if len(f) < 4:
        return float("nan")

    slope, _ = np.polyfit(np.log(f), np.log(p), 1)
    beta = -float(slope)
    return beta


def dfa_alpha_1d(signal: np.ndarray, scales: np.ndarray | None = None) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 32:
        return float("nan")

    x = x - np.mean(x)
    y = np.cumsum(x)
    n = len(y)

    if scales is None:
        scales = np.array([4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48], dtype=int)
        scales = scales[scales < n // 2]

    flucts = []
    valid_scales = []

    for s in scales:
        n_windows = n // s
        if n_windows < 2:
            continue

        y_cut = y[:n_windows * s]
        windows = y_cut.reshape(n_windows, s)

        t = np.arange(s, dtype=float)
        rms_vals = []

        for w in windows:
            coeffs = np.polyfit(t, w, 1)
            trend = np.polyval(coeffs, t)
            rms = np.sqrt(np.mean((w - trend) ** 2))
            if np.isfinite(rms) and rms > 0:
                rms_vals.append(rms)

        if rms_vals:
            flucts.append(np.mean(rms_vals))
            valid_scales.append(s)

    if len(valid_scales) < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(valid_scales), np.log(flucts), 1)
    return float(slope)


def higuchi_fd_1d(signal: np.ndarray, kmax: int = 10) -> float:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < max(16, kmax + 2):
        return float("nan")

    lk = []
    ks = []

    for k in range(1, kmax + 1):
        lm_vals = []

        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue

            diff = np.abs(np.diff(x[idx]))
            norm = (n - 1) / (((n - m - 1) // k) * k) if ((n - m - 1) // k) > 0 else np.nan
            length = norm * diff.sum()

            if np.isfinite(length) and length > 0:
                lm_vals.append(length)

        if lm_vals:
            lk.append(np.mean(lm_vals))
            ks.append(k)

    if len(ks) < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(1.0 / np.array(ks)), np.log(np.array(lk)), 1)
    return float(slope)


def sliding_window_dfa_features(
    signal: np.ndarray,
    window: int = 80,
    step: int = 10,
) -> tuple[float, float, float]:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < window:
        return float("nan"), float("nan"), float("nan")

    vals = []
    starts = list(range(0, len(x) - window + 1, step))

    for s in starts:
        seg = x[s:s + window]
        h = dfa_alpha_1d(seg)
        if np.isfinite(h):
            vals.append(h)

    vals = np.asarray(vals, dtype=float)

    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")

    hvar = float(np.var(vals))
    hrange = float(np.max(vals) - np.min(vals))
    trend_x = np.arange(len(vals), dtype=float)
    htrend, _ = np.polyfit(trend_x, vals, 1)

    return hvar, hrange, float(htrend)


def subject_mean_spectral_slope(ts_subject: np.ndarray, tr: float = 1.0) -> float:
    vals = [spectral_slope_1d(ts_subject[:, r], tr=tr) for r in range(ts_subject.shape[1])]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals))


def subject_mean_dfa(ts_subject: np.ndarray) -> float:
    vals = [dfa_alpha_1d(ts_subject[:, r]) for r in range(ts_subject.shape[1])]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals))


def subject_mean_hfd(ts_subject: np.ndarray, kmax: int = 10) -> float:
    vals = [higuchi_fd_1d(ts_subject[:, r], kmax=kmax) for r in range(ts_subject.shape[1])]
    vals = np.asarray(vals, dtype=float)
    return float(np.nanmean(vals))


def subject_mean_sliding_window_dfa(ts_subject: np.ndarray) -> tuple[float, float, float]:
    hvars = []
    hranges = []
    htrends = []

    for r in range(ts_subject.shape[1]):
        hvar, hrange, htrend = sliding_window_dfa_features(ts_subject[:, r])
        hvars.append(hvar)
        hranges.append(hrange)
        htrends.append(htrend)

    hvars = np.asarray(hvars, dtype=float)
    hranges = np.asarray(hranges, dtype=float)
    htrends = np.asarray(htrends, dtype=float)

    return (
        float(np.nanmean(hvars)),
        float(np.nanmean(hranges)),
        float(np.nanmean(htrends)),
    )