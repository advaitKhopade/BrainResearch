from __future__ import annotations

import numpy as np


def _boxcount(binary_img: np.ndarray, k: int) -> int:
    h, w = binary_img.shape
    h_trim = h - (h % k)
    w_trim = w - (w % k)
    img = binary_img[:h_trim, :w_trim]

    if h_trim == 0 or w_trim == 0:
        return 0

    blocks = img.reshape(h_trim // k, k, w_trim // k, k)
    block_sum = blocks.sum(axis=(1, 3))
    return int(np.sum(block_sum > 0))


def boxcount_fd_matrix(
    matrix: np.ndarray,
    threshold: float = 0.30,
    use_abs: bool = True,
    exclude_diagonal: bool = True,
) -> float:
    x = np.asarray(matrix, dtype=float).copy()

    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("matrix must be square 2D")

    if use_abs:
        x = np.abs(x)

    if exclude_diagonal:
        np.fill_diagonal(x, 0.0)

    binary = (x >= threshold).astype(np.uint8)

    n = min(binary.shape)
    max_pow = int(np.floor(np.log2(n)))
    sizes = [2 ** p for p in range(max_pow, 1, -1)]

    counts = []
    inv_sizes = []

    for k in sizes:
        c = _boxcount(binary, k)
        if c > 0:
            counts.append(c)
            inv_sizes.append(1.0 / k)

    if len(counts) < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(inv_sizes), np.log(counts), 1)
    return float(slope)


def graph_features(
    matrix: np.ndarray,
    threshold: float = 0.30,
    use_abs: bool = True,
) -> dict:
    x = np.asarray(matrix, dtype=float).copy()

    if use_abs:
        x = np.abs(x)

    np.fill_diagonal(x, 0.0)

    A = (x >= threshold).astype(np.uint8)
    n = A.shape[0]

    degrees = A.sum(axis=1).astype(float)
    density = float(A.sum() / (n * (n - 1)))

    deg_sum = degrees.sum()
    if deg_sum > 0:
        p = degrees / deg_sum
        p = p[p > 0]
        degree_entropy = float(-(p * np.log(p)).sum())
    else:
        degree_entropy = float("nan")

    return {
        "graph_density": density,
        "mean_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees)),
        "degree_entropy": degree_entropy,
    }


def multi_threshold_graph_scaling(
    matrix: np.ndarray,
    thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60),
    use_abs: bool = True,
) -> float:
    x = np.asarray(matrix, dtype=float).copy()

    if use_abs:
        x = np.abs(x)

    np.fill_diagonal(x, 0.0)

    valid_t = []
    edge_counts = []

    for t in thresholds:
        A = (x >= t).astype(np.uint8)
        m = int(A.sum())
        if m > 0:
            valid_t.append(t)
            edge_counts.append(m)

    if len(valid_t) < 2:
        return float("nan")

    slope, _ = np.polyfit(np.log(valid_t), np.log(edge_counts), 1)
    return float(slope)