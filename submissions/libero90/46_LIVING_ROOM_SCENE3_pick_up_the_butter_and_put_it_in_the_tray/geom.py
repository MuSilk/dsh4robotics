#!/usr/bin/env python3
"""Small geometry/vision helpers (no scipy on this rig: numpy 1.19 + PIL only).

    from geom import components, quat_to_mat, rotvec
"""
import numpy as np


def label(mask):
    """4-connected labelling: returns (int32 HxW labels, count)."""
    mask = np.ascontiguousarray(mask, dtype=bool)
    h, w = mask.shape
    flat = mask.ravel()
    lab = np.zeros(h * w, dtype=np.int32)
    cur = 0
    for start in np.flatnonzero(flat):
        if lab[start]:
            continue
        cur += 1
        lab[start] = cur
        stack = [int(start)]
        while stack:
            p = stack.pop()
            r, c = divmod(p, w)
            if r > 0:
                q = p - w
                if flat[q] and not lab[q]:
                    lab[q] = cur
                    stack.append(q)
            if r < h - 1:
                q = p + w
                if flat[q] and not lab[q]:
                    lab[q] = cur
                    stack.append(q)
            if c > 0:
                q = p - 1
                if flat[q] and not lab[q]:
                    lab[q] = cur
                    stack.append(q)
            if c < w - 1:
                q = p + 1
                if flat[q] and not lab[q]:
                    lab[q] = cur
                    stack.append(q)
    return lab.reshape(h, w), cur


def components(mask, min_px=100):
    """List of (N,2) int arrays of (row, col) for each blob with >= min_px pixels."""
    lab, n = label(mask)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())
    out = []
    for i in np.flatnonzero(sizes >= min_px):
        if i == 0:
            continue
        ys, xs = np.nonzero(lab == i)
        out.append(np.stack([ys, xs], -1))
    return out


def quat_to_mat(q):
    """robosuite xyzw quaternion -> 3x3 rotation (columns are the tool axes)."""
    x, y, z, w = np.asarray(q, dtype=float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec(R):
    """(axis, angle) of a rotation matrix."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-6:
        return np.zeros(3), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return axis, ang
