#!/usr/bin/env python3
"""LIBERO-90 task 56: pick up the butter and put it in the tray.

Everything is measured in the episode: DELTA from the two finger tips, the butter
from its depth cluster (flat box) + label colour, the tray from its interior floor.

Calibration note: rig.gripper_tip()'s colour mask (mean>105) stops at the wrist
block because the fingers below it are darker; that biases DELTA_z by ~0.11 m and
makes the table land at 0.32 instead of 0.43. Here the near-depth blob that hangs
from row 0 gives both finger tips, and DELTA is checked by the jaw gap (~0.078 m)
and the table height (~0.43 m).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
if _CODE not in sys.path:
    sys.path.append(_CODE)

import rig                    # noqa: E402
from geom import components   # noqa: E402

# code/inspect.py shadows the stdlib module that numpy imports lazily; rig.py puts
# its own directory on sys.path, so drop it again now that rig/geom are imported.
sys.path = [p for p in sys.path
            if not (p and os.path.abspath(p) == os.path.abspath(_CODE))]
sys.modules.pop("inspect", None)

OPEN, CLOSE = rig.OPEN, rig.CLOSE
JAW_OPEN = 0.078
TABLE_BAND = (0.36, 0.50)
GOOD_JAW = 0.010


def log(*a):
    print("[butter]", *a, flush=True)


# ------------------------------------------------------------------ calibration
def _col_runs(cols):
    cols = np.unique(cols)
    out = []
    if len(cols) == 0:
        return out
    s = p = cols[0]
    for c in cols[1:]:
        if c != p + 1:
            out.append((s, p))
            s = c
        p = c
    out.append((s, p))
    return out


def _fingertips(pts, z, mask, min_px=400):
    """Both finger tips of the blob that hangs in from the top of the agentview."""
    comps = [c for c in components(mask, min_px=min_px)
             if c[:, 0].min() <= 4 and c[:, 0].max() < 300
             and (c[:, 1].max() - c[:, 1].min()) < 250]
    if not comps:
        return []
    g = max(comps, key=len)
    low = g[g[:, 0] >= g[:, 0].max() - 11]
    tips = []
    for a, b in _col_runs(low[:, 1]):
        sel = low[(low[:, 1] >= a) & (low[:, 1] <= b)]
        m = sel[sel[:, 0] >= sel[:, 0].max() - 1]
        tips.append(pts[m[:, 0], m[:, 1]].mean(0))
    return tips


def solve_delta(ctx, tag="calib"):
    """DELTA with eef_pt = depth_pt + DELTA, validated by jaw gap and table height."""
    agent, _ = rig.views(ctx, tag)
    pts, z, dep, Rc, tc = rig.depth(ctx)
    eef = rig.observe(ctx)["p"]
    tz = rig.table_plane(pts, z)
    valid = (z > 0.05) & (z < 4.0)
    top = z[0:6, :]
    top = top[(top > 0.05) & (top < 4.0)]
    z0 = float(np.percentile(top, 10)) if top.size else 0.60
    best = None
    for margin in (0.15, 0.13, 0.17, 0.11, 0.20, 0.25):
        tips = _fingertips(pts, z, valid & (z < z0 + margin) & (z > 0.25))
        if len(tips) != 2:
            continue
        sep = float(np.linalg.norm(tips[0] - tips[1]))
        if not (0.05 <= sep <= 0.11):
            continue
        d = eef - 0.5 * (tips[0] + tips[1])
        table = float(tz + d[2])
        ok = TABLE_BAND[0] <= table <= TABLE_BAND[1]
        score = abs(sep - JAW_OPEN) + (0.0 if ok else 1.0)
        if best is None or score < best[0]:
            best = (score, d, table, sep, ok)
        if ok and abs(sep - JAW_OPEN) < 0.012:
            break
    if best is None:
        d, tz2, checks = rig.solve_delta(ctx, verbose=False)
        log("WARN fingertip fallback, checks", checks)
        return d, tz, float(tz + d[2])
    _, d, table, sep, ok = best
    log("DELTA", np.round(d, 3), "| jaw gap", round(sep, 4),
        "| table", round(table, 3), "| checks ok", ok)
    return d, tz, table


# ------------------------------------------------------------------ perception
def analyse(objs, agent, table):
    rows = []
    for o in objs:
        px = agent[o["pix"][:, 0], o["pix"][:, 1]]
        c = px.mean(0)
        rows.append({"o": o, "h": float(o["top"] - table), "rgb": c,
                     "rb": float(c[0] - c[2]), "n": int(o["n"])})
    return rows


def find_butter(rows):
    """Flat box (a few cm tall) with the red/orange label; the cream-cheese stick
    next to it is blue/white and scores negative on R-B."""
    near = [r for r in rows if abs(r["o"]["xy"][0]) < 0.55 and abs(r["o"]["xy"][1]) < 0.65
            and r["n"] >= 250]
    cands = [r for r in near if 0.010 <= r["h"] <= 0.060]
    if not cands:
        cands = near
    if not cands:
        return None
    cands.sort(key=lambda r: -r["rb"])
    return cands[0]


def find_tray(ctx, pts, z, tz, delta, rows, table):
    """Container: the rim cluster with the widest footprint, then its interior floor."""
    best = None
    for r in rows:
        o = r["o"]
        if abs(o["xy"][0]) > 0.6 or abs(o["xy"][1]) > 0.7 or r["h"] < 0.05:
            continue
        xyz = pts[o["pix"][:, 0], o["pix"][:, 1]] + delta
        w = float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])))
        if best is None or w > best[0]:
            best = (w, o)
    floor = None
    if best is not None:
        floor = rig.floor_region(pts, z, tz, delta, bbox=best[1]["bbox"])
    if floor is None:
        floor = rig.floor_region(pts, z, tz, delta)
    return floor


# ------------------------------------------------------------------ motion
def descend(ctx, xy, z_tgt, dz=0.010, max_blocks=40, cap=0.6):
    """Lower with the jaws open until z_tgt, stopping if the arm stops following."""
    p = rig.observe(ctx)["p"]
    for _ in range(max_blocks):
        if p[2] <= z_tgt + 0.0015:
            break
        tgt = np.array([xy[0], xy[1], max(p[2] - dz, z_tgt)], dtype=float)
        q = rig.goto(ctx, tgt, OPEN, tol=0.002, blocks=1, n=5, cap=cap)
        if abs(q[2] - tgt[2]) > 0.004 and abs(q[2] - p[2]) < 0.0015:
            log("stall at", round(q[2], 4))
            return q[2], True
        p = q
    return p[2], False


def pick(ctx, xy, top):
    rig.goto(ctx, [xy[0], xy[1], top + 0.15], OPEN)
    rig.goto(ctx, [xy[0], xy[1], top + 0.030], OPEN, tol=0.006, blocks=8, n=5, cap=0.7)
    jaw = 0.0
    for attempt in range(2):
        z_tgt = top - 0.012 - 0.010 * attempt
        z, stalled = descend(ctx, xy, z_tgt)
        jaw = rig.close(ctx, 8)
        log("grasp attempt", attempt, "z", round(z, 3), "stalled", stalled, "jaw", round(jaw, 4))
        if jaw >= GOOD_JAW:
            break
        rig.open_(ctx, 5)
        rig.goto(ctx, [xy[0], xy[1], z_tgt + 0.02], OPEN, tol=0.004, blocks=4)
    rig.goto(ctx, [xy[0], xy[1], top + 0.14], CLOSE, blocks=10, n=5)
    return rig.observe(ctx)["jaw"] >= GOOD_JAW, jaw


def patch_height(pts, z, delta, table, xy, rad=0.022):
    """Median height (above the table) of the surface seen around xy in the eef frame.

    Before the placement this reads the tray floor (~0.02); after it, the butter's
    top (~floor + butter thickness).  Works even when the butter merges into the
    tray's pixel cluster, which is what happens once it is inside.
    """
    valid = (z > 0.05) & (z < 4.0)
    q = pts + delta
    m = (valid & (np.abs(q[..., 0] - xy[0]) < rad) & (np.abs(q[..., 1] - xy[1]) < rad)
         & (q[..., 2] - table < 0.12))
    if int(m.sum()) < 25:
        return None, int(m.sum())
    return float(np.median(q[..., 2][m]) - table), int(m.sum())


def main(ctx):
    st = rig.observe(ctx)
    log("eef", np.round(st["p"], 3), "jaw", round(st["jaw"], 4), "remaining", st["remaining"])
    if st["remaining"] is not None and st["remaining"] <= 0:
        return

    delta, tz, table = solve_delta(ctx)
    pts, z, dep, Rc, tc = rig.depth(ctx)
    objs, agent, wrist = rig.scene_objects(ctx, delta, tz)
    rows = analyse(objs, agent, table)
    for r in rows:
        log("obj n=%d h=%.3f rb=%.0f xy=%s top=%.3f bbox=%s"
            % (r["n"], r["h"], r["rb"], np.round(r["o"]["xy"], 3), r["o"]["top"], r["o"]["bbox"]))

    butter = find_butter(rows)
    floor = find_tray(ctx, pts, z, tz, delta, rows, table)
    if butter is None or floor is None:
        log("ABORT: butter", butter is not None, "floor", floor is not None)
        return
    bxy = np.asarray(butter["o"]["xy"], dtype=float)
    btop = float(butter["o"]["top"])
    log("BUTTER xy", np.round(bxy, 3), "top", round(btop, 3), "h", round(butter["h"], 3),
        "| TRAY floor", np.round(floor["xy"], 3), "z", round(floor["z"], 3),
        "xr", np.round(floor["xr"], 3), "yr", np.round(floor["yr"], 3))
    h_pre, n_pre = patch_height(pts, z, delta, table, floor["xy"])
    log("tray floor patch height before", None if h_pre is None else round(h_pre, 3), "n", n_pre)

    ok, jaw = pick(ctx, bxy, btop)
    log("picked", ok, "jaw", round(jaw, 4))
    if not ok:
        log("ABORT: grasp failed")
        return

    travel = max(0.58, btop + 0.16)
    rig.goto(ctx, [floor["xy"][0], floor["xy"][1], travel], CLOSE, blocks=18, n=5)
    z_rel = floor["z"] + 0.020
    rig.goto(ctx, [floor["xy"][0], floor["xy"][1], z_rel], CLOSE, tol=0.004, blocks=10, n=4, cap=0.6)
    rig.open_(ctx, 6)
    rig.goto(ctx, [floor["xy"][0], floor["xy"][1], z_rel + 0.16], OPEN, blocks=12, n=5)
    log("released at z", round(z_rel, 3))

    # geometric check: the surface under the release point rose by the object thickness
    pts2, z2, dep2, _, _ = rig.depth(ctx)
    h_post, n_post = patch_height(pts2, z2, delta, table, floor["xy"])
    want = (float(floor["z"]) - table) + float(butter["h"])
    rise = None if (h_pre is None or h_post is None) else h_post - h_pre
    log("CHECK patch before", None if h_pre is None else round(h_pre, 3),
        "after", None if h_post is None else round(h_post, 3),
        "rise", None if rise is None else round(rise, 3),
        "expected", round(want, 3), "n", n_post)
    if h_post is not None and ((rise is not None and 0.010 <= rise <= 0.060)
                               or abs(h_post - want) < 0.020):
        log("CHECK OK: butter in the tray")
        return

    objs2, agent2, _ = rig.scene_objects(ctx, delta, tz)
    rows2 = analyse(objs2, agent2, table)
    hit, best = None, None
    for r in rows2:
        o = r["o"]
        if abs(o["xy"][0]) > 0.5:            # ignore the far horizon blobs
            continue
        ins = rig.inside(o["xy"], floor["xr"], floor["yr"])
        log("  after: n=%d h=%.3f inside=%s xy=%s top=%.3f"
            % (o["n"], r["h"], ins, np.round(o["xy"], 3), o["top"]))
        if not ins or not (0.004 <= r["h"] <= 0.060):
            continue                          # the tray's own rim is 0.09 tall
        err = abs(float(o["top"]) - (float(floor["z"]) + float(butter["h"])))
        if best is None or err < best[0]:
            best = (err, o)
    hit = best[1] if best is not None else None
    if hit is None:
        log("CHECK FAIL: nothing inside the tray")
        return
    dt = float(hit["top"] - (floor["z"] + butter["h"]))
    log("CHECK in-tray top", round(float(hit["top"]), 3), "expected",
        round(float(floor["z"] + butter["h"]), 3), "err", round(dt, 4),
        "n", hit["n"], "xy", np.round(hit["xy"], 3))
    log("done")
