#!/usr/bin/env python3
"""Which cluster is the butter, which is the tray - and where exactly to drop.

Geometry proposes, imajev decides, depth measures:

    rows  = scene_id.candidates(objs, table, "butter")   # flat boxes (2 on task 56)
    row, how = scene_id.identify(rows, agent, "butter")  # imajev on 2 crops
    floor = scene_id.container_floor(pts, z, tz, delta, tray_row, table)
    rig.goto(ctx, [floor["xy_safe"][0], floor["xy_safe"][1], 0.60], rig.CLOSE)

Why not pure geometry for the tray: the 46/50 rollout on task 56 lost 4 scenes on
"the tall cluster with the widest footprint".  A horizon/wall blob can win that
contest (they are up to 0.3 m tall and wide), and the release point was the median
of the visible floor pixels, which slides onto a rail or a wall when the tray is
clipped by the right edge of the agentview - it always is, x=511/512 there.
"""
import numpy as np

import rig
import vlm

# ---------------------------------------------------------------- shortlists
def analyse(objs, agent, table):
    """One row per cluster: height above the table, mean colour, red-minus-blue.

    h (not the absolute top) is what separates kinds: on task 56 the butter and the
    cream-cheese stick are both 0.029-0.030 tall, the cans 0.091, the tray rim 0.093.
    rb is what separates the butter (+54) from the cream cheese (-21).
    """
    rows = []
    for o in objs:
        px = agent[o["pix"][:, 0], o["pix"][:, 1]]
        c = px.mean(0)
        rows.append({"o": o, "h": float(o["top"] - table), "rgb": c,
                     "rb": float(c[0] - c[2]), "n": int(o["n"])})
    return rows


def _reachable(row, x=0.60, y=0.70):
    return abs(row["o"]["xy"][0]) <= x and abs(row["o"]["xy"][1]) <= y


def candidates(rows, kind):
    """Geometric shortlist. 'butter' -> flat boxes, 'tray' -> tall wide clusters."""
    out = []
    for r in rows:
        if r["n"] < 250 or not _reachable(r):
            continue
        if kind == "butter" and 0.010 <= r["h"] <= 0.060:
            out.append(r)
        if kind == "tray" and 0.055 <= r["h"] <= 0.16 and r["n"] >= 600:
            out.append(r)
    return out


def footprint(row, pts, delta):
    xyz = pts[row["o"]["pix"][:, 0], row["o"]["pix"][:, 1]] + delta
    return float(max(np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1])))


BUTTER_Q = {"which": {
    "type": "choice",
    "instructions": "Two upscaled crops of single objects lying on a table, from a "
                    "robot's camera. Which one is a pack of butter - a small box whose "
                    "label says BUTTER and is mostly warm yellow/orange?",
    "criteria": {"first": "the FIRST picture is butter",
                 "second": "the SECOND picture is butter",
                 "neither": "neither picture shows butter"}}}

TRAY_Q = {"which": {
    "type": "choice",
    "instructions": "Two upscaled crops from a robot's table view. Which one is an open "
                    "wooden tray or box - raised walls with a flat interior floor you "
                    "could drop something into?",
    "criteria": {"first": "the FIRST picture is the tray",
                 "second": "the SECOND picture is the tray",
                 "neither": "neither picture shows a tray"}}}


def ask_pair(a, b, question, agent, tag="id"):
    """Ask imajev which of two clusters is the one. 0, 1, or None (no answer)."""
    if not vlm.available():
        return None
    pa = rig.crop(agent, a["o"]["bbox"], "%s_A.png" % tag, scale=4)
    pb = rig.crop(agent, b["o"]["bbox"], "%s_B.png" % tag, scale=4)
    try:
        body = vlm.decide(question, images=[pa, pb])
    except Exception as exc:                     # socket died mid-episode: keep going
        print("imajev failed:", exc, flush=True)
        return None
    val, p = vlm.answer(body, "which")
    print("imajev", tag, "->", val, "unknown", p, flush=True)
    if val is None:
        return None
    if str(val).lower() in ("first", "1", "a", "left"):
        return 0
    if str(val).lower() in ("second", "2", "b", "right"):
        return 1
    return None


def identify(rows, table, agent, kind, pts=None, delta=None, tag=None):
    """Pick the butter / tray row. Returns (row, how).

    imajev is only asked when geometry leaves a real choice (>=2 candidates), so a
    quiet scene costs zero model calls.  `how` is 'imajev', 'single', or 'geometry'.
    """
    cands = candidates(rows, kind)
    if not cands:
        return None, "none"
    if kind == "butter":
        cands.sort(key=lambda r: -r["rb"])
    elif pts is not None and delta is not None:
        cands.sort(key=lambda r: -footprint(r, pts, delta))
    else:
        cands.sort(key=lambda r: -r["n"])
    if len(cands) == 1:
        return cands[0], "single"
    tag = tag or kind
    pick = ask_pair(cands[0], cands[1], BUTTER_Q if kind == "butter" else TRAY_Q, agent, tag)
    if pick is None:
        return cands[0], "geometry"
    return cands[pick], "imajev"


# ---------------------------------------------------------------- container
def _dilate(mask, k):
    m = mask
    for _ in range(int(k)):
        p = np.pad(m, 1, constant_values=False)
        m = (p[2:, 1:-1] | p[:-2, 1:-1] | p[1:-1, 2:] | p[1:-1, :-2]
             | p[2:, 2:] | p[2:, :-2] | p[:-2, 2:] | p[:-2, :-2])
    return m


def _erode(mask, k):
    m = mask
    for _ in range(int(k)):
        p = np.pad(m, 1, constant_values=False)
        m = (p[2:, 1:-1] & p[:-2, 1:-1] & p[1:-1, 2:] & p[1:-1, :-2]
             & p[2:, 2:] & p[2:, :-2] & p[:-2, 2:] & p[:-2, :-2])
    return m


def clear_point(xyz, cell=0.003, margin=0.030):
    """Interior point at least `margin` from the walls/rails/holes of a floor patch.

    Close the occupancy first (depth samples sit ~2 mm apart on a 3 mm grid, so a
    bare erosion would eat the patch through pinholes), then erode by `margin` and
    take the eroded cell nearest its centroid, snapped to a real floor point.
    Falls back to the median for a patch too thin to hold the margin.
    """
    xy = np.asarray(xyz, dtype=float)[:, :2]
    if len(xy) < 12:
        return xy.mean(0) if len(xy) else np.zeros(2)
    lo = xy.min(0) - 2 * cell
    hi = xy.max(0) + 2 * cell
    nx = max(int(np.ceil((hi[0] - lo[0]) / cell)), 1)
    ny = max(int(np.ceil((hi[1] - lo[1]) / cell)), 1)
    occ = np.zeros((nx, ny), bool)
    ix = np.clip(((xy[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    occ[ix, iy] = True
    occ = _erode(_dilate(occ, 1), 1)              # closing: fill 1-cell pinholes
    for m in (margin, margin * 0.5, cell):
        k = max(int(round(m / cell)), 1)
        er = _erode(occ, k)
        if er.any():
            idx = np.nonzero(er)
            cent = np.array([idx[0].mean(), idx[1].mean()])
            d = (idx[0] - cent[0]) ** 2 + (idx[1] - cent[1]) ** 2
            j = int(np.argmin(d))
            p = np.array([lo[0] + idx[0][j] * cell, lo[1] + idx[1][j] * cell])
            return xy[int(np.argmin(((xy - p) ** 2).sum(1)))]
    return np.median(xy, axis=0)


def container_floor(pts, z, tz, delta, row, table,
                    min_n=1200, min_span=0.060, zlo=0.004, zhi=0.040, margin=0.035):
    """Interior floor of the chosen container, with a validated release point.

    'checks' says whether it really looks like a container: a floor big enough to
    drop into, floating 4-40 mm above the table (a tray floor or a basket bottom),
    and wide enough that a 3.5 cm inset still exists.  'ok' is all of them.
    'xy_safe' is the max-clearance release point, never the raw median.
    """
    floor = rig.floor_region(pts, z, tz, delta, bbox=row["o"]["bbox"])
    if floor is None:
        return None
    xyz = pts[floor["pix"][:, 0], floor["pix"][:, 1]] + delta
    span_x = float(floor["xr"][1] - floor["xr"][0])
    span_y = float(floor["yr"][1] - floor["yr"][0])
    floor["xy_safe"] = clear_point(xyz, margin=margin)
    floor["checks"] = {
        "area": int(floor["n"]) >= min_n,
        "span_x": span_x >= min_span,
        "span_y": span_y >= min_span,
        "floor_above_table": zlo <= float(floor["z"]) - table <= zhi,
    }
    floor["ok"] = all(bool(v) for v in floor["checks"].values())
    return floor


def best_container(pts, z, tz, delta, rows, table, agent=None, tag="tray", margin=0.035):
    """Container floor picked from the shortlist, imajev-confirmed when it can help.

    Returns (floor, how).  `how` is 'imajev', 'single', 'geometry', 'global' or
    'none'.  Candidates are tried in order and the first one that passes the
    checks wins; if none passes, the largest one is returned anyway with
    floor['ok'] False - a marginal release can still score, an aborted one cannot.
    """
    cands = candidates(rows, "tray")
    if cands:
        cands.sort(key=lambda r: -footprint(r, pts, delta))
    how = "single" if len(cands) == 1 else "geometry"
    if len(cands) > 1 and agent is not None:
        pick = ask_pair(cands[0], cands[1], TRAY_Q, agent, tag)
        if pick is not None:
            cands = [cands[pick]] + [c for i, c in enumerate(cands[:2]) if i != pick] + cands[2:]
            how = "imajev"
    if not cands:
        floor = rig.floor_region(pts, z, tz, delta)
        return floor, ("global" if floor is not None else "none")
    best = None
    for row in cands:
        floor = container_floor(pts, z, tz, delta, row, table, margin=margin)
        if floor is None:
            continue
        if floor["ok"]:
            return floor, how
        if best is None or int(floor["n"]) > int(best["n"]):
            best = floor
    return best, how
