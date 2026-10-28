"""LIBERO-90 task 22 - CLOSE THE BOTTOM DRAWER of the cabinet (KITCHEN_SCENE4).

This draw's cabinet faces -y, i.e. the drawers come OUT toward -y and must be pushed IN
toward +y.  That is the mirror of the chest all of SKILL.md's measured numbers were taken on
(KITCHEN_SCENE2, fronts at y -0.23, out = +y), so **every y-relation below is written in an
out-coordinate u = S*y** with `S = +1` for that chest and `S = -1` here, and `S` is *measured*
in `probe` (the cabinet's front plane is the strongest vertical face; the drawers open toward
the camera side of it, and the camera is at rig.T_RIG = (0.637, -0.012, 1.659)).

Measured on this draw's opening depth call (0 rows, `probe0.npz`):
    table z            0.9498
    cabinet fronts     y +0.216  (one flush plane, z 0.965..1.176, 3100 px in the 1 mm bin)
    open panel         y +0.067  (z 0.958..1.030)        -> 0.149 m of travel to home
    its handle bar     front y +0.036 = panel - 0.031, x -0.079..+0.010, z 0.992..1.005
    drawer floor       z 0.973 (bright, so `Scene.arm` swallows it - see `floor_visible`)
    the bar above      z 1.103..1.147  -> this drawer IS the bottom one

The recipe is `tasks/close-drawer.md` + SKILL §7 and §10.1, in the one form that fits a LOW
drawer.  The hand block's front face is 0.111 m ahead of eef_pos (the fingers only 0.069) and
spans eef_z+0.082..+0.15, so aiming the *hand* at the bar needs eef_z = bar_z0 - 0.094 = 0.898
here, i.e. the pads would be under the table: on this chest the hand-face recipe is impossible
and the measured one is the PITCHED FINGER TIPS - rotate in place to the hook pitch, push at
`eef_z = bar_zc - 0.031` from `panel + 0.10`, 24 rows to home (SKILL §10.1, task 6's close).

The pitch is mirrored with the scene: `yaw_quat(Q_HOOK, 180)` makes the fingers lead in +y and
puts the hand block behind and above them, so nothing but the two fingertips ever approaches
the panel.  Everything else - the aim (the bar, not the panel), the stop (the drawer's OWN
front plane, because a push has no force signal), the +z freeze proof and the 1 mm verdict -
is SKILL §7 unchanged.

Stages (all re-runnable on the same episode):

    ENTRY=probe   0 rows: table, axis+sign, fronts' plane, open panel + z band, bar,
                  reference plane, drawer floor, and the gripper rig at the start pose
    ENTRY=close   rotate in place -> hover -> descend -> measure the FINGER TIP vs the bar
                  (0 rows) -> push to the plane stop -> +z proof
    ENTRY=verify  retreat and take the verdict signals + a frame
    ENTRY=run     probe (if absent) -> close -> verify
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lib"))
import rig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments", "data")
LOGS = os.path.join(ROOT, "experiments", "logs")
PLAN = os.path.join(DATA, "close22.json")

QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])
Q_HOOK = rig.Q_HOOK
CAM = rig.T_RIG                      # camera centre in the base frame (= drawer "out" side)

TIPS_DZ = -0.031                     # eef_z = bar_zc + TIPS_DZ  (§10.1, task 6's close)
HAND_Z0 = 0.082                      # hand block underside = eef_z + HAND_Z0
HAND_MARGIN = 0.094                  # hand-face aim would be eef_z = bar_z0 - 0.094
HOME_TOL = 0.0025                    # panel-meets-reference tolerance, from depth
PAD_DZ = 0.0431


def _log(msg):
    print(msg, flush=True)


def _load():
    with open(PLAN) as f:
        return json.load(f)


def _save(m):
    os.makedirs(DATA, exist_ok=True)
    with open(PLAN, "w") as f:
        json.dump(m, f, indent=1, default=float)


def steps_left(ctx, tag=""):
    try:
        p = json.loads(ctx.call("libero_status")["content"][0]["text"])
        n = p["episode"]["remaining_episode_steps"]
    except Exception as exc:
        _log("  (status failed: %s)" % exc)
        return None
    _log("  remaining_episode_steps = %s %s" % (n, tag))
    return n


def _quat_deg(qa, qb):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


# ============================================================== the mirror
# S = +1: "out" is +y (SKILL's chest).  S = -1: "out" is -y (this scene).
# u(p) = S*p[1] is the outward coordinate; pushing home drives u DOWN.
def uu(S, y):
    return S * float(y)


# ============================================================== motion helpers
def rotate_in_place(ctx, q_tgt, gripper=-1.0, max_rows=26, tol=0.10, verbose=True):
    """Rotation-only rows to `q_tgt` (a position goto cannot rotate; SKILL §2)."""
    payload = rig.observe(ctx)
    for i in range(int(max_rows)):
        e = rig.rotvec_err(rig.quat_of(payload), q_tgt)
        if float(np.linalg.norm(e)) < tol:
            if verbose:
                _log("  rotate converged |err|=%.3f rad after %d rows" % (
                    float(np.linalg.norm(e)), i))
            return payload, True
        a = np.zeros(7)
        a[3:6] = np.clip(e / 0.30, -1, 1)
        a[6] = gripper
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        if verbose and i % 5 == 0:
            _log("  rotate %2d |err|=%.3f eef=%s" % (
                i, float(np.linalg.norm(rig.rotvec_err(rig.quat_of(payload), q_tgt))),
                np.round(rig.pose_of(payload), 4)))
    e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(payload), q_tgt)))
    if verbose:
        _log("  rotate ended |err|=%.3f rad eef=%s" % (e, np.round(rig.pose_of(payload), 4)))
    return payload, e < tol


def finger_profile(s, S, x, bar_zc, bar_front, half_x=0.070, dz_lo=-0.045, dz_hi=0.090):
    """Where the FINGERS are, in the pitched pose.  0 rows, one depth call.

    The out-most arm pixel in front of the bar is the fingertip (at the hook pitch the
    fingers lead and the hand block sits behind them), so its height IS the contact height.
    `S*(y - bar_front) > 0.006` keeps the bar itself out of the mask (a handle bar is
    achromatic, hence inside `Scene.arm`), and the z window keeps the table out.
    """
    rel = S * (s.Y - float(bar_front))
    m = (s.arm & (np.abs(s.X - float(x)) < half_x)
         & (s.Z > bar_zc + dz_lo) & (s.Z < bar_zc + dz_hi))
    out = {"n_all": int(m.sum())}
    if m.sum() < 60:
        return out
    p = s.pts[m]
    out["z_all"] = (round(float(p[:, 2].min()), 4), round(float(p[:, 2].max()), 4))
    f = m & (rel > 0.006)
    out["n_front"] = int(f.sum())
    if f.sum() >= 40:
        q = s.pts[f]
        uu_max = float(np.max(S * q[:, 1]))
        tip = q[S * q[:, 1] > uu_max - 0.006]
        out["u_tip"] = round(uu_max, 4)
        out["lead_vs_bar"] = round(uu_max - uu(float(S), bar_front), 4)
        out["z_tip"] = (round(float(np.percentile(tip[:, 2], 5)), 4),
                        round(float(np.percentile(tip[:, 2], 95)), 4))
        out["x_tip"] = (round(float(tip[:, 0].min()), 4), round(float(tip[:, 0].max()), 4))
    rows = []
    for z0 in np.arange(bar_zc + dz_lo, bar_zc + dz_hi, 0.008):
        q = p[(p[:, 2] >= z0) & (p[:, 2] < z0 + 0.008)]
        if len(q) < 20:
            continue
        rows.append((round(float(z0), 3), int(len(q)),
                     round(float(np.max(S * q[:, 1])), 4)))
    out["rows"] = rows
    return out


def ybin_scene(s, band, xwin, ywin, min_pixels=60):
    """1 mm y-face bins in a window - `rig.face_bins` for axis y (`S` handled by the caller)."""
    return rig.face_bins(s, 1, (float(band[0]), float(band[1])),
                         ((float(xwin[0]), float(xwin[1])), (float(ywin[0]), float(ywin[1]))),
                         min_pixels=int(min_pixels))


def front_bin(s, S, band, xwin, ywin, min_pixels=200):
    """The OUT-most 1 mm y face in the band (what a closing push must move) - 0 rows."""
    rows = ybin_scene(s, band, xwin, ywin, min_pixels=min_pixels)
    if not rows:
        return None
    return max(rows, key=lambda t: S * t[0])


def face_extent(s, y0, band, xwin, tol=0.0018, nmin=0.85):
    """(z_lo, z_hi, x_lo, x_hi, n) of the pixels on the y=y0 face, plus the normal's sign."""
    n = rig._normal(s.pts)
    m = ((np.abs(s.Y - float(y0)) < tol) & (np.abs(n[:, :, 1]) > nmin)
         & (s.X > xwin[0]) & (s.X < xwin[1])
         & (s.Z > band[0]) & (s.Z < band[1]))
    if m.sum() < 30:
        return None
    p = s.pts[m]
    return {"z": (float(np.percentile(p[:, 2], 1)), float(np.percentile(p[:, 2], 99))),
            "x": (float(np.percentile(p[:, 0], 1)), float(np.percentile(p[:, 0], 99))),
            "n": int(m.sum()), "ny": float(n[:, :, 1][m].mean())}


def find_bar(s, S, panel_y, band_z, xwin, min_px=120):
    """The handle bar attached to `panel_y`: achromatic pixels on its OUT side, z-bucketed.

    A bar is ~3 cm proud of its panel, ~1 cm tall and achromatic (mean ~95), hence inside
    `Scene.arm`; that is why this is a colour+geometry test and not a face test.
    """
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    rel = S * (s.Y - float(panel_y))
    m = (((mx - mn) < 25) & (mean > 70) & (mean < 170)
         & (rel > 0.004) & (rel < 0.075)
         & (s.Z > float(band_z[0]) - 0.02) & (s.Z < float(band_z[1]) + 0.02)
         & (s.X > xwin[0]) & (s.X < xwin[1]))
    if s.eef is not None:
        m &= (np.linalg.norm(s.pts - s.eef, axis=2) > 0.10)
    if m.sum() < min_px:
        return None
    edges = np.arange(float(band_z[0]) - 0.02, float(band_z[1]) + 0.021, 0.004)
    h, e = np.histogram(s.Z[m], edges)
    best = None
    i = 0
    while i < len(h):
        if h[i] < 8:
            i += 1
            continue
        j = i
        while j + 1 < len(h) and h[j + 1] >= 8:
            j += 1
        sel = m & (s.Z >= e[i]) & (s.Z < e[j + 1])
        i = j + 1
        if sel.sum() < min_px:
            continue
        p = s.pts[sel]
        cand = {"n": int(sel.sum()),
                "front_u": float(np.max(S * p[:, 1])),
                "front_y": float(S * np.max(S * p[:, 1])),
                "z": (float(p[:, 2].min()), float(p[:, 2].max())),
                "zc": float(0.5 * (p[:, 2].min() + p[:, 2].max())),
                "x": (float(p[:, 0].min()), float(p[:, 0].max())),
                "mean": float(mean[sel].mean())}
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def floor_visible(s, xwin, ywin, zlo, zhi):
    """Bright pixels (the drawer's white interior floor) inside the box window.  0 rows.

    The interior is rgb ~ (224,222,223): mean > 200 with chroma < 20, which excludes the
    wooden table (mean ~182, chroma ~35).  While the drawer is out the interior is visible;
    once it is shut it is not.  This is SKILL §7's "the drawer-floor plane is gone" signal,
    adapted because on this asset the floor falls inside `Scene.arm` and `planes()` drops it.
    """
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    m = ((mean > 200) & ((mx - mn) < 20) & (s.Z > float(zlo)) & (s.Z < float(zhi))
         & (s.X > xwin[0]) & (s.X < xwin[1]) & (s.Y > ywin[0]) & (s.Y < ywin[1]))
    return int(m.sum())


def panel_now(s, S, band, xwin, ywin, ref_y=None, min_pixels=150):
    """(front-most panel plane, its bin count, all bins) for the drawer being pushed."""
    rows = ybin_scene(s, band, xwin, ywin, min_pixels=60)
    out = {"rows": [(round(float(a), 4), int(b)) for a, b in rows[:8]],
           "panel_y": None, "n": 0, "off": None, "home": None}
    if not rows:
        return out
    cand = [r for r in rows if r[1] >= min_pixels]
    if not cand:
        cand = rows[:1]
    y, n = max(cand, key=lambda t: S * t[0])
    out["panel_y"], out["n"] = float(y), int(n)
    if ref_y is not None:
        out["off"] = uu(S, y) - uu(S, ref_y)
        out["home"] = bool(abs(out["off"]) <= HOME_TOL)
    return out


# ============================================================== probe (0 rows)
def probe(ctx):
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    s = rig.snapshot(ctx, tag="probe0", outdir=DATA)
    _log("start pose eef=%s quat=%s gq=%s" % (np.round(s.eef, 4), np.round(s.quat, 4),
                                             np.round(s.gq, 4)))
    tz = s.table_z()
    _log("table_z %s (rig check: the table unprojects to ONE flat plane at the height"
         " origin)" % (tz,))
    _log("quat builder check: quat_from_fingers((0,0,-1),(0,-1,0)) vs the live wrist-down"
         " = %.1f deg (want <= 5)" % _quat_deg(rig.quat_from_fingers((0, 0, -1), (0, -1, 0)),
                                              s.quat))
    _log("--- gripper rig at the start pose (wrist down)")
    pads = {}
    if s.eef is not None and s.eef[2] > 1.10:
        for half in (0.030, 0.045, 0.050, 0.060, 0.075):
            p = s.pads(half=half)
            _log("   pads(half=%.3f) %s" % (half, p))
            if p is not None:
                pads[str(half)] = float(p["offset"])
    else:
        _log("   !! probe from a LOW pose - carrying the stored pad_dz forward")
    jaw = s.jaw()
    _log("   jaw %s" % (jaw,))
    hl = s.hand_lead()
    _log("   hand_lead %s  (SKILL: 0.111 m ahead, z +0.082..+0.15 -> what really pushes)"
         % (hl,))

    # ---- 1. the drawer's frame: axis, OUT sign and the fronts' plane (rig.drawer_axis)
    fr = rig.drawer_axis(s)
    if fr is None:
        _log("!! no vertical fronts at all - cannot find the drawer bank")
        return False
    ax, S, flush = fr["i"], float(fr["S"]), float(fr["front"])
    _log("drawer frame: axis %s, OUT %+d %s, fronts %+.4f (%d px in one 5 mm bin),"
         " fronts z %.4f..%.4f" % (fr["axis"], fr["S"], fr["axis"], flush, fr["n"],
                                   fr["front_z"][0], fr["front_z"][1]))
    if fr["axis"] != "y":
        _log("!! the drawer axis is %s, not y - this controller mirrors y only; use"
             " rig.hook_quat_toward() for the pusher and mirror the geometry by hand"
             % fr["axis"])
        return False
    xwin, ywin, zwin = fr["win"][0], fr["win"][1], fr["zwin"]
    _log("window x %s y %s z %s" % (np.round(xwin, 3), np.round(ywin, 3),
                                    np.round(zwin, 4)))

    # ---- 2. the 1 mm y-face histogram: the fronts' flush plane, and the panel that is OUT
    rows = ybin_scene(s, zwin, xwin, ywin, min_pixels=100)
    _log("--- 1 mm y-face histogram (z %.3f..%.3f)" % zwin)
    for y, c in rows[:12]:
        _log("   y=%+.4f n=%d" % (y, c))
    if not rows:
        _log("no y faces at all - abort")
        return False
    ref_y = max(rows, key=lambda t: t[1])[0]          # biggest = the CLOSED fronts
    cand = [r for r in rows if r[1] >= 200 and (uu(S, r[0]) - uu(S, ref_y)) > 0.020]
    if not cand:
        _log("no drawer is out (every face is at the closed plane %+.4f) -> already HOME"
             % ref_y)
        _save(dict(table_z=float(tz), S=float(S), ref_y=float(ref_y), already_closed=True,
                   xwin=list(xwin), ywin=list(ywin), zwin=list(zwin), pads=pads))
        steps_left(ctx, "(after probe)")
        return True
    open_y, open_n = max(cand, key=lambda t: uu(S, t[0]))
    _log("closed fronts' plane (reference) y=%+.4f (n=%d) ; OPEN panel y=%+.4f (n=%d) ;"
         " travel %.4f m" % (ref_y, dict(rows)[ref_y], open_y, open_n,
                             uu(S, open_y) - uu(S, ref_y)))

    # ---- 3. the open panel's own z band
    ext = face_extent(s, open_y, zwin, xwin, tol=0.0018)
    if ext is None or ext["n"] < 60:
        _log("open-panel face only %s px - widening the y tolerance"
             % (None if ext is None else ext["n"]))
        ext = face_extent(s, open_y, zwin, xwin, tol=0.004)
    panel_z = ext["z"]
    band = (panel_z[0] + 0.002, panel_z[1] - 0.002)
    _log("open panel: z %.4f..%.4f (n=%d, mean n_y %+.2f) x %.4f..%.4f" % (
        panel_z[0], panel_z[1], ext["n"], ext["ny"], ext["x"][0], ext["x"][1]))

    # ---- 4. its handle bar (the only per-drawer identifier)
    bar = find_bar(s, S, open_y, panel_z, xwin)
    if bar is None:
        _log("NO BAR on the open panel - abort")
        return False
    dz_bar = uu(S, bar["front_y"]) - uu(S, open_y)
    _log("bar: front y=%+.4f (%.4f proud of the panel), z %.4f..%.4f (zc %.4f),"
         " x %.4f..%.4f (mid %.4f), n=%d, mean %.0f" % (
             bar["front_y"], dz_bar, bar["z"][0], bar["z"][1], bar["zc"],
             bar["x"][0], bar["x"][1], 0.5 * (bar["x"][0] + bar["x"][1]), bar["n"],
             bar["mean"]))
    zb0, zb1, zbc = bar["z"][0], bar["z"][1], bar["zc"]
    # is this the BOTTOM drawer?  any bar of a closed drawer below it?
    above_y0 = panel_z[1] + 0.010
    rows_above = ybin_scene(s, (above_y0, zwin[1]), xwin, ywin, min_pixels=200)
    _log("faces above this panel (z %.3f..%.3f): %s" % (
        above_y0, zwin[1], [(round(float(a), 4), int(b)) for a, b in rows_above[:5]]))
    bar_above = find_bar(s, S, ref_y, (above_y0, 1.30), xwin)
    _log("bar of the drawer ABOVE (searched on the closed plane): %s" % (
        None if bar_above is None else
        "z %.4f..%.4f" % (bar_above["z"][0], bar_above["z"][1]),))
    is_bottom = bool(bar_above is None or bar_above["zc"] > zbc + 0.02)
    _log("=> this drawer IS the bottom one: %s (its bar zc %.4f)" % (is_bottom, zbc))

    # ---- 5. the drawer floor (bright interior) and its visibility window
    box_x = (0.5 * (bar["x"][0] + bar["x"][1]) - 0.09, 0.5 * (bar["x"][0] + bar["x"][1]) + 0.09)
    box_y = (min(open_y, ref_y) - 0.005, max(open_y, ref_y) + 0.055)
    fz = None
    mm = s.rgb.mean(2)
    fm = ((mm > 180) & (s.Z > float(tz) + 0.010) & (s.Z < panel_z[1])
          & (s.X > box_x[0]) & (s.X < box_x[1]) & (s.Y > box_y[0]) & (s.Y < box_y[1]))
    if fm.sum() > 200:
        fz = float(np.median(s.Z[fm]))
    _log("drawer interior: %d bright px, floor z %s ; window x %s y %s" % (
        int(fm.sum()), None if fz is None else round(fz, 4),
        np.round(box_x, 3), np.round(box_y, 3)))
    floor_n0 = floor_visible(s, box_x, box_y, float(tz) + 0.010, panel_z[1])

    # ---- 6. which push?
    hand_z = zb0 - HAND_MARGIN
    hand_ok = bool(hand_z + PAD_DZ > float(tz) + 0.005 and hand_z + HAND_Z0 > float(tz) + 0.010)
    mode = "hand" if hand_ok else "tips"
    q_push = Q_HOOK if S > 0 else rig.yaw_quat(Q_HOOK, 180.0)
    _log("PUSH MODE %s : hand-face aim eef_z %.4f -> pads %.4f / underside %.4f ; table"
         " %.4f ; bar band %.4f..%.4f" % (
             mode, hand_z, hand_z + PAD_DZ, hand_z + HAND_Z0, float(tz), zb0, zb1))
    _log("  q_push = %s (fingers lead toward %+d y), %.1f deg from Q_HOOK" % (
        np.round(q_push, 4), -S, _quat_deg(q_push, Q_HOOK)))
    _log("  tips recipe: eef_z = bar_zc + %.3f = %.4f ; x = bar x mid = %.4f" % (
        TIPS_DZ, zbc + TIPS_DZ, 0.5 * (bar["x"][0] + bar["x"][1])))

    m = dict(frame={k: (v if not isinstance(v, tuple) else list(v)) for k, v in fr.items()},
             S=float(S), table_z=float(tz), xwin=list(xwin), ywin=list(ywin),
             zwin=list(zwin), ref_y=float(ref_y), open_panel_y=float(open_y),
             panel_band=[float(band[0]), float(band[1])],
             panel_z=[float(panel_z[0]), float(panel_z[1])],
             travel=float(uu(S, open_y) - uu(S, ref_y)),
             dz_bar=float(dz_bar), bar_front=float(bar["front_y"]),
             bar_z0=float(zb0), bar_z1=float(zb1), bar_zc=float(zbc),
             bar_x0=float(bar["x"][0]), bar_x1=float(bar["x"][1]),
             bar_x_mid=float(0.5 * (bar["x"][0] + bar["x"][1])),
             bar_n=int(bar["n"]), is_bottom=is_bottom,
             box_x=list(box_x), box_y=list(box_y), floor_z=fz, floor_n0=int(floor_n0),
             mode=mode, hand_z=float(hand_z), tips_z=float(zbc + TIPS_DZ),
             q_push=[float(v) for v in q_push],
             pads=pads, already_closed=False,
             jaw=None if jaw is None else {k: (v if not isinstance(v, np.ndarray)
                                               else [float(x) for x in v])
                                          for k, v in jaw.items()},
             hand_lead=None if hl is None else {k: float(v) for k, v in hl.items()})
    _save(m)
    _log("saved %s" % PLAN)
    steps_left(ctx, "(after probe)")
    return True


def selftest(ctx=None):
    """Re-run the whole METROLOGY offline on a saved snapshot - 0 rows, no episode needed.

        ENTRY=selftest python3 code/lib/run_controller.py

    Loads `experiments/data/probe0.npz` (the open drawer) and, if present,
    `experiments/data/final_clear.npz` (the same drawer shut) and re-derives every number the
    probe/report used, then compares against the plan.  This is how a change to `rig.py` or to
    this file's measurement code can be checked **after the episode is over** -- and how the
    sign of a scene can be re-checked for free.
    """
    m = _load() if os.path.exists(PLAN) else {}
    so = rig.Scene.load(os.path.join(DATA, "probe0.npz"))
    fr = rig.drawer_axis(so)
    _log("frame: axis=%s S=%+d front=%+.4f n=%d" % (fr["axis"], fr["S"], fr["front"], fr["n"]))
    ok = True
    if m:
        ok &= abs(fr["front"] - float(m["ref_y"])) < 0.003
        _log("  front vs the plan's reference: %+.4f vs %+.4f -> %s" % (
            fr["front"], m["ref_y"], "OK" if ok else "MISMATCH"))
    band = tuple(m["panel_band"]) if m else (0.973, 1.031)
    pf = rig.front_face_on(so, fr, band)
    _log("  open panel (OUT-most 1 mm bin): %s" % (None if pf is None else
                                                   (round(pf[0], 4), pf[1]),))
    bars = rig.handle_bars_on(so, fr, pf[0] if pf else fr["front"], (fr["zwin"][0], 1.30))
    _log("  bars on it: %s" % ([(round(b["front_u"], 4), round(b["zc"], 4), b["n"])
                                for b in bars],))
    if bars and m:
        ok &= abs(bars[0]["zc"] - float(m["bar_zc"])) < 0.005
        _log("  bar zc vs the plan: %.4f vs %.4f (4 mm z-buckets) -> %s" % (
            bars[0]["zc"], m["bar_zc"],
            "OK" if abs(bars[0]["zc"] - float(m["bar_zc"])) < 0.005 else "MISMATCH"))
        dz = (bars[0]["front_u"] - float(fr["S"]) * pf[0]) if pf else float("nan")
        ok &= abs(dz - float(m["dz_bar"])) < 0.004
        _log("  bar protrusion vs the plan's dz_bar: %.4f vs %.4f -> %s" % (
            dz, m["dz_bar"], "OK" if abs(dz - float(m["dz_bar"])) < 0.004 else "MISMATCH"))
    path2 = os.path.join(DATA, "final_clear.npz")
    if os.path.exists(path2):
        sc = rig.Scene.load(path2)
        off, n, home = rig.panel_home_on(sc, fr, float(m["ref_y"]), band)
        _log("  SHUT panel: off %+.4f n=%d home=%s (u_panel - u_ref)" % (
            off, n, home))
        ok &= bool(home)
        cnt = None
        if m.get("box_x"):
            cnt = rig.colour_class_count(sc, (224, 222, 223),
                                         (tuple(m["box_x"]), tuple(m["box_y"])),
                                         (m["table_z"] + 0.010, m["panel_z"][1]))
            _log("  interior px shut: %s (was %s when out)" % (cnt, m.get("floor_n0")))
            ok &= int(cnt) < 200
    _log("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return bool(ok)


def look(ctx):
    """0-row re-read of the drawer's plane state (free, and the frame for the eye)."""
    tag = os.environ.get("TAG", "look")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    m = _load()
    S = m["S"]
    _log("snapshot %s eef=%s quat=%s gq=%s" % (tag, np.round(s.eef, 4),
                                               np.round(s.quat, 4), np.round(s.gq, 4)))
    ps = panel_now(s, S, m["panel_band"], m["xwin"], m["ywin"], ref_y=m["ref_y"])
    _log("panel_now %s" % (ps,))
    _log("floor_visible %d (was %d)" % (
        floor_visible(s, m["box_x"], m["box_y"], m["table_z"] + 0.010, m["panel_z"][1]),
        m["floor_n0"]))
    return s


# ============================================================== close
def close(ctx):
    m = _load()
    if m.get("already_closed"):
        _log("close: the probe found nothing out - already home")
        return True
    S = float(m["S"])
    band = tuple(m["panel_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    s = rig.snapshot(ctx, tag="close_a", outdir=DATA)
    ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
    _log("close start: panel_now %s" % (ps,))
    panel_y = ps["panel_y"] if ps["panel_y"] is not None else m["open_panel_y"]
    off = uu(S, panel_y) - uu(S, m["ref_y"])
    _log("panel y=%+.4f -> %.4f m still to travel (reference %+.4f)" % (
        panel_y, off, m["ref_y"]))
    if off <= HOME_TOL:
        _log("already home")
        return True
    bar_front = panel_y + S * float(m["dz_bar"])      # the bar moves only along y
    zbc = float(m["bar_zc"])
    x_t = float(m["bar_x_mid"])
    q_push = np.array(m["q_push"], dtype=np.float64)
    if m["mode"] == "hand":
        return _close_hand(ctx, m, panel_y, bar_front, x_t, band, xwin, ywin)
    return _close_tips(ctx, m, panel_y, bar_front, x_t, zbc, q_push, band, xwin, ywin)


def _close_hand(ctx, m, panel_y, bar_front, x_t, band, xwin, ywin):
    """SKILL §7 with the hand's face on the bar - only reachable when the bar is high."""
    S = float(m["S"])
    z_push = float(m["hand_z"])
    y_out = float(bar_front) + S * 0.16
    _log("HAND mode: eef_z %.4f (face %.4f.. on the bar band %.4f..%.4f) x %.4f" % (
        z_push, z_push + HAND_Z0, m["bar_z0"], m["bar_z1"], x_t))
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1], max(p[2], m["bar_z1"] + 0.20)], QUAT_DOWN, -1.0, tol=0.010,
             max_rows=12, tag="clear ")
    rig.goto(ctx, [x_t, y_out, z_push + 0.12], QUAT_DOWN, -1.0, tol=0.008, max_rows=22,
             tag="pre-y ")
    rig.goto(ctx, [x_t, y_out, z_push], QUAT_DOWN, -1.0, tol=0.006, max_rows=14, tag="pre-z ")
    # rig.push_to_stop drives -y only; mirror by looping here
    payload = rig.observe(ctx)
    dead = 0
    for i in range(36):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -S
        a[0] = float(np.clip((x_t - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((z_push - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), QUAT_DOWN) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        ps = panel_now(rig.snapshot(ctx, tag="push_i", outdir=DATA), S, band, xwin, ywin,
                       ref_y=m["ref_y"]) if (i % 4 == 0 or dead >= 3) else None
        if ps is not None:
            _log("  push %2d eef=%s moved=%.4f dead=%d panel %s off %s home=%s" % (
                i, np.round(q, 4), moved, dead,
                None if ps["panel_y"] is None else round(ps["panel_y"], 4),
                None if ps["off"] is None else round(ps["off"], 4), ps["home"]))
            if ps["home"]:
                break
        if dead >= 3:
            _log("  push STALLED at eef=%s" % np.round(q, 4))
            break
    steps_left(ctx, "(after push)")
    return True


def _close_tips(ctx, m, panel_y, bar_front, x_t, zbc, q_push, band, xwin, ywin):
    """The LOW-drawer recipe: pitched fingertips at z = bar_zc - 0.031, push home, stop on
    the drawer's own front plane (a push has no force signal)."""
    S = float(m["S"])
    z_push = zbc + TIPS_DZ
    _log("TIPS mode: eef_z %.4f (bar %.4f..%.4f, zc %.4f), x %.4f, bar front y=%+.4f" % (
        z_push, m["bar_z0"], m["bar_z1"], zbc, x_t, bar_front))

    # 1. LEVEL first, at the CURRENT xy and folded up (SKILL §2: "level -> rise -> cross ->
    #    descend", and an orientation error stalls every axis; a rotation-only row still
    #    walks the arm, so every later target is re-aimed from the MEASURED pose)
    p = rig.pose_of(rig.observe(ctx))
    jp = rig.observe(ctx).get("observation", {}).get("joint_pos")
    _log("joints at the start of the approach: %s" % (None if jp is None else np.round(jp, 3),))
    rig.goto(ctx, [p[0], p[1], max(p[2], 1.34)], QUAT_DOWN, -1.0, tol=0.010, max_rows=16,
             tag="fold ")
    rotate_in_place(ctx, q_push, -1.0, max_rows=40, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    for k in range(3):
        e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q_push)))
        if e < 0.06:
            break
        _log("  rot_err %.3f still large - another rotation-only pass" % e)
        rotate_in_place(ctx, q_push, -1.0, max_rows=20, tol=0.04)
    _log("after rotate eef=%s rot_err=%.3f" % (
        np.round(rig.pose_of(rig.observe(ctx)), 4),
        float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q_push)))))

    # 2. CROSS at the folded height, then descend vertically in two stages (a diagonal in
    #    -x re-stretches the elbow and stalls: SKILL §2 "short reach near the base")
    y_hover = float(bar_front) + S * 0.11          # 11 cm OUT of the bar's front face
    rig.goto(ctx, [x_t, y_hover, max(1.30, z_push + 0.30)], q_push, -1.0, tol=0.008,
             max_rows=26, tag="cross ")
    p = rig.pose_of(rig.observe(ctx))
    _log("crossed to eef=%s (residual %s)" % (
        np.round(p, 4), np.round(np.array([x_t, y_hover, 1.30]) - p, 4)))
    rig.goto(ctx, [x_t, y_hover, z_push + 0.10], q_push, -1.0, tol=0.008, max_rows=18,
             tag="desc1 ")
    rig.goto(ctx, [x_t, y_hover, z_push], q_push, -1.0, tol=0.006, max_rows=14,
             tag="desc2 ")
    p = rig.pose_of(rig.observe(ctx))
    _log("pre-push eef=%s ; aim (%.4f, %.4f, %.4f) ; z_push %.4f" % (
        np.round(p, 4), x_t, y_hover, z_push, z_push))

    # 3. BUY THE VERDICT for 0 rows: where is the fingertip against the bar's own band?
    s = rig.snapshot(ctx, tag="prepush", outdir=DATA)
    prof = finger_profile(s, S, x_t, zbc, bar_front)
    _log("finger_profile: n_all=%s z_all=%s n_front=%s" % (
        prof.get("n_all"), prof.get("z_all"), prof.get("n_front")))
    if "u_tip" in prof:
        _log("  fingertip: u=%+.4f (lead %+.4f), z %s, x %s" % (
            prof["u_tip"], prof["lead_vs_bar"], prof.get("z_tip"), prof.get("x_tip")))
    for r in prof.get("rows", []):
        _log("   z %+.3f n=%4d  u_max %+.4f" % r)
    if "z_tip" in prof:
        z_tip = 0.5 * (prof["z_tip"][0] + prof["z_tip"][1])
        dz = zbc - z_tip
        _log("  fingertip centre z %.4f vs bar zc %.4f -> dz %+.4f" % (z_tip, zbc, dz))
        if abs(dz) > 0.008:
            z_new = z_push + float(np.clip(dz, -0.030, 0.030))
            _log("  re-aiming z to %.4f (%.4f above the table)" % (
                z_new, z_new - float(m["table_z"])))
            rig.goto(ctx, [x_t, y_hover, z_new], q_push, -1.0, tol=0.005, max_rows=12,
                     tag="tip-z ")
            z_push = z_new
        xt = prof.get("x_tip")
        if xt is not None:
            xc = 0.5 * (xt[0] + xt[1])
            dx = float(m["bar_x_mid"]) - xc
            if abs(dx) > 0.008:
                x_new = x_t + float(np.clip(dx, -0.03, 0.03))
                _log("  tips span x %.4f..%.4f (centre %.4f) vs bar mid %.4f -> x %+.4f"
                     " (pads straddle the bar: it is only %.3f m long)" % (
                         xt[0], xt[1], xc, m["bar_x_mid"], dx, m["bar_x1"] - m["bar_x0"]))
                rig.goto(ctx, [x_new, y_hover, z_push], q_push, -1.0, tol=0.005,
                         max_rows=10, tag="tip-x ")
                x_t = x_new
    else:
        _log("  no fingertip pixels OUT of the bar - the fingers may already be at the bar;"
             " pushing from here")

    # 4. the push: saturated inward row, x/z servoed, quaternion servoed, stop on the plane
    payload = rig.observe(ctx)
    dead, home = 0, False
    n_rows = int(os.environ.get("ROWS", 44))
    for i in range(n_rows):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -S
        a[0] = float(np.clip((x_t - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((z_push - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_push) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if i % 4 == 0 or dead >= 3:
            s = rig.snapshot(ctx, tag="push_i", outdir=DATA)
            ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
            _log("  push %2d eef=%s moved=%.4f dead=%d panel %s off %s home=%s ; bins %s" % (
                i, np.round(q, 4), moved, dead,
                None if ps["panel_y"] is None else round(ps["panel_y"], 4),
                None if ps["off"] is None else round(ps["off"], 4), ps["home"],
                ps["rows"][:3]))
            if ps["home"]:
                _log("  panel is HOME after %d rows" % i)
                home = True
                break
        if dead >= 3:
            _log("  push STALLED at eef=%s after %d rows" % (np.round(q, 4), i))
            break

    # 5. one +z row: a freeze rises ~0, a real contact rises a few mm (SKILL §7)
    p = rig.pose_of(rig.observe(ctx))
    payload, _ = rig.step(ctx, [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0]], verbose=False)
    rose = float(rig.pose_of(payload)[2] - p[2])
    _log("  stall check: +z row rose %.4f m -> %s" % (
        rose, "FROZEN (controller), not the drawer" if abs(rose) < 0.002
        else "real contact with something"))
    if abs(rose) >= 0.002:
        p = rig.pose_of(payload)
        rig.step(ctx, [[0.0, 0.0, float(np.clip((z_push - p[2]) / 0.010, -1, 1)),
                        0.0, 0.0, 0.0, -1.0]], verbose=False)
    _log("close rows done; home(plane)=%s" % home)
    steps_left(ctx, "(after push)")
    return home


# ============================================================== verdict
def verify(ctx, tag="verify0"):
    m = _load()
    S = float(m["S"])
    band = tuple(m["panel_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    p = rig.pose_of(rig.observe(ctx))
    q_hold = np.array(m["q_push"], dtype=np.float64)
    _log("retreat from eef=%s : OUT %.2f first (the fingers may be inside the cabinet),"
         " then +z" % (np.round(p, 4), 0.24))
    # pull straight out along -u first: the fingers point INWARD, so this slides them out
    # of the pocket without dragging the shut drawer, and keeps them below its bar
    rig.goto(ctx, [p[0], p[1] + S * 0.24, p[2]], q_hold, -1.0, tol=0.012, max_rows=18,
             tag="retreat out ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1], max(1.20, p[2] + 0.20)], q_hold, -1.0, tol=0.012, max_rows=14,
             tag="retreat z ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0] + 0.06, p[1] + S * 0.08, p[2]], q_hold, -1.0, tol=0.012, max_rows=10,
             tag="retreat y ")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
    _log("SIGNAL 1 (the pushed panel's own plane): y=%s n=%d off %s m home=%s ; bins %s" % (
        None if ps["panel_y"] is None else round(ps["panel_y"], 4), ps["n"],
        None if ps["off"] is None else round(ps["off"], 4), ps["home"], ps["rows"][:5]))
    _log("           reference (the closed fronts) y=%+.4f ; travel was %.4f m" % (
        m["ref_y"], m["travel"]))
    fv = floor_visible(s, m["box_x"], m["box_y"], m["table_z"] + 0.010, m["panel_z"][1])
    _log("SIGNAL 2 (drawer interior visible): %d bright px (was %d when out) -> %s" % (
        fv, m["floor_n0"],
        "interior INVISIBLE -> shut" if fv < max(200, 0.25 * m["floor_n0"])
        else "interior still visible -> still out"))
    # the bar must be back inside the closed-fronts window
    bar = find_bar(s, S, m["ref_y"], m["panel_z"], xwin)
    _log("SIGNAL 3 (bar back on the closed plane): %s" % (
        None if bar is None else "front y=%+.4f z %.4f..%.4f n=%d" % (
            bar["front_y"], bar["z"][0], bar["z"][1], bar["n"]),))
    if bar is not None:
        exp = float(m["ref_y"]) + S * float(m["dz_bar"])
        _log("           a bar on the closed plane sits at y=%+.4f (reference %.4f %+.4f) ;"
             " measured %+.4f" % (exp, m["ref_y"], S * m["dz_bar"], bar["front_y"]))
        du = uu(S, bar["front_y"]) - uu(S, m["ref_y"])      # a CLOSED bar is dz_bar OUT
        _log("           the bar front is %+.4f m (u) from the reference; a closed bar is"
             " %.4f -> %s (error %+.4f)" % (
                 du, m["dz_bar"],
                 "home" if abs(du - float(m["dz_bar"])) < 0.005 else "NOT where a closed bar is",
                 du - float(m["dz_bar"])))
    ok = bool(ps["home"] is True and fv < max(200, 0.25 * m["floor_n0"]))
    _log("VERDICT: %s  (panel off home %s ; interior px %d vs %d)" % (
        "CLOSED / HOME" if ok else "NOT HOME",
        None if ps["off"] is None else round(ps["off"], 4), fv, m["floor_n0"]))
    rig.save_frame(tag, outdir=DATA)
    m["verify_" + tag] = {"panel_y": ps["panel_y"], "off": ps["off"], "home": ps["home"],
                          "floor_px": int(fv), "floor_px0": int(m["floor_n0"]), "ok": ok}
    _save(m)
    steps_left(ctx, "(after %s)" % tag)
    return ok


def run(ctx):
    if not os.path.exists(PLAN):
        _log("=== PROBE ===")
        if not probe(ctx):
            return False
    _log("=== CLOSE ===")
    close(ctx)
    _log("=== VERIFY ===")
    ok = verify(ctx, tag="verify1")
    if not ok:
        _log("not home yet - one more measured push")
        close(ctx)
        ok = verify(ctx, tag="verify1b")
    return bool(ok)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    _log("ENTRY %s" % entry)
    t0 = time.time()
    fn = globals().get(entry)
    if fn is None or entry.startswith("_") or entry in ("rig",):
        raise SystemExit("unknown ENTRY %r" % entry)
    out = fn(ctx)
    _log("entry %s done in %.1fs -> %r" % (entry, time.time() - t0, out))
    return out
