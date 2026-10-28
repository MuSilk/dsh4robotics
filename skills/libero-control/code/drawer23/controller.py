"""LIBERO-90 task 23 - KITCHEN_SCENE4: CLOSE THE BOTTOM DRAWER, then OPEN THE TOP ONE.

Two clauses, one episode (1000 rows).  Both halves are the *measured* recipes of SKILL.md and
tasks/close-drawer.md / tasks/open-drawer.md, mirrored onto this chest.

THE MIRROR.  `probe` measures the chest with `rig.drawer_axis`: on this asset (task 22 measured
it, and `probe` re-measures it every draw) the fronts are the +y face of the chest, so the
drawers open toward **-y** (`S = -1`) and closing is a push in **+y**.  Everything is written in
the out-coordinate `u = S*y` (more u = further out) so the +y-only relations of SKILL §7 mirror
untouched:

    open bottom panel   u_open  > u_ref  (out)      panel plane the close must drive to u_ref
    handle bar          u_bar = u_panel + dz_bar (~0.031), one per drawer, LOWEST = bottom
    press onto a bar    u_bar - 0.015  (i.e. 1.5 cm INTO the cabinet from the bar's front)
    hook orientation    q_push = rig.hook_quat_toward((0, -S, 0))  -> yaw(Q_HOOK, 180) here
    pull profile        dy = S*0.85 (i.e. -0.85), dz = -0.35, jaws OPEN

THE CLOSE (bottom drawer, low, so the hand-face recipe is impossible: it needs
eef_z = bar_z0 - 0.094 = 0.90 with the table at 0.95).  Recipe that PASSED on this exact asset
(task 22, 155 rows): fold -> rotate in place to the hook pitch -> cross at z 1.30 -> descend in
two stages -> read the FINGER TIP against the bar's own band (0 rows) -> saturated push with the
panel plane sampled every 4 rows -> stop when the drawer's OWN front plane is home -> one +z row
to prove the stall was the drawer and not a freeze -> retreat OUT first, then up, then the
4-signal verdict (panel plane, interior pixels collapsed, bar back on the closed plane).

THE OPEN (top drawer).  A wrist-down grasp of a bar is geometrically impossible (the hand
block reaches 0.111 m ahead of eef_pos with its underside at eef_z+0.082, so a pad-band pose on
a bar buries the hand's face in the drawer front and every descent dies with all axes dead while
droll still turns).  So the fingers HOOK the bar with the jaws open and the wrist pitched:
DIAGONAL press to (bar_x_mid, u_bar - 0.015, bar_z0 - 0.045) - five millimetres is the whole
height margin and it fails UPWARD only - then 6 rows of dy S*0.85 / dz -0.35 as the PROBE (its
`move` and its blocked `dz_actual` are the only in-loop contact evidence a drag has), then the
same profile in 8-row bursts, checking the drawer's own front plane after each, to the stop
(0.16 m of travel; "open enough" is the full range, not 70 %).

Stages (each re-runnable on the same episode; ENTRY=run chains them):

    ENTRY=probe    0 rows: the whole chest - axis + OUT sign, fronts' plane, the open panel and
                   its band, its bar, the closed drawers' bars (the TOP one = max zc), the top
                   drawer's band, the interior colour/count, the rig checks
    ENTRY=close    rotate in place -> cross -> descend -> read the fingertip (0 rows) -> push
    ENTRY=cverify  retreat OUT then up, the 4-signal close verdict
    ENTRY=openhook rotate/aim, read the fingertips on the TOP bar (0 rows), diagonal press,
                   6-row biased probe
    ENTRY=pull     pull bursts to the stop (re-entrant; re-presses if the fingers slipped)
    ENTRY=overify  retreat, the opening verdict (panel out >= 0.14 m in the top band)
    ENTRY=run      probe|close|cverify|openhook|pull|overify
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
PLAN = os.path.join(DATA, "drawer23.json")

QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])
Q_HOOK = rig.Q_HOOK
TIPS_DZ = -0.031          # close with the tips: eef_z = bar_zc + TIPS_DZ (task 6 / task 22)
PRESS_DZ = rig.HOOK_PRESS_DZ     # open: eef_z = bar_z0 - 0.045  (5 mm window, fails upward)
PRESS_DY = 0.015          # press 1.5 cm INTO the cabinet past the bar's front face
DZ_BAR_TYP = 0.031        # bar proud of its panel (~0.032 on the other chest, 0.0316 here)
OPEN_OK = 0.140           # "open" = the panel is at least this far out of its home plane
HOME_TOL = 0.0025


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


def uu(S, y):
    """The out-coordinate: more u = further OUT of the cabinet."""
    return float(S) * float(y)


# ============================================================== measurement helpers (0 rows)
def face_extent(s, y0, band, xwin, tol=0.0018, nmin=0.85):
    """(z_lo,z_hi,x_lo,x_hi,n,mean_n) of the pixels on the y=y0 face - the panel's own band."""
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


def find_bar(s, S, panel_y, band_z, xwin, min_px=120, reach=0.075, drop_arm=True):
    """The handle bar attached to `panel_y`, z-bucketed.  A bar is achromatic (mean 70-170),
    ~3 cm proud of its panel and ~1 cm tall, hence inside `Scene.arm` - a colour+geometry test."""
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    rel = S * (s.Y - float(panel_y))
    m = (((mx - mn) < 25) & (mean > 70) & (mean < 170)
         & (rel > 0.004) & (rel < float(reach))
         & (s.Z > float(band_z[0])) & (s.Z < float(band_z[1]))
         & (s.X > xwin[0]) & (s.X < xwin[1]))
    if s.eef is not None and drop_arm:
        m &= (np.linalg.norm(s.pts - s.eef, axis=2) > 0.10)
    if m.sum() < min_px:
        return None
    edges = np.arange(float(band_z[0]), float(band_z[1]) + 1e-9, 0.004)
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
                "front_u": float(np.percentile(S * p[:, 1], 98)),
                "z": (float(p[:, 2].min()), float(p[:, 2].max())),
                "zc": float(0.5 * (p[:, 2].min() + p[:, 2].max())),
                "x": (float(np.percentile(p[:, 0], 2.0)),
                      float(np.percentile(p[:, 0], 98.0))),
                "mean": float(mean[sel].mean())}
        cand["front_y"] = float(S * cand["front_u"])
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def bar_hist(s, S, panel_y, band_z, xwin, reach=0.075, drop_arm=True, bin_mm=4.0):
    """Raw achromatic-z histogram in front of `panel_y` - to SEE whether two bars merged."""
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    rel = S * (s.Y - float(panel_y))
    m = (((mx - mn) < 25) & (mean > 70) & (mean < 170)
         & (rel > 0.004) & (rel < float(reach))
         & (s.Z > float(band_z[0])) & (s.Z < float(band_z[1]))
         & (s.X > xwin[0]) & (s.X < xwin[1]))
    if s.eef is not None and drop_arm:
        m &= (np.linalg.norm(s.pts - s.eef, axis=2) > 0.10)
    out = []
    z0 = float(band_z[0])
    while z0 < float(band_z[1]):
        sel = m & (s.Z >= z0) & (s.Z < z0 + 0.001 * bin_mm)
        if sel.sum() >= 4:
            p = s.pts[sel]
            out.append((round(z0, 3), int(sel.sum()),
                        round(float(np.percentile(S * p[:, 1], 98)), 4),
                        round(float(p[:, 0].min()), 3), round(float(p[:, 0].max()), 3)))
        z0 += 0.001 * bin_mm
    return out


def ybin_scene(s, band, xwin, ywin, min_pixels=60):
    return rig.face_bins(s, 1, (float(band[0]), float(band[1])),
                         ((float(xwin[0]), float(xwin[1])), (float(ywin[0]), float(ywin[1]))),
                         min_pixels=int(min_pixels))


def panel_now(s, S, band, xwin, ywin, ref_y=None, min_pixels=150):
    """(OUT-most panel plane in this band, its count, offset from `ref_y`, home?) - 0 rows."""
    rows = ybin_scene(s, band, xwin, ywin, min_pixels=60)
    out = {"rows": [(round(float(a), 4), int(b)) for a, b in rows[:8]],
           "panel_y": None, "n": 0, "off": None, "home": None}
    if not rows:
        return out
    cand = [r for r in rows if r[1] >= min_pixels] or rows[:1]
    y, n = max(cand, key=lambda t: S * t[0])
    out["panel_y"], out["n"] = float(y), int(n)
    if ref_y is not None:
        out["off"] = uu(S, y) - uu(S, ref_y)
        out["home"] = bool(-0.006 <= out["off"] <= HOME_TOL)
    return out


def floor_visible(s, xwin, ywin, zlo, zhi):
    """Bright interior pixels (the box's white floor) inside the window - the close signal 2.
    mean > 200 with chroma < 20 excludes the wooden table (mean ~182, chroma ~35)."""
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    m = ((mean > 195) & ((mx - mn) < 22) & (s.Z > float(zlo)) & (s.Z < float(zhi))
         & (s.X > xwin[0]) & (s.X < xwin[1]) & (s.Y > ywin[0]) & (s.Y < ywin[1]))
    return int(m.sum())


def finger_profile(s, S, x, bar_zc, bar_front_y, half_x=0.075, dz_lo=-0.050, dz_hi=0.075):
    """Where the FINGERS are, in the pitched pose.  0 rows, one depth call.

    The out-most arm pixel in front of the bar is the fingertip (at the hook pitch the fingers
    lead and the hand block sits behind and above them), so its z IS the contact height.
    `S*(y - bar_front) > 0.004` keeps the bar itself (achromatic -> arm) out of the mask."""
    rel = S * (s.Y - float(bar_front_y))
    m = (s.arm & (np.abs(s.X - float(x)) < half_x)
         & (s.Z > float(bar_zc) + dz_lo) & (s.Z < float(bar_zc) + dz_hi))
    out = {"n_all": int(m.sum())}
    if m.sum() < 60:
        return out
    p = s.pts[m]
    out["z_all"] = (round(float(p[:, 2].min()), 4), round(float(p[:, 2].max()), 4))
    f = m & (rel > 0.004)
    out["n_front"] = int(f.sum())
    if f.sum() >= 40:
        q = s.pts[f]
        umax = float(np.max(S * q[:, 1]))
        tip = q[S * q[:, 1] > umax - 0.006]
        out["u_tip"] = round(umax, 4)
        out["lead_vs_bar"] = round(umax - uu(S, bar_front_y), 4)
        out["z_tip"] = (round(float(np.percentile(tip[:, 2], 5)), 4),
                        round(float(np.percentile(tip[:, 2], 95)), 4))
        out["x_tip"] = (round(float(tip[:, 0].min()), 4), round(float(tip[:, 0].max()), 4))
    rows = []
    for z0 in np.arange(float(bar_zc) + dz_lo, float(bar_zc) + dz_hi, 0.008):
        q = p[(p[:, 2] >= z0) & (p[:, 2] < z0 + 0.008)]
        if len(q) < 20:
            continue
        rows.append((round(float(z0), 3), int(len(q)),
                     round(float(np.max(S * q[:, 1])), 4)))
    out["rows"] = rows
    return out


# ============================================================== motion helpers
def rotate_in_place(ctx, q_tgt, gripper=-1.0, max_rows=26, tol=0.04, verbose=True):
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


def level_then_rise(ctx, q_push, z_up=1.34):
    """SKILL §2's order: LEVEL -> RISE at the current xy.  A rotation-only row still walks the
    arm 4-8 cm, so after it every later target is re-aimed from the MEASURED pose."""
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1], max(p[2], z_up)], QUAT_DOWN, -1.0, tol=0.010, max_rows=16,
             tag="fold ")
    rotate_in_place(ctx, q_push, -1.0, max_rows=44, tol=0.04)
    for _ in range(3):
        e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q_push)))
        if e < 0.06:
            break
        _log("  rot_err %.3f still large - another rotation-only pass" % e)
        rotate_in_place(ctx, q_push, -1.0, max_rows=20, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    _log("after rotate eef=%s rot_err=%.3f" % (
        np.round(p, 4),
        float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q_push)))))
    return p


def approach(ctx):
    """TEST/DIAG entry: cross high to (XT, YT) then descend to ZT with stall reporting.

    The measured table is crowded on this draw (a speckled bowl at (0.018,-0.078) r 0.054 top
    1.0006, a bottle, two leaning boards), and the first tips descent stalled at eef_z 1.109
    with NOTHING in the depth under the hand - so the corridor is chosen empirically here and
    the numbers are printed rather than assumed.
    """
    m = _load()
    S = float(m["S"])
    q = np.array(m["q_push"], dtype=np.float64)
    x = float(os.environ.get("XT", -0.035))
    y = float(os.environ.get("YT", -0.01))
    z = float(os.environ.get("ZT", m["tips_z"]))
    rotate_in_place(ctx, q, -1.0, max_rows=12, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    _log("approach from eef=%s -> (%.4f, %+.4f, %.4f)" % (np.round(p, 4), x, y, z))
    if p[2] < 1.15:
        rig.goto(ctx, [p[0], p[1], 1.22], q, -1.0, tol=0.008, max_rows=14, tag="rise ")
        p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [x, y, max(p[2], 1.20)], q, -1.0, tol=0.010, max_rows=22, tag="cross ")
    p = rig.pose_of(rig.observe(ctx))
    _log("crossed: eef=%s residual=%s" % (np.round(p, 4),
                                          np.round(np.array([x, y, max(p[2], 1.20)]) - p, 4)))
    payload, ok = rig.goto(ctx, [x, y, z], q, -1.0, tol=0.005, max_rows=28, stall_rows=4,
                           tag="desc ")
    p = rig.pose_of(payload)
    _log("DESC end eef=%s residual=%s done=%s (z left %.4f)" % (
        np.round(p, 4), np.round(np.array([x, y, z]) - p, 4), ok, float(z - p[2])))
    s = rig.snapshot(ctx, tag="approach", outdir=DATA)
    ps = panel_now(s, S, tuple(m["panel_band"]), tuple(m["xwin"]), tuple(m["ywin"]),
                   ref_y=m["ref_y"])
    _log("panel_now %s" % (ps,))
    bf = float(ps["panel_y"] if ps["panel_y"] is not None else m["open_panel_y"]) + \
        S * float(m["bar"]["front_u"] - uu(S, m["open_panel_y"]))
    prof = finger_profile(s, S, float(p[0]), float(m["bar"]["zc"]), bf)
    _log("finger_profile n_all=%s z_all=%s n_front=%s" % (
        prof.get("n_all"), prof.get("z_all"), prof.get("n_front")))
    if "u_tip" in prof:
        _log("  fingertip u=%+.4f lead %+.4f z %s x %s" % (
            prof["u_tip"], prof["lead_vs_bar"], prof.get("z_tip"), prof.get("x_tip")))
    for r in prof.get("rows", []):
        _log("   z %+.3f n=%4d u_max %+.4f" % r)
    m["approach_eef"] = [float(v) for v in p]
    m["approach_ok"] = bool(ok)
    _save(m)
    steps_left(ctx, "(after approach)")
    return ok


def push_from_here(ctx):
    """Push the bottom drawer home from the CURRENT pose (whatever z that is)."""
    m = _load()
    S = float(m["S"])
    band = tuple(m["panel_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    q = np.array(m["q_push"], dtype=np.float64)
    p0 = rig.pose_of(rig.observe(ctx))
    x_default = 0.5 * (float(m["bar"]["x"][0]) + float(m["bar"]["x"][1]))
    x_t = float(os.environ.get("XT", x_default))
    z_t = float(os.environ.get("ZT", p0[2]))
    _log("push from eef=%s ; hold x=%.4f z=%.4f, saturated +y" % (np.round(p0, 4), x_t, z_t))
    payload = rig.observe(ctx)
    dead, home = 0, False
    for i in range(int(os.environ.get("ROWS", 40))):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -S
        a[0] = float(np.clip((x_t - p[0]) / 0.010, -0.5, 0.5))
        a[2] = float(np.clip((z_t - p[2]) / 0.010, -0.5, 0.5))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        qq = rig.pose_of(payload)
        moved = float(np.max(np.abs(qq - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if i % 4 == 0 or dead >= 3:
            ps = panel_now(rig.snapshot(ctx, tag="push2_i", outdir=DATA), S, band, xwin, ywin,
                           ref_y=m["ref_y"])
            _log("  push %2d eef=%s moved=%.4f dead=%d panel %s off %s home=%s" % (
                i, np.round(qq, 4), moved, dead,
                None if ps["panel_y"] is None else round(ps["panel_y"], 4),
                None if ps["off"] is None else round(ps["off"], 4), ps["home"]))
            if ps["home"]:
                home = True
                _log("  panel HOME after %d rows" % i)
                break
        if dead >= 3:
            _log("  STALLED at eef=%s after %d rows" % (np.round(qq, 4), i))
            break
    steps_left(ctx, "(after push2)")
    return home


def descend_smart(ctx, x, y, z_target, q, step=0.030, max_steps=8, tag="ds"):
    """Descend in `step` slices, RE-LEVELLING the wrist between slices (SKILL §2).

    Measured on this draw: a straight `goto` down 0.23 m let `rot_err` grow 0.026 -> 0.120 and
    the arm slowed 13 mm/row -> 0.6 mm/row and stopped 10 cm above the target -- with nothing
    in the depth under the hand. That is the OSC trading orientation for position, not a
    contact. So: descend, then rotate in place back to `q`, repeat, and only believe a stall
    whose `rot_err` is already < 0.05.
    """
    payload = rig.observe(ctx)
    z = float(rig.pose_of(payload)[2])
    for k in range(int(max_steps)):
        if z <= float(z_target) + 0.004:
            break
        zk = max(float(z_target), z - float(step))
        payload, ok = rig.goto(ctx, [float(x), float(y), zk], q, -1.0, tol=0.004,
                               max_rows=10, stall_rows=4, tag="%s%d " % (tag, k))
        p = rig.pose_of(payload)
        e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(payload), q)))
        _log("  slice %d -> eef=%s (dz %+.4f, rot_err %.3f)" % (
            k, np.round(p, 4), float(p[2] - z), e))
        moved = abs(float(p[2] - z))
        z = float(p[2])
        if moved < 0.002:
            _log("  no descent in slice %d (rot_err %.3f) - re-levelling" % (k, e))
        rotate_in_place(ctx, q, -1.0, max_rows=10, tol=0.04)
        payload = rig.observe(ctx)
        p = rig.pose_of(payload)
        if abs(float(p[2] - z)) > 0.004:
            _log("  !! the rotation-only pass walked z by %+.4f" % float(p[2] - z))
        z = float(p[2])
    return payload, (z <= float(z_target) + 0.010)


def desc2(ctx):
    """Continue the tips descent from the CURRENT xy to tips_z, re-levelling every 3 cm."""
    m = _load()
    S = float(m["S"])
    q = np.array(m["q_push"], dtype=np.float64)
    x = float(os.environ.get("XT", m["approach_eef"][0] if m.get("approach_eef") else -0.035))
    y = float(os.environ.get("YT", -0.030))
    z = float(os.environ.get("ZT", m["tips_z"]))
    p0 = rig.pose_of(rig.observe(ctx))
    _log("desc2 from eef=%s -> (%.4f, %+.4f, %.4f)" % (np.round(p0, 4), x, y, z))
    rig.goto(ctx, [x, y, p0[2]], q, -1.0, tol=0.006, max_rows=12, tag="align ")
    payload, ok = descend_smart(ctx, x, y, z, q)
    p = rig.pose_of(payload)
    _log("desc2 end eef=%s residual=%s ok=%s" % (np.round(p, 4),
                                                 np.round(np.array([x, y, z]) - p, 4), ok))
    s = rig.snapshot(ctx, tag="desc2", outdir=DATA)
    ps = panel_now(s, S, tuple(m["panel_band"]), tuple(m["xwin"]), tuple(m["ywin"]),
                   ref_y=m["ref_y"])
    _log("panel_now %s" % (ps,))
    bf = float(ps["panel_y"] if ps["panel_y"] is not None else m["open_panel_y"]) + \
        S * (float(m["bar"]["front_u"]) - uu(S, m["open_panel_y"]))
    _log("bar front now y=%+.4f (panel + dz_bar)" % bf)
    prof = finger_profile(s, S, float(p[0]), float(m["bar"]["zc"]), bf)
    if "u_tip" in prof:
        _log("  fingertip u=%+.4f lead %+.4f z %s x %s" % (
            prof["u_tip"], prof["lead_vs_bar"], prof.get("z_tip"), prof.get("x_tip")))
    for r in prof.get("rows", []):
        _log("   z %+.3f n=%4d u_max %+.4f" % r)
    steps_left(ctx, "(after desc2)")
    return ok


def desc3(ctx):
    """THE CORRIDOR THAT FITS THIS DRAW: down in front of the leaning boards, out, then down.

    Measured on this episode: a vertical tips descent with the hook pitch stalls at
    eef_z 1.109 (hover y -0.068) / 1.067 (y -0.010) with `rot_err` climbing to 0.12 and the
    Y DRIFTING +y as it stops, and nothing at all under the fingers in the depth. The blocking
    part is the HAND BLOCK's far corner, which at this pitch trails 0.19 m BEHIND (-y) and
    ~0.015 m BELOW the eef, and the two leaning cutting boards in front of this drawer
    (brown, top edge z ~1.04-1.08 at y -0.19..-0.24, x -0.15..+0.02). Below that edge the
    boards lean away over the table, so a low horizontal move passes UNDER them.

    Legs: (1) descend at y = +0.03 -> the palm stays at y -0.16, in front of the boards;
    (2) move OUT to y = -0.03 at z 1.045, where the fingers are still above the drawer front's
    top (1.0385) and the palm is under the boards; (3) descend to the tips height, the palm
    now 10 cm under the boards' surface; then the +y push.
    """
    m = _load()
    S = float(m["S"])
    q = np.array(m["q_push"], dtype=np.float64)
    x = float(os.environ.get("XT", -0.035))
    y_hi = float(os.environ.get("YHI", 0.030))
    y_lo = float(os.environ.get("YLO", -0.030))
    z_mid = float(os.environ.get("ZMID", 1.045))
    z = float(os.environ.get("ZT", m["tips_z"]))
    p = rig.pose_of(rig.observe(ctx))
    _log("desc3 from eef=%s -> (x %.4f, y_hi %+.4f @ z %.4f, y_lo %+.4f @ z %.4f)" % (
        np.round(p, 4), x, y_hi, z_mid, y_lo, z))
    # level, and get to (x, y_hi, z_mid) from wherever we are
    rotate_in_place(ctx, q, -1.0, max_rows=12, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    if abs(p[1] - y_hi) > 0.010 or abs(p[0] - x) > 0.010 or p[2] < z_mid - 0.02:
        if p[2] < z_mid + 0.10:
            rig.goto(ctx, [p[0], p[1], z_mid + 0.12], q, -1.0, tol=0.008, max_rows=12,
                     tag="rise ")
        rig.goto(ctx, [x, y_hi, z_mid + 0.12], q, -1.0, tol=0.008, max_rows=20, tag="cross ")
        rig.goto(ctx, [x, y_hi, z_mid], q, -1.0, tol=0.005, max_rows=10, tag="leg1 ")
    p = rig.pose_of(rig.observe(ctx))
    _log("leg1 eef=%s (aim %.4f, %+.4f, %.4f)" % (np.round(p, 4), x, y_hi, z_mid))
    # leg 2: OUT at z_mid
    for k in range(6):
        p = rig.pose_of(rig.observe(ctx))
        if abs(p[1] - y_lo) < 0.006:
            break
        rig.goto(ctx, [x, y_lo, z_mid], q, -1.0, tol=0.005, max_rows=8, tag="leg2%d " % k)
        e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q)))
        if e > 0.06:
            rotate_in_place(ctx, q, -1.0, max_rows=8, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    _log("leg2 eef=%s (aim %.4f, %+.4f, %.4f) residual %s" % (
        np.round(p, 4), x, y_lo, z_mid, np.round(np.array([x, y_lo, z_mid]) - p, 4)))
    # leg 3: down to the tips height
    payload, ok = descend_smart(ctx, x, y_lo, z, q, step=0.025)
    p = rig.pose_of(payload)
    _log("leg3 eef=%s residual %s ok=%s" % (np.round(p, 4),
                                            np.round(np.array([x, y_lo, z]) - p, 4), ok))
    s = rig.snapshot(ctx, tag="desc3", outdir=DATA)
    ps = panel_now(s, S, tuple(m["panel_band"]), tuple(m["xwin"]), tuple(m["ywin"]),
                   ref_y=m["ref_y"])
    _log("panel_now %s" % (ps,))
    bf = float(ps["panel_y"] if ps["panel_y"] is not None else m["open_panel_y"]) + \
        S * (float(m["bar"]["front_u"]) - uu(S, m["open_panel_y"]))
    _log("bar front now y=%+.4f" % bf)
    prof = finger_profile(s, S, float(p[0]), float(m["bar"]["zc"]), bf)
    for r in prof.get("rows", []):
        _log("   z %+.3f n=%4d u_max %+.4f" % r)
    if "u_tip" in prof:
        _log("  out-most arm: u=%+.4f lead %+.4f z %s x %s" % (
            prof["u_tip"], prof["lead_vs_bar"], prof.get("z_tip"), prof.get("x_tip")))
    steps_left(ctx, "(after desc3)")
    return ok


def pressdiag(ctx):
    """Re-press the TOP bar at the measured target and READ the pose (0 rows of measurement)."""
    m = _load()
    S = float(m["S"])
    q = np.array(m["q_push"], dtype=np.float64)
    top = m["top"]
    x = float(os.environ.get("XT", top["x_mid"]))
    zp = float(os.environ.get("ZPRESS", m["press_z"]))
    yy = float(os.environ.get("YPRESS", m["press_y"]))
    _log("pressdiag: aim (%.4f, %+.4f, %.4f) ; bar z %.4f..%.4f front y=%+.4f" % (
        x, yy, zp, top["z0"], top["z1"], top["front_y"]))
    rotate_in_place(ctx, q, -1.0, max_rows=10, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [x, p[1], max(p[2], zp + 0.10)], q, -1.0, tol=0.008, max_rows=12, tag="up ")
    rig.goto(ctx, [x, float(top["front_y"]) - S * 0.16, zp + 0.06], q, -1.0, tol=0.008,
             max_rows=22, tag="pre ")
    payload, _ = rig.goto(ctx, [x, yy, zp], q, -1.0, tol=0.004, max_rows=26, stall_rows=4,
                          tag="press ")
    p = rig.pose_of(payload)
    _log("PRESS at eef=%s residual=%s" % (np.round(p, 4),
                                          np.round(np.array([x, yy, zp]) - p, 4)))
    s = rig.snapshot(ctx, tag="presspose", outdir=DATA)
    _log("arm near the bar band (z %.3f..%.3f):" % (float(top["z0"]) - 0.05,
                                                    float(top["z1"]) + 0.05))
    for z0 in np.arange(float(top["z0"]) - 0.06, float(top["z1"]) + 0.08, 0.01):
        mm = s.arm & (s.Z >= z0) & (s.Z < z0 + 0.01) & (np.abs(s.X - p[0]) < 0.09)
        if mm.sum() > 20:
            q2 = s.pts[mm]
            _log("   z %+.3f n=%4d  y %+.3f..%+.3f  x %+.3f..%+.3f" % (
                z0, mm.sum(), q2[:, 1].min(), q2[:, 1].max(), q2[:, 0].min(), q2[:, 0].max()))
    bar = find_bar(s, S, float(m["ref_y"]), (float(top["z0"]) - 0.01, float(top["z1"]) + 0.01),
                   tuple(m["xwin"]), min_px=60, reach=0.075, drop_arm=True)
    _log("bar re-read on the closed plane: %s" % (
        None if bar is None else "front y=%+.4f z %.4f..%.4f x %.4f..%.4f n=%d" % (
            bar["front_y"], bar["z"][0], bar["z"][1], bar["x"][0], bar["x"][1], bar["n"]),))
    # wrist camera (0 rows) - it looks along the finger axis
    import urllib.request
    try:
        d = urllib.request.urlopen("http://127.0.0.1:8766/wrist.jpg", timeout=10).read()
        path = os.path.join(DATA, "wrist_presspose.jpg")
        with open(path, "wb") as f:
            f.write(d)
        _log("wrist frame %s" % path)
    except Exception as exc:
        _log("wrist frame failed: %s" % exc)
    steps_left(ctx, "(after pressdiag)")
    return p


def pp(ctx):
    """One biased probe-pull (6 rows) from the CURRENT pose - the engagement verdict."""
    p = rig.pose_of(rig.observe(ctx))
    m = _load()
    _log("pp from eef=%s (top bar z %.4f..%.4f, front y=%+.4f)" % (
        np.round(p, 4), m["top"]["z0"], m["top"]["z1"], m["top"]["front_y"]))
    res = probe_pull(ctx, float(p[0]), float(p[2]), tag="probe2", rows=6)
    steps_left(ctx, "(after pp)")
    return res["engaged"]


# ============================================================== probe (0 rows)
def probe(ctx):
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    s = rig.snapshot(ctx, tag="probe0", outdir=DATA)
    _log("start pose eef=%s quat=%s gq=%s" % (np.round(s.eef, 4), np.round(s.quat, 4),
                                             np.round(s.gq, 4)))
    jp = rig.observe(ctx).get("observation", {}).get("joint_pos")
    _log("joints %s (j4 pinned at -0.07 = straight elbow, SKILL §2)" % (
        None if jp is None else np.round(jp, 3),))
    tz = s.table_z()
    _log("table_z %s" % (tz,))
    _log("quat builder check: quat_from_fingers((0,0,-1),(0,-1,0)) vs live wrist-down = %.1f deg"
         " (want <= 5)" % _quat_deg(rig.quat_from_fingers((0, 0, -1), (0, -1, 0)), s.quat))
    _log("--- gripper rig at the start pose")
    pads = {}
    if s.eef is not None and s.eef[2] > 1.10:
        for half in (0.030, 0.045, 0.050, 0.060, 0.075):
            p = s.pads(half=half)
            _log("   pads(half=%.3f) %s" % (half, p))
            if p is not None:
                pads[str(half)] = float(p["offset"])
    else:
        _log("   !! LOW start pose - not re-deriving the pad offset")
    _log("   jaw %s" % (s.jaw(),))
    hl = s.hand_lead()
    _log("   hand_lead %s (0.111 m ahead, underside eef_z+0.082 -> what really pushes)"
         % (hl,))

    fr = rig.drawer_axis(s)
    if fr is None:
        _log("!! no vertical fronts - abort")
        return False
    ax, S = fr["axis"], float(fr["S"])
    _log("drawer frame: axis %s, OUT %+d (u = %+d*%s), fronts %+.4f (%d px), fronts z"
         " %.4f..%.4f" % (ax, fr["S"], fr["S"], ax, fr["front"], fr["n"],
                          fr["front_z"][0], fr["front_z"][1]))
    if ax != "y":
        _log("!! axis %s, not y - this controller mirrors y only" % ax)
        return False
    xwin, ywin, zwin = fr["win"][0], fr["win"][1], fr["zwin"]
    _log("window x %s y %s z %s" % (np.round(xwin, 3), np.round(ywin, 3), np.round(zwin, 4)))

    # ---- the 1 mm y-face histogram: the flush fronts' plane and every panel that is OUT
    rows = ybin_scene(s, zwin, xwin, ywin, min_pixels=100)
    _log("--- 1 mm y-face histogram over the whole front (z %.3f..%.3f)" % zwin)
    for y, c in rows[:14]:
        _log("   y=%+.4f u=%+.4f n=%d" % (y, uu(S, y), c))
    if not rows:
        _log("no y faces - abort")
        return False
    ref_y = max(rows, key=lambda t: t[1])[0]
    outs = [r for r in rows if r[1] >= 200 and (uu(S, r[0]) - uu(S, ref_y)) > 0.020]
    _log("closed fronts' plane (reference) y=%+.4f n=%d" % (ref_y, dict(rows)[ref_y]))
    _log("panels OUT of it: %s" % ([(round(r[0], 4), r[1], round(uu(S, r[0]) - uu(S, ref_y), 4))
                                    for r in outs],))
    if not outs:
        _log("!! nothing is out - the bottom drawer is already shut; skipping the close")
    open_y = None
    panel_z = band = None
    if outs:
        open_y = min(outs, key=lambda t: t[0])[0] if S < 0 else max(outs, key=lambda t: t[0])[0]
        open_n = dict(rows)[open_y]
        _log("OPEN panel y=%+.4f (n=%d) -> %.4f m of travel to home" % (
            open_y, open_n, uu(S, open_y) - uu(S, ref_y)))
        ext = face_extent(s, open_y, zwin, xwin, tol=0.0018)
        if ext is None or ext["n"] < 60:
            _log("panel face only %s px - widening y tol" % (
                None if ext is None else ext["n"]))
            ext = face_extent(s, open_y, zwin, xwin, tol=0.004)
        panel_z = ext["z"]
        band = (panel_z[0] + 0.002, panel_z[1] - 0.002)
        _log("open panel: z %.4f..%.4f (n=%d, mean n_y %+.2f) x %.4f..%.4f" % (
            panel_z[0], panel_z[1], ext["n"], ext["ny"], ext["x"][0], ext["x"][1]))

    # ---- the OPEN drawer's own bar (needed by the close)
    bar = bot = None
    if open_y is not None:
        bar = find_bar(s, S, open_y, panel_z, xwin)
        if bar is None:
            _log("!! NO BAR on the open panel")
        else:
            _log("open drawer's bar: front u=%+.4f -> y=%+.4f (%.4f proud of the panel), z"
                 " %.4f..%.4f (zc %.4f), x %.4f..%.4f (mid %.4f), n=%d, mean %.0f" % (
                     bar["front_u"], bar["front_y"], uu(S, bar["front_y"]) - uu(S, open_y),
                     bar["z"][0], bar["z"][1], bar["zc"], bar["x"][0], bar["x"][1],
                     0.5 * (bar["x"][0] + bar["x"][1]), bar["n"], bar["mean"]))
            bot = dict(bar=bar)

    # ---- the CLOSED drawers' bars: one per drawer, LOWEST = bottom, HIGHEST = TOP
    zbars0 = (panel_z[1] + 0.012) if panel_z is not None else zwin[0]
    zbars1 = float(zwin[1])
    _log("--- achromatic z histogram in front of the CLOSED plane %+.4f (z %.3f..%.3f)" % (
        ref_y, zbars0, zbars1))
    for r in bar_hist(s, S, ref_y, (zbars0, zbars1), xwin):
        _log("   z %+.3f n=%4d u=%+.4f x %+.3f..%+.3f" % r)
    closed_bars = rig.handle_bars_on(s, fr, ref_y, (zbars0, zbars1), min_px=80, reach=0.075)
    _log("handle_bars_on(closed plane): %d" % len(closed_bars))
    for i, b in enumerate(closed_bars):
        _log("  [%d] n=%4d front_u=%+.4f (y=%+.4f, %.4f proud) z %.4f..%.4f (zc %.4f)"
             " x %.4f..%.4f (mid %.4f)" % (
                 i, b["n"], b["front_u"], b["front_off"],
                 b["front_u"] - uu(S, ref_y), b["z"][0], b["z"][1], b["zc"],
                 b["x"][0], b["x"][1], 0.5 * (b["x"][0] + b["x"][1])))
    if not closed_bars:
        _log("!! no closed bars - cannot aim the open")
        return False
    top = max(closed_bars, key=lambda b: b["zc"])
    mid = [b for b in closed_bars if b is not top]
    _log("=> TOP drawer's bar: zc %.4f, band %.4f..%.4f, front y=%+.4f, x %.4f..%.4f" % (
        top["zc"], top["z"][0], top["z"][1], top["front_off"], top["x"][0], top["x"][1]))
    _log("   other closed bars: %s" % ([(round(b["zc"], 4), round(b["z"][1] - b["z"][0], 4))
                                        for b in mid],))
    top_band = (float(top["z"][0]) - 0.045, float(top["z"][1]) + 0.006)
    _log("   top drawer's band (bar widened 4.5 cm down / 0.6 cm up): %s" % (
        np.round(top_band, 4),))

    # ---- the open drawer's interior (close signal 2) and floor
    box_x = (0.5 * (bar["x"][0] + bar["x"][1]) - 0.09,
             0.5 * (bar["x"][0] + bar["x"][1]) + 0.09) if bar else xwin
    box_y = (min(open_y, ref_y) - 0.005, max(open_y, ref_y) + 0.055) if open_y else ywin
    fz = None
    mm = s.rgb.mean(2)
    fm = ((mm > 180) & (s.Z > float(tz) + 0.010) & (s.Z < (panel_z[1] if panel_z else 1.10))
          & (s.X > box_x[0]) & (s.X < box_x[1]) & (s.Y > box_y[0]) & (s.Y < box_y[1]))
    if fm.sum() > 200:
        fz = float(np.median(s.Z[fm]))
    floor_n0 = floor_visible(s, box_x, box_y, float(tz) + 0.010,
                             (panel_z[1] if panel_z else 1.10))
    rgb0 = None
    if fm.sum() > 200:
        rgb0 = [float(v) for v in np.median(s.rgb[fm], axis=0)]
    _log("open drawer interior: %d bright px, floor z %s, colour %s ; window x %s y %s" % (
        int(fm.sum()), None if fz is None else round(fz, 4), rgb0,
        np.round(box_x, 3), np.round(box_y, 3)))
    _log("   floor_visible(mean>195,chroma<22) = %d px while OUT" % floor_n0)

    # ---- the pushers
    q_push = rig.hook_quat_toward((0.0, -S, 0.0))
    _log("q_push = %s (fingers lead %+d y); %.1f deg from Q_HOOK" % (
        np.round(q_push, 4), -S, _quat_deg(q_push, Q_HOOK)))
    hand_z = (top["z"][0] - 0.094) if closed_bars else None
    _log("pusher geometry: hand-face aim for the TOP bar would be eef_z %.4f (bar_z0-0.094)"
         " with the table at %.4f -> %s" % (
             hand_z, float(tz), "possible" if hand_z > float(tz) + 0.05 else "impossible"))
    _log("  CLOSE (bottom): tips at eef_z = bar_zc + %.3f = %.4f ; x = bar x mid = %.4f ;"
         " start %.2f OUT of the bar" % (
             TIPS_DZ, bar["zc"] + TIPS_DZ if bar else float("nan"),
             0.5 * (bar["x"][0] + bar["x"][1]) if bar else float("nan"), 0.11))
    _log("  OPEN (top): press at eef_z = bar_z0 + %.3f = %.4f, u = u_bar - %.3f ; pull"
         " dy %+.2f dz -0.35, jaws OPEN" % (
             PRESS_DZ, float(top["z"][0]) + PRESS_DZ, PRESS_DY, S * 0.85))

    m = dict(frame={k: (v if not isinstance(v, tuple) else list(v)) for k, v in fr.items()},
             S=float(S), table_z=float(tz), xwin=list(xwin), ywin=list(ywin),
             zwin=list(zwin), ref_y=float(ref_y),
             open_panel_y=None if open_y is None else float(open_y),
             panel_band=None if band is None else [float(band[0]), float(band[1])],
             panel_z=None if panel_z is None else [float(panel_z[0]), float(panel_z[1])],
             travel=None if open_y is None else float(uu(S, open_y) - uu(S, ref_y)),
             bar=None if bar is None else {k: (list(v) if isinstance(v, tuple) else float(v))
                                           for k, v in bar.items()},
             tips_z=None if bar is None else float(bar["zc"] + TIPS_DZ),
             closed_bars=[{k: (list(v) if isinstance(v, tuple) else float(v))
                           for k, v in b.items()} for b in closed_bars],
             top=dict(z0=float(top["z"][0]), z1=float(top["z"][1]), zc=float(top["zc"]),
                      front_y=float(top["front_off"]), front_u=float(top["front_u"]),
                      x0=float(top["x"][0]), x1=float(top["x"][1]),
                      x_mid=float(0.5 * (top["x"][0] + top["x"][1])), n=int(top["n"])),
             top_band=[float(top_band[0]), float(top_band[1])],
             press_z=float(top["z"][0]) + PRESS_DZ,
             press_y=float(top["front_off"]) - S * PRESS_DY,
             box_x=list(box_x), box_y=list(box_y), floor_z=fz, floor_rgb=rgb0,
             floor_n0=int(floor_n0), q_push=[float(v) for v in q_push], pads=pads,
             bottom_open=bool(open_y is not None))
    _save(m)
    _log("saved %s" % PLAN)
    steps_left(ctx, "(after probe)")
    return True


def look(ctx):
    """0-row re-read of the panel state + a frame."""
    tag = os.environ.get("TAG", "look")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    m = _load()
    _log("snapshot %s eef=%s quat=%s gq=%s" % (tag, np.round(s.eef, 4), np.round(s.quat, 4),
                                               np.round(s.gq, 4)))
    if m.get("panel_band"):
        _log("bottom band panel_now %s" % (panel_now(s, m["S"], tuple(m["panel_band"]),
                                                    tuple(m["xwin"]), tuple(m["ywin"]),
                                                    ref_y=m["ref_y"]),))
    _log("top band panel_now %s" % (panel_now(s, m["S"], tuple(m["top_band"]),
                                             tuple(m["xwin"]), tuple(m["ywin"]),
                                             ref_y=m["ref_y"]),))
    _log("top-bar faces: %s" % ([(round(float(a), 4), int(b)) for a, b in
                                 ybin_scene(s, m["top_band"], m["xwin"], m["ywin"],
                                            min_pixels=60)[:6]],))
    return tag


# ============================================================== CLOSE (bottom drawer)
def close(ctx):
    m = _load()
    if not m.get("bottom_open"):
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
    _log("panel y=%+.4f -> %.4f m to travel (reference %+.4f)" % (panel_y, off, m["ref_y"]))
    if off <= HOME_TOL:
        _log("already home")
        return True
    z_push = float(m["tips_z"])
    x_t = 0.5 * (m["bar"]["x"][0] + m["bar"]["x"][1])
    bar_front = panel_y + S * float(m["bar"]["front_u"] - uu(S, m["open_panel_y"]))
    q_push = np.array(m["q_push"], dtype=np.float64)
    zbc = float(m["bar"]["zc"])
    _log("TIPS mode: eef_z %.4f (bar %.4f..%.4f, zc %.4f), x %.4f, bar front y=%+.4f" % (
        z_push, m["bar"]["z"][0], m["bar"]["z"][1], zbc, x_t, bar_front))

    # 1. LEVEL at the current xy, then RISE (SKILL §2), rotate in place to the hook pitch
    level_then_rise(ctx, q_push, z_up=1.34)

    # 2. CROSS at the folded height, then descend vertically in two stages
    y_hover = float(bar_front) + S * 0.11          # 11 cm OUT of the bar's front face
    rig.goto(ctx, [x_t, y_hover, max(1.30, z_push + 0.30)], q_push, -1.0, tol=0.008,
             max_rows=26, tag="cross ")
    p = rig.pose_of(rig.observe(ctx))
    _log("crossed to eef=%s (residual %s)" % (
        np.round(p, 4), np.round(np.array([x_t, y_hover, max(1.30, z_push + 0.30)]) - p, 4)))
    rig.goto(ctx, [x_t, y_hover, z_push + 0.10], q_push, -1.0, tol=0.008, max_rows=18,
             tag="desc1 ")
    rig.goto(ctx, [x_t, y_hover, z_push], q_push, -1.0, tol=0.006, max_rows=14,
             tag="desc2 ")
    p = rig.pose_of(rig.observe(ctx))
    _log("pre-push eef=%s ; aim (%.4f, %.4f, %.4f)" % (np.round(p, 4), x_t, y_hover, z_push))

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
            dx = x_t - xc
            if abs(dx) > 0.008:
                x_new = x_t + float(np.clip(dx, -0.03, 0.03))
                _log("  tips span x %.4f..%.4f (centre %.4f) vs aim %.4f -> x %+.4f" % (
                    xt[0], xt[1], xc, x_t, dx))
                rig.goto(ctx, [x_new, y_hover, z_push], q_push, -1.0, tol=0.005,
                         max_rows=10, tag="tip-x ")
                x_t = x_new
    else:
        _log("  no fingertip pixels OUT of the bar - pushing from here")

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
    steps_left(ctx, "(after close push)")
    return home


def cverify(ctx, tag="cverify0"):
    """Retreat OUT first (the fingers point inward), then up; then the 4 close signals."""
    m = _load()
    if not m.get("bottom_open"):
        _log("cverify: nothing was out")
        return True
    S = float(m["S"])
    band = tuple(m["panel_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    q_hold = np.array(m["q_push"], dtype=np.float64)
    p = rig.pose_of(rig.observe(ctx))
    _log("retreat from eef=%s : OUT 0.24 first, then up" % (np.round(p, 4),))
    rig.goto(ctx, [p[0], p[1] + S * 0.24, p[2]], q_hold, -1.0, tol=0.012, max_rows=18,
             tag="retreat out ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1], max(1.22, p[2] + 0.20)], q_hold, -1.0, tol=0.012, max_rows=14,
             tag="retreat z ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0] + 0.10, p[1] + S * 0.10, p[2]], q_hold, -1.0, tol=0.014, max_rows=12,
             tag="retreat y ")

    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
    _log("SIGNAL 1 (the pushed panel's own plane): y=%s n=%d off %s m home=%s ; bins %s" % (
        None if ps["panel_y"] is None else round(ps["panel_y"], 4), ps["n"],
        None if ps["off"] is None else round(ps["off"], 4), ps["home"], ps["rows"][:5]))
    off, n, home1 = rig.panel_home_on(s, m["frame"], float(m["ref_y"]), band)
    _log("           panel_home_on: off %+.4f n=%d home=%s (asymmetric: -6 mm..+1.5 mm)" % (
        off, n, home1))
    fv = floor_visible(s, m["box_x"], m["box_y"], m["table_z"] + 0.010, m["panel_z"][1])
    _log("SIGNAL 2 (drawer interior visible): %d bright px (was %d out) -> %s" % (
        fv, m["floor_n0"],
        "interior GONE -> shut" if fv < max(150, 0.25 * m["floor_n0"]) else "still visible"))
    bar = find_bar(s, S, m["ref_y"], m["panel_z"], xwin)
    _log("SIGNAL 3 (bar back on the closed plane): %s" % (
        None if bar is None else "front y=%+.4f z %.4f..%.4f n=%d -> u - u_ref = %+.4f"
        " (a closed bar is %+.4f)" % (
            bar["front_y"], bar["z"][0], bar["z"][1], bar["n"],
            uu(S, bar["front_y"]) - uu(S, m["ref_y"]),
            float(m["bar"]["front_u"]) - uu(S, m["open_panel_y"]))),)
    rows = ybin_scene(s, band, xwin, ywin, min_pixels=150)
    front = [r for r in rows if uu(S, r[0]) - uu(S, m["ref_y"]) > 0.005]
    _log("SIGNAL 4 (nothing in front of the fronts except the bars' sliver): %s" % (
        [(round(float(a), 4), int(b)) for a, b in front[:4]],))
    ok = bool(home1 and fv < max(150, 0.25 * m["floor_n0"]))
    _log("CLOSE VERDICT: %s (panel_home %s ; interior %d vs %d ; plane off %+.4f)" % (
        "CLOSED/HOME" if ok else "NOT HOME", home1, fv, m["floor_n0"], off))
    rig.save_frame(tag, outdir=DATA)
    m["cverify"] = {"off": float(off), "n": int(n), "home": bool(home1),
                    "floor_px": int(fv), "floor_px0": int(m["floor_n0"]), "ok": ok}
    _save(m)
    steps_left(ctx, "(after cverify)")
    return ok


# ============================================================== OPEN (top drawer)
def openhook(ctx):
    """Aim at the TOP bar, read the fingertips there (0 rows), DIAGONAL press, 6-row probe."""
    m = _load()
    S = float(m["S"])
    top = m["top"]
    x_t = float(top["x_mid"])
    bf = float(top["front_y"])
    z_press = float(m["press_z"])
    y_press = float(m["press_y"])
    q_push = np.array(m["q_push"], dtype=np.float64)
    band = tuple(m["top_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    _log("openhook: TOP bar z %.4f..%.4f (zc %.4f) front y=%+.4f x %.4f..%.4f ; press target"
         " (%.4f, %+.4f, %.4f)" % (top["z0"], top["z1"], top["zc"], bf, top["x0"], top["x1"],
                                   x_t, y_press, z_press))

    # 0. where is the top panel now?
    s = rig.snapshot(ctx, tag="open_a", outdir=DATA)
    ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
    _log("host panel_now %s" % (ps,))
    if ps["panel_y"] is not None and ps["off"] is not None and ps["off"] >= OPEN_OK:
        _log("the top drawer is ALREADY out %.4f m" % ps["off"])
        return True

    # 1. RISE holding the CURRENT quaternion (never rotate while translating), then pass the
    #    wrist to q_push in place and re-read the pose (a rotation-only row walks the arm).
    p = rig.pose_of(rig.observe(ctx))
    q_cur = rig.quat_of(rig.observe(ctx))
    if p[2] < 1.32:
        rig.goto(ctx, [p[0], p[1], 1.32], q_cur, -1.0, tol=0.010, max_rows=14, tag="rise ")
    rotate_in_place(ctx, q_push, -1.0, max_rows=20, tol=0.04)
    p = rig.pose_of(rig.observe(ctx))
    _log("after rise+rotate eef=%s rot_err=%.3f" % (
        np.round(p, 4),
        float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), q_push)))))

    # 2. CROSS at the folded height, then descend in front of the bar in two stages
    y_pre = bf + S * 0.16
    z_hi = max(1.30, z_press + 0.22)
    rig.goto(ctx, [x_t, y_pre, z_hi], q_push, -1.0, tol=0.008, max_rows=26, tag="cross ")
    p = rig.pose_of(rig.observe(ctx))
    _log("crossed to eef=%s (residual %s)" % (
        np.round(p, 4), np.round(np.array([x_t, y_pre, z_hi]) - p, 4)))
    rig.goto(ctx, [x_t, y_pre, z_press + 0.06], q_push, -1.0, tol=0.008, max_rows=16,
             tag="desc1 ")
    p = rig.pose_of(rig.observe(ctx))
    _log("pre-press eef=%s (aim %.4f, %+.4f, %.4f)" % (
        np.round(p, 4), x_t, y_pre, z_press + 0.06))

    # 3. BUY THE VERDICT for 0 rows: the fingertips vs the bar's own band
    s = rig.snapshot(ctx, tag="prepress", outdir=DATA)
    prof = finger_profile(s, S, x_t, float(top["zc"]), bf)
    _log("finger_profile: n_all=%s z_all=%s n_front=%s" % (
        prof.get("n_all"), prof.get("z_all"), prof.get("n_front")))
    if "u_tip" in prof:
        _log("  fingertip: u=%+.4f (lead %+.4f), z %s, x %s" % (
            prof["u_tip"], prof["lead_vs_bar"], prof.get("z_tip"), prof.get("x_tip")))
    for r in prof.get("rows", []):
        _log("   z %+.3f n=%4d u_max %+.4f" % r)

    # 4. the DIAGONAL press: -u and -z together, until it stalls (that is CONTACT)
    tgt = np.array([x_t, y_press, z_press])
    payload, conv = rig.goto(ctx, tgt, q_push, -1.0, tol=0.004, max_rows=26, stall_rows=5,
                             tag="press")
    p = rig.pose_of(payload)
    res = tgt - p
    _log("PRESS end eef=%s target=%s residual=%s -> y residual %+.4f is %s" % (
        np.round(p, 4), np.round(tgt, 4), np.round(res, 4), res[1],
        "CONTACT (want 10-12 mm)" if abs(res[1]) > 0.005 else "A MISS (converged through air)"))
    m["press_pose"] = [float(v) for v in p]
    m["press_res"] = [float(v) for v in res]
    _save(m)

    # 5. the PROBE: 6 rows of the pull we will actually run
    probe_pull(ctx, x_t, float(p[2]), tag="probe1", rows=6)
    steps_left(ctx, "(after openhook)")
    return True


def probe_pull(ctx, x, z, tag="probe_pull", rows=6, min_move=0.005):
    m = _load()
    S = float(m["S"])
    res = rig.probe_pull_biased(ctx, np.array(m["q_push"], dtype=np.float64), float(x),
                                float(z), band_z=tuple(m["top_band"]),
                                xybox=(tuple(m["xwin"]), tuple(m["ywin"])),
                                rows=int(rows), dy=S * 0.85, dz=-0.35,
                                min_move=float(min_move), tag=tag, S=S)
    # `rig.probe_pull_biased` scores `move_u = S*dY` itself now; `move` is kept raw only for
    # the log (on a -y-open chest it is NEGATIVE for a drawer that opened).
    move_u = res.get("move_u", None if res["move"] is None else float(S) * float(res["move"]))
    engaged = bool(res.get("engaged", move_u is not None and move_u >= float(min_move)))
    _log("  probe_pull(%d rows, dy %+.2f/dz -0.35): y %s -> %s  move %s m OUT  dz_actual %+.4f"
         " (z %s) -> engaged=%s" % (
             rows, S * 0.85,
             None if res["y_before"] is None else round(res["y_before"], 4),
             None if res["y_after"] is None else round(res["y_after"], 4),
             None if move_u is None else round(move_u, 4), res["dz_actual"],
             "BLOCKED" if res["blocked_z"] else "free", engaged))
    res["move_u"] = move_u
    res["engaged"] = engaged
    if res["y_after"] is not None:
        m["top_panel_y"] = float(res["y_after"])
        _save(m)
    return res


def pull_to_stop(ctx, x, z, batches=5, rows=8, out_target=OPEN_OK, tag="pull"):
    m = _load()
    S = float(m["S"])
    band = tuple(m["top_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    q_push = np.array(m["q_push"], dtype=np.float64)
    f = ybin_scene(rig.snapshot(ctx, tag=tag + "_a", outdir=DATA), band, xwin, ywin,
                   min_pixels=150)
    prev = None if not f else float(max(f, key=lambda t: S * t[0])[0])
    _log("  pull start: top panel y=%s (ref %+.4f -> out %s)" % (
        None if prev is None else round(prev, 4), m["ref_y"],
        None if prev is None else round(uu(S, prev) - uu(S, m["ref_y"]), 4)))
    for i in range(int(batches)):
        rig.pull_burst(ctx, q_push, x=float(x), z=float(z), rows=int(rows), dy=S * 0.85,
                       dz=-0.35)
        s = rig.snapshot(ctx, tag="%s_%d" % (tag, i), outdir=DATA)
        f = ybin_scene(s, band, xwin, ywin, min_pixels=150)
        y = None if not f else float(max(f, key=lambda t: S * t[0])[0])
        e = rig.pose_of(rig.observe(ctx))
        out = None if y is None else uu(S, y) - uu(S, m["ref_y"])
        d = None if (y is None or prev is None) else uu(S, y) - uu(S, prev)
        _log("  burst %d eef=%s panel y=%s out %s (moved %s)" % (
            i, np.round(e, 4), None if y is None else round(y, 4),
            None if out is None else round(out, 4), None if d is None else round(d, 4)))
        if y is not None:
            m["top_panel_y"] = y
            _save(m)
        if out is not None and out >= float(out_target):
            _log("  top drawer out %.4f m -> at/near the stop" % out)
            return out
        if d is not None and d < 0.002:
            _log("  panel stopped advancing at %s m out (the fingers left the bar)" % (
                None if out is None else round(out, 4)))
            return out
        prev = y if y is not None else prev
    return None if prev is None else uu(S, prev) - uu(S, m["ref_y"])


def pull(ctx):
    """Re-entrant: pull bursts from wherever the fingers are; re-press if the panel is short."""
    m = _load()
    S = float(m["S"])
    band = tuple(m["top_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    z = None
    if os.environ.get("ZT"):
        z = float(os.environ["ZT"])
    elif m.get("press_pose"):
        z = float(m["press_pose"][2])
    else:
        z = float(m["press_z"])
    x = float(m["top"]["x_mid"])
    out = pull_to_stop(ctx, x=x, z=z)
    if out is not None and out >= OPEN_OK:
        _log("pull: TOP drawer out %.4f m" % out)
        return out
    _log("pull: short (%s) -> re-measured press loop" % (
        None if out is None else round(out, 4),))
    for cyc in range(2):
        s = rig.snapshot(ctx, tag="opencyc%d" % cyc, outdir=DATA)
        ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
        _log("  cycle %d: top panel_now %s" % (cyc, ps))
        if ps["panel_y"] is None:
            return out
        panel_y = float(ps["panel_y"])
        # the bar moves only along y: bar front = panel + dz_bar (in u)
        u_bar = uu(S, panel_y) + (uu(S, float(m["top"]["front_y"])) - uu(S, m["ref_y"]))
        bf = S * u_bar
        y_aim = bf - S * PRESS_DY
        x_t = float(m["top"]["x_mid"])
        z_p = float(m["top"]["z0"]) + PRESS_DZ - 0.005 * cyc
        _log("  re-press: bar front y=%+.4f -> aim (%.4f, %+.4f, %.4f)" % (bf, x_t, y_aim, z_p))
        level_then_rise(ctx, np.array(m["q_push"], dtype=np.float64), z_up=1.34)
        rig.goto(ctx, [x_t, bf + S * 0.16, max(1.30, z_p + 0.22)],
                 np.array(m["q_push"], dtype=np.float64), -1.0, tol=0.008, max_rows=22,
                 tag="rp cross ")
        rig.goto(ctx, [x_t, bf + S * 0.16, z_p + 0.06], np.array(m["q_push"], dtype=np.float64),
                 -1.0, tol=0.008, max_rows=16, tag="rp desc ")
        tgt = np.array([x_t, y_aim, z_p])
        payload, _ = rig.goto(ctx, tgt, np.array(m["q_push"], dtype=np.float64), -1.0,
                              tol=0.004, max_rows=26, stall_rows=5, tag="rp press ")
        p = rig.pose_of(payload)
        _log("  re-press end eef=%s residual=%s" % (np.round(p, 4), np.round(tgt - p, 4)))
        res = probe_pull(ctx, x_t, float(p[2]), tag="probe_re%d" % cyc, rows=6)
        if res["engaged"]:
            out = pull_to_stop(ctx, x=x_t, z=float(p[2]))
            if out is not None and out >= OPEN_OK:
                return out
    return out


def overify(ctx, tag="overify0"):
    """Retreat clear, then the opening verdict: the TOP drawer's own panel plane is out."""
    m = _load()
    S = float(m["S"])
    band = tuple(m["top_band"])
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    q_hold = np.array(m["q_push"], dtype=np.float64)
    p = rig.pose_of(rig.observe(ctx))
    _log("retreat from eef=%s" % (np.round(p, 4),))
    rig.goto(ctx, [p[0], p[1] + S * 0.16, p[2] + 0.06], q_hold, -1.0, tol=0.012, max_rows=16,
             tag="retreat out ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0] + 0.10, p[1], max(1.25, p[2] + 0.18)], q_hold, -1.0, tol=0.014,
             max_rows=14, tag="retreat z ")

    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
    _log("OPEN SIGNAL 1 (top drawer's own band): y=%s n=%d off %s m ; bins %s" % (
        None if ps["panel_y"] is None else round(ps["panel_y"], 4), ps["n"],
        None if ps["off"] is None else round(ps["off"], 4), ps["rows"][:6]))
    _log("   reference (the closed fronts' flush plane) y=%+.4f ; a full slide is ~0.15-0.16 m"
         % m["ref_y"])
    bars = rig.handle_bars_on(s, m["frame"], float(m["ref_y"]), (m["table_z"] + 0.005,
                                                               m["ref_y"] * 0 + 1.30),
                              min_px=80, reach=0.30)
    _log("OPEN SIGNAL 2 (bars with reach 0.30): %s" % (
        [(round(b["front_u"], 4), round(b["zc"], 4), b["n"]) for b in bars],))
    near = [b for b in bars if abs(b["zc"] - float(m["top"]["zc"])) < 0.02]
    _log("   the bar at the TOP band: %s (expect front u ~ panel + %.4f)" % (
        None if not near else "front_u %+.4f at zc %.4f -> %+.4f OUT of the reference" % (
            near[0]["front_u"], near[0]["zc"],
            near[0]["front_u"] - uu(S, m["ref_y"])), DZ_BAR_TYP))
    ok = bool(ps["off"] is not None and ps["off"] >= OPEN_OK)
    _log("OPEN VERDICT: %s (top panel out %s m)" % (
        "OPEN" if ok else "NOT OPEN YET", None if ps["off"] is None else round(ps["off"], 4)))
    rig.save_frame(tag, outdir=DATA)
    m["overify"] = {"panel_y": ps["panel_y"], "off": ps["off"], "ok": ok}
    _save(m)
    steps_left(ctx, "(after overify)")
    return ok


def score(ctx, tag="score"):
    """BOTH goal clauses, from one depth call, before `finish` (SKILL §0.7: one shot)."""
    m = _load()
    S = float(m["S"])
    fr = m["frame"]
    xwin, ywin = tuple(m["xwin"]), tuple(m["ywin"])
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    _log("=== score at eef=%s" % (np.round(s.eef, 4),))
    ok = True
    if m.get("bottom_open"):
        band = tuple(m["panel_band"])
        ps = panel_now(s, S, band, xwin, ywin, ref_y=m["ref_y"])
        off, n, home = rig.panel_home_on(s, fr, float(m["ref_y"]), band)
        fv = floor_visible(s, m["box_x"], m["box_y"], m["table_z"] + 0.010, m["panel_z"][1])
        _log("CLOSE clause: bottom panel %s (off %s) ; panel_home_on off %+.4f home=%s ;"
             " interior %d px (was %d) -> %s" % (
                 None if ps["panel_y"] is None else round(ps["panel_y"], 4),
                 None if ps["off"] is None else round(ps["off"], 4), off, home, fv,
                 m["floor_n0"], "shut" if fv < max(150, 0.25 * m["floor_n0"]) else "STILL OPEN"))
        ok = ok and bool(home) and fv < max(150, 0.25 * m["floor_n0"])
    band_t = tuple(m["top_band"])
    ts = panel_now(s, S, band_t, xwin, ywin, ref_y=m["ref_y"])
    _log("OPEN clause: top panel %s (out %s m ; a full slide is 0.15-0.16)" % (
        None if ts["panel_y"] is None else round(ts["panel_y"], 4),
        None if ts["off"] is None else round(ts["off"], 4)))
    ok = ok and bool(ts["off"] is not None and ts["off"] >= OPEN_OK)
    # a bar out in front of the closed plane at the TOP band's z = the top drawer's handle
    bars = rig.handle_bars_on(s, fr, float(m["ref_y"]), (m["table_z"] + 0.005, 1.30),
                              min_px=80, reach=0.30, drop_arm=False)
    _log("bars with reach 0.30 (no arm drop): %s" % (
        [(round(b["front_u"], 4), round(b["zc"], 4), b["n"]) for b in bars],))
    near = [b for b in bars if abs(b["zc"] - float(m["top"]["zc"])) < 0.015]
    if near:
        b = min(near, key=lambda b: b["front_u"])       # the OUT-most in u
        out_b = b["front_u"] - uu(S, m["ref_y"])
        _log("   the TOP bar is at u %+.4f = %+.4f m OUT of the closed plane (a closed bar is"
             " %+.4f)" % (b["front_u"], out_b, float(m["top"]["front_u"]) - uu(S, m["ref_y"])))
    _log("SCORE: %s" % ("BOTH CLAUSES TRUE" if ok else "ONE CLAUSE NOT MET"))
    rig.save_frame(tag, outdir=DATA)
    steps_left(ctx, "(after score)")
    return ok


# ============================================================== run
def run(ctx):
    if not os.path.exists(PLAN):
        _log("=== PROBE ===")
        if not probe(ctx):
            return False
    _log("=== CLOSE (bottom drawer) ===")
    close(ctx)
    _log("=== CVERIFY ===")
    okc = cverify(ctx, tag="cverify1")
    if not okc:
        _log("not home yet - one more measured push")
        close(ctx)
        okc = cverify(ctx, tag="cverify1b")
    _log("=== OPENHOOK (top drawer) ===")
    openhook(ctx)
    _log("=== PULL ===")
    pull(ctx)
    _log("=== OVERIFY ===")
    oko = overify(ctx, tag="overify1")
    _log("close ok=%s open ok=%s" % (okc, oko))
    return bool(okc and oko)


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
