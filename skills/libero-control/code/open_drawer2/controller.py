"""LIBERO-90 task 6 - OPEN THE BOTTOM DRAWER of the kitchen chest.

Stages, dispatched by ENTRY (experiments/run_controller.py):

    ENTRY=probe      0 rows: snapshot + chest_window/front_face_y/handle_bars + rig checks
    ENTRY=hook       rotate to the hook pitch, measure the finger offset AT THAT PITCH,
                     pre-hook, DIAGONAL press onto the bottom bar, 4-row probe pull
    ENTRY=pull       (re-entrant) probe pull + pull bursts to the drawer's stop
    ENTRY=verify     retreat clear, then the plane verdicts
    ENTRY=close      shut the drawer with the finger tips (own code) - verdict on the panel
    ENTRY=reopen     press + probe pull + pull to the stop again (own code)
    ENTRY=verify2    final short-retreat verdict + frame
    ENTRY=run        probe -> hook -> probe_pull/pull -> verify -> close -> reopen -> verify2

Every placement number is measured on THIS draw in `probe`; z bands / travel come from the
asset (the task's 50 demo traces: the slide joint runs 0 -> -0.1605 m, the hook stalls at
eef_z 0.965 +/- 0.006 against a bar centre of 1.000, and the pull is dy +0.7..0.9 / dz
-0.3..-0.7 with the jaws OPEN).

The hook orientation is `rig.Q_HOOK` (finger (0.26,-0.855,-0.455), jaw (0.933,0.10,0.34)) -
15 deg away from an idealised (1,-3,-2)/(1,0,0) build, which is enough for an idealised press
to slide past the bar without a word.  NO DEMO FILE IS NEEDED: the trace's numbers are recorded
in `rig` (SKILL §10.1), and this file was run to a 632-row pass with them.

Measured on the pass, in one place, because every one of these has a 5 mm window:
  press      eef_z = bar_z0 - 0.045 = 0.9499, eef_y = bar_front - 0.015, 21-26 rows
             -> stalls at (-0.027, -0.2166, 0.9516) with a 10-12 mm y residual = CONTACT
  probe      6 rows of dy +0.85 / dz -0.35 -> panel +4.07 cm, dz_actual -0.0011 (blocked)
  pull       the same profile in 8-row bursts -> +6.3, +5.9 cm -> 0.1609 m out (the stop)
A z-held probe (rig.probe_pull) and any press at 0.955-0.975 converged and moved the panel
0.000 m; that is what cost this episode ~225 rows.
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
MEAS = os.path.join(DATA, "meas.json")

QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])

# measured hook orientation (trace rows 85-105 of this task's demos)
Q_HOOK = rig.quat_from_fingers((0.260, -0.855, -0.455), (0.933, 0.10, 0.34))

DZ_BAR_TYP = 0.0322      # bar front - panel front (re-measured every draw in `probe`)
TRAVEL = 0.1605          # slide joint range of this drawer (asset; demo states[:, 26])
OUT_OK = 0.150           # "open" = the panel is this far out of its home plane


def _log(msg):
    print(msg, flush=True)


def _load():
    with open(MEAS) as f:
        return json.load(f)


def _save(m):
    os.makedirs(DATA, exist_ok=True)
    with open(MEAS, "w") as f:
        json.dump(m, f, indent=1, default=float)


def _quat_deg(qa, qb):
    return float(np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(qa, qb)))))))


def _xybox(m):
    return (tuple(m["xybox"][0]), tuple(m["xybox"][1]))


# ------------------------------------------------------------------ motion
def rotate_in_place(ctx, q_tgt, gripper=-1.0, max_rows=30, tol=0.08, verbose=True):
    """Rotation-only rows: a large pitch commanded while translating stalls and the error GROWS."""
    payload = rig.observe(ctx)
    for i in range(int(max_rows)):
        e = rig.rotvec_err(rig.quat_of(payload), q_tgt)
        if np.linalg.norm(e) < tol:
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


# ------------------------------------------------------------------ probe
def probe(ctx):
    os.makedirs(DATA, exist_ok=True)
    s = rig.snapshot(ctx, tag="probe0", outdir=DATA)
    _log("start pose eef=%s quat=%s gq=%s" % (np.round(s.eef, 4), np.round(s.quat, 4),
                                              np.round(s.gq, 4)))
    tz = s.table_z()
    _log("table_z %s" % (tz,))
    _log("quat builder check: quat_from_fingers((0,0,-1),(0,-1,0)) vs measured wrist-down"
         " = %.1f deg (want <= 5)" % _quat_deg(rig.quat_from_fingers((0, 0, -1), (0, -1, 0)),
                                              s.quat))
    _log("--- gripper rig at the start pose (wrist down)")
    for half in (0.030, 0.045, 0.060):
        _log("   pads(half=%.3f) %s" % (half, s.pads(half=half)))
    _log("   jaw %s" % (s.jaw(),))
    _log("   hand_lead %s" % (s.hand_lead(),))

    xybox, band_z, counter_z, counter_y = rig.chest_window(s)
    _log("chest_window xybox=%s band_z=%s counter_z=%.4f counter_y=%.4f" % (
        np.round(xybox, 3), np.round(band_z, 4), counter_z, counter_y))
    tf = s.top_face_front_y()
    _log("top_face_front_y %s" % (None if tf is None else tuple(np.round(tf, 4)),))
    cf = rig.front_face_y(s, band_z=band_z, xybox=xybox, min_pixels=150)
    _log("closed_front (fronts' flush plane) %s" % (None if cf is None else
                                                    tuple(np.round(cf, 4)),))
    _log("--- 1 mm y-face histogram in the front band")
    for y, n in rig.y_faces_mm(s, band_z=band_z, xybox=xybox, min_pixels=40)[:8]:
        _log("   y=%+.4f n=%d" % (y, n))

    bars = s.handle_bars(panel_y=cf[0], zrange=(tz + 0.005, counter_z - 0.01), xbox=xybox[0])
    _log("--- handle bars (one per drawer; the LOWEST is the bottom drawer): %d" % len(bars))
    for i, b in enumerate(bars):
        _log("  [%d] n=%4d front_y=%.4f z %.4f..%.4f (zc %.4f) x %.4f..%.4f (mid %.4f)" % (
            i, b["n"], b["front_y"], b["z0"], b["z1"], b["zc"], b["x0"], b["x1"],
            0.5 * (b["x0"] + b["x1"])))
    if not bars:
        _log("NO BARS - abort probe")
        return False
    bottom = min(bars, key=lambda b: b["zc"])
    drawer_band = rig.drawer_band_for_bar(bottom)
    own = rig.front_face_y(s, band_z=drawer_band, xybox=xybox, min_pixels=150)
    own_ref = float(own[0]) if own else float(cf[0])
    dz_bar = float(bottom["front_y"]) - own_ref
    _log("bottom drawer: bar zc=%.4f band=%s ; bar front=%.4f ; own panel front=%.4f ;"
         " DZ_BAR=%.4f" % (bottom["zc"], np.round(drawer_band, 4), bottom["front_y"],
                           own_ref, dz_bar))

    m = dict(xybox=[list(xybox[0]), list(xybox[1])], band_z=list(band_z),
             counter_z=float(counter_z), counter_y=float(counter_y),
             table_z=None if tz is None else float(tz),
             closed_front=float(cf[0]), own_ref=own_ref, dz_bar=dz_bar,
             bar_x_mid=0.5 * (bottom["x0"] + bottom["x1"]),
             bottom_z0=float(bottom["z0"]), bottom_z1=float(bottom["z1"]),
             bottom_zc=float(bottom["zc"]), bar_front0=float(bottom["front_y"]),
             drawer_band=[float(drawer_band[0]), float(drawer_band[1])],
             n_bars=len(bars))
    _save(m)
    _log("saved %s" % MEAS)
    return True


def look(ctx):
    """0-row snapshot + frame under $TAG."""
    tag = os.environ.get("TAG", "look")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    _log("snapshot %s eef=%s quat=%s gq=%s" % (tag, np.round(s.eef, 4), np.round(s.quat, 4),
                                               np.round(s.gq, 4)))
    return tag


# ------------------------------------------------------------------ hook + press
def measure_pitch_offset(ctx, m):
    """Finger-surface offset at the hook pitch, in free space and >= 12 cm above the table."""
    goto_clear = [m["bar_x_mid"], m["bar_front0"] + 0.16, 1.10]
    rig.goto(ctx, goto_clear, Q_HOOK, -1.0, tol=0.008, max_rows=16, tag="clear ")
    s = rig.snapshot(ctx, tag="pitch_clear", outdir=DATA)
    off, log = rig.pitch_offset(s)
    _log("pitch_offset at eef=%s -> %s log=%s" % (
        np.round(s.eef, 4), None if off is None else round(off, 4), log))
    _log("   (demo cross-check: stall eef_z %.3f against bar zc %.4f -> offset ~%.3f)" % (
        m["bottom_zc"] - 0.035, m["bottom_zc"], 0.035))
    m["pitch_offset"] = None if off is None else float(off)
    _save(m)
    return off


def press(ctx, x, bar_front, z_press, y_off=-0.015, max_rows=26, stall_rows=5, tag="press"):
    """DIAGONAL press onto the bar's top-front: -y and -z together, until it stalls (contact).

    A constant-height insert converges with a sub-mm residual and touches nothing; a press
    that is still creeping when it ends is the one that bites.
    """
    rig.goto(ctx, [x, bar_front + 0.16, 1.00], Q_HOOK, -1.0, tol=0.008, max_rows=12,
             tag=tag + " pre ")
    tgt = np.array([x, bar_front + float(y_off), float(z_press)])
    payload, ok = rig.goto(ctx, tgt, Q_HOOK, -1.0, tol=0.004, max_rows=max_rows,
                           stall_rows=stall_rows, tag=tag + " ")
    p = rig.pose_of(payload)
    _log("  press end eef=%s target=%s residual=%s (a y residual > 3 mm is CONTACT)"
         % (np.round(p, 4), np.round(tgt, 4), np.round(tgt - p, 4)))
    return p, ok


def probe_pull(ctx, x, z, tag="probe_pull", rows=6, min_move=0.005):
    """6 rows of the DEMO pull profile + the drawer's own panel before/after.

    Delegates to `rig.probe_pull_biased`, which carries both signals: the panel move and
    `dz_actual` (blocked when the fingers are on the bar, the full -0.029 m when they are not).
    A pure +y pull is a false negative on a bar - measured 0.000 m at 0.950/0.955/0.965/0.971,
    including the pose a biased burst pulled 4.4 cm from.
    """
    m = _load()
    res = rig.probe_pull_biased(ctx, Q_HOOK, float(x), float(z), band_z=m["drawer_band"],
                                xybox=_xybox(m), rows=int(rows), min_move=float(min_move),
                                tag=tag)
    _log("  probe_pull(%d rows, dy .85/dz -.35): y %s -> %s  move %s  dz_actual %+.4f "
         "(z %s) -> engaged=%s" % (
             rows, None if res["y_before"] is None else round(res["y_before"], 4),
             None if res["y_after"] is None else round(res["y_after"], 4),
             None if res["move"] is None else round(res["move"], 4), res["dz_actual"],
             "BLOCKED" if res["blocked_z"] else "free", res["engaged"]))
    if res["y_after"] is not None:
        m["panel_y"] = float(res["y_after"])
        _save(m)
    return res


def pull_to_stop(ctx, x, z, out_target=OUT_OK, batches=5, rows=8, tag="pull"):
    """Pull +y (dz -0.35) in bursts; stop at the drawer's stop, not at 'open enough'."""
    m = _load()
    band, xybox = m["drawer_band"], _xybox(m)
    f = rig.front_face_y(rig.snapshot(ctx, tag=tag + "_a", outdir=DATA), band_z=band,
                         xybox=xybox, min_pixels=150)
    prev = None if f is None else float(f[0])
    _log("  pull start: panel y=%s (own_ref %.4f -> out %s)" % (
        None if prev is None else round(prev, 4), m["own_ref"],
        None if prev is None else round(prev - m["own_ref"], 4)))
    for i in range(int(batches)):
        rig.pull_burst(ctx, Q_HOOK, x=float(x), z=float(z), rows=int(rows))
        s = rig.snapshot(ctx, tag="%s_%d" % (tag, i), outdir=DATA)
        f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
        y = None if f is None else float(f[0])
        e = rig.pose_of(rig.observe(ctx))
        out = None if y is None else y - m["own_ref"]
        d = None if (y is None or prev is None) else y - prev
        _log("  burst %d eef=%s panel y=%s out %s (moved %s)" % (
            i, np.round(e, 4), None if y is None else round(y, 4),
            None if out is None else round(out, 4), None if d is None else round(d, 4)))
        if y is not None:
            m["panel_y"] = y
            _save(m)
        if out is not None and out >= float(out_target):
            _log("  drawer out %.4f m -> at the stop" % out)
            return out
        if d is not None and d < 0.002:
            _log("  panel stopped advancing at %.4f m out (fingers out of the handle)" % (
                None if out is None else out))
            return out
        prev = y if y is not None else prev
    return None if prev is None else prev - m["own_ref"]


def open_once(ctx, z_base=None, sweep=(0.0, -0.005, -0.010), max_cycles=3):
    """Measure the panel, DIAGONAL press onto the bar, PROBE the pull, then pull to the stop.

    `z_base` is `bar_z0 + rig.HOOK_PRESS_DZ` = 0.9499 on this chest (rig.hook_press_z).  At that
    height the press stalls with a 10-12 mm y residual and the biased burst moves the panel
    4.4 cm with the commanded -z blocked; 5 mm higher it converges in y, touches nothing and the
    burst takes the full 2.9 cm of z.  Five millimetres is the whole margin and it fails upward
    only, so the fallback sweep goes DOWN in 5 mm steps, never +/-1 cm.

    Repeats (re-measure -> press -> probe -> pull) until the drawer is at its stop: after each
    pull burst the drawer has moved and the bar's front is `panel + DZ_BAR` again.
    """
    m = _load()
    band, xybox = m["drawer_band"], _xybox(m)
    x = float(m["bar_x_mid"])
    out = None
    for cyc in range(int(max_cycles)):
        s = rig.snapshot(ctx, tag="open_cyc%d" % cyc, outdir=DATA)
        f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
        if f is None:
            _log("open_once: cannot see the drawer's own front plane")
            return out
        panel_y = float(f[0])
        bar_front = float(rig.bar_front_from_panel(panel_y, m["dz_bar"]))
        out = panel_y - m["own_ref"]
        _log("--- cycle %d: panel y=%.4f (out %.4f) -> bar front %.4f ; x=%.4f ; bar z %.4f..%.4f"
             % (cyc, panel_y, out, bar_front, x, m["bottom_z0"], m["bottom_z1"]))
        if out >= OUT_OK:
            _log("  already at the stop")
            return out
        zb = (float(m["bottom_z0"]) + rig.HOOK_PRESS_DZ) if z_base is None else float(z_base)
        heights = [zb + d for d in (sweep if cyc == 0 else (0.0,))]
        for z in heights:
            _log("--- press at eef_z %.4f (bar zc %.4f)" % (z, m["bottom_zc"]))
            p, ok = press(ctx, x, bar_front, z)
            res = probe_pull(ctx, x=x, z=float(p[2]))
            if res["engaged"]:
                _log("  ENGAGED (panel moved %.4f m) - pulling" % res["move"])
                out = pull_to_stop(ctx, x=x, z=float(p[2]))
                break
            _log("  not engaged at eef_z %.4f (move %s) - re-pressing" % (
                p[2], None if res["move"] is None else round(res["move"], 4)))
    return out


def pull_more(ctx):
    """Continue pulling from wherever the fingers are (no re-press), then finish if short."""
    m = _load()
    p = rig.pose_of(rig.observe(ctx))
    _log("pull_more from eef=%s" % np.round(p, 4))
    out = pull_to_stop(ctx, x=float(p[0]), z=float(p[2]), out_target=OUT_OK, batches=4,
                       rows=8, tag="pm")
    _log("pull_more out=%s" % (None if out is None else round(out, 4)))
    if out is not None and out >= OUT_OK:
        return out
    _log("short - re-hooking (measured press loop)")
    return open_once(ctx)


def hook(ctx):
    """Fresh arm: rotate to the hook pitch, measure the offset, then the diagonal press."""
    m = _load()
    q0 = rig.quat_of(rig.observe(ctx))
    _log("wrist now %.1f deg from q_hook; rotating in place" % _quat_deg(q0, Q_HOOK))
    rig.goto(ctx, [m["bar_x_mid"], m["bar_front0"] + 0.20, m["bottom_zc"] + 0.16], QUAT_DOWN,
             -1.0, tol=0.010, max_rows=18, tag="prehook ")
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=30)
    measure_pitch_offset(ctx, m)
    s = rig.snapshot(ctx, tag="hook_ready", outdir=DATA)
    _log("hook pose eef=%s quat=%s (%.1f deg from q_hook)" % (
        np.round(s.eef, 4), np.round(s.quat, 4), _quat_deg(s.quat, Q_HOOK)))
    _log("  hand_envelope(dy_lo,dy_hi,dz_lo,dz_hi,n) %s" % (s.arm_envelope(),))
    return True


# ------------------------------------------------------------------ verdicts

def wrist_frame(tag="wrist"):
    """Fetch the wrist camera JPEG (0 rows)."""
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "wrist_%s.jpg" % tag)
    with urllib.request.urlopen("http://127.0.0.1:8766/wrist.jpg", timeout=10) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    _log("wrist frame %s" % path)
    return path


def diag(ctx):
    """Goto the press pose at $ZPRESS (default the measured base) and snapshot for offline work."""
    m = _load()
    z = float(os.environ.get("ZPRESS", m["bottom_z0"] - 0.030))
    x, bf = float(m["bar_x_mid"]), float(m["bar_front0"])
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=10, tol=0.10)
    p, ok = press(ctx, x, bf, z, max_rows=22, stall_rows=5)
    s = rig.snapshot(ctx, tag=os.environ.get("TAG", "at_press"), outdir=DATA)
    _log("at press eef=%s quat=%s" % (np.round(s.eef, 4), np.round(s.quat, 4)))
    _log("  pads %s / %s / %s" % (s.pads(0.030), s.pads(0.040), s.pads(0.045)))
    _log("  arm_envelope %s" % (s.arm_envelope(),))
    _log("  horizontal_arm_faces %s" % (s.horizontal_arm_faces(),))
    wrist_frame(os.environ.get("TAG", "at_press"))
    return True


def try_pull(ctx):
    """Press at $ZPRESS/$YOFF then run ONE pull burst (dy $DY, dz $DZ, $ROWS rows) and check
    the drawer's own panel plane.  The z response separates contact from a miss: with the
    fingers on the bar the commanded -z is blocked (the demos hold eef_z ~0.950 through the
    whole pull); with no contact the tips walk down the panel's face."""
    m = _load()
    band, xybox = m["drawer_band"], _xybox(m)
    z = float(os.environ.get("ZPRESS", m["bottom_z0"] - 0.030))
    yoff = float(os.environ.get("YOFF", -0.005))
    dy = float(os.environ.get("DY", 0.85))
    dz = float(os.environ.get("DZ", -0.35))
    rows = int(os.environ.get("ROWS", 8))
    x, bf = float(m["bar_x_mid"]), float(m["bar_front0"])
    f0 = rig.front_face_y(rig.snapshot(ctx, tag="tp_a", outdir=DATA), band_z=band,
                          xybox=xybox, min_pixels=150)
    y0 = None if f0 is None else float(f0[0])
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=8, tol=0.10)
    rig.goto(ctx, [x, bf + 0.16, 1.00], Q_HOOK, -1.0, tol=0.008, max_rows=14, tag="tp pre ")
    tgt = np.array([x, bf + yoff, z])
    payload, conv = rig.goto(ctx, tgt, Q_HOOK, -1.0, tol=0.004, max_rows=26, stall_rows=5,
                             tag="tp press ")
    p = rig.pose_of(payload)
    _log("  press eef=%s target=%s residual=%s" % (np.round(p, 4), np.round(tgt, 4),
                                                   np.round(tgt - p, 4)))
    payload = rig.pull_burst(ctx, Q_HOOK, x=x, z=float(p[2]), rows=rows, dy=dy, dz=dz)
    e = rig.pose_of(payload)
    s = rig.snapshot(ctx, tag="tp_b", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    y1 = None if f is None else float(f[0])
    _log("  burst dy=%.2f dz=%.2f rows=%d: eef %s -> %s (dz_actual %+.4f)"
         % (dy, dz, rows, np.round(p, 4), np.round(e, 4), e[2] - p[2]))
    _log("  panel %s -> %s  MOVE %s" % (
        None if y0 is None else round(y0, 4), None if y1 is None else round(y1, 4),
        None if (y0 is None or y1 is None) else round(y1 - y0, 4)))
    wrist_frame("try_pull")
    return None if (y0 is None or y1 is None) else y1 - y0


def ascend(ctx):
    """Press into the pocket under the bar, then command pure +z until the BAR blocks it.

    The stall height is the finger's top surface against the bar's underside
    (bar bottom 0.9949 on this asset), i.e. it measures the pitched finger-top offset with the
    arm itself - the one number the wrist-down `pads()` cannot give (§3)."""
    m = _load()
    z = float(os.environ.get("ZPRESS", m["bottom_z0"] - 0.030))
    yoff = float(os.environ.get("YOFF", -0.018))
    x, bf = float(m["bar_x_mid"]), float(m["bar_front0"])
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=8, tol=0.10)
    rig.goto(ctx, [x, bf + 0.16, 1.00], Q_HOOK, -1.0, tol=0.008, max_rows=14, tag="as pre ")
    tgt = np.array([x, bf + yoff, z])
    payload, _ = rig.goto(ctx, tgt, Q_HOOK, -1.0, tol=0.004, max_rows=26, stall_rows=5,
                          tag="as press ")
    p = rig.pose_of(payload)
    _log("  pressed to eef=%s (residual %s)" % (np.round(p, 4), np.round(tgt - p, 4)))
    dead = 0
    for i in range(int(os.environ.get("ROWS", 12))):
        a = np.zeros(7)
        a[2] = 1.0
        a[0] = float(np.clip((x - p[0]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), Q_HOOK) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        _log("  ascend %2d eef=%s dz_actual=%+.4f dead=%d" % (i, np.round(q, 4), q[2] - p[2],
                                                             dead))
        p = q
        if dead >= 3:
            break
    _log("ASCENT STALL at eef_z=%.4f ; finger-top offset vs bar bottom %.4f = %.4f" % (
        p[2], m["bottom_z0"], m["bottom_z0"] - p[2]))
    wrist_frame("ascend")
    return float(p[2])

def verify(ctx, tag="verify0", retreat=(0.10, 0.13), tol=0.010, short=False):
    m = _load()
    band, xybox = m["drawer_band"], _xybox(m)
    p = rig.pose_of(rig.observe(ctx))
    _log("retreat from eef=%s (+z %.2f then +y %.2f)" % (np.round(p, 4), retreat[0], retreat[1]))
    rig.goto(ctx, [p[0], p[1], p[2] + retreat[0]], Q_HOOK, -1.0, tol=tol,
             max_rows=8 if short else 14, tag="retreat z ")
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1] + retreat[1], p[2]], Q_HOOK, -1.0, tol=tol,
             max_rows=10 if short else 16, tag="retreat y ")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    y = None if f is None else float(f[0])
    out = None if y is None else y - m["own_ref"]
    _log("SIGNAL 1 - this drawer's own front plane y=%s vs own_ref %.4f -> out %s m" % (
        None if y is None else round(y, 4), m["own_ref"],
        None if out is None else round(out, 4)))
    bars = s.handle_bars(panel_y=m["closed_front"], zrange=(m["table_z"] + 0.005,
                                                           m["counter_z"] - 0.01),
                         xbox=xybox[0], reach=0.30)
    low = [b for b in bars if b["zc"] < m["bottom_zc"] + 0.02]
    _log("SIGNAL 2 - bars with the closed plane as reference (reach 0.30): %d ; lowest %s"
         " (expect front ~ panel + %.4f)" % (
             len(bars), None if not low else round(low[0]["front_y"], 4), m["dz_bar"]))
    closed_bars = s.handle_bars(panel_y=m["closed_front"], zrange=(m["table_z"] + 0.005,
                                                                  m["counter_z"] - 0.01),
                                xbox=xybox[0], reach=0.09)
    _log("SIGNAL 3 - bars in the closed window (reach 0.09): %d (was %d)" % (
        len(closed_bars), m["n_bars"]))
    tf = s.top_face_front_y()
    _log("counter/rim front y=%s (closed %.4f)" % (
        None if tf is None else round(tf[1], 4), m["counter_y"]))
    rig.save_frame(tag, outdir=DATA)
    ok = bool(out is not None and out >= OUT_OK)
    _log("VERDICT: %s (panel out %s m of the %.4f m travel)" % (
        "OPEN" if ok else "NOT OPEN YET", None if out is None else round(out, 4), TRAVEL))
    m["verify_" + tag] = {"y": y, "out": out, "open": ok}
    _save(m)
    return ok


# ------------------------------------------------------------------ close (own code)
def close(ctx, max_rows=44, check_every=6):
    """Shut the bottom drawer with the PITCHED finger tips; verdict = the panel's own plane.

    The hand's face (eef_z <= bar_top - 0.088) would need eef_z 0.917, i.e. under the table
    (bar band 0.995..1.005, table 0.950), so the tips push instead: at this pitch they are
    ~3 cm above eef_z -> z = bar_zc - 0.031.  A push has no force signal; the panel plane is
    the only stop.
    """
    m = _load()
    band, xybox = m["drawer_band"], _xybox(m)
    s = rig.snapshot(ctx, tag="before_close", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    if f is None:
        _log("close: cannot see the drawer's own front plane")
        return False
    panel_now = float(f[0])
    x = float(m["bar_x_mid"])
    z = float(m["bottom_zc"] - 0.031)
    _log("close: panel y=%.4f (out %.4f) ; x=%.4f ; eef_z=%.4f -> tips ~%.4f (bar %.4f..%.4f)"
         % (panel_now, panel_now - m["own_ref"], x, z, z + 0.030, m["bottom_z0"],
            m["bottom_z1"]))
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=8, tol=0.10)
    rig.goto(ctx, [x, panel_now + 0.10, z + 0.09], Q_HOOK, -1.0, tol=0.008, max_rows=24,
             tag="close pre ")
    rig.goto(ctx, [x, panel_now + 0.10, z], Q_HOOK, -1.0, tol=0.006, max_rows=16,
             tag="close pre ")
    payload = rig.observe(ctx)
    dead, y = 0, None
    for i in range(int(max_rows)):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -1.0
        a[0] = float(np.clip((x - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((z - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), Q_HOOK) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if i % int(check_every) == 0 or dead >= 3:
            s = rig.snapshot(ctx, tag="close_i", outdir=DATA)
            f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
            y = None if f is None else float(f[0])
            _log("  push %2d eef=%s moved=%.4f dead=%d panel y=%s (off home %s)" % (
                i, np.round(q, 4), moved, dead, None if y is None else round(y, 4),
                None if y is None else round(y - m["own_ref"], 4)))
            if y is not None and abs(y - m["own_ref"]) <= 0.002:
                _log("  panel is home")
                break
        if dead >= 3:
            _log("  push STALLED at eef=%s" % np.round(q, 4))
            break
    s = rig.snapshot(ctx, tag="closed", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    panel_final = None if f is None else float(f[0])
    ok = bool(panel_final is not None and abs(panel_final - m["own_ref"]) <= 0.004)
    _log("CLOSE VERDICT: panel y=%s vs closed plane %.4f -> home=%s" % (
        None if panel_final is None else round(panel_final, 4), m["own_ref"], ok))
    rig.save_frame("closed", outdir=DATA)
    m["closed_panel_y"] = panel_final
    _save(m)
    return ok


def reopen(ctx):
    """Own-code reopen: rotate, press the bar diagonally (bar front = panel + DZ_BAR), probe,
    pull to the stop.  The panel is re-measured, so this works from any part-open position."""
    m = _load()
    _log("reopen: q_hook, dz_bar=%.4f" % m["dz_bar"])
    rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=10, tol=0.10)
    out = open_once(ctx)
    _log("reopen result: out=%s" % (None if out is None else round(out, 4)))
    return out


def verify2(ctx):
    return verify(ctx, tag="verify2", retreat=(0.10, 0.13), tol=0.014, short=True)


# ------------------------------------------------------------------ run
def run(ctx):
    """probe -> hook -> press/probe/pull -> verify.  The close-and-reopen cycle is NOT part of
    the pass: it exists to purge a replayed demo trace from the final state (SKILL §0.7), and
    with the HDF5 gone there is nothing to replay.  Set CYCLE=1 to run it anyway (it is a good
    end-to-end check, and both halves are measured: 28 rows to shut, ~50 to reopen).
    """
    if not os.path.exists(MEAS):
        _log("=== PROBE ===")
        if not probe(ctx):
            return False
    _log("=== HOOK ===")
    hook(ctx)
    _log("=== OPEN (press + biased probe + pull to stop) ===")
    out = open_once(ctx)
    _log("=== VERIFY ===")
    ok = verify(ctx, tag="verify1")
    _log("open out=%s verify=%s" % (None if out is None else round(out, 4), ok))
    if not ok:
        _log("not open yet - one more measured press/pull")
        out = open_once(ctx)
        ok = verify(ctx, tag="verify1b")
        _log("second attempt: out=%s verify=%s" % (
            None if out is None else round(out, 4), ok))
        if not ok:
            return False
    if os.environ.get("CYCLE"):
        _log("=== CLOSE (own code) ===")
        _log("close -> %s" % close(ctx))
        _log("=== REOPEN (own code) ===")
        out2 = reopen(ctx)
        _log("=== VERIFY2 ===")
        ok2 = verify2(ctx)
        _log("reopen out=%s verify2=%s" % (None if out2 is None else round(out2, 4), ok2))
        if not ok2:
            _log("reopen short - one more measured press/pull")
            out2 = open_once(ctx)
            ok2 = verify2(ctx)
        return bool(ok2)
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
