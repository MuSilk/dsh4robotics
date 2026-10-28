"""Controller for LIBERO-90 task 4: put the butter at the FRONT in the top drawer of the
cabinet and close it.

Stages are dispatched by the ENTRY env var from experiments/run_controller.py, so each one
is runnable on its own on the same open episode:

    ENTRY=probe   zero-step measurement (saves experiments/data/probe0.npz)
    ENTRY=grasp   pick the front butter off the table
    ENTRY=place   carry it over the open drawer mouth, lower, release, verify column
    ENTRY=close   push the drawer shut, stop on the eef stall, verdict at 1.5 mm
    ENTRY=run     grasp -> place -> close in one run

Every number below was measured on this episode's opening depth call (init_state 31,
libero90_interactive_31_e28b47cf) and is listed with its provenance; the physical
constants come from `code/lib/rig.py`.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lib"))
import rig  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "experiments", "data")

# wrist-down quaternion (rig, measured): w ~ 0 with a vector near 1 is this pose
QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])

# ---- placement numbers, measured in `probe` on this episode (0 rows) ------------------
TABLE_Z = 0.9498          # table plane, height origin (rig check: one flat plane)
FLOOR_Z = 1.113           # drawer floor plane (n=5643)
RIM_Z = 1.175             # counter top / drawer rim (n=10747)
PANEL_Y = -0.0771         # open top drawer's front panel (1 mm y_faces_mm)
BODY_Y = -0.2281          # cabinet body front face
REF_Y = -0.2294           # the CLOSED lower drawer's panel = "joint home" reference
BAR_Y = -0.0455           # handle bar front face (95th pct of y)
BAR_X = (-0.065, 0.024)   # handle bar span across the push axis
BAR_X_MID = -0.0205
# pan band is floor..rim; the pad band is eef_z+0.043 .. eef_z+0.065, so aim mid-band
PUSH_EEF_Z = 0.5 * (FLOOR_Z + RIM_Z) - 0.054          # = 1.090
# front butter (the +x member of the identical pair; -x is "at the back")
BUTTER_XY = (-0.0192, 0.2097)   # top-face centroid, zmax 0.9677 (pair: -0.0968, 0.1907)
BUTTER_TOP = 0.9677
BUTTER_H = BUTTER_TOP - TABLE_Z                        # 0.0179 m tall
# where the butter should end up: deep in the box, mouth floor 1.113 over
# x -0.13..0.01, y -0.24..-0.10; 9.5 cm behind the panel's inner face
DROP_XY = (-0.050, -0.170)

PAD_DZ = 0.0426           # pad bottom - eef_z, measured this episode (probe, open jaws)
JAW_DX = -0.0170          # finger-pair centre = eef + (JAW_DX, JAW_DY); probe -0.0197/-0.0138
JAW_DY = -0.0145          # vs rig -0.0140/-0.0151 -> take the mean, the pair is good to 5 mm


def _log(msg):
    print(msg, flush=True)


def _eef_for_jaw(xy):
    """eef position whose JAW CENTRE sits on `xy` (eef_pos is not the centre of the jaws)."""
    return np.array([xy[0] - JAW_DX, xy[1] - JAW_DY])


# ----------------------------------------------------------------- measurement
def butter_mask(s, xy=None, half=0.09, zlo=None, zhi=1.03):
    """The wrapped butter sticks: chromatic and mid-bright, on the table.

    Cabinet faces are dark achromatic, the table is bright wood (mean > 150), the bowl is
    achromatic speckle, the arm is achromatic.  mean < 150 alone already excludes the
    table, which is what keeps this test honest after the butter is lifted.
    """
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    zlo = TABLE_Z + 0.0008 if zlo is None else zlo
    m = ((s.Z > zlo) & (s.Z < zhi) & (mean > 20) & (mean < 150) & ((mx - mn) > 35)
         & ~s.arm)
    if xy is not None:
        m &= (np.abs(s.X - xy[0]) < half) & (np.abs(s.Y - xy[1]) < half)
    else:
        m &= (s.X > -0.30) & (s.X < 0.16) & (s.Y > -0.08) & (s.Y < 0.30)
    return m


def butter_at(s, xy, half=0.09):
    """Top-face centroid + zmax of the butter near `xy`, or None if it is not there."""
    keep, info = s.blob(butter_mask(s, xy=xy, half=half), min_pixels=80)
    if keep is None or info["n"] < 80:
        return None
    p = s.pts[keep]
    top = p[p[:, 2] > p[:, 2].max() - 0.002]
    info["top_xy"] = (float(top[:, 0].mean()), float(top[:, 1].mean()))
    info["top_n"] = int(len(top))
    info["area"] = 0.5 * ((info["x"][1] - info["x"][0]) * (info["y"][1] - info["y"][0]))
    return info


def probe(ctx):
    os.makedirs(DATA, exist_ok=True)
    s = rig.snapshot(ctx, tag="probe0", outdir=DATA)
    s.save(os.path.join(DATA, "probe0.npz"))
    _log("eef %s quat %s gq %s" % (np.round(s.eef, 4), np.round(s.quat, 4),
                                   np.round(s.gq, 4)))
    _log("table_z %.4f" % (s.table_z() or -1))
    _log("--- planes (axis, offset, n, span1, span2)")
    for p in s.planes(top=8):
        _log("   %s=%+.3f n=%4d %s %s" % (p[0], p[1], p[2], np.round(p[3], 3),
                                          np.round(p[4], 3)))
    _log("pads %s" % (s.pads(),))
    _log("jaw %s" % (s.jaw(),))
    _log("closed_reference_y %s" % (rig.closed_reference_y(s, 0.5 * (FLOOR_Z + RIM_Z)),))
    _log("--- butters (sorted by x; +x = front)")
    out = []
    work = butter_mask(s).copy()
    while work.sum() > 80:
        keep, info = s.blob(work, min_pixels=80)
        if keep is None:
            break
        work &= ~keep
        out.append(info)
    for i in sorted(out, key=lambda d: d["centre"][0]):
        _log("   n=%4d centre=%s x%s y%s zmax=%.4f" % (
            i["n"], np.round(i["centre"], 4), np.round(i["x"], 3), np.round(i["y"], 3),
            i["zmax"]))
    _log("--- free columns over the drawer mouth (x, y, max_z)")
    for x, y, z in rig.box_free_columns(s, -0.10, 0.0, -0.21, -0.11, step=0.02):
        _log("   x=%+.3f y=%+.3f maxz=%s" % (x, y, None if z is None else round(z, 3)))


# ----------------------------------------------------------------- grasp
def grasp(ctx):
    """Pick the front butter off the table (pad bottom 6 mm above the table)."""
    s = rig.snapshot(ctx, tag="grasp0", outdir=DATA)
    info = butter_at(s, BUTTER_XY, half=0.09)
    if info is None:
        _log("butter not found near %s" % (BUTTER_XY,))
        return False
    cx, cy = info["top_xy"]
    _log("front butter: n=%d top_xy=(%.4f, %.4f) zmax=%.4f (h=%.4f)" % (
        info["n"], cx, cy, info["zmax"], info["zmax"] - TABLE_Z))
    xy = _eef_for_jaw((cx, cy))
    z_grasp = TABLE_Z + 0.006 - PAD_DZ          # pad bottom 6 mm above the table
    tgt = np.array([xy[0], xy[1], z_grasp])
    _log("grasp target eef=%s (jaw centre -> (%.4f, %.4f), pad bottom %.4f)" % (
        np.round(tgt, 4), cx, cy, TABLE_Z + 0.006))
    # over the object, high, then straight down: the diagonal would cross the bowl rim
    rig.servo(ctx, [tgt[0], tgt[1], 1.20], QUAT_DOWN, -1.0, max_rows=40, tol=0.008)
    payload, _ = rig.servo(ctx, tgt, QUAT_DOWN, -1.0, max_rows=30, tol=0.005, zclip=0.9)
    _log("at grasp pose eef=%s residual=%s" % (
        np.round(rig.pose_of(payload), 4), np.round(tgt - rig.pose_of(payload), 4)))
    payload, _complete, gq = rig.close_grip(ctx)
    _log("jaws settled at gq=%.4f (a 4.4 cm stick pinches at ~0.022)" % gq)
    p = rig.pose_of(payload)
    payload, _ = rig.servo(ctx, [p[0], p[1], p[2] + 0.11], QUAT_DOWN, 1.0, max_rows=16,
                           tol=0.006)
    s2 = rig.snapshot(ctx, tag="lift0", outdir=DATA)
    gone = butter_at(s2, (cx, cy), half=0.06)
    _log("after lift: eef=%s gq=%.4f old-spot butter=%s" % (
        np.round(rig.pose_of(payload), 4), rig.gq_of(payload),
        "GONE" if gone is None else "STILL THERE (n=%d zmax=%.4f)" % (gone["n"], gone["zmax"])))
    return gone is None


# ----------------------------------------------------------------- place
def place(ctx):
    """Carry the butter above the drawer mouth, lower until the hand stalls, release."""
    p = rig.pose_of(rig.observe(ctx))
    gq = rig.gq_of(rig.observe(ctx))
    _log("place start eef=%s gq=%.4f" % (np.round(p, 4), gq))
    carry_z = RIM_Z + 0.095
    xy = _eef_for_jaw(DROP_XY)
    _log("drop: jaw centre -> %s, eef -> (%s, %s, %.3f)" % (
        DROP_XY, round(xy[0], 4), round(xy[1], 4), carry_z))
    rig.servo(ctx, [p[0], p[1], carry_z], QUAT_DOWN, 1.0, max_rows=14, tol=0.008)
    rig.servo(ctx, [xy[0], xy[1], carry_z], QUAT_DOWN, 1.0, max_rows=45, tol=0.008)
    # descend into the box; the hand meets the mouth ~2.5-3 cm above the floor and stalls
    payload, _ = rig.servo(ctx, [xy[0], xy[1], FLOOR_Z + BUTTER_H + 0.001 - 0.045],
                           QUAT_DOWN, 1.0, max_rows=24, tol=0.005, zclip=0.9)
    p = rig.pose_of(payload)
    _log("descent stopped eef=%s z-FLOOR=%.4f (a stall 2-3 cm up is the hand at the mouth)"
         % (np.round(p, 4), p[2] - FLOOR_Z))
    # release with a small +z bias so the pads clear the butter
    for _ in range(6):
        payload, _c = rig.step(ctx, [[0.0, 0.0, 0.35, 0.0, 0.0, 0.0, -1.0]], verbose=False)
    p = rig.pose_of(payload)
    _log("released at eef=%s gq=%.4f" % (np.round(p, 4), rig.gq_of(payload)))
    # retreat: up out of the mouth, then out along +y, then clear
    rig.servo(ctx, [p[0], p[1], RIM_Z + 0.10], QUAT_DOWN, -1.0, max_rows=14, tol=0.008)
    p = rig.pose_of(rig.observe(ctx))
    rig.servo(ctx, [p[0], 0.06, RIM_Z + 0.10], QUAT_DOWN, -1.0, max_rows=20, tol=0.010)
    s = rig.snapshot(ctx, tag="placed", outdir=DATA)
    col = rig.column_max_z(s, DROP_XY[0], DROP_XY[1], half=0.012)
    want = FLOOR_Z + BUTTER_H
    _log("column at drop point: max_z=%s  want ~%.4f (floor %.3f + butter %.4f)" % (
        None if col is None else round(col, 4), want, FLOOR_Z, BUTTER_H))
    _log("in the box: %s" % ("YES" if col is not None and abs(col - want) < 0.020 else
                             "NO/UNCLEAR"))
    return col is not None and abs(col - want) < 0.020


# ----------------------------------------------------------------- close
def close(ctx):
    """Push the drawer shut: stop on the JOINT (eef stall), verdict at 1.5 mm."""
    p = rig.pose_of(rig.observe(ctx))
    _log("close start eef=%s" % np.round(p, 4))
    # get out of the mouth and rise above the rim before swinging to the bar
    rig.servo(ctx, [p[0], p[1], RIM_Z + 0.075], QUAT_DOWN, -1.0, max_rows=14, tol=0.008)
    x_push = BAR_X_MID - JAW_DX
    rig.servo(ctx, [x_push, 0.065, RIM_Z + 0.06], QUAT_DOWN, -1.0, max_rows=24, tol=0.008)
    payload, _ = rig.servo(ctx, [x_push, 0.065, PUSH_EEF_Z], QUAT_DOWN, -1.0, max_rows=18,
                           tol=0.005, zclip=0.9)
    _log("pre-push eef=%s (bar front y=%.4f, x mid %.4f)" % (
        np.round(rig.pose_of(payload), 4), BAR_Y, BAR_X_MID))
    payload, ok, log = rig.push_drawer_closed(
        ctx, QUAT_DOWN, PANEL_Y, BODY_Y, PUSH_EEF_Z, x_push, gripper=-1.0,
        rows_per_batch=5, max_batches=10, ref_y=REF_Y)
    _log("push_drawer_closed ok=%s" % ok)
    return ok


# --- the push that actually finishes it: press the BAR, not the panel -----------------
def push_arm_to_stop(ctx, q_tgt, z_target, x_target, max_rows=34, stall_rows=3,
                     stall_tol=0.0015):
    """Push -y with x/z held until the eef has been dead for `stall_rows` rows.

    Measured on this rig: the gripper's HAND BLOCK reaches 0.111 m ahead of `eef_pos`
    (the finger pads only 0.069 m), and the hand's front face spans z = eef_z+0.082 ..
    eef_z+0.15.  At eef_z 1.09 that face is entirely above the panel's top edge (1.173),
    so the drawer is pushed by the panel's top rim and the hand's underside rides the
    counter's front edge (y -0.226) -> the push stalls ~4 mm proud of home.  Aiming the
    face at the BAR (z 1.139..1.147, protruding 3.2 cm to y -0.193) needs
    eef_z+0.082 < 1.139, i.e. eef_z <= 1.057; 1.045 leaves 12 mm of margin and keeps the
    hand's face 3 cm clear of the cabinet front when the drawer is home.
    """
    payload = rig.observe(ctx)
    dead, stalled = 0, False
    for i in range(int(max_rows)):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -1.0
        a[0] = float(np.clip((x_target - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((z_target - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        dead = dead + 1 if moved < stall_tol else 0
        _log("  push row eef=%s moved=%.4f dead=%d" % (np.round(q, 4), moved, dead))
        if dead >= stall_rows:
            stalled = True
            break
    froze = None
    if stalled:
        p = rig.pose_of(payload)
        payload, _ = rig.step(ctx, [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0]], verbose=False)
        rose = float(rig.pose_of(payload)[2] - p[2])
        froze = bool(abs(rose) < 0.002)
        _log("  stall check: +z row rose %.4f m -> %s" % (
            rose, "FROZEN" if froze else "real contact"))
        p = rig.pose_of(payload)
        rig.step(ctx, [[0.0, 0.0, float(np.clip((z_target - p[2]) / 0.010, -1, 1)),
                        0.0, 0.0, 0.0, -1.0]], verbose=False)
    return payload, stalled, froze


def close2(ctx):
    """Close, 2nd attempt: same travel, but press the handle bar with the hand lower."""
    p = rig.pose_of(rig.observe(ctx))
    _log("close2 start eef=%s" % np.round(p, 4))
    z_push = 1.045          # hand's front face then spans 1.127..1.195, over the bar band
    x_push = BAR_X_MID - JAW_DX
    rig.servo(ctx, [p[0], p[1], z_push], QUAT_DOWN, -1.0, max_rows=16, tol=0.006)
    _log("pre-push eef=%s (bar front %.4f, x mid %.4f)" % (
        np.round(rig.pose_of(rig.observe(ctx)), 4), BAR_Y, BAR_X_MID))
    payload, stalled, froze = push_arm_to_stop(ctx, QUAT_DOWN, z_push, x_push)
    s = rig.snapshot(ctx, tag="push2", outdir=DATA)
    s.save(os.path.join(DATA, "push2.npz"))
    faces, coarse = rig.drawer_closed(s, PANEL_Y, BODY_Y, band_z=PUSH_EEF_Z)
    y_face, n_face, home = rig.panel_home(s, REF_Y, band_z=(1.105, 1.176))
    _log("faces(y,n) in the panel band: %s" % (faces,))
    _log("front face y=%s n=%d ref_y=%.4f home=%s  stalled=%s froze=%s" % (
        y_face, n_face, REF_Y, home, stalled, froze))
    ok = bool(stalled) and not froze and bool(home)
    _log("VERDICT close2: %s" % ("HOME" if ok else "NOT HOME"))
    return ok


def run(ctx):
    _log("=== GRASP ===")
    if not grasp(ctx):
        _log("grasp did not confirm; stopping so the stage can be re-run")
        return
    _log("=== PLACE ===")
    if not place(ctx):
        _log("place did not confirm; stopping")
        return
    _log("=== CLOSE ===")
    close(ctx)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    _log("ENTRY %s" % entry)
    t0 = time.time()
    globals()[entry](ctx)
    _log("entry %s done in %.1fs" % (entry, time.time() - t0))
