"""Controller for LIBERO-90 task 5: put the chocolate pudding in the top drawer of the
cabinet and close it.

Stages, dispatched by ENTRY (see experiments/run_controller.py):

    ENTRY=probe    0 rows: snapshot + every measurement in SKILL.md §4
    ENTRY=grasp    pick the chocolate pudding off the table
    ENTRY=place    carry it above the open drawer mouth, lower, release, verify the column
    ENTRY=close    push the drawer shut: stop on the eef stall, 1 mm verdict
    ENTRY=run      grasp -> place -> close

Numbers marked (rig) are physical constants from code/lib/rig.py; numbers marked
(placement) were measured in `probe` on THIS episode (init_state 2,
libero90_interactive_2_9166f15c) and do not transfer.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lib"))
import rig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "experiments", "data")

# wrist-down quaternion (rig): w near 0 with a vector near 1 is this pose
QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])

# ---- placement numbers, measured in `probe` on THIS episode (0 rows) ------------------
TABLE_Z = 0.9498          # table plane = height origin (rig check: one flat plane)
FLOOR_Z = 1.113           # drawer floor (free columns + the 1.115 z-plane, n=5635)
RIM_Z = 1.175             # counter top / drawer rim (z-plane n=10749)
PANEL_Y = -0.0784         # the OPEN top drawer's front panel (1 mm y_faces_mm, n=577)
BODY_Y = -0.2300          # cabinet body front face (y-plane n=1647)
REF_Y = -0.2289           # the CLOSED lower drawer's panel = "joint home" (n=765)
BAR_Y = -0.0460           # handle bar front face (n=52 at 1 mm)
BAR_X_MID = -0.0205       # bar spans x -0.0648..+0.0239
BAR_Z_BOT = 1.1354        # bar band z 1.1354..1.1469 (its front face)
# the hand's front face spans eef_z+0.082..+0.13, so eef_z 1.040 puts it at 1.122..1.170,
# inside the bar band and below the panel's top edge (1.173)
PUSH_EEF_Z = 1.040

# ---- the chocolate pudding: the UNIQUE dark-brown box (the other two are an identical
# pair of butters, 7.6x3.9x1.8, colour (103,54,31) vs this one's (70,49,40))
PUD_TOP = 0.9779          # zmax of its top face
PUD_H = PUD_TOP - TABLE_Z  # 0.0281 m tall
PUD_JAW_XY = (-0.0131, 0.0440)   # top-face centroid -> put the JAW PAIR here
PUD_FOOT = (0.0812, 0.0486)      # x, y footprint (jaws close along y -> 4.9 cm wide)

# where it goes: deep in the box.  Free mouth (floor 1.113) measured over
# x -0.105..+0.015, y -0.205..-0.115; the panel's inner face is at y ~-0.10.
DROP_JAW_XY = (-0.055, -0.160)   # 6 cm behind the panel, 4.5 cm clear of both walls

PAD_DZ = 0.0405           # pad bottom - eef_z, measured this episode (stable for
                          # mean>20..70 and box half 0.03..0.06; mean>90 reads 0.0526)
JAW_DX = -0.0171          # finger-pair centre = eef + (JAW_DX, JAW_DY).  Measured
JAW_DY = -0.0127          # (-0.0202, -0.0103) here, rig -0.0140/-0.0151 -> take the mean


def _log(msg):
    print(msg, flush=True)


def _eef_for_jaw(xy):
    """eef position whose JAW CENTRE sits on `xy` (eef_pos is not the centre of the jaws)."""
    return np.array([xy[0] - JAW_DX, xy[1] - JAW_DY])


def goto(ctx, target, q_tgt, gripper, tol=0.006, max_rows=40, rate=0.011, kick=0.30,
         stall_rows=6, verbose=True):
    """rig.servo plus a ROW-ALTERNATING roll kick, with stall detection.

    SKILL §2: a row can go inert (0.000-0.001 m while translating) at particular heights and
    it costs the whole task if the approach silently ends short.  An *undoctored* kick winds
    the wrist, so the kick is added on top of the quaternion servo, which pulls back every
    row; alternating its sign keeps the mean at zero.  Measured on this episode: the first
    descent froze dead at eef_z 1.0643 (all axes, +z too) and only a sideways move cleared
    it, so `stall_rows` rows without progress is reported as a stall and the caller retries.
    """
    target = np.asarray(target, dtype=np.float64)
    payload = rig.observe(ctx)
    dead = 0
    for i in range(int(max_rows)):
        p = rig.pose_of(payload)
        if np.max(np.abs(target - p)) < tol:
            if verbose:
                print("  goto converged err=%s" % np.round(target - p, 4), flush=True)
            return payload, True
        a = np.zeros(7)
        a[:3] = np.clip((target - p) / rate, -1, 1)
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[3] = float(np.clip(a[3] + (kick if i % 2 == 0 else -kick), -1, 1))
        a[6] = gripper
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if verbose and (i % 4 == 0 or dead):
            print("  goto %2d eef=%s err=%s moved=%.4f" % (
                i, np.round(q, 4), np.round(target - q, 4), moved), flush=True)
        if dead >= stall_rows:
            print("  goto STALLED at eef=%s err=%s (steps %d-%d dead)" % (
                np.round(q, 4), np.round(target - q, 4), dead - stall_rows, i), flush=True)
            return payload, False
    if verbose:
        print("  goto hit max_rows, residual=%s" % np.round(target - rig.pose_of(payload), 4),
              flush=True)
    return payload, False


# ----------------------------------------------------------------- probe
def probe(ctx):
    os.makedirs(DATA, exist_ok=True)
    s = rig.snapshot(ctx, tag="probe0", outdir=DATA)
    _log("eef %s quat %s gq %s" % (np.round(s.eef, 4), np.round(s.quat, 4), np.round(s.gq, 4)))
    _log("table_z %.4f" % (s.table_z() or -1))
    _log("--- planes (axis, offset, n, span_a, span_b)")
    for p in s.planes(top=8):
        _log("   %s=%+.3f n=%4d %s %s" % (p[0], p[1], p[2], np.round(p[3], 3),
                                          np.round(p[4], 3)))
    _log("--- gripper geometry (measured, not assumed)")
    for half in (0.03, 0.045, 0.06):
        _log("   pads(half=%.3f) %s" % (half, s.pads(half=half)))
    _log("   jaw %s" % (s.jaw(),))
    _log("   hand_lead %s" % (s.hand_lead(),))
    _log("--- drawer faces at 1 mm in the panel band")
    for y, n in rig.y_faces_mm(s, band_z=(1.105, 1.19), min_pixels=40)[:8]:
        _log("   y=%+.4f n=%d" % (y, n))
    _log("closed_reference_y %s" % (rig.closed_reference_y(s, 0.5 * (FLOOR_Z + RIM_Z)),))
    _log("panel_face %s" % (rig.panel_face(s),))
    _log("top_face_front_y %s" % (s.top_face_front_y(),))
    _log("--- objects on the table (sorted by x; +x = front)")
    for i in s.objects_on_table(min_pixels=60):
        _log("   n=%4d centre=%s top_xy=%s x=%s y=%s zmax=%.4f h=%.4f" % (
            i["n"], np.round(i["centre"], 4), np.round(i["top_xy"], 4), np.round(i["x"], 3),
            np.round(i["y"], 3), i["zmax"], i["zmax"] - (s.table_z() or 0)))
    _log("--- handle bar (achromatic, so inside `arm`)")
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    m = ((mx - mn) < 25) & (mean > 70) & (s.Z > 1.10) & (s.Z < 1.22) \
        & (s.Y > -0.075) & (s.Y < -0.03) & (s.X > -0.20) & (s.X < 0.10)
    if m.sum():
        p = s.pts[m]
        _log("   n=%d x %.4f..%.4f y %.4f..%.4f z %.4f..%.4f" % (
            m.sum(), p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
            p[:, 2].min(), p[:, 2].max()))
        _log("   front y (95th pct) %.4f  z band %.4f..%.4f  x mid %.4f" % (
            np.percentile(p[:, 1], 95), p[:, 2].min(), p[:, 2].max(), p[:, 0].mean()))
    _log("--- free columns over the drawer mouth (x, y, max_z)")
    for x, y, z in rig.box_free_columns(s, -0.135, 0.045, -0.235, -0.085, step=0.015,
                                        half=0.012):
        _log("   x=%+.3f y=%+.3f maxz=%s" % (x, y, None if z is None else round(z, 3)))


# ----------------------------------------------------------------- pudding mask
def pud_mask(s, xy=None, half=0.06):
    """The chocolate pudding: dark, chromatic (not achromatic arm/cabinet, not the table).

    Its own colour is (70, 49, 40): saturated relative to its brightness, which separates
    it from the cabinet faces (dark AND achromatic) and from the bright wood table.
    """
    mx, mn, mean = s.rgb.max(2), s.rgb.min(2), s.rgb.mean(2)
    m = ((s.Z > TABLE_Z + 0.0015) & (s.Z < TABLE_Z + 0.06) & ~s.arm
         & (mean > 30) & (mean < 150) & ((mx - mn) > 18))
    if xy is None:
        m &= (s.X > -0.30) & (s.X < 0.16) & (s.Y > -0.08) & (s.Y < 0.30)
    else:
        m &= (np.abs(s.X - xy[0]) < half) & (np.abs(s.Y - xy[1]) < half)
    return m


def pud_at(s, xy=None, half=0.06):
    keep, info = s.blob(pud_mask(s, xy=xy, half=half), min_pixels=60)
    if keep is None or info["n"] < 60:
        return None
    p = s.pts[keep]
    top = p[p[:, 2] > p[:, 2].max() - 0.002]
    info["top_xy"] = (float(top[:, 0].mean()), float(top[:, 1].mean()))
    info["mid_xy"] = (0.5 * (info["x"][0] + info["x"][1]),
                      0.5 * (info["y"][0] + info["y"][1]))
    return info


# ----------------------------------------------------------------- grasp
def grasp(ctx):
    """Pick the pudding off the table: jaw pair on the top face, pad bottom 1.3 cm up."""
    s = rig.snapshot(ctx, tag="grasp0", outdir=DATA)
    info = pud_at(s, PUD_JAW_XY, half=0.06)
    if info is None:
        _log("pudding not found near %s" % (PUD_JAW_XY,))
        return False
    cx, cy = info["top_xy"]
    _log("pudding: n=%d top_xy=(%.4f, %.4f) zmax=%.4f h=%.4f foot x%s y%s" % (
        info["n"], cx, cy, info["zmax"], info["zmax"] - TABLE_Z, np.round(info["x"], 3),
        np.round(info["y"], 3)))
    xy = _eef_for_jaw((cx, cy))
    pad_bottom = TABLE_Z + 0.013           # 1.3 cm above the table, 1.5 cm below its top
    z_grasp = pad_bottom - PAD_DZ
    tgt = np.array([xy[0], xy[1], z_grasp])
    _log("grasp target eef=%s (jaw centre -> (%.4f, %.4f), pad bottom %.4f, top %.4f)" % (
        np.round(tgt, 4), cx, cy, pad_bottom, info["zmax"]))
    # THE HAND BLOCK'S FRONT FACE IS THE CONSTRAINT, NOT THE PADS.  A straight descent at
    # the grasp y froze dead at eef_z 1.0645 on two attempts: the hand's face (eef_y-0.111
    # = -0.056) is 1 cm BEHIND the handle bar's front face (y -0.046) and its bottom edge
    # is eef_z+0.082 = 1.1465, i.e. 0.4 mm above the bar's top face (1.1469) -> the arm
    # lands on the bar and every further -z row is a contact, not a freeze.  So: descend
    # clear in +y, get the hand's TOP (eef_z+0.13) below the bar's bottom (1.135), and only
    # then slide in -y at a height the pads still clear the object's top by 5 cm.
    y_clear = BAR_Y + 0.111 + 0.020        # hand's face 2 cm in front of the bar
    z_under = 0.990                        # hand top 1.120 < bar bottom 1.135
    _log("approach: descend at eef_y=%.4f (hand clear of the bar), slide in at z=%.3f" % (
        y_clear, z_under))
    ok = False
    for attempt in range(3):
        goto(ctx, [tgt[0], y_clear, 1.16], QUAT_DOWN, -1.0, tol=0.008, max_rows=30)
        goto(ctx, [tgt[0], y_clear, z_under], QUAT_DOWN, -1.0, tol=0.006, max_rows=20)
        goto(ctx, [tgt[0], tgt[1], z_under], QUAT_DOWN, -1.0, tol=0.006, max_rows=12)
        payload, ok = goto(ctx, tgt, QUAT_DOWN, -1.0, tol=0.005, max_rows=16, kick=0.30)
        if ok:
            break
        p = rig.pose_of(payload)
        _log("descent attempt %d stalled at z %.4f; nudging +y and retrying" % (
            attempt, p[2]))
        rig.step(ctx, [[0.0, 0.5, 0.2, 0.0, 0.0, 0.0, -1.0]], verbose=False)
    _log("at grasp pose eef=%s residual=%s (a capped servo ends silently SHORT)" % (
        np.round(rig.pose_of(payload), 4), np.round(tgt - rig.pose_of(payload), 4)))
    payload, _c, gq = rig.close_grip(ctx)
    _log("jaws settled at gq=%.4f (a 4.9 cm box pinches at ~0.024; 0.002 = nothing)" % gq)
    p = rig.pose_of(payload)
    payload, _ = rig.servo(ctx, [p[0], p[1], p[2] + 0.11], QUAT_DOWN, 1.0, max_rows=16,
                           tol=0.006)
    s2 = rig.snapshot(ctx, tag="lift0", outdir=DATA)
    gone = pud_at(s2, (cx, cy), half=0.05)
    _log("after lift: eef=%s gq=%.4f old-spot=%s" % (
        np.round(rig.pose_of(payload), 4), rig.gq_of(payload),
        "GONE" if gone is None else "STILL THERE (n=%d zmax=%.4f)" % (gone["n"], gone["zmax"])))
    rig.save_frame("lift0", outdir=DATA)
    return gone is None


# ----------------------------------------------------------------- place
def place(ctx):
    """Carry the pudding above the mouth, lower until the hand stalls, release, verify."""
    obs = rig.observe(ctx)
    p, gq = rig.pose_of(obs), rig.gq_of(obs)
    _log("place start eef=%s gq=%.4f" % (np.round(p, 4), gq))
    carry_z = RIM_Z + 0.055          # object bottom ~1.26, 8 cm above the 1.175 rim
    xy = _eef_for_jaw(DROP_JAW_XY)
    _log("drop: jaw centre -> %s, eef -> (%.4f, %.4f, %.3f)" % (DROP_JAW_XY, xy[0], xy[1],
                                                                carry_z))
    goto(ctx, [p[0], p[1], carry_z], QUAT_DOWN, 1.0, tol=0.008, max_rows=26, kick=0.15)
    goto(ctx, [xy[0], xy[1], carry_z], QUAT_DOWN, 1.0, tol=0.008, max_rows=26, kick=0.15)
    # descend into the box; the hand's underside meets the counter/back rim ~2-3 cm above
    # the floor (the hand's bottom is eef_z+0.082 and the rim is at 1.175) and stalls there
    payload, ok = goto(ctx, [xy[0], xy[1], 1.092], QUAT_DOWN, 1.0, tol=0.006, max_rows=20,
                       kick=0.15, stall_rows=4)
    p = rig.pose_of(payload)
    _log("descent stopped eef=%s  z-FLOOR=%.4f (a stall 2-3 cm up is the hand at the mouth)"
         % (np.round(p, 4), p[2] - FLOOR_Z))
    # release with a small +z bias so the pads clear the pudding
    for _ in range(6):
        payload, _c = rig.step(ctx, [[0.0, 0.0, 0.35, 0.0, 0.0, 0.0, -1.0]], verbose=False)
    p = rig.pose_of(payload)
    _log("released at eef=%s gq=%.4f" % (np.round(p, 4), rig.gq_of(payload)))
    # retreat: up out of the mouth, then out along +y, then clear
    goto(ctx, [p[0], p[1], RIM_Z + 0.105], QUAT_DOWN, -1.0, tol=0.008, max_rows=16, kick=0.15)
    p = rig.pose_of(rig.observe(ctx))
    goto(ctx, [p[0], 0.065, RIM_Z + 0.105], QUAT_DOWN, -1.0, tol=0.010, max_rows=22, kick=0.15)
    s = rig.snapshot(ctx, tag="placed", outdir=DATA)
    col = rig.column_max_z(s, DROP_JAW_XY[0], DROP_JAW_XY[1], half=0.012)
    want = FLOOR_Z + PUD_H
    _log("column at the drop point: max_z=%s  want ~%.4f (floor %.3f + pudding %.4f)" % (
        None if col is None else round(col, 4), want, FLOOR_Z, PUD_H))
    old = pud_at(s, PUD_JAW_XY, half=0.05)
    _log("old table spot: %s" % ("EMPTY" if old is None else
                                 "STILL THERE (n=%d zmax=%.4f)" % (old["n"], old["zmax"])))
    rig.save_frame("placed", outdir=DATA)
    ok = col is not None and abs(col - want) < 0.020
    _log("in the box: %s" % ("YES" if ok else "NO/UNCLEAR"))
    return ok


# ----------------------------------------------------------------- close
def close(ctx):
    """Push the drawer shut: stop on the JOINT (eef stall), 1 mm verdict."""
    p = rig.pose_of(rig.observe(ctx))
    _log("close start eef=%s" % np.round(p, 4))
    # get out of the mouth and rise above the rim before swinging to the bar
    rig.servo(ctx, [p[0], p[1], RIM_Z + 0.075], QUAT_DOWN, -1.0, max_rows=14, tol=0.008)
    x_push = BAR_X_MID - JAW_DX
    rig.servo(ctx, [x_push, 0.075, RIM_Z + 0.05], QUAT_DOWN, -1.0, max_rows=24, tol=0.008)
    payload, _ = rig.servo(ctx, [x_push, 0.075, PUSH_EEF_Z], QUAT_DOWN, -1.0, max_rows=22,
                           tol=0.005, zclip=0.9)
    _log("pre-push eef=%s (bar front y=%.4f, bar x mid %.4f, bar z %.4f..%.4f)" % (
        np.round(rig.pose_of(payload), 4), BAR_Y, BAR_X_MID, BAR_Z_BOT, 1.1469))
    payload, ok, log = rig.push_to_stop(
        ctx, QUAT_DOWN, PUSH_EEF_Z, x_push, PANEL_Y, BODY_Y, ref_y=REF_Y,
        max_rows=46, stall_rows=3, stall_tol=0.0015, floor_z=FLOOR_Z)
    _log("push_to_stop ok=%s" % ok)
    rig.save_frame("closed", outdir=DATA)
    return ok


def run(ctx):
    _log("=== GRASP ===")
    if not grasp(ctx):
        _log("grasp did not confirm; stop so the stage can be re-run")
        return
    _log("=== PLACE ===")
    if not place(ctx):
        _log("place did not confirm; stop so the stage can be re-run")
        return
    _log("=== CLOSE ===")
    close(ctx)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    _log("ENTRY %s" % entry)
    t0 = time.time()
    fn = globals().get(entry)
    if fn is None or entry.startswith("_"):
        raise SystemExit("unknown ENTRY %r" % entry)
    out = fn(ctx)
    _log("entry %s done in %.1fs -> %r" % (entry, time.time() - t0, out))
    return out
