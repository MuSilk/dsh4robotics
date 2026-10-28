"""LIBERO-90 task 18 -- KITCHEN_SCENE3: put the FRYING PAN on the STOVE.

Scene this draw (0 rows of depth, `probe`):
  * frying pan: black disc + long handle, rim circle c=(-0.0776, -0.2492) R=0.0981,
    rim plane z=0.9870 at its axis, pan 0.0372 tall (base on the table 0.9498), tilt 0.4 deg.
    Outer wall is a strong cone (r 0.075 @ z0.955 -> 0.098 @ z0.985, ~38 deg from vertical),
    so the only place it can be pinched is the lip.
  * stove: square plate, top face x -0.16..+0.03, y 0.09..0.28, plate z 0.9747, and the spiral
    burner is RAISED 5 mm (z 0.9800, r ~0.076) -> the pan's base rests on the spiral.
  * moka pot at x 0.02..0.07, y -0.07..0.01, top 1.06 -- an obstacle only while carrying.
  * a small black pan/pot sits BEHIND the stove at x -0.26..-0.165, y 0.13..0.22, top 1.009
    -> the pan's rim must stay clear of it.

Measured on the first lift: a pan of R 0.098 pinched at its -y lip does NOT hang level -- it
settles at 27.5 deg (n = (0, +0.462, +0.887), +y side low; 74 deg is the free-hang angle, so
the pinch holds 2/3 of it) and that is a 3.5 cm drop of its low edge, so the placement has to
be levelled or compensated or the pan lands on one edge.

Stages (each runnable on the same open episode):
    ENTRY=probe   0 rows: scene ledger, pan rim, stove top, jaw/pad offsets
    ENTRY=pre     fold up and hover above the pan's -y rim
    ENTRY=grasp   descend onto the rim, close until settled, lift 0.13, measure what is held
    ENTRY=fix     level the carried pan by rotating the TOOL, capture q_hold
    ENTRY=carry   carry over the stove at q_hold, align the measured pan centre on TARGET
    ENTRY=place   closed-loop descent on the measured LOW edge, release, retreat
    ENTRY=look    0 rows: re-measure the pan (carried or free)
    ENTRY=score   0 rows: is the pan on the stove
    ENTRY=run     pre -> grasp -> fix -> carry -> place -> score
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "code", "lib"))
import rig  # noqa: E402

STATE = os.path.join(ROOT, "experiments", "pan18")
PLAN = os.path.join(STATE, "plan.json")

# ---- measured on this episode's probe (0 rows) -----------------------------------------
TABLE = 0.9498
PAN_C = np.array([-0.0776, -0.2492])      # rim circle centre, pan on the table
PAN_R = 0.0981
PAN_ZC = 0.9870                            # rim PLANE z at the pan's axis
PAN_H = PAN_ZC - TABLE                     # 0.0372: base plane -> rim plane
R_BASE = 0.075                             # pan base outer radius (r 0.075 @ z 0.955)
JDX, JDY = -0.0203, -0.0133                # jaw-pair centre - eef (this episode's probe)
PAD_DZ_M = 0.0417                          # pad bottom - eef_z (this episode's probe)
STOVE_TOP = 0.9747                         # plate
STOVE_SPIRAL = 0.9800                      # raised spiral burner = the real support surface
STOVE_X = (-0.160, 0.032)
STOVE_Y = (0.090, 0.280)
TARGET = np.array([-0.055, 0.185])         # pan CENTRE over the stove
Q_DOWN = np.array([0.9995, 0.0038, -0.0326, 0.0006])
CARRY_Z = 1.130                            # eef_z while carrying (base clears the moka pot)


def log(m):
    print(m, flush=True)


def ensure_dirs():
    os.makedirs(STATE, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "experiments", "data"), exist_ok=True)


def load_plan():
    try:
        with open(PLAN) as f:
            return json.load(f)
    except Exception:
        return {}


def save_plan(d):
    ensure_dirs()
    cur = load_plan()
    cur.update(d)
    with open(PLAN, "w") as f:
        json.dump(cur, f, indent=1)


def qhold():
    return np.array(load_plan().get("q_hold", list(Q_DOWN)), dtype=np.float64)


# ------------------------------------------------------------------ quaternion helpers
def level_quat(n_cur, q_cur):
    """Tool quaternion that takes the pan's rim normal `n_cur` to +z (world-frame rotation).

    `rig.level_quat` (SKILL §2): `q_axis_angle(n x z, acos(n_z))` LEFT-multiplied onto the
    current quat, then driven by `rig.rotate_to` (gated on rot_err), and the result is `q_hold`
    for every later row -- a later `goto` with `Q_DOWN` re-tilts the pan 28 deg in one move.
    """
    return rig.level_quat(n_cur, q_cur)


# ---------------------------------------------------------------- pan metrology (0 rows)
def pan_ring(s, jaw=None, centre=None, box=0.20, z_abs=None, quiet=False):
    """Rim circle+plane of the BLACK pan, carried, released, or standing.  Returns dict|None.

    `rig.carried_black_ring` (§4c) does the mask and the fit: the z window is what makes it work
    on a CARRIED pan, because the pan's own table shadow is at 0.9498 and the wrist's dark blocks
    sit above eef_z+0.09, so [eef−0.085, eef+0.080] keeps the pan and drops both.  Measured on
    this episode's first lift: without it the blob was the pan + its shadow + the wrist
    (8263 px) and no ring fit survived.  `centre`/`z_abs` switch it to a STANDING object (its
    own centre, table_z .. rim), where the eef-relative window is wrong by construction.

    Adds `base` (zc − h_pan) and `base_low` (the pan's LOWEST point, for the descent §2/§4).
    """
    eef = s.eef
    if eef is None:
        return None
    jx = JDX if jaw is None else float(jaw[0] - eef[0])
    jy = JDY if jaw is None else float(jaw[1] - eef[1])
    r = rig.carried_black_ring(s, jx, jy, centre=centre, box=box, Rlo=0.085, Rhi=0.115,
                               z_abs=z_abs, pad=0.033, tilt_max=45.0, quiet=True)
    if r is None:
        if not quiet:
            print("  pan_ring: no black blob with a usable %s ring" % ("pan" if centre is None
                                                                      else "pan"), flush=True)
        return None
    r["base"] = r["zc"] - PAN_H
    r["base_low"] = rig.base_low(r["base"], R_BASE, r["tilt"])
    if not quiet:
        print("  pan: c=%s R=%.4f zc=%.4f base=%.4f base_low=%.4f tilt=%.1f n=(%+.2f,%+.2f,"
              "%+.2f) arc=%.0f res=%.1fmm n=%d | centre-eef=%s zc-eef=%+.4f %s" % (
                  np.round(r["centre"], 4), r["R"], r["zc"], r["base"], r["base_low"],
                  r["tilt"], r["normal"][0], r["normal"][1], r["normal"][2], r["arc"],
                  r["res"] * 1000.0, r["n"], np.round(r["d_xy"], 4), r["d_z"],
                  "" if r["ok"] else "<-- SUSPECT"), flush=True)
    return r


def grasp_target():
    return rig.rim_grasp_target({"centre": PAN_C, "R": PAN_R, "rim_z": PAN_ZC},
                                jaw_dx=JDX, jaw_dy=JDY, dz=-0.055, side=-1.0)


def stove_planes(s):
    """Is the stove still where the probe put it?  Plate + raised spiral."""
    m = (s.z < 1.6) & (s.Z > 0.965) & (s.Z < 0.990) & (s.X > -0.20) & (s.X < 0.10) \
        & (s.Y > 0.05) & (s.Y < 0.33)
    p = s.pts[m]
    if len(p) < 200:
        return None
    return {"n": int(len(p)), "zmax": float(p[:, 2].max()),
            "plate": float(np.percentile(p[:, 2], 25)),
            "x": (float(p[:, 0].min()), float(p[:, 0].max())),
            "y": (float(p[:, 1].min()), float(p[:, 1].max()))}


# --------------------------------------------------------------------------- stages
def probe(ctx):
    s = rig.snapshot(ctx, tag="probe0")
    log("eef %s quat %s gq %s" % (np.round(s.eef, 4), np.round(s.quat, 4), s.gq))
    log("table_z %.4f" % s.table_z())
    log("--- planes")
    for p in s.planes():
        log("  %s" % (p,))
    log("--- pads %s" % (s.pads(),))
    log("--- jaw %s" % (s.jaw(),))
    log("--- stove %s" % (stove_planes(s),))
    log("--- pan (standing on the table)")
    pan_ring(s, centre=PAN_C, box=0.14, z_abs=(0.93, 1.03), quiet=False)
    log("--- grasp target %s" % ({k: np.round(v, 4) for k, v in grasp_target().items()
                                 if k != "side"},))
    for o in s.objects_on_table():
        log("  obj centre=%s top_xy=%s zmax=%.4f n=%d" % (
            np.round(o["centre"], 3), np.round(o["top_xy"], 3), o["zmax"], o["n"]))
    return "probe"


def pre(ctx):
    """Fold up, then hover 10 cm above the pan's -y rim.  No contact."""
    ensure_dirs()
    q = qhold()
    t = grasp_target()
    log("grasp: jaw %s eef %s pad_bottom %.4f (rim plane %.4f, rim top ~0.9896)" % (
        np.round(t["jaw"], 4), np.round(t["eef"], 4), t["pad_bottom"], PAN_ZC))
    p = rig.pose_of(rig.observe(ctx))
    rig.goto(ctx, [p[0], p[1], 1.300], q, -1.0, max_rows=16, tag="[up]")
    rig.goto(ctx, [t["eef"][0], t["eef"][1], 1.280], q, -1.0, max_rows=48, tag="[over]")
    payload, done = rig.goto(ctx, [t["eef"][0], t["eef"][1], 1.030], q, -1.0,
                             max_rows=30, tag="[hover]")
    p = rig.pose_of(payload)
    obs = rig.observe(ctx)
    log("hover at %s  residual %s  done=%s  j4=%.3f  quat=%s" % (
        np.round(p, 4), np.round(np.array(t["eef"]) - p, 4), done,
        obs["joint_pos"][3], np.round(rig.quat_of(obs), 4)))
    save_plan({"grasp_eef": list(map(float, t["eef"])),
               "grasp_jaw": list(map(float, t["jaw"]))})
    return "pre"


def grasp(ctx):
    """Descend onto the -y rim, close until the jaws settle, lift 0.13, measure."""
    ensure_dirs()
    q = qhold()
    t = grasp_target()
    payload, done = rig.goto(ctx, t["eef"], q, -1.0, max_rows=22, tol=0.004, tag="[seat]")
    p = rig.pose_of(payload)
    log("seat residual %s done=%s" % (np.round(np.array(t["eef"]) - p, 4), done))
    payload, done, gq = rig.close_grip(ctx)
    log("closed: gq %.4f (gap %.1f mm)" % (gq, 2000.0 * gq))
    p = rig.pose_of(payload)
    rig.goto(ctx, [p[0], p[1], p[2] + 0.130], q, 1.0, max_rows=18, tol=0.008,
             tag="[lift]")
    s = rig.snapshot(ctx, tag="lifted")
    r = pan_ring(s)
    if r is None:
        log("!! nothing black-and-round in the jaws -- the pinch failed")
        return "grasp FAILED"
    log("held: centre=eef%s  base %.4f  tilt %.1f  -> %s" % (
        np.round(r["d_xy"], 4), r["base"], r["tilt"], "OK" if r["ok"] else "SUSPECT"))
    save_plan({"gq": gq, "d_xy": list(map(float, r["d_xy"])), "d_z": float(r["d_z"]),
               "tilt": float(r["tilt"])})
    rig.save_frame("lifted")
    return "grasp %s" % ("OK" if r["ok"] else "SUSPECT")


def fix(ctx):
    """Level the carried pan: rotate the TOOL by the rotation that takes the rim normal to +z.

    The grip is a clamp, so the pan follows the wrist -- but a rotation-only row also walks
    the arm 4-8 cm, so every later row must pass the CAPTURED quaternion (SKILL 2).
    """
    for k in range(3):
        s = rig.snapshot(ctx, tag="fix%d" % k)
        r = pan_ring(s)
        if r is None:
            return "fix: lost the pan"
        if r["tilt"] < 5.0:
            log("fix: level enough (%.1f deg)" % r["tilt"])
            break
        q = qhold()
        q_new, deg = level_quat(r["normal"], rig.quat_of(rig.observe(ctx)))
        p = rig.pose_of(rig.observe(ctx))
        log("fix %d: tilt %.1f -> rotating the tool %.1f deg about (normal x z); hold %s"
            % (k, r["tilt"], deg, np.round(p, 4)))
        rig.rotate_to(ctx, q_new, p, gripper=1.0, max_rows=22, tol=0.03,
                      tag="[level%d]" % k)
        save_plan({"q_hold": list(map(float, q_new))})
        p2 = rig.pose_of(rig.observe(ctx))
        log("   after the rotation eef %s (walked %s)" % (
            np.round(p2, 4), np.round(p2 - p, 4)))
        rig.goto(ctx, [p2[0], p2[1], max(p2[2], CARRY_Z)], q_new, 1.0, max_rows=14,
                 tol=0.006, tag="[rise]")
    s = rig.snapshot(ctx, tag="fixed")
    r = pan_ring(s)
    if r is not None:
        log("fix done: tilt %.1f, base %.4f, base_low %.4f" % (
            r["tilt"], r["base"], r["base_low"]))
        save_plan({"d_xy": list(map(float, r["d_xy"])), "d_z": float(r["d_z"]),
                   "tilt": float(r["tilt"])})
    rig.save_frame("fixed")
    return "fix"


def carry(ctx):
    """Carry over the stove at q_hold/CARRY_Z, then align the pan's CENTRE on TARGET."""
    q = qhold()
    s = rig.snapshot(ctx, tag="carry0")
    r = pan_ring(s)
    if r is None:
        return "carry: lost the pan"
    aim = np.array([TARGET[0] - r["d_xy"][0], TARGET[1] - r["d_xy"][1], CARRY_Z])
    log("carry -> eef %s (pan centre -> %s)" % (np.round(aim, 4), np.round(TARGET, 4)))
    rig.goto(ctx, [s.eef[0], s.eef[1], CARRY_Z], q, 1.0, max_rows=14, tag="[rise]")
    rig.goto(ctx, aim, q, 1.0, max_rows=70, tag="[carry]")
    for k in range(3):
        s = rig.snapshot(ctx, tag="align%d" % k)
        r = pan_ring(s)
        if r is None:
            return "carry: lost the pan while aligning"
        corr = np.array([TARGET[0] - r["d_xy"][0], TARGET[1] - r["d_xy"][1], CARRY_Z])
        err = corr - s.eef
        log("align %d: pan centre %s (err %s) tilt %.1f" % (
            k, np.round(r["centre"][:2], 4), np.round(err[:2], 4), r["tilt"]))
        if np.max(np.abs(err[:2])) < 0.005:
            break
        rig.goto(ctx, corr, q, 1.0, max_rows=16, tol=0.004, tag="[align]")
    rig.save_frame("over_stove")
    return "carry"


def _descend_step(ctx, xy, q, goal, max_rows=8):
    """One closed-loop step on the MEASURED LOW edge of the pan.  Returns (scene, ring)."""
    s = rig.snapshot(ctx, tag="d")
    r = pan_ring(s)
    if r is None:
        return s, None
    log("   base_low %.4f (goal %.4f, %+.1f mm) tilt %.1f base %.4f zc-eef %+.4f" % (
        r["base_low"], goal, 1000.0 * (r["base_low"] - goal), r["tilt"], r["base"], r["d_z"]))
    if r["base_low"] <= goal + 0.0005:
        return s, r
    z_t = s.eef[2] + (goal - r["base_low"])
    z_t = max(z_t, s.eef[2] - 0.008)
    rig.goto(ctx, [xy[0], xy[1], z_t], q, 1.0, max_rows=max_rows, tol=0.002, tag="[d]")
    return s, r


def place(ctx):
    """Descend on the measured LOW edge onto the spiral, verify, release, retreat."""
    q = qhold()
    goal = STOVE_SPIRAL + 0.0010
    s = rig.snapshot(ctx, tag="place0")
    r = pan_ring(s)
    if r is None:
        return "place: lost the pan"
    xy = np.array([TARGET[0] - r["d_xy"][0], TARGET[1] - r["d_xy"][1]])
    # coarse move: predicted eef_z for the goal, stopped 2 cm short
    z_goal = s.eef[2] + (goal - r["base_low"])
    log("descend: eef_xy %s, pan centre %s, base_low %.4f -> %.4f (eef_z %.4f -> %.4f)" % (
        np.round(xy, 4), np.round(r["centre"][:2], 4), r["base_low"], goal, s.eef[2], z_goal))
    rig.goto(ctx, [xy[0], xy[1], z_goal + 0.020], q, 1.0, max_rows=30, tol=0.004,
             tag="[coarse]")
    for i in range(12):
        s, r = _descend_step(ctx, xy, q, goal, max_rows=6)
        if r is None or r["base_low"] <= goal + 0.0005:
            break
    if r is None:
        return "place: lost the pan during the descent"
    log("at the bottom: base %.4f base_low %.4f vs spiral %.4f (%+.1f mm), tilt %.1f, "
        "centre %s" % (r["base"], r["base_low"], STOVE_SPIRAL,
                       1000.0 * (r["base_low"] - STOVE_SPIRAL), r["tilt"],
                       np.round(r["centre"][:2], 4)))
    rig.save_frame("about_to_release")
    # release: open the jaws while holding the pose
    for i in range(14):
        payload, _ = rig.step(ctx, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], verbose=False)
        if i % 5 == 4:
            log("   release row %d gq %.4f" % (i, rig.gq_of(payload)))
    p = rig.pose_of(payload)
    rig.goto(ctx, [p[0], p[1], p[2] + 0.140], q, -1.0, max_rows=18, tol=0.008,
             tag="[retreat]")
    rig.save_frame("released")
    return "place"


def look(ctx):
    s = rig.snapshot(ctx, tag=os.environ.get("TAG", "look"))
    pan_ring(s)
    log("stove %s" % (stove_planes(s),))
    return "look"


def score(ctx):
    s = rig.snapshot(ctx, tag="score")
    r = pan_ring(s, quiet=False)
    st = stove_planes(s)
    log("stove %s" % (st,))
    if r is None:
        log("SCORE: no pan found")
        return "score: no pan"
    cx, cy = float(r["centre"][0]), float(r["centre"][1])
    over = STOVE_X[0] <= cx <= STOVE_X[1] and STOVE_Y[0] <= cy <= STOVE_Y[1]
    log("SCORE: pan centre (%+.4f,%+.4f)  base %.4f base_low %.4f tilt %.1f" % (
        cx, cy, r["base"], r["base_low"], r["tilt"]))
    log("SCORE: base - plate = %+.1f mm ; base_low - spiral = %+.1f mm ; centre over the "
        "stove face = %s" % (1000.0 * (r["base"] - STOVE_TOP),
                             1000.0 * (r["base_low"] - STOVE_SPIRAL), over))
    log("SCORE: verdict %s" % ("ON THE STOVE" if (over and r["base_low"] < STOVE_TOP + 0.020)
                               else "NOT on the stove"))
    return "score"


def run(ctx):
    for st in (pre, grasp, fix, carry, place):
        out = st(ctx)
        log("== %s -> %s" % (st.__name__, out))
        if "FAILED" in str(out) or "lost" in str(out):
            return out
    return score(ctx)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    return globals()[entry](ctx)
