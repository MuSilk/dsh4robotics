"""LIBERO-90 task 17 -- stack the MIDDLE black bowl on the BACK black bowl.

Design, from SKILL.md 0.7 + tasks/stack-bowl.md (all of it bought by earlier episodes):

  * the clause reads CONTACT AT THE LOWER BOWL'S RIM PLANE.  Sunk in (-13 mm) is False,
    a held clearance (+9 mm, no contact) is False; base ON the rim ring (-0.3 / -0.8 mm)
    is the only state that has ever been produced.  A RELEASE there does not survive
    (#4: -14.8 mm, 9.7 deg on release; foot ring r ~ 0.028 on a 0.053-0.0566 annulus is
    one-point support outboard of the COM).  So the ending kept here is the HELD one:
    press until the measured base plane is AT the rim, keep the jaws closed, score from
    depth, finish.
  * the destination is chosen by DISTANCE TO THE LEDGER, never by a measured height.
    x orders the bowls: front = max x, middle = median x, back = min x.

Measured on this episode's probe (0 rows):
    table_z 0.9498, h_bowl 0.04952 (three bowls agree to 0.1 mm)
    back   c=(-0.1653, 0.0354) R 0.0546 zc 0.99933     <- destination
    middle c=(-0.0586, 0.1847) R 0.0545 zc 0.99932     <- payload
    front  c=( 0.0903, 0.1164) R 0.0546 zc 0.99925
    annulus of the back bowl 0.0535 .. 0.0560 -> window 0.0255..0.0280, d = 0.0267
    pad bottom - eef_z 0.0407 (flat at 0.03/0.045/0.06), jaw pair - eef (-0.0204,-0.0123)

THE ONE ARITHMETIC THAT MUST NOT BE GOT WRONG (it cost this run a nested bowl):
    carried rim plane zc = eef_z + rel_z ;  carried base = zc - h_bowl
    =>  eef_z for a wanted base B is   B + h_bowl - rel_z
(`B - rel_z` puts the bowl h_bowl = 49.5 mm into the destination.)

Stages, each runnable alone on the same open episode:

    ENTRY=probe  0 rows: bowls, rim planes, h_bowl, annulus, pad/jaw offsets
    ENTRY=grasp  pinch the MIDDLE bowl's -y rim, lift 0.13, measure what is in the jaws
    ENTRY=fix    lift clear, LEVEL the bowl (17 deg lock), hover 40 mm over the BACK bowl
    ENTRY=seat   closed-loop descent on the MEASURED base plane; press 1 mm; report
    ENTRY=look   0 rows: ledger check -- every grey blob + the carried bowl
    ENTRY=score  0 rows: score on(upper, lower) from depth
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lib"))
import rig  # noqa: E402

DATA = os.path.join(ROOT, "experiments", "data")
STATE = os.path.join(ROOT, "experiments", "stack17f")
QF = os.path.join(STATE, "plan.json")

# the carried bowl hangs on the eef's +y side after a -y rim pinch (SKILL 4b)
CARRY_HINT = (-0.020, 0.038)
D_DEFAULT = 0.0267


def log(m):
    print(m, flush=True)


def ensure_dirs():
    for d in (DATA, STATE):
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------ quaternion / geometry
def q_axis_angle(axis, ang):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9 or abs(ang) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis / n
    s = np.sin(0.5 * ang)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * ang)])


def plane_normal(p):
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    return n / (np.linalg.norm(n) + 1e-12)


def circle_in_plane(rp, n):
    """Circle fit of 3-D points in a plane with normal n -> centre/R/zc/arc/res."""
    c0 = rp.mean(0)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n, a)
    e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    d = rp - c0
    u, v = d @ e1, d @ e2
    A = np.c_[2 * u, 2 * v, np.ones(len(u))]
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cu, cv = float(sol[0]), float(sol[1])
    R = float(np.sqrt(max(sol[2] + cu * cu + cv * cv, 1e-9)))
    centre = c0 + cu * e1 + cv * e2
    res = float(np.sqrt(np.mean((np.hypot(u - cu, v - cv) - R) ** 2)))
    ang = np.sort(np.mod(np.degrees(np.arctan2(v - cv, u - cu)), 360.0))
    arc = float(360.0 - np.diff(np.r_[ang, ang[0] + 360.0]).max())
    return centre, R, float(centre[2]), arc, res


def robust_ring(scene, mask, Rlo=0.045, Rhi=0.066):
    """Rim fit that does not need a good seed: brute-force the tilt, then refine.

    The +z-seeded coarse-to-fine schedule (SKILL 4b's fit_ring3d) fails on a bowl hanging
    at a pivot tilt -- the top band is then only the near arc, the plane fit returns a
    50-70 deg normal and the schedule collapses (measured here: true R 0.0542 / tilt 12.6 /
    arc 181 / res 0.7 mm, while the +z seed gave R 0.016 / tilt 41).  A 12 x 24 grid over
    (tilt, azimuth) is ~300 circle fits on ~2000 points, costs no rows, and recovers the
    ring at 3 deg, at 10 deg and at 12.6 deg.
    """
    p = scene.pts[mask]
    if len(p) < 120:
        return None
    best = None
    for t in np.arange(0.0, 34.0, 3.0):
        tr = np.radians(t)
        for k in range(24):
            az = 2.0 * np.pi * k / 24.0
            n = np.array([np.sin(tr) * np.cos(az), np.sin(tr) * np.sin(az), np.cos(tr)])
            h = p @ n
            rp = p[h > h.max() - 0.003]
            if len(rp) < 80:
                continue
            _c, R, _zc, _arc, res = circle_in_plane(rp, n)
            if Rlo < R < Rhi and res < 0.0015:
                if best is None or len(rp) > best[0]:
                    best = (len(rp), n)
    if best is None:
        return None
    n = best[1]
    out = None
    for b in (0.008, 0.006, 0.005, 0.004, 0.003):
        h = p @ n
        rp = p[h > h.max() - b]
        if len(rp) < 60:
            break
        nn = plane_normal(rp)
        if nn[2] < 0:
            nn = -nn
        centre, R, zc, arc, res = circle_in_plane(rp, nn)
        n = nn
        out = {"centre": centre, "R": R, "zc": zc, "rim_z": float(p[:, 2].max()),
               "normal": nn, "tilt": float(np.degrees(np.arccos(np.clip(nn[2], -1, 1)))),
               "arc": arc, "res": res, "band": b, "n_ring": int(len(rp))}
    return out


def level_align(s, r):
    """Quaternion that makes the carried bowl level: rotate the TOOL by the bowl's tilt."""
    n = r["normal"]
    axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
    sn = float(np.linalg.norm(axis))
    ang = float(np.arctan2(sn, float(n[2])))
    if sn < 1e-6 or abs(np.degrees(ang)) < 0.5:
        return np.asarray(s.quat, dtype=np.float64), 0.0
    return (rig.qmul(q_axis_angle(axis, ang), np.asarray(s.quat, dtype=np.float64)),
            float(np.degrees(ang)))


# ------------------------------------------------------------------ plan bookkeeping
def save_plan(plan):
    ensure_dirs()

    def plain(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v

    with open(QF, "w") as f:
        json.dump(plain(plan), f, indent=1)


def load_plan():
    with open(QF) as f:
        return json.load(f)


def bowl(plan, name):
    for b in plan["bowls"]:
        if b["name"] == name:
            return b
    raise KeyError(name)


def eef_for_base(base, rel_z, h):
    """THE formula: eef_z that puts the carried bowl's BASE PLANE at `base`."""
    return float(base) + float(h) - float(rel_z)


# ------------------------------------------------------------------ masks / measurement
def bowl_candidates(scene, tz):
    """Grey speckled blobs on the table, as recs with their rim fit, sorted by x."""
    mx, mn, mean = scene.rgb.max(2), scene.rgb.min(2), scene.rgb.mean(2)
    m = ((scene.Z > tz + 0.012) & (scene.Z < tz + 0.085)
         & (mean > 60) & (mean < 188) & ((mx - mn) < 45)
         & (scene.X > -0.40) & (scene.X < 0.22)
         & (scene.Y > -0.30) & (scene.Y < 0.38))
    out, work = [], m.copy()
    while work.sum() >= 200:
        keep, info = scene.blob(work, min_pixels=200)
        if keep is None:
            break
        work &= ~keep
        r = rig.rim_plane(scene, keep)
        rec = {"mask": keep, "blob": info, "rim": r,
               "rgb": np.round(scene.rgb[keep].mean(0), 0)}
        if r is not None:
            rec.update({"R": r["R"], "zc": r["zc"], "centre": r["centre"],
                        "tilt": r["tilt"], "arc": r["arc"], "dz": r["zc"] - tz})
        out.append(rec)
    bow = [c for c in out if c["rim"] is not None and 0.045 < c["R"] < 0.072
           and 0.040 < c["dz"] < 0.062 and c["arc"] > 90]
    bow.sort(key=lambda c: c["centre"][0])
    return bow, out


def measure_carried(s, plan, hint=CARRY_HINT, z_lo=None, z_pred=None, quiet=False):
    """Geometry-only carried-bowl measurement (SKILL 4b), made robust to the HAND BLOCK.

    The mask is a box around the expected bowl centre, a z window, and a 36 mm exclusion
    cylinder around the MEASURED jaw-pair centre.  The z window is the hard part: the
    carried bowl's rim sits eef+0.05..eef+0.075, but at a pitched tool the hand block's
    underside reaches down to eef+0.02, so a window with a fixed generous top measures the
    HAND (circle fits of R 0.07-0.14 with long arcs).  So: try a ladder of windows, keep
    every fit whose R matches the probe's rim radius to 2.5 mm with a small residual, and
    take the one with the most ring pixels.  z_lo (an absolute floor, = destination rim
    + 4 mm) keeps the destination bowl's own rim out of every window.
    """
    eef = s.eef
    jx = plan.get("jaw_dx", rig.JAW_DX)
    jy = plan.get("jaw_dy", rig.JAW_DY)
    jaw = np.array([eef[0] + jx, eef[1] + jy])
    hxy = np.array([eef[0] + hint[0], eef[1] + hint[1]])
    R0 = plan.get("R_upper") or bowl(plan, "middle")["R"]
    lo_abs = -1e9 if z_lo is None else float(z_lo)
    if z_pred is None:
        spans = [(-0.020, 0.070), (-0.020, 0.085), (-0.020, 0.100),
                 (-0.020, 0.115), (0.0, 0.060)]
    else:
        spans = [(-0.020, 0.012), (-0.030, 0.030), (-0.045, 0.050), (-0.020, 0.075),
                 (-0.020, 0.100)]
    cands = []
    for (dn, up) in spans:
        ref = eef[2] if z_pred is None else float(z_pred)
        zlo, zhi = max(lo_abs, ref + dn), ref + up
        if zhi <= zlo:
            continue
        m = ((np.abs(s.X - hxy[0]) < 0.085) & (np.abs(s.Y - hxy[1]) < 0.085)
             & (s.Z > zlo) & (s.Z < zhi)
             & (np.hypot(s.X - jaw[0], s.Y - jaw[1]) > 0.036))
        keep, info = s.blob(m, min_pixels=150)
        if keep is None:
            continue
        r = robust_ring(s, keep, Rlo=max(0.040, R0 - 0.010), Rhi=min(0.070, R0 + 0.010))
        if r is None:
            continue
        r["n"] = int(info["n"])
        r["win"] = (round(zlo, 4), round(zhi, 4))
        r["dev"] = abs(r["R"] - R0)
        cands.append(r)
    good = [c for c in cands if c["dev"] < 0.0025 and c["res"] < 0.0013 and c["arc"] > 110
            and np.linalg.norm(c["centre"][:2] - hxy) < 0.045]
    if not good:
        if not quiet:
            log("  carried: NO usable ring fit (%d candidates: %s)" % (
                len(cands), [(round(c["R"], 3), round(c["zc"], 3), round(c["arc"], 0))
                             for c in cands]))
        return None
    good.sort(key=lambda c: -c["n_ring"])
    r = good[0]
    h = plan["h_bowl"]
    r["hint"] = hxy
    r["base"] = r["zc"] - h
    r["ok"] = True
    r["d_xy"] = r["centre"][:2] - eef[:2]
    r["d_z"] = r["zc"] - eef[2]
    if not quiet:
        log("  carried: c=%s R=%.4f zc=%.4f base=%.4f tilt=%.1f arc=%.0f res=%.1fmm "
            "nring=%d win=%s" % (
                np.round(r["centre"], 4), r["R"], r["zc"], r["base"], r["tilt"],
                r["arc"], r["res"] * 1000.0, r["n_ring"], r["win"]))
        log("           rel centre-eef=%s  rel zc-eef=%+.4f  (%d candidates)" % (
            np.round(r["d_xy"], 4), r["d_z"], len(good)))
        for c in good[1:]:
            log("           alt: zc=%.4f R=%.4f arc=%.0f tilt=%.1f win=%s" % (
                c["zc"], c["R"], c["arc"], c["tilt"], c["win"]))
    return r


def dest_now(s, plan, quiet=False):
    """Re-identify the destination bowl: rig.rim_plane + a fixed-R cross-check against the
    ledger.  It is invisible once the carried bowl is within 5 mm of it, so do it early."""
    b = bowl(plan, "back")
    c = np.array(b["centre"])
    zc0 = b["zc"]
    m = ((np.abs(s.X - c[0]) < 0.075) & (np.abs(s.Y - c[1]) < 0.075)
         & (s.Z > zc0 - 0.030) & (s.Z < zc0 + 0.006))
    keep, info = s.blob(m, min_pixels=200)
    out = {"centre": c, "zc": zc0, "R": b["R"], "moved": 0.0, "source": "ledger"}
    if keep is None:
        if not quiet:
            log("  dest: no blob at the ledger spot (%s) -> ledger" % info)
        return out
    r = rig.rim_plane(s, keep)
    if r is None or not (0.045 < r["R"] < 0.072) or r["arc"] < 90:
        if not quiet:
            log("  dest: fit unusable (%s) -> ledger" % (
                None if r is None else (round(r["R"], 4), round(r["arc"], 0))))
        return out
    moved = float(np.linalg.norm(r["centre"] - c))
    dz = float(r["zc"] - zc0)
    if not quiet:
        log("  dest: c=%s R=%.4f zc=%.4f tilt=%.1f arc=%.0f moved=%.1fmm dz=%+.1fmm" % (
            np.round(r["centre"], 4), r["R"], r["zc"], r["tilt"], r["arc"],
            moved * 1000.0, dz * 1000.0))
    if moved > 0.012:
        log("  dest: WARNING the fit is %.1f mm from the ledger -- keeping the ledger"
            % (moved * 1000.0))
    else:
        out.update({"centre": r["centre"], "zc": r["zc"], "R": r["R"],
                    "source": "fit", "tilt": r["tilt"]})
    out["moved"] = moved
    out["dz"] = dz
    return out


def annulus(scene, mask):
    """Radii of the pixels in the top 1.5 mm of a rim -> (lo, hi) of the support ring."""
    p = scene.pts[mask]
    if len(p) < 100:
        return None
    top = p[p[:, 2] > p[:, 2].max() - 0.0015]
    if len(top) < 40:
        return None
    r = rig.rim_plane(scene, mask)
    if r is None:
        return None
    rr = np.hypot(top[:, 0] - r["centre"][0], top[:, 1] - r["centre"][1])
    return float(np.percentile(rr, 5)), float(np.percentile(rr, 95)), r


# ------------------------------------------------------------------ motion helpers
def rotate_to(ctx, q_tgt, hold_xyz, gripper=1.0, max_rows=16, tol=0.05, tag="[rot]",
              verbose=True):
    """Rotate IN PLACE until the orientation error is under tol.

    A `goto` whose only error is rotational returns on row 0 (it gates on position) and
    never rotates at all -- that left this run's bowl at a 14 deg pivot tilt.
    """
    payload = rig.observe(ctx)
    hold = np.asarray(hold_xyz, dtype=np.float64)
    for i in range(int(max_rows)):
        err = rig.rotvec_err(rig.quat_of(payload), q_tgt)
        if np.linalg.norm(err) < tol:
            if verbose:
                log("  %s converged rot_err=%.3f after %d rows" % (
                    tag, float(np.linalg.norm(err)), i))
            return payload, True
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[:3] = np.clip((hold - p) / 0.011, -0.5, 0.5)
        a[3:6] = np.clip(err / 0.30, -1, 1)
        a[6] = float(gripper)
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        if verbose and i % 3 == 0:
            log("  %s %2d rot_err=%.3f eef=%s" % (tag, i, float(np.linalg.norm(err)),
                                                  np.round(rig.pose_of(payload), 4)))
    if verbose:
        log("  %s hit max_rows rot_err=%.3f" % (
            tag, float(np.linalg.norm(rig.rotvec_err(rig.quat_of(payload), q_tgt)))))
    return payload, False


def descend_rows(ctx, z_target, q_cur, gripper, step=0.0012, max_rows=40, floor_z=None,
                 hold_xy=None):
    """Rows of ~`step` m down to z_target, never past floor_z.  Returns (payload, rows)."""
    payload = rig.observe(ctx)
    rows = 0
    while rows < int(max_rows):
        p = rig.pose_of(payload)
        err = float(z_target - p[2])
        if err > -1e-5:
            break
        if floor_z is not None and p[2] <= floor_z + 1e-6:
            log("  descend: at the hard floor eef_z %.4f" % p[2])
            break
        a = np.zeros(7)
        if hold_xy is not None:
            a[0] = float(np.clip((hold_xy[0] - p[0]) / 0.011, -0.35, 0.35))
            a[1] = float(np.clip((hold_xy[1] - p[1]) / 0.011, -0.35, 0.35))
        a[2] = float(np.clip(err / 0.011, -step / 0.011, 0.0))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_cur) / 0.5, -1, 1)
        a[6] = float(gripper)
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        rows += 1
    return payload, rows


# ------------------------------------------------------------------ stages
def probe(ctx):
    ensure_dirs()
    s = rig.snapshot(ctx, tag="probe0")
    tz = s.table_z()
    log("eef %s  quat %s  gq %.4f" % (np.round(s.eef, 4), np.round(s.quat, 4), abs(s.gq[0])))
    log("table_z %.4f" % tz)
    for half in (0.030, 0.045, 0.060):
        log("pads half=%.3f %s" % (half, s.pads(half=half)))
    j = s.jaw()
    log("jaw %s" % ({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in (j or {}).items()}))
    log("hand_lead %s" % (s.hand_lead(),))
    for p in s.planes(axes="z", zrange=(0.94, 1.10))[:4]:
        log("plane %s" % (p,))

    bows, allb = bowl_candidates(s, tz)
    log("grey blobs:")
    for c in allb:
        log("   n=%4d x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f] rgb=%s R=%s zc=%s" % (
            c["blob"]["n"], c["blob"]["x"][0], c["blob"]["x"][1],
            c["blob"]["y"][0], c["blob"]["y"][1], c["blob"]["z"][0], c["blob"]["z"][1],
            c["rgb"], None if "R" not in c else round(c["R"], 4),
            None if "zc" not in c else round(c["zc"], 5)))
    if len(bows) < 3:
        log("!! only %d bowl-like blobs -- stop and look" % len(bows))
        return
    h, hs = rig.bowl_height(s, [c["mask"] for c in bows], table_z=tz)
    log("h_bowl median=%.5f per-bowl=%s" % (h, np.round(hs, 5)))

    # x orders them: front = max x, middle = median x, back = min x (SKILL 4)
    names = ["back", "middle", "front"]
    plan = {"table_z": tz, "h_bowl": float(h), "bowls": [], "d": D_DEFAULT,
            "jaw_dx": float(j["centre_dx"]) if j else rig.JAW_DX,
            "jaw_dy": float(j["centre_dy"]) if j else rig.JAW_DY,
            "pad_dz": float(s.pads()["offset"]) if s.pads() else rig.PAD_DZ}
    for i in range(3):
        c = bows[i]
        plan["bowls"].append({"name": names[i], "centre": list(map(float, c["centre"])),
                              "R": c["R"], "zc": c["zc"], "rim_z": c["rim"]["rim_z"],
                              "tilt": c["tilt"], "arc": c["arc"], "n": c["blob"]["n"]})
    plan["R_upper"] = bowl(plan, "middle")["R"]
    ab = annulus(s, bows[0]["mask"])          # destination = the BACK bowl = min x
    if ab:
        lo, hi, _r = ab
        plan["annulus"] = [lo, hi]
        plan["d"] = float(0.5 * (lo + hi) - 0.028)
        log("annulus radii %.4f .. %.4f (ring %.1f mm) -> window %.4f .. %.4f, d = %.4f" % (
            lo, hi, (hi - lo) * 1000.0, lo - 0.028, hi - 0.028, plan["d"]))
    for b in plan["bowls"]:
        log("  %-6s c=%s R=%.4f zc=%.5f rim_z=%.5f tilt=%.1f arc=%.0f n=%d" % (
            b["name"], np.round(b["centre"], 4), b["R"], b["zc"], b["rim_z"],
            b["tilt"], b["arc"], b["n"]))
    mid = bowl(plan, "middle")
    t = rig.rim_grasp_target({"centre": np.array(mid["centre"]), "R": mid["R"],
                              "rim_z": mid["zc"]}, plan["jaw_dx"], plan["jaw_dy"],
                             dz=-0.055, side=-1.0)
    log("grasp target: jaw %s eef %s pad_bottom %.4f (rim plane %.4f)" % (
        np.round(t["jaw"], 4), np.round(t["eef"], 4), t["pad_bottom"], mid["zc"]))
    save_plan(plan)


def grasp(ctx):
    plan = load_plan()
    s0 = rig.snapshot(ctx, tag="grasp0")
    q_down = s0.quat
    log("start eef %s quat %s joints=%s" % (
        np.round(s0.eef, 4), np.round(q_down, 4),
        np.round(rig.observe(ctx).get("joint_pos", []), 2)))
    mid = bowl(plan, "middle")
    t = rig.rim_grasp_target({"centre": np.array(mid["centre"]), "R": mid["R"],
                              "rim_z": mid["zc"]}, plan["jaw_dx"], plan["jaw_dy"],
                             dz=-0.055, side=-1.0)
    eef_t = t["eef"]
    log("grasp: jaw -> %s eef -> %s" % (np.round(t["jaw"], 4), np.round(eef_t, 4)))
    rig.goto(ctx, [eef_t[0], eef_t[1], eef_t[2] + 0.14], q_down, -1.0, tol=0.006,
             max_rows=26, tag="[above]")
    payload, done = rig.goto(ctx, eef_t, q_down, -1.0, tol=0.004, max_rows=22, tag="[down]")
    p = rig.pose_of(payload)
    log("  at grasp pose eef %s err=%s done=%s" % (
        np.round(p, 4), np.round(eef_t - p, 4), done))
    payload, complete, gq = rig.close_grip(ctx, max_rows=26)
    log("  jaws settled gq=%.4f" % gq)
    p = rig.pose_of(payload)
    rig.goto(ctx, [p[0], p[1], p[2] + 0.13], q_down, 1.0, tol=0.008, max_rows=18,
             tag="[lift]")
    s = rig.snapshot(ctx, tag="grasp_lift")
    gone, info = rig.object_gone(s, tuple(mid["centre"]))
    log("  source spot gone=%s zmax=%s" % (
        gone, None if info.get("n", 0) == 0 else round(info["zmax"], 4)))
    r = measure_carried(s, plan)
    if r is not None:
        log("  lift: base %.4f  gq %.4f" % (r["base"], gq))
    if gq <= 0.0022:
        log("  !! gq <= 0.0022: closed on nothing (SKILL 2) -- do not carry")


def fix(ctx):
    """Recover from the pivot tilt: lift clear, LEVEL, hover over the destination.

    Idempotent: run it again if the bowl tilts on any later move.
    """
    plan = load_plan()
    d = plan.get("d", D_DEFAULT)
    h = plan["h_bowl"]
    dest = bowl(plan, "back")
    rim = dest["zc"]
    q_hold = np.array(plan["q_hold"]) if "q_hold" in plan else None
    s = rig.snapshot(ctx, tag="fix0")
    r = measure_carried(s, plan, z_lo=rim + 0.004)
    if r is None:
        log("!! fix: cannot see the carried bowl at the start")
        return
    log("fix: eef %s, carried base %.4f (%+.1f mm vs rim %.4f), tilt %.1f deg" % (
        np.round(s.eef, 4), r["base"], 1000.0 * (r["base"] - rim), rim, r["tilt"]))
    if q_hold is None:
        q_hold = np.asarray(s.quat, dtype=np.float64)
    # 1. lift so the carried base is 60 mm above the destination rim
    z_up = eef_for_base(rim + 0.060, r["d_z"], h)
    log("  lift to eef_z %.4f (base = rim + 60 mm)" % z_up)
    payload, done = rig.goto(ctx, [s.eef[0], s.eef[1], z_up], q_hold, 1.0, tol=0.005,
                             max_rows=14, stall_rows=10, tag="[clear]")
    q_hold = rig.quat_of(payload)
    # 2. level: rotate the tool by the bowl's measured tilt, re-measure, repeat
    for i in range(3):
        s = rig.snapshot(ctx, tag="fix_l%d" % i)
        r = measure_carried(s, plan, z_lo=rim + 0.004)
        if r is None:
            log("!! fix: cannot see the carried bowl while levelling")
            return
        if r["tilt"] < 2.0:
            log("  fix: level (tilt %.1f deg) -- q_hold %s" % (r["tilt"], np.round(s.quat, 4)))
            q_hold = np.asarray(s.quat, dtype=np.float64)
            break
        q_tgt, deg = level_align(s, r)
        log("  fix: bowl tilt %.1f deg -> rotating the tool %.1f deg" % (r["tilt"], deg))
        payload, ok = rotate_to(ctx, q_tgt, s.eef, gripper=1.0, max_rows=16, tol=0.05,
                                tag="[level%d]" % i)
        q_hold = rig.quat_of(payload)
    # 3. final measurement in the level pose
    s = rig.snapshot(ctx, tag="fix_lv")
    r = measure_carried(s, plan, z_lo=rim + 0.004)
    if r is None:
        log("!! fix: carried bowl unmeasurable at the level pose")
        return
    rel_xy, rel_z = r["d_xy"], r["d_z"]
    log("  LEVEL: base %.4f (%+.1f mm vs rim), tilt %.1f deg, rel centre-eef %s rel zc-eef "
        "%+.4f" % (r["base"], 1000.0 * (r["base"] - rim), r["tilt"],
                   np.round(rel_xy, 4), rel_z))
    plan["q_hold"] = list(map(float, q_hold))
    plan["rel_xy"] = list(map(float, rel_xy))
    plan["rel_z"] = float(rel_z)
    save_plan(plan)
    # 4. hover over the destination with the carried base 40 mm above the rim plane
    dest = dest_now(s, plan)
    tgt_c = np.array([dest["centre"][0], dest["centre"][1] + d])
    eef_xy = tgt_c - rel_xy
    z_hover = eef_for_base(dest["zc"] + 0.040, rel_z, h)
    log("  aim: dest %s + d %.4f in y -> carried centre %s; eef %s z %.4f" % (
        np.round(dest["centre"], 4), d, np.round(tgt_c, 4), np.round(eef_xy, 4), z_hover))
    payload, done = rig.goto(ctx, [eef_xy[0], eef_xy[1], z_hover], np.array(plan["q_hold"]),
                             1.0, tol=0.004, max_rows=30, kick=0.2, tag="[hover]")
    p = rig.pose_of(payload)
    log("  hover: eef %s err=%s done=%s" % (
        np.round(p, 4), np.round(np.array([eef_xy[0], eef_xy[1], z_hover]) - p, 4), done))
    s = rig.snapshot(ctx, tag="fix_hover")
    r = measure_carried(s, plan, z_lo=rim + 0.004, z_pred=p[2] + rel_z)
    if r is not None:
        dd = dest_now(s, plan, quiet=True)
        log("  hover check: base %.4f rim %.4f -> %+.1f mm, lateral %.1f mm, tilt %.1f deg" % (
            r["base"], dd["zc"], 1000.0 * (r["base"] - dd["zc"]),
            1000.0 * float(np.linalg.norm(r["centre"][:2] - np.array(dd["centre"]))),
            r["tilt"]))


def seat(ctx):
    """CLOSED-LOOP descent: every step is decided on the MEASURED carried base plane."""
    plan = load_plan()
    h = plan["h_bowl"]
    q_hold = np.array(plan["q_hold"])
    d = plan.get("d", D_DEFAULT)
    rel_z = plan.get("rel_z", 0.060)
    prev = None
    for step_i in range(20):
        s = rig.snapshot(ctx, tag="seat_s%d" % step_i)
        dest = dest_now(s, plan)
        rim = dest["zc"]
        # wide ladder from the eef (the narrow z_pred window is not reliable -- it clips
        # the ring when the bowl has slid in the grip), destination rim kept out by z_lo
        r = measure_carried(s, plan, z_lo=rim + 0.004)
        if r is None:
            log("!! seat: carried bowl unmeasurable at step %d (eef %s)" % (
                step_i, np.round(s.eef, 4)))
            return
        rel_z = r["d_z"]
        rel_xy = r["d_xy"]
        lat = float(np.linalg.norm(r["centre"][:2] - np.array(dest["centre"])))
        log("seat %2d: eef %s base %.4f rim %.4f -> %+.1f mm  lateral %.1f mm  tilt %.1f "
            "deg" % (step_i, np.round(s.eef, 4), r["base"], rim,
                     1000.0 * (r["base"] - rim), 1000.0 * lat, r["tilt"]))
        if r["base"] < rim - 0.010:
            log("!! seat: base is %.1f mm BELOW the rim -- abort (guard)" % (
                1000.0 * (rim - r["base"])))
            return
        if r["tilt"] > 5.0 and r["base"] > rim + 0.004:
            log("  seat: tilt %.1f deg -- re-level with `fix` before descending further"
                % r["tilt"])
            return
        stalled = (prev is not None and (prev[0] - r["base"]) < 0.0006
                   and (prev[1] - s.eef[2]) > 0.0012
                   and r["base"] < rim + 0.006)
        if stalled:
            log("  STALL: the arm moved %.1f mm but the base only %.1f mm -- contact"
                % (1000.0 * (prev[1] - s.eef[2]), 1000.0 * (prev[0] - r["base"])))
        if r["base"] <= rim + 0.0010 or stalled:
            log("  ==> base at the rim plane.  Press 1.2 mm with the jaws CLOSED and read "
                "the CONTACT TEST (does the base follow the arm?).")
            p = s.eef
            base0 = r["base"]
            z_floor = eef_for_base(rim - 0.003, rel_z, h)
            payload, rows = descend_rows(ctx, p[2] - 0.0012, q_hold, 1.0, step=0.0004,
                                         max_rows=6, floor_z=z_floor,
                                         hold_xy=(p[0], p[1]))
            eef1 = rig.pose_of(payload)
            s2 = rig.snapshot(ctx, tag="seat_press")
            r2 = measure_carried(s2, plan, z_lo=rim + 0.004)
            if r2 is not None:
                lat = float(np.linalg.norm(r2["centre"][:2] - np.array(dest["centre"])))
                dd = dest_now(s2, plan, quiet=True)
                d_eef = eef1[2] - p[2]
                d_base = r2["base"] - base0
                log("  PRESS: base %.4f rim %.4f -> %+.1f mm, lateral %.1f mm, tilt %.1f "
                    "deg, eef_z %.4f, gq %.4f" % (
                        r2["base"], dd["zc"], 1000.0 * (r2["base"] - dd["zc"]),
                        1000.0 * lat, r2["tilt"], s2.eef[2], abs(s2.gq[0])))
                log("  CONTACT TEST: deef %+.2f mm vs dbase %+.2f mm -> %s" % (
                    1000.0 * d_eef, 1000.0 * d_base,
                    "CONTACT (the bowl resisted)" if d_base - d_eef > 0.0006
                    else "NO CONTACT (the base followed the arm freely)"))
                plan["final"] = {"base": float(r2["base"]), "rim": float(dd["zc"]),
                                 "delta": float(r2["base"] - dd["zc"]),
                                 "lateral": lat, "tilt": float(r2["tilt"]),
                                 "d": d, "contact": bool(d_base - d_eef > 0.0006)}
                save_plan(plan)
            else:
                plan["final"] = {"base": float(r["base"]), "rim": float(rim),
                                 "delta": float(r["base"] - rim), "lateral": lat,
                                 "tilt": float(r["tilt"]), "d": d}
                save_plan(plan)
            return
        # one bounded descent step toward base = rim + 0.3 mm; small steps, because the
        # bowl SLIPS in the lip pinch at the moment of contact (measured: 11 mm in one
        # 6 mm step, which put the base 3.8 mm below the rim)
        target_base = max(rim + 0.0003, r["base"] - 0.0025)
        z_t = eef_for_base(target_base, rel_z, h)
        tgt_c = np.array([dest["centre"][0], dest["centre"][1] + d])
        eef_xy = tgt_c - rel_xy
        p = s.eef
        if abs(eef_xy[0] - p[0]) > 0.004 or abs(eef_xy[1] - p[1]) > 0.004:
            log("  re-aim xy: carried centre is %.1f mm off the aim point" % (
                1000.0 * float(np.linalg.norm(p[:2] + rel_xy - tgt_c))))
            payload, done = rig.goto(ctx, [eef_xy[0], eef_xy[1], p[2]], q_hold, 1.0,
                                     tol=0.003, max_rows=8, tag="[aim]")
            p = rig.pose_of(payload)
        payload, rows = descend_rows(ctx, z_t, q_hold, 1.0, step=0.0010, max_rows=4,
                                    hold_xy=(eef_xy[0], eef_xy[1]))
        p = rig.pose_of(payload)
        log("  step: %d rows to eef_z %.4f (commanded base %.4f)" % (
            rows, p[2], p[2] + rel_z - h))
        prev = (r["base"], s.eef[2])
    log("!! seat: out of steps")


def nudge(ctx):
    """Lift, shift the aim by D (lateral offset from the destination axis), hover.

    The offset window is set by the foot ring (r ~ 0.028) against the annulus
    (0.0535-0.0560): contact needs d + r_foot > 0.0535.  Measured at d = 0.0267 the base
    followed the arm freely for 1 mm at the rim plane -- NO CONTACT -- so the foot was
    over the throat.  D raises it.
    """
    plan = load_plan()
    h = plan["h_bowl"]
    dest = bowl(plan, "back")
    rim = dest["zc"]
    d = float(os.environ.get("D", plan.get("d", D_DEFAULT)))
    q_hold = np.array(plan["q_hold"])
    s = rig.snapshot(ctx, tag="nudge0")
    r = measure_carried(s, plan, z_lo=rim + 0.004)
    if r is None:
        log("!! nudge: cannot see the carried bowl")
        return
    log("nudge: base %+.1f mm vs rim, tilt %.1f, d was %.4f -> %.4f" % (
        1000.0 * (r["base"] - rim), r["tilt"], plan.get("d", D_DEFAULT), d))
    z_up = eef_for_base(rim + 0.050, r["d_z"], h)
    rig.goto(ctx, [s.eef[0], s.eef[1], z_up], q_hold, 1.0, tol=0.005, max_rows=12,
             stall_rows=10, tag="[lift]")
    for i in range(3):
        s = rig.snapshot(ctx, tag="nudge_l%d" % i)
        r = measure_carried(s, plan, z_lo=rim + 0.004)
        if r is None:
            log("!! nudge: carried bowl lost while lifting")
            return
        if r["tilt"] < 2.5:
            break
        q_tgt, deg = level_align(s, r)
        log("  nudge: tilt %.1f deg -> rotate the tool %.1f deg" % (r["tilt"], deg))
        payload, ok = rotate_to(ctx, q_tgt, s.eef, gripper=1.0, max_rows=16, tol=0.05,
                               tag="[lvl%d]" % i)
        q_hold = rig.quat_of(payload)
    s = rig.snapshot(ctx, tag="nudge_lv")
    r = measure_carried(s, plan, z_lo=rim + 0.004)
    if r is None:
        log("!! nudge: unmeasurable at the level pose")
        return
    q_hold = np.asarray(s.quat, dtype=np.float64)
    dest = dest_now(s, plan)
    tgt_c = np.array([dest["centre"][0], dest["centre"][1] + d])
    eef_xy = tgt_c - r["d_xy"]
    z_h = eef_for_base(dest["zc"] + 0.028, r["d_z"], h)
    log("  aim d=%.4f: carried centre %s (measured %s), eef -> %s z %.4f (tilt %.1f)"
        % (d, np.round(tgt_c, 4), np.round(r["centre"][:2], 4), np.round(eef_xy, 4),
           z_h, r["tilt"]))
    payload, done = rig.goto(ctx, [eef_xy[0], eef_xy[1], z_h], q_hold, 1.0, tol=0.0015,
                             max_rows=22, kick=0.2, tag="[aim]")
    p = rig.pose_of(payload)
    log("  aim residual %s" % (np.round(np.array([eef_xy[0], eef_xy[1], z_h]) - p, 4),))
    if max(abs(eef_xy[0] - p[0]), abs(eef_xy[1] - p[1])) > 0.002:
        payload, done = rig.goto(ctx, [eef_xy[0], eef_xy[1], p[2]], q_hold, 1.0, tol=0.0015,
                                 max_rows=10, tag="[aim2]")
        p = rig.pose_of(payload)
    s = rig.snapshot(ctx, tag="nudge_hover")
    r = measure_carried(s, plan, z_lo=dest["zc"] + 0.004)
    plan["q_hold"] = list(map(float, q_hold))
    plan["d"] = d
    if r is not None:
        lat = float(np.linalg.norm(r["centre"][:2] - np.array(dest["centre"])))
        plan["rel_z"] = float(r["d_z"])
        log("  hover: eef %s base %+.1f mm vs rim, lateral %.1f mm, tilt %.1f deg" % (
            np.round(p, 4), 1000.0 * (r["base"] - dest["zc"]), 1000.0 * lat, r["tilt"]))
    save_plan(plan)


def score(ctx):
    """0 rows: score on(upper, lower) from depth, from the pose the episode finishes in."""
    plan = load_plan()
    s = rig.snapshot(ctx, tag="score")
    dest = dest_now(s, plan)
    rim = dest["zc"]
    r = measure_carried(s, plan, z_lo=rim + 0.004)
    if r is None:
        log("!! carried bowl unmeasurable -- cannot score")
        return
    lat = float(np.linalg.norm(r["centre"][:2] - np.array(dest["centre"])))
    gone, info = rig.object_gone(s, tuple(bowl(plan, "middle")["centre"]))
    log("SCORE  upper base %.5f  lower rim %.5f  delta %+.2f mm  lateral %.1f mm  "
        "tilt %.1f deg" % (r["base"], rim, 1000.0 * (r["base"] - rim), 1000.0 * lat,
                           r["tilt"]))
    log("       height_ok=%s  lateral_ok=%s  tilt_ok=%s  source_empty=%s  dest_moved=%.1fmm" % (
        abs(r["base"] - rim) < 0.0025, lat < 0.015, r["tilt"] <= 6.0, gone,
        1000.0 * dest["moved"]))
    log("       gq %.4f  eef %s" % (abs(s.gq[0]), np.round(s.eef, 4)))


def look(ctx):
    """0 rows: ledger check -- every grey blob, the ledger, and the carried bowl."""
    plan = load_plan()
    s = rig.snapshot(ctx, tag="look")
    tz = plan["table_z"]
    log("eef %s gq %.4f quat %s" % (np.round(s.eef, 4), abs(s.gq[0]), np.round(s.quat, 4)))
    bows, allb = bowl_candidates(s, tz)
    for c in allb:
        log("   n=%4d x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f] rgb=%s R=%s zc=%s" % (
            c["blob"]["n"], c["blob"]["x"][0], c["blob"]["x"][1],
            c["blob"]["y"][0], c["blob"]["y"][1], c["blob"]["z"][0], c["blob"]["z"][1],
            c["rgb"], None if "R" not in c else round(c["R"], 4),
            None if "zc" not in c else round(c["zc"], 5)))
    for b in plan["bowls"]:
        log("   ledger %-6s c=%s zc=%.5f" % (b["name"], np.round(b["centre"], 4), b["zc"]))
    measure_carried(s, plan, z_lo=bowl(plan, "back")["zc"] + 0.004)
    if "final" in plan:
        log("   last final: %s" % (plan["final"],))


def run(ctx):
    probe(ctx)
    grasp(ctx)
    fix(ctx)
    seat(ctx)
    score(ctx)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    log("ENTRY %s  t=%s" % (entry, time.strftime("%H:%M:%S")))
    t0 = time.time()
    globals()[entry](ctx)
    log("entry %s done in %.1fs" % (entry, time.time() - t0))


if __name__ == "__main__":
    main(None)
