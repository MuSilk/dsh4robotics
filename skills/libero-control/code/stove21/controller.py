"""libero_90 task 21 (KITCHEN_SCENE3): turn on the stove, then put the frying pan on it.

RESULT of the run this file finished: **776/1000 rows, task_complete true** — knob -1.9 deg ->
+51 deg with the burner lamp 0 -> 1444 px, pan base 0.9801 on the 0.9800 burner, tilt 0.7 deg,
pan centre 17 mm off the cook region.

Two clauses, measured independently:
    (turnon flat_stove_1)                                <- a KNOB: the state is an ANGLE
    (on chefmate_8_frypan_1 flat_stove_1_cook_region)     <- a named REGION: the pan's CENTRE

Stages (each runnable alone, ENTRY=<stage>) — **the order that works**:
    probe   0 rows   table / pads ladder / jaw / stove plate + raised burner / knob axis+fin
                     + the fin's n-face-vs-s profile / standing pan rim + h_pan + floor_dz.
                     GATED on eef_z > 1.10: from a low pose pads()/jaw() return the SURFACE
                     and the plan's pad_dz/jaw are poisoned (that cost 7 cm of aim error).
    dry     0 rows   print what the plan implies (poses, targets) and move nothing
    knob    rows     corner-push ratchet.  S_CONTACT=0.040 -> the fin's 24 mm-thick end:
                     2 pushes = +45 deg.  S_CONTACT=0.022 (the thin sheet) = 1-5 deg/push.
    klook   0 rows   lamp + fin azimuth with the arm clear
    pre     rows     fold to z 1.30, then cross at 1.28, then hover over the pan's -y lip
    grasp   rows     descend, settle-close (gq 0.0102 on a 5 mm lip), lift 0.13
    fix     rows     level_quat -> rotate_to (29.3 -> 0.1 deg in 11 rows) -> q_hold persisted
    carry   rows     pan centre -> burner centre, high, re-aimed from a fresh fit
    place4  rows     **THE ENDING THAT WORKED**: level in place -> rise at the CURRENT xy to
                     1.42 -> cross at 1.42 -> descend in two stages -> release on
                     eef_z = support + 1 mm - 0.013 + R_base*sin(tilt)
    score   0 rows   the goal predicates from depth, with the arm clear
    look    0 rows   fin azimuth + lamp + pose

Metrology helpers to reuse (all 0 rows): the pads-offset box ladder in `probe`, `jaw_fixed`,
`pan_fit` (a Hough that survives the handle), `carried_base` (base plane from the pan's dark
INTERIOR FLOOR), `lip_fit` (centre + lip z from the lip ring in a narrow window around a
prediction), `pan_state` (a RELEASED pan), `pan_geom` (both estimators, averaged), `push_dir`
(servo a push in metres; its residual is the contact sensor).

Counter-examples, kept deliberately and NOT to be run: the pinch-yaw knob stages, the
rigid-rotation push, the fingertip `slide` (it cannot reach), and `place`/`place2`/`place3`
(earlier endings that release on vision which can lock onto a slice of the pan's 38 deg cone).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "code/lib")
import rig

OUT = "experiments/stove21"
PLAN_PATH = os.path.join(OUT, "plan.json")
Q_DOWN = np.array([0.9995, -0.010, -0.029, 0.002])
Q_DOWN = Q_DOWN / np.linalg.norm(Q_DOWN)

# ---------------------------------------------------------------- small helpers
def rem(ctx, tag=""):
    st = rig.json.loads(ctx.call("libero_status")["content"][0]["text"])
    e = st["episode"]
    print("  [%s] step %d  remaining %d" % (tag, e["step_id"], e["remaining_episode_steps"]),
          flush=True)
    return e["remaining_episode_steps"]


def save_plan(p):
    os.makedirs(OUT, exist_ok=True)
    q = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            q[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            q[k] = float(v)
        else:
            q[k] = v
    with open(PLAN_PATH, "w") as f:
        json.dump(q, f, indent=1)


def load_plan(must_have=()):
    with open(PLAN_PATH) as f:
        p = json.load(f)
    for k in must_have:
        if k not in p:
            raise KeyError("plan has no %r - run ENTRY=probe first" % k)
    return p


def qz(v):
    return np.round(np.asarray(v, dtype=np.float64), 4)


# ---------------------------------------------------------------- probe (0 rows)
def dark_mask(s, black=85, chroma=30):
    return ((s.mean < black) & ((s.rgb.max(2) - s.rgb.min(2)) < chroma) & (s.z < 1.6))


def dark_blobs(s, min_pixels=300, bbox=((-0.45, 0.25), (-0.45, 0.35)), z=(0.90, 1.30)):
    m = dark_mask(s)
    m &= (s.X > bbox[0][0]) & (s.X < bbox[0][1]) & (s.Y > bbox[1][0]) & (s.Y < bbox[1][1])
    m &= (s.Z > z[0]) & (s.Z < z[1])
    out, work = [], m.copy()
    while work.sum() >= min_pixels:
        keep, info = s.blob(work, min_pixels=min_pixels)
        if keep is None:
            break
        work &= ~keep
        info["mask"] = keep
        out.append(info)
    return sorted(out, key=lambda i: -i["n"])


def hough_rim(p, hint, half_c=0.06, step=0.005, Rr=(0.090, 0.112), dR=0.005):
    """Coarse (cx, cy, R) Hough over an annulus-width band, then trimmed LSQ."""
    xs = np.arange(hint[0] - half_c, hint[0] + half_c + 1e-9, step)
    ys = np.arange(hint[1] - half_c, hint[1] + half_c + 1e-9, step)
    Rs = np.arange(Rr[0], Rr[1] + 1e-9, 0.002)
    edges = np.r_[Rs - dR, Rs[-1] + dR]
    best = None
    for cx in xs:
        for cy in ys:
            d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
            cnt = np.histogram(d, bins=edges)[0]
            k = int(np.argmax(cnt))
            sc = cnt[k] / np.sqrt(Rs[k])
            if best is None or sc > best[0]:
                best = (float(sc), float(cx), float(cy), float(Rs[k]))
    cx, cy, R = best[1], best[2], best[3]
    sel = None
    for it in range(5):
        d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
        sel = np.abs(d - R) < (0.012 if it == 0 else 0.005)
        if sel.sum() < 80:
            break
        A = np.c_[2 * p[sel, 0], 2 * p[sel, 1], np.ones(int(sel.sum()))]
        b = p[sel, 0] ** 2 + p[sel, 1] ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        R = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9)))
    d = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    sel = np.abs(d - R) < 0.005
    q = p[sel]
    n = rig.plane_normal(q)
    if n[2] < 0:
        n = -n
    c = q.mean(0)
    zc = float(c[2] - (n[0] * (cx - c[0]) + n[1] * (cy - c[1])) / n[2])
    res = float(np.sqrt(np.mean((np.hypot(q[:, 0] - cx, q[:, 1] - cy) - R) ** 2)))
    ang = np.sort(np.mod(np.degrees(np.arctan2(q[:, 1] - cy, q[:, 0] - cx)), 360.0))
    arc = float(360.0 - np.diff(np.r_[ang, ang[0] + 360.0]).max())
    return {"centre": np.array([cx, cy]), "R": R, "zc": zc,
            "tilt": float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
            "arc": arc, "res": res, "n": int(sel.sum()), "normal": n}


def pan_fit(s, hint, table, half=0.17, Rr=(0.088, 0.112), quiet=False, zband=None):
    """Standing pan's rim: a Hough on the rim band beats robust_ring (the handle wins the top
    band).  Two passes: coarse over the blob, then a refit with the HANDLE side cut away."""
    m = dark_mask(s)
    m &= (np.abs(s.X - hint[0]) < half) & (np.abs(s.Y - hint[1]) < half)
    zb = zband if zband is not None else (table + 0.028, table + 0.046)
    m &= (s.Z > zb[0]) & (s.Z < zb[1])
    q = s.pts[m]
    if len(q) < 400:
        print("  pan_fit: only %d px in the rim band" % len(q), flush=True)
        return None
    c0 = np.array([np.median(q[:, 0]), np.median(q[:, 1])])
    r = hough_rim(q, c0, half_c=0.09, step=0.008, Rr=Rr, dR=0.006)
    print("  pan_fit coarse: c=%s R=%.4f zc=%.4f tilt=%.1f arc=%.0f res=%.1fmm n=%d" % (
        qz(r["centre"]), r["R"], r["zc"], r["tilt"], r["arc"], r["res"] * 1000, r["n"]),
        flush=True)
    q2 = q[q[:, 1] < r["centre"][1] + 0.05]          # cut the handle side
    r2 = hough_rim(q2, r["centre"], half_c=0.03, step=0.003, Rr=(r["R"] - 0.012, r["R"] + 0.012),
                   dR=0.004)
    if not quiet:
        print("  pan_fit: c=%s R=%.4f zc=%.4f tilt=%.1f arc=%.0f res=%.1fmm n=%d" % (
            qz(r2["centre"]), r2["R"], r2["zc"], r2["tilt"], r2["arc"], r2["res"] * 1000,
            r2["n"]), flush=True)
    return r2


def radial_profile(s, centre, table, nb=8, half=0.105):
    """median radius per 3 mm z bin above the table -> the cone's own r(z) (0 rows)."""
    m = dark_mask(s) & (np.abs(s.X - centre[0]) < half) & (np.abs(s.Y - centre[1]) < half)
    p = s.pts[m]
    out = []
    for i in range(nb):
        lo = table + 0.003 + i * 0.003
        hi = lo + 0.003
        q = p[(p[:, 2] > lo) & (p[:, 2] < hi)]
        if len(q) < 40:
            continue
        rr = np.hypot(q[:, 0] - centre[0], q[:, 1] - centre[1])
        out.append((round(0.5 * (lo + hi) - table, 4), float(np.median(rr)), int(len(q))))
    return out


def jaw_fixed(s, pad_dz, half=0.12, span=0.030, min_pixels=30):
    """`Scene.jaw()` with the MEASURED pad offset (the default pads(half=0.045) reads 6 mm
    high on this rig and then the pad band sits above the fingertips: gap 0.091 vs 0.068)."""
    if s.eef is None:
        return None
    pd = float(s.eef[2]) + float(pad_dz)
    m = (s.arm & (np.abs(s.X - s.eef[0]) < half) & (np.abs(s.Y - s.eef[1]) < half)
         & (s.Z > pd - 0.004) & (s.Z < pd + span))
    p = s.pts[m]
    if len(p) < 2 * min_pixels:
        return None
    ym = np.median(p[:, 1])
    lo, hi = p[p[:, 1] <= ym], p[p[:, 1] > ym]
    if len(lo) < min_pixels or len(hi) < min_pixels:
        return None
    cx = 0.5 * (lo[:, 0].mean() + hi[:, 0].mean())
    cy = 0.5 * (lo[:, 1].mean() + hi[:, 1].mean())
    return {"centre": np.array([cx, cy]), "centre_dx": cx - s.eef[0],
            "centre_dy": cy - s.eef[1], "gap": abs(hi[:, 1].mean() - lo[:, 1].mean()),
            "pad_bottom": pd, "n": int(m.sum())}


def probe(ctx, tag="probe0"):
    try:
        old_plan = load_plan()
    except Exception:
        old_plan = {}
    s = rig.snapshot(ctx, tag=tag)
    plan = {}
    o = rig.observe(ctx)
    print("== pose", flush=True)
    print("  eef %s gq %.4f" % (qz(s.eef), float(abs(s.gq[0]))), flush=True)
    plan["gq_open"] = (float(abs(s.gq[0])) if clear else
                       old_plan.get("gq_open", float(abs(s.gq[0]))))
    print("  quat %s" % qz(s.quat), flush=True)
    j = o.get("joint_pos")
    if j is not None:
        print("  joints %s" % np.round(j, 3), flush=True)
    plan["q_down"] = [float(v) for v in s.quat]

    mean = s.mean
    m = (mean > 150) & (s.Z > 0.90) & (s.Z < 1.00)
    if m.sum() > 800:
        vals = np.round(s.Z[m] / 0.002) * 0.002
        uq, ct = np.unique(vals, return_counts=True)
        table = float(uq[np.argmax(ct)])
    else:
        table = float(s.table_z())
    plan["table"] = table
    print("== table_z %.4f (n=%d)" % (table, int(m.sum())), flush=True)

    clear = bool(s.eef[2] > 1.10)
    if not clear:
        print("  !! probe from a LOW pose (eef_z %.3f): pads()/jaw() would return the SURFACE "
              "(SKILL 3), so they are NOT updated this run" % s.eef[2], flush=True)
    print("== pads ladder", flush=True)
    for h in (0.045, 0.050, 0.060, 0.075):
        pd = s.pads(half=h)
        print("  half %.3f -> %s" % (h, None if pd is None else
                                     {k: round(v, 4) for k, v in pd.items()}), flush=True)
    pad_dz = rig.PAD_DZ
    for h in (0.050, 0.060, 0.075):
        pd = s.pads(half=h)
        if pd is not None:
            pad_dz = float(pd["offset"])
    if clear or not old_plan.get("pad_dz"):
        plan["pad_dz"] = pad_dz
    else:
        plan["pad_dz"] = old_plan["pad_dz"]
    print("  pad_dz := %.4f" % plan["pad_dz"], flush=True)
    jw = (jaw_fixed(s, pad_dz) or s.jaw()) if clear else None
    if jw is not None:
        print("  jaw c=%s dx=%+.4f dy=%+.4f gap=%.4f (2gq %.4f)" % (
            qz(jw["centre"]), jw["centre_dx"], jw["centre_dy"], jw["gap"],
            jw.get("gap_expected", float("nan"))), flush=True)
        plan["jaw_dx"] = float(jw["centre_dx"])
        plan["jaw_dy"] = float(jw["centre_dy"])
    elif old_plan.get("jaw_dx") is not None:
        plan["jaw_dx"] = old_plan["jaw_dx"]
        plan["jaw_dy"] = old_plan["jaw_dy"]
    else:
        plan["jaw_dx"], plan["jaw_dy"] = rig.JAW_DX, rig.JAW_DY

    # ---- stove: pale plate (mode of z), then the DARK raised spiral above it
    grey = (mean > 95) & ((s.rgb.max(2) - s.rgb.min(2)) < 45) & (s.X > -0.45)
    plate = rig.plane_mode(s, ((-0.28, 0.10), (0.00, 0.36)),
                           zrange=(table + 0.008, table + 0.045), mask=grey)
    if plate is None:
        raise RuntimeError("no stove plate found")
    plan["plate"] = {"z": plate["z"], "x": list(plate["x"]), "y": list(plate["y"])}
    print("== plate z %.4f x[%.3f %.3f] y[%.3f %.3f] n=%d" % (
        plate["z"], plate["x"][0], plate["x"][1], plate["y"][0], plate["y"][1], plate["n"]),
        flush=True)
    (px0, px1), (py0, py1) = plate["x"], plate["y"]
    mb = ((s.Z > plate["z"] + 0.0025) & (s.Z < plate["z"] + 0.020)
          & (s.X > px0 + 0.005) & (s.X < px1 - 0.005)
          & (s.Y > py0 + 0.005) & (s.Y < py1 - 0.005))
    keep, info = s.blob(mb, min_pixels=150)
    if keep is None:
        raise RuntimeError("no raised burner found")
    bp = s.pts[keep]
    rr = np.hypot(bp[:, 0] - bp[:, 0].mean(), bp[:, 1] - bp[:, 1].mean())
    bz = np.round(bp[:, 2] / 0.001) * 0.001
    uq, ct = np.unique(bz, return_counts=True)
    burn = {"centre": np.array([bp[:, 0].mean(), bp[:, 1].mean()]),
            "z": float(uq[np.argmax(ct)]), "zmax": float(bp[:, 2].max()),
            "r": float(np.percentile(rr, 90)), "n": int(keep.sum())}
    print("== raised burner c=%s z %.4f r %.3f n=%d" % (
        qz(burn["centre"]), burn["z"], burn["r"], burn["n"]), flush=True)
    plan["burner"] = {"cx": float(burn["centre"][0]), "cy": float(burn["centre"][1]),
                      "z": burn["z"], "r": burn["r"]}

    # ---- knob: DARK blob BEHIND the plate (the burner is dark too and 5 mm above it)
    mk = dark_mask(s, black=95)
    mk &= (s.X < px0 + 0.01) & (s.X > px0 - 0.20) & (s.Y > py0 - 0.08) & (s.Y < py1 + 0.06)
    mk &= (s.Z > table + 0.010) & (s.Z < table + 0.14)
    keep, info = s.blob(mk, min_pixels=200)
    if keep is None:
        raise RuntimeError("no knob blob found")
    kp = s.pts[keep]
    zs = np.round(kp[:, 2] / 0.001) * 0.001
    uq, ct = np.unique(zs, return_counts=True)
    order = np.argsort(uq)
    print("== knob blob n=%d x[%.3f %.3f] y[%.3f %.3f] z[%.4f %.4f]" % (
        keep.sum(), kp[:, 0].min(), kp[:, 0].max(), kp[:, 1].min(), kp[:, 1].max(),
        kp[:, 2].min(), kp[:, 2].max()), flush=True)
    print("   knob z histogram (top 10 by z):", flush=True)
    for i in order[-10:]:
        print("     z %.4f n %d" % (uq[i], ct[i]), flush=True)
    fin_top = float(kp[:, 2].max())
    top = kp[kp[:, 2] > fin_top - 0.0025]
    axis = np.array([top[:, 0].mean(), top[:, 1].mean()])
    fin_top = float(np.median(top[:, 2]))
    print("   fin top %.4f  axis %s  n %d" % (fin_top, qz(axis), len(top)), flush=True)
    fa = rig.flat_face_az(s, axis, fin_top, dz=0.004, quiet=False)
    if fa is None:
        print("   RETRY flat_face_az with dz 0.006", flush=True)
        fa = rig.flat_face_az(s, axis, fin_top, dz=0.006, quiet=False)
    if fa is None:
        raise RuntimeError("no fin top face")
    axis = np.asarray(fa["centroid"], dtype=np.float64)
    hz = kp[(kp[:, 2] > fin_top - 0.055) & (kp[:, 2] < fin_top - 0.008)]
    hub = None
    if len(hz) > 100:
        v = np.round(hz[:, 2] / 0.002) * 0.002
        u, c = np.unique(v, return_counts=True)
        hub = float(u[np.argmax(c)])
    az0 = float(fa["az"])
    prev0 = (old_plan.get("knob") or {}).get("az0")
    if prev0 is not None and abs(rig.az_delta(az0, prev0)) > 3.0:
        print("  fin is not at its rest azimuth (%+.1f vs recorded az0 %+.1f) - keeping the "
              "recorded az0 and this az as the current one" % (az0, prev0), flush=True)
        az0 = float(prev0)
    plan["knob"] = {"ax": float(axis[0]), "ay": float(axis[1]),
                    "fin_top": fin_top, "az0": az0, "hub_top": hub,
                    "half_len_top": float(fa["half_len"]), "half_w_top": float(fa["half_w"])}
    print("== knob axis %s fin_top %.4f az %+.1f hub_top %s half_len %.3f half_w %.3f" % (
        qz(axis), fin_top, fa["az"], hub, fa["half_len"], fa["half_w"]), flush=True)

    # the blade's reach at PAD height (the top plate is much longer than the blade)
    hub_top = hub if hub is not None else fin_top - 0.037
    z0 = hub_top + 0.008
    az_r = np.radians(fa["az"])
    u = np.array([np.cos(az_r), np.sin(az_r)])
    n = np.array([-np.sin(az_r), np.cos(az_r)])
    pk = dark_mask(s, black=95)
    pk &= (np.hypot(s.X - axis[0], s.Y - axis[1]) < 0.06)
    pk &= (s.Z > z0) & (s.Z < fin_top - 0.002)
    kpb = s.pts[pk]
    if len(kpb) > 50:
        rel = kpb[:, :2] - axis
        sv = rel @ u
        nv = rel @ n
        print("   at pad z (%.3f..%.3f): s in [%.3f %.3f] n in [%.3f %.3f] n=%d" % (
            z0, fin_top - 0.002, sv.min(), sv.max(), nv.min(), nv.max(), len(kpb)), flush=True)
        plan["knob"]["blade_s"] = [float(sv.min()), float(sv.max())]
        plan["knob"]["blade_n"] = [float(nv.min()), float(nv.max())]
        sel = np.abs(sv - 0.022) < 0.016          # the face the pad will push, at u = +0.022
        if sel.sum() > 20:
            n_face = float(np.percentile(nv[sel], 2))
            plan["knob"]["fin_nface"] = n_face
            print("   fin -n face at u=+0.022: n %+.4f (n=%d)  [world y %.4f]" % (
                n_face, int(sel.sum()), axis[1] + n_face * n[1]), flush=True)
        prof = []
        for sc in np.arange(0.005, 0.051, 0.005):
            q2 = np.abs(sv - sc) < 0.008
            if q2.sum() > 15:
                prof.append([float(round(sc, 3)), float(np.percentile(nv[q2], 2)), int(q2.sum())])
        plan["knob"]["nface_s"] = prof
        print("   n-face vs s: %s" % "  ".join("%.3f:%+.4f/%d" % (a, b, c) for a, b, c in prof),
              flush=True)
    # the -y side face of the fin in the pad's x window (the camera cannot see the +y face)
    sel = ((np.abs(s.X - (axis[0] + 0.022 * u[0])) < 0.020) & (np.abs(s.Y - axis[1]) < 0.06)
           & (s.Z > z0) & (s.Z < fin_top - 0.002) & dark_mask(s, black=95))
    if sel.sum() > 30:
        yy = s.Y[sel]
        print("   fin at that x window: y[%.4f %.4f] (n=%d)" % (
            yy.min(), yy.max(), int(sel.sum())), flush=True)
        plan["knob"]["fin_y"] = [float(yy.min()), float(yy.max())]

    # ---- pan: the biggest dark blob away from the stove
    blobs = dark_blobs(s)
    print("== dark blobs", flush=True)
    pan_hint = None
    for b in blobs[:6]:
        print("   n %5d c %s x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]" % (
            b["n"], qz(b["centre"]), b["x"][0], b["x"][1], b["y"][0], b["y"][1],
            b["z"][0], b["z"][1]), flush=True)
        span = max(b["x"][1] - b["x"][0], b["y"][1] - b["y"][0])
        if b["centre"][1] < -0.05 and span > 0.14 and b["z"][0] < table + 0.02:
            if pan_hint is None or b["n"] > pan_hint["n"]:
                pan_hint = b
    if pan_hint is None:
        raise RuntimeError("no pan-like dark blob")
    plan["pan_hint"] = [float(pan_hint["centre"][0]), float(pan_hint["centre"][1])]
    print("== pan hint %s" % qz(pan_hint["centre"]), flush=True)

    hpan_guess = 0.036
    pf = pan_fit(s, pan_hint["centre"], table)
    if pf is None:
        raise RuntimeError("pan rim fit failed")
    prof = radial_profile(s, pf["centre"], table)
    print("   r(z) profile: %s" % prof, flush=True)
    R_base = min(prof, key=lambda t: abs(t[0] - 0.008))[1] if prof else 0.075
    # the two CALIBRATED features for a CARRIED pan (see pan_geom): the lip's face and the
    # dark interior's floor.  `zc` of a dark-band Hough is a slice of the cone, 2 cm low.
    cc = pf["centre"]
    rr = np.hypot(s.X - cc[0], s.Y - cc[1])
    ml = ((rr > 0.085) & (rr < 0.112) & (s.Y < cc[1] + 0.085)
          & (s.Z > table + 0.02) & (s.Z < table + 0.06))
    mf = (s.mean < 85) & (rr < 0.055) & (s.Z > table + 0.001) & (s.Z < table + 0.03)
    lip = float(np.percentile(s.Z[ml], 99.0)) if ml.sum() > 200 else pf["zc"]
    flo = float(np.percentile(s.Z[mf], 10.0)) if mf.sum() > 200 else table + 0.002
    h_pan = float(lip - table)
    floor_dz = float(flo - table)
    print("   standing pan: lip face %.4f (h_pan %.4f)  floor %.4f (floor_dz %.4f)" % (
        lip, h_pan, flo, floor_dz), flush=True)
    plan["pan"] = {"cx": float(pf["centre"][0]), "cy": float(pf["centre"][1]),
                   "R": float(pf["R"]), "zc": float(pf["zc"]), "h_pan": h_pan,
                   "floor_dz": floor_dz,
                   "tilt": float(pf["tilt"]), "R_base": float(R_base)}
    print("== pan c=%s R %.4f zc %.4f h_pan %.4f tilt %.1f R_base %.4f arc %.0f res %.1fmm"
          % (qz(pf["centre"]), pf["R"], pf["zc"], h_pan, pf["tilt"], R_base, pf["arc"],
             pf["res"] * 1000), flush=True)

    # ---- the lamp, before anything covers the burner
    path = rig.save_frame("probe0")
    red = rig.red_pixels(path)
    plan["red0"] = red
    print("== red px (whole frame) %s" % red, flush=True)

    save_plan(plan)
    print("== plan -> %s" % PLAN_PATH, flush=True)
    rem(ctx, "probe")
    return plan


# ---------------------------------------------------------------- knob (the turn-on)
def go(ctx, target, q_tgt, gripper, tol=0.006, max_rows=None, rate=0.011):
    """`rig.goto` with a row cap sized from the distance (a cap ends silently short)."""
    target = np.asarray(target, dtype=np.float64)
    d = float(np.max(np.abs(target - rig.pose_of(rig.observe(ctx)))))
    mr = int(max_rows) if max_rows else int(min(45, d / 0.010 + 6))
    return rig.goto(ctx, target, q_tgt, gripper, tol=tol, max_rows=mr)


def push_dir(ctx, start, dxy, metres, q_tgt, gripper=-1.0, max_rows=30, tol=0.0025,
             verbose=True):
    """Servo the eef `metres` along the world direction `dxy` (z held).  Residual = contact."""
    dxy = np.asarray(dxy, dtype=np.float64)
    dxy = dxy / (np.linalg.norm(dxy) + 1e-12)
    tgt = np.asarray(start, dtype=np.float64) + np.array([dxy[0] * metres,
                                                          dxy[1] * metres, 0.0])
    payload = rig.observe(ctx)
    rows, dead = 0, 0
    for i in range(int(max_rows)):
        p = rig.pose_of(payload)
        err = tgt - p
        if np.linalg.norm(err[:2]) < tol:
            if verbose:
                print("  push reached err=%s after %d rows" % (qz(err[:2]), i), flush=True)
            payload = rig.observe(ctx)
            p2 = rig.pose_of(payload)
            res = float(np.dot(tgt[:2] - p2[:2], dxy))
            print("  push residual along dir %+.4f m  %s" % (
                res, "CONTACT" if res > 0.004 else "EMPTY AIR (slid past?)"), flush=True)
            return payload, res
        a = np.zeros(7)
        a[0] = float(np.clip(err[0] / 0.011, -1, 1))
        a[1] = float(np.clip(err[1] / 0.011, -1, 1))
        a[2] = float(np.clip(err[2] / 0.011, -0.3, 0.3))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = float(gripper)
        prev = p
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        p = rig.pose_of(payload)
        moved = float(np.max(np.abs(p - prev)))
        dead = dead + 1 if moved < 0.0015 else 0
        rows += 1
        if dead >= 4:
            print("  push STALLED after %d rows at eef=%s" % (rows, qz(p)), flush=True)
            break
    p = rig.pose_of(payload)
    res = float(np.dot(tgt[:2] - p[:2], dxy))
    print("  push ended residual along dir %+.4f m after %d rows (eef %s)%s" % (
        res, rows, qz(p), "  <- CONTACT" if res > 0.004 else "  <- EMPTY AIR"), flush=True)
    return payload, res


def knob_pose(plan, az):
    """eef/jaw target for a +n push at the fin's +u end (CCW), tool yawed to `az`.

    The fin's -n face is measured ONCE in the probe in the fin's own frame (`fin_nface`,
    relative to the axis, at u = +0.022) and that offset is invariant under the fin's own
    rotation - so the pose for any azimuth is axis + s*u + (fin_nface - clear + gq)*n.
    """
    k = plan["knob"]
    a = np.array([k["ax"], k["ay"]])
    ar = np.radians(az)
    u = np.array([np.cos(ar), np.sin(ar)])
    n = np.array([-np.sin(ar), np.cos(ar)])
    gq = float(plan.get("gq_open", 0.034))
    s = float(os.environ.get("S_CONTACT", "0.022"))
    clear = float(os.environ.get("CLEAR", "0.007"))
    prof = k.get("nface_s")
    if prof:
        arr = np.array(prof, dtype=np.float64)
        n_face = float(np.interp(s, arr[:, 0], arr[:, 1]))
        if not (-0.025 < n_face < -0.004):
            n_face = float(k.get("fin_nface", -0.012))
    else:
        n_face = float(k.get("fin_nface", -0.012))
    jaw = a + s * u + (n_face - clear + gq) * n
    yaw = rig.yaw_quat(np.array(plan["q_down"]), az)
    jaw_off = np.array([np.cos(ar) * plan["jaw_dx"] - np.sin(ar) * plan["jaw_dy"],
                        np.sin(ar) * plan["jaw_dx"] + np.cos(ar) * plan["jaw_dy"]])
    eef = jaw - jaw_off
    z = (k["hub_top"] if k.get("hub_top") else k["fin_top"] - 0.037) + 0.010 - plan["pad_dz"]
    return {"eef": np.array([eef[0], eef[1], z]), "q": yaw, "jaw": jaw, "s": s,
            "n": n, "u": u, "z": z, "n_face": n_face, "jaw_n": n_face - clear + gq}


def klook(ctx, plan, tag):
    """0-row read: fin azimuth + lamp with the arm where it is."""
    s = rig.snapshot(ctx, tag=tag)
    k = plan["knob"]
    fa = rig.flat_face_az(s, np.array([k["ax"], k["ay"]]), k["fin_top"], dz=0.004,
                          quiet=False)
    path = rig.save_frame(tag)
    red = rig.red_pixels(path)
    az = None if fa is None else fa["az"]
    print("  [%s] fin az %s  red %s px" % (tag, az, red), flush=True)
    return s, az, red


def knob(ctx, plan=None):
    plan = plan or load_plan(("knob", "table"))
    k = plan["knob"]
    az0 = float(k["az0"])
    az = az0
    red0 = plan.get("red0")
    red = red0
    cap = int(os.environ.get("KCAP", "200"))
    max_cycles = int(os.environ.get("KCYC", "8"))
    push_m = float(os.environ.get("PUSH", "0.030"))
    target = float(os.environ.get("KTARGET", "40"))
    done = 0
    print("== knob: axis %s fin_top %.4f az0 %+.1f red0 %s nface %s" % (
        (round(k["ax"], 4), round(k["ay"], 4)), k["fin_top"], az0, red0,
        k.get("fin_nface")), flush=True)
    _p = rig.pose_of(rig.observe(ctx))
    if _p[1] > 0.12 and _p[2] < 1.12:          # the arm hides the fin's top from the camera
        rig.goto(ctx, np.array([_p[0] + 0.02, _p[1] - 0.06, _p[2] + 0.10]), np.array(plan["q_down"]),
                 -1.0, tol=0.010, max_rows=14)
    _s, az_now, red_now = klook(ctx, plan, "knob_start")
    if az_now is not None:
        print("  measured az %+.1f -> turned %+.1f from az0 (red %s)" % (
            az_now, rig.az_delta(az_now, az0), red_now), flush=True)
        if abs(rig.az_delta(az_now, az0)) > 2.5:
            az = az_now
            red = red_now
    for cyc in range(max_cycles):
        if done > cap:
            print("  KNOB CAP %d rows reached - stop" % cap, flush=True)
            break
        pp = knob_pose(plan, az)
        print("  -- cycle %d: az %+.1f  eef %s  jaw %s  pad_bottom %.4f" % (
            cyc, az, qz(pp["eef"]), qz(pp["jaw"]), pp["z"] + plan["pad_dz"]), flush=True)
        pl = rig.observe(ctx)
        q_err = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(pl), pp["q"])))
        if q_err > 0.08:
            pl, ok = rig.rotate_to(ctx, pp["q"], rig.pose_of(pl), gripper=-1.0, max_rows=8,
                                   tol=0.06)
            done += 8 if not ok else 5
        p = rig.pose_of(rig.observe(ctx))
        if np.max(np.abs(p[:2] - pp["eef"][:2])) > 0.010 or p[2] < pp["z"] + 0.05:
            pl, ok = go(ctx, np.array([pp["eef"][0], pp["eef"][1], pp["z"] + 0.12]), pp["q"],
                        -1.0, tol=0.008)
            done += 12
            print("   hover %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
        pl, ok = go(ctx, pp["eef"], pp["q"], -1.0, tol=0.004, max_rows=18)
        done += 18 if not ok else 10
        print("   at %s ok=%s (rot %.3f)" % (qz(rig.pose_of(pl)), ok,
              float(np.linalg.norm(rig.rotvec_err(rig.quat_of(pl), pp["q"])))), flush=True)
        start = rig.pose_of(pl)
        pl, res = push_dir(ctx, start, pp["n"], push_m, pp["q"], gripper=-1.0, max_rows=26)
        done += 26 if abs(res) > 0.02 else 10
        rem(ctx, "knob-push%d" % cyc)
        # back off along the push axis BEFORE lifting (a vertical retreat drags the fin)
        back = start[:2] - pp["n"] * 0.022
        rig.goto(ctx, np.array([back[0], back[1], start[2] + 0.045]), pp["q"], -1.0,
                 tol=0.006, max_rows=12)
        done += 10
        s, az_new, red = klook(ctx, plan, "knob%d" % cyc)
        if az_new is None:
            print("  lost the fin face - retry the same az", flush=True)
            continue
        d = rig.az_delta(az_new, az)
        az = az_new
        turned = rig.az_delta(az, az0)
        print("  -- cycle %d: az %+.1f (d %+.1f)  turned %+.1f  red %s (was %s)" % (
            cyc, az, d, turned, red, red0), flush=True)
        if abs(turned) >= target:
            print("  knob target %.0f deg reached" % target, flush=True)
            break
    plan["knob"]["az_final"] = float(az)
    plan["knob"]["az_turned"] = float(rig.az_delta(az, az0))
    plan["red_final"] = red
    save_plan(plan)
    rem(ctx, "knob")
    return plan


# ---------------------------------------------------------------- the pan
def pan_geom(s, plan, quiet=False, eef_rel=True):
    """CARRIED pan: centre + base plane, tied to the STANDING pan measured in the probe.

    The dark-band ring fit is a SLICE of the pan's 38 deg cone, NOT its rim: measured on
    this draw the slice locks at r 0.083 while the pan's lip is at r 0.097, and the slice
    sits ~2 cm BELOW the lip - so `zc - h_pan` is 2 cm low (it would bury the pan).  Two
    calibrated estimators instead, each a feature seen on the standing pan too:

      lip   = p99 z over the lip annulus (r 0.085..0.112), handle side (+y) cut, pads cut
              -> base = lip - h_pan      (h_pan = the lip face above the base, standing)
      floor = p10 z of the dark interior (r < 0.055) -> base = floor - floor_dz
    """
    p = plan["pan"]
    r = rig.carried_black_ring(s, plan["jaw_dx"], plan["jaw_dy"], Rlo=0.085, Rhi=0.115,
                              zlo=-0.090, zhi=0.030, quiet=True)
    if r is None:
        if not quiet:
            print("  pan_geom: no ring fit", flush=True)
        return None
    c = r["centre"][:2]
    eef = s.eef
    jaw = np.array([eef[0] + plan["jaw_dx"], eef[1] + plan["jaw_dy"]])
    rr = np.hypot(s.X - c[0], s.Y - c[1])
    dj = np.hypot(s.X - jaw[0], s.Y - jaw[1])
    ml = ((rr > 0.085) & (rr < 0.112) & (s.Y < c[1] + 0.085) & (dj > 0.045)
          & (s.Z > eef[2] - 0.10) & (s.Z < eef[2] + 0.075))
    mf = ((s.mean < 85) & (rr < 0.055) & (dj > 0.045)
          & (s.Z > eef[2] - 0.12) & (s.Z < eef[2] + 0.060))
    lip = float(np.percentile(s.Z[ml], 99.0)) if ml.sum() > 200 else None
    flo = float(np.percentile(s.Z[mf], 10.0)) if mf.sum() > 200 else None
    bl = None if lip is None else lip - float(p["h_pan"])
    bf = None if flo is None else flo - float(p["floor_dz"])
    bs = [b for b in (bl, bf) if b is not None]
    if not bs:
        return None
    out = {"centre": c, "R": r["R"], "tilt": r["tilt"], "lip": lip, "floor": flo,
           "base_lip": bl, "base_floor": bf, "base": float(np.mean(bs)),
           "d_xy": c - eef[:2], "n_lip": int(ml.sum()), "n_floor": int(mf.sum())}
    if not quiet:
        print("  pan_geom c=%s d_xy=%s | lip %.4f -> base %s | floor %.4f -> base %s "
              "| base %.4f tilt %.1f R %.4f" % (
                  qz(c), qz(out["d_xy"]), lip if lip else 0,
                  "None" if bl is None else "%+.4f" % bl, flo if flo else 0,
                  "None" if bf is None else "%+.4f" % bf, out["base"], r["tilt"], r["R"]),
              flush=True)
    return out


def pan_state(s, plan, hint, zb, quiet=False):
    """A RELEASED / standing pan: centre from a Hough on its lip band, base from the lip."""
    p = plan["pan"]
    r = pan_fit(s, hint, plan["table"], zband=zb)
    if r is None:
        return None
    c = r["centre"][:2]
    rr = np.hypot(s.X - c[0], s.Y - c[1])
    ml = ((rr > 0.085) & (rr < 0.112) & (s.Y < c[1] + 0.085)
          & (s.Z > zb[0] - 0.02) & (s.Z < zb[1] + 0.04))
    lip = float(np.percentile(s.Z[ml], 99.0)) if ml.sum() > 200 else None
    base = None if lip is None else lip - float(p["h_pan"])
    out = {"centre": c, "R": r["R"], "tilt": r["tilt"], "lip": lip, "base": base,
           "n": int(ml.sum())}
    if not quiet:
        print("  pan_state c=%s R %.4f lip %s base %s tilt %.1f" % (
            qz(c), r["R"], None if lip is None else round(lip, 4),
            None if base is None else round(base, 4), r["tilt"]), flush=True)
    return out


def pre(ctx, plan=None):
    """Fold UP first, then cross to the pan's -y lip, then hover 10 cm above it."""
    plan = plan or load_plan(("pan", "table", "q_down"))
    pan = plan["pan"]
    tgt = rig.rim_grasp_target({"centre": np.array([pan["cx"], pan["cy"]]),
                                "R": pan["R"], "rim_z": pan["zc"]},
                               jaw_dx=plan["jaw_dx"], jaw_dy=plan["jaw_dy"],
                               dz=-0.055, side=-1.0)
    plan["grasp_eef"] = [float(v) for v in tgt["eef"]]
    save_plan(plan)
    qd = np.array(plan["q_down"])
    print("== pre: lip grasp eef %s jaw %s pad %.4f" % (
        qz(tgt["eef"]), qz(tgt["jaw"]), tgt["pad_bottom"]), flush=True)
    p = rig.pose_of(rig.observe(ctx))
    pl, ok = go(ctx, np.array([p[0], p[1], 1.30]), qd, -1.0, tol=0.010, max_rows=20)
    print("  fold: %s" % qz(rig.pose_of(pl)), flush=True)
    t = tgt["eef"]
    pl, ok = go(ctx, np.array([t[0], t[1], 1.28]), qd, -1.0, tol=0.008, max_rows=75)
    print("  cross: %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
    pl, ok = go(ctx, np.array([t[0], t[1], t[2] + 0.10]), qd, -1.0, tol=0.006,
                max_rows=20)
    print("  hover: %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
    rem(ctx, "pre")


def pan_ring(ctx, plan, tag, extra_z=0.06):
    """Carried-pan ring fit with the arm where it is (0 rows besides the depth call)."""
    s = rig.snapshot(ctx, tag=tag)
    r = rig.carried_black_ring(s, plan["jaw_dx"], plan["jaw_dy"], Rlo=0.085, Rhi=0.115,
                              zlo=-0.090, zhi=0.030)   # +0.08 lets the WRIST blocks in
    return s, r


def grasp(ctx, plan=None):
    plan = plan or load_plan(("pan", "q_down"))
    qd = np.array(plan["q_down"])
    t = np.array(plan["grasp_eef"])
    pl, ok = go(ctx, t, qd, -1.0, tol=0.004, max_rows=28)
    print("== grasp: at %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
    rem(ctx, "grasp-down")
    pl, ok, gq = rig.close_grip(ctx)
    print("  close: gq %.4f" % gq, flush=True)
    p = rig.pose_of(pl)
    rig.goto(ctx, np.array([p[0], p[1], p[2] + 0.13]), qd, 1.0, tol=0.008, max_rows=16)
    print("  lift: %s" % qz(rig.pose_of(rig.observe(ctx))), flush=True)
    s, r = pan_ring(ctx, plan, "lift0")
    if r is not None:
        plan["pan"]["carried"] = {"R": r["R"], "zc": r["zc"], "tilt": r["tilt"],
                                  "dx": float(r["d_xy"][0]), "dy": float(r["d_xy"][1]),
                                  "dz": float(r["d_z"])}
        save_plan(plan)
    rem(ctx, "grasp")


def fix(ctx, plan=None):
    """Level the carried pan and persist q_hold."""
    plan = plan or load_plan(("pan",))
    s = rig.snapshot(ctx, tag="fix0")
    r = rig.carried_black_ring(s, plan["jaw_dx"], plan["jaw_dy"], Rlo=0.085, Rhi=0.115,
                              zlo=-0.090, zhi=0.030)
    if r is None or not r.get("ok"):
        print("  pan ring SUSPECT - not levelling blind", flush=True)
        return
    q_new, deg = rig.level_quat(r["normal"], rig.quat_of(rig.observe(ctx)))
    print("== fix: tilt %.1f deg -> levelling rotation of %.1f deg" % (r["tilt"], deg),
          flush=True)
    p = rig.pose_of(rig.observe(ctx))
    pl, ok = rig.rotate_to(ctx, q_new, p, gripper=1.0, max_rows=18, tol=0.05)
    q_hold = rig.quat_of(pl)
    s2, r2 = pan_ring(ctx, plan, "fix1")
    if r2 is not None:
        print("  after level: tilt %.1f  d_xy %s d_z %+.4f" % (
            r2["tilt"], qz(r2["d_xy"]), r2["d_z"]), flush=True)
    plan["q_hold"] = [float(v) for v in q_hold]
    plan["pan"]["carried"] = {"R": r2["R"] if r2 else r["R"],
                              "zc": r2["zc"] if r2 else r["zc"],
                              "tilt": r2["tilt"] if r2 else r["tilt"],
                              "dx": float(r2["d_xy"][0]) if r2 else float(r["d_xy"][0]),
                              "dy": float(r2["d_xy"][1]) if r2 else float(r["d_xy"][1]),
                              "dz": float(r2["d_z"]) if r2 else float(r["d_z"])}
    save_plan(plan)
    rem(ctx, "fix")


def carry(ctx, plan=None):
    """Carry the pan to the burner centre, high, then re-aim from a fresh fit."""
    plan = plan or load_plan(("pan", "q_hold", "burner"))
    qh = np.array(plan["q_hold"])
    b = plan["burner"]
    c = plan["pan"]["carried"]
    p = rig.pose_of(rig.observe(ctx))
    goal_xy = np.array([b["cx"] - c["dx"], b["cy"] - c["dy"]])
    print("== carry: pan centre target (%+.4f, %+.4f) -> eef xy %s" % (
        b["cx"], b["cy"], qz(goal_xy)), flush=True)
    # RISE AT THE CURRENT XY FIRST, then cross at that height, then descend: a single goto
    # that descends and crosses at once re-stretches the elbow and stops ~12 mm short in x
    # with rot_err ~0.3 (measured: stuck at (0.03, 0.10, 1.25) with j4 = -0.070).
    pl, ok = go(ctx, np.array([p[0], p[1], 1.42]), qh, 1.0, tol=0.010, max_rows=22)
    print("  up: %s j4=%s" % (qz(rig.pose_of(pl)),
                              np.round(rig.observe(ctx).get("joint_pos")[3], 2)), flush=True)
    for k in range(2):
        pl, ok = go(ctx, np.array([goal_xy[0], goal_xy[1], 1.42]), qh, 1.0, tol=0.007,
                    max_rows=30)
        print("  across %d: %s e2=%s" % (k, qz(rig.pose_of(pl)),
                                         qz(goal_xy - rig.pose_of(pl)[:2])), flush=True)
        if np.max(np.abs(goal_xy - rig.pose_of(pl)[:2])) < 0.012:
            break
    rem(ctx, "carry")
    pl, ok = go(ctx, np.array([goal_xy[0], goal_xy[1], 1.24]), qh, 1.0, tol=0.008,
                max_rows=30)
    print("  down to 1.24: %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
    rem(ctx, "carry-down")
    # re-aim from a fresh fit of the carried pan
    s, r = pan_ring(ctx, plan, "carry1")
    if r is not None:
        p = rig.pose_of(rig.observe(ctx))
        d = np.array([r["d_xy"][0], r["d_xy"][1]])
        want = np.array([b["cx"], b["cy"]]) - d
        err = want - p[:2]
        print("  re-aim: pan centre %s (want %s) -> xy err %s" % (
            qz(r["centre"][:2]), qz([b["cx"], b["cy"]]), qz(err)), flush=True)
        if np.max(np.abs(err)) > 0.008:
            pl, ok = rig.goto(ctx, np.array([want[0], want[1], p[2]]), qh, 1.0, tol=0.005,
                              max_rows=20)
            print("  re-aim: eef %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
        plan["pan"]["carried"] = {"R": r["R"], "zc": r["zc"], "tilt": r["tilt"],
                                  "dx": float(r["d_xy"][0]), "dy": float(r["d_xy"][1]),
                                  "dz": float(r["d_z"])}
        save_plan(plan)
    rem(ctx, "carry-reaim")


def carried_base(s, plan, quiet=False):
    """A LEVEL carried pan: BASE PLANE from its dark interior FLOOR (the one unambiguous
    feature), plus the tilt from that floor's own plane.

    Everything else on this asset is ambiguous: the dark-band ring fit locks onto a slice of
    the 38 deg cone (R 0.083 at 2 cm below the lip), and a z-band sweep cannot tell the lip
    from a wall slice (a horizontal cut of a level cone is also a circle).  The floor is a
    disc 1.9 mm above the base plane, calibrated on the same feature on the standing pan.
    """
    p = plan["pan"]
    eef = s.eef
    hint = np.array([eef[0] + float(os.environ.get("PAN_DX", "-0.044")),
                     eef[1] + float(os.environ.get("PAN_DY", "0.083"))])
    jaw = np.array([eef[0] + plan["jaw_dx"], eef[1] + plan["jaw_dy"]])
    rr = np.hypot(s.X - hint[0], s.Y - hint[1])
    dj = np.hypot(s.X - jaw[0], s.Y - jaw[1])
    mf = ((s.mean < 85) & (rr < 0.055) & (dj > 0.045)
          & (s.Z > eef[2] - 0.12) & (s.Z < eef[2] + 0.060))
    if mf.sum() < 300:
        if not quiet:
            print("  carried_base: only %d floor px" % int(mf.sum()), flush=True)
        return None
    floor = float(np.percentile(s.Z[mf], 10.0))
    q = s.pts[mf & (s.Z < floor + 0.006)]
    n = rig.plane_normal(q) if len(q) > 200 else None
    if n is not None and n[2] < 0:
        n = -n
    out = {"hint": hint, "floor": floor, "n": int(mf.sum()),
           "base": floor - float(p["floor_dz"]),
           "tilt": None if n is None else float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
           "normal": n}
    if not quiet:
        print("  carried_base floor %.4f -> base %.4f  tilt %s  centre~%s n=%d" % (
            floor, out["base"], None if out["tilt"] is None else round(out["tilt"], 1),
            qz(hint), int(mf.sum())), flush=True)
    return out


def place(ctx, plan=None):
    """Level, align, then descend onto the burner and release on a MEASURED base plane.

    Order matters: a 0.3 rad orientation error stalls the arm completely (OSC trades
    orientation for position), so the wrist is re-levelled IN PLACE before any translation;
    the pan is clamped, so levelling the tool levels the pan.
    """
    plan = plan or load_plan(("pan", "q_hold", "burner"))
    qh = np.array(plan["q_hold"])
    b = plan["burner"]
    support = float(b["z"])
    bc = np.array([b["cx"], b["cy"]])
    dxy = np.array([-0.044, 0.083])
    goal = support + 0.002
    print("== place: support %.4f goal base %.4f h_pan %.4f floor_dz %.4f d_xy* %s" % (
        support, goal, plan["pan"]["h_pan"], plan["pan"]["floor_dz"], qz(dxy)), flush=True)
    p = rig.pose_of(rig.observe(ctx))
    e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), qh)))
    print("  rot_err %.3f" % e, flush=True)
    if e > 0.06:
        pl, ok = rig.rotate_to(ctx, qh, p, gripper=1.0, max_rows=16, tol=0.05)
        print("  levelled: rot_err %.3f ok=%s eef %s" % (
            float(np.linalg.norm(rig.rotvec_err(rig.quat_of(pl), qh))), ok,
            qz(rig.pose_of(pl))), flush=True)
        rem(ctx, "place-level")
    g = carried_base(rig.snapshot(ctx, tag="place0"), plan)
    if g is None:
        print("  no base measurement - abort", flush=True)
        return None
    p = rig.pose_of(rig.observe(ctx))
    centre = p[:2] + dxy
    print("  pan centre ~%s (off burner %s)  base %.4f (d %+.4f) tilt %s" % (
        qz(centre), qz(centre - bc), g["base"], g["base"] - support, g["tilt"]), flush=True)
    # ---- xy: one correction, at the current height (a 10 mm miss is inside the 2 cm clause)
    want = bc - dxy
    xy_err = want - p[:2]
    print("  xy err %s" % qz(xy_err), flush=True)
    if np.max(np.abs(xy_err)) > 0.030:
        print("  too far - rising to 1.40 first, then across", flush=True)
        go(ctx, np.array([p[0], p[1], 1.40]), qh, 1.0, tol=0.010, max_rows=24)
        pl, ok = go(ctx, np.array([want[0], want[1], 1.40]), qh, 1.0, tol=0.008, max_rows=30)
        print("  across at 1.40: %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
        rem(ctx, "place-across")
    elif np.max(np.abs(xy_err)) > 0.005:
        pl, ok = go(ctx, np.array([want[0], want[1], p[2]]), qh, 1.0, tol=0.004,
                    max_rows=22)
        print("  xy nudge: %s ok=%s (stall here is the short-reach dead zone; 10 mm is ok)"
              % (qz(rig.pose_of(pl)), ok), flush=True)
        rem(ctx, "place-xy")
    # ---- descend in two stages, re-measuring between them
    p = rig.pose_of(rig.observe(ctx))
    for stage in (0, 1, 2, 3):
        z_t = p[2] + (goal - g["base"]) - (0.010 if p[2] - goal > 0.05 else 0.0)
        if z_t >= p[2] - 0.002:
            print("  no more descent needed (base %.4f vs goal %.4f)" % (g["base"], goal),
                  flush=True)
            break
        pl, ok = go(ctx, np.array([p[0], p[1], z_t]), qh, 1.0, tol=0.004, max_rows=45)
        p = rig.pose_of(pl)
        print("  descend %d: eef_z %.4f (wanted %.4f) ok=%s" % (
            stage, p[2], z_t, ok), flush=True)
        g2 = carried_base(rig.snapshot(ctx, tag="place_d%d" % stage), plan)
        rem(ctx, "place-d%d" % stage)
        if g2 is None:
            print("  lost the measurement - abort rather than release blind", flush=True)
            return None
        g = g2
        print("  base %.4f (support %.4f, d %+.4f) tilt %s" % (
            g["base"], support, g["base"] - support, g["tilt"]), flush=True)
        if g["base"] - support < 0.006:
            break
    # ---- press 1 mm: contact = the base stops following the eef
    pB = rig.pose_of(rig.observe(ctx))
    gB0 = g
    for k in range(3):
        pl, _ = rig.goto(ctx, np.array([pB[0], pB[1], pB[2] - 0.0015]), qh, 1.0, tol=0.0005,
                         max_rows=3)
        pC = rig.pose_of(pl)
        gC = carried_base(rig.snapshot(ctx, tag="place_press%d" % k), plan)
        if gC is None:
            print("  press: no measurement - abort", flush=True)
            return None
        print("  press %d: d_eef %+.4f  d_base %+.4f -> %s | base %.4f (d support %+.4f) "
              "tilt %s" % (k, pC[2] - pB[2], gC["base"] - gB0["base"],
                           "CONTACT" if (pB[2] - pC[2]) - (gB0["base"] - gC["base"]) > 0.0003
                           else "free", gC["base"], gC["base"] - support, gC["tilt"]),
              flush=True)
        gB0 = g
        g = gC
        pB = pC
        if g["base"] - support < 0.0015:
            print("  at the burner", flush=True)
            break
        rem(ctx, "place-press%d" % k)
    centre = rig.pose_of(rig.observe(ctx))[:2] + dxy
    off = centre - bc
    ok_rel = ((g["base"] - support) < 0.005) and np.max(np.abs(off)) < 0.020 and \
        (g["tilt"] is None or g["tilt"] < 6.0)
    print("  RELEASE? %s  base %+.4f  off %s (|.| %.4f)  tilt %s" % (
        bool(ok_rel), g["base"] - support, qz(off), float(np.hypot(*off)), g["tilt"]),
        flush=True)
    if not ok_rel:
        print("  NOT releasing", flush=True)
        return None
    pl = rig.hold(ctx, 14, gripper=-1.0)
    print("  opened: eef %s gq %.4f" % (qz(rig.pose_of(pl)), float(abs(rig.gq_of(pl)))),
          flush=True)
    p = rig.pose_of(pl)
    rig.goto(ctx, np.array([p[0], p[1], p[2] + 0.15]), qh, -1.0, tol=0.008, max_rows=18)
    print("  retreat: %s" % qz(rig.pose_of(rig.observe(ctx))), flush=True)
    rem(ctx, "placed")
    return plan


def lip_fit(s, plan, zc_pred, half_z=0.014, quiet=False):
    """(centre, lip z, R, tilt) of a carried pan from its LIP ring, searched in a NARROW
    z window around a prediction.

    The window is the whole trick: a horizontal cut of the pan's 38 deg cone is also a
    circle, but every cone slice sits BELOW the lip (r 0.088 at 1.6 cm down, r 0.098 at the
    lip), so a +-14 mm window around the lip's predicted height keeps the real ring and
    drops the slices.  The prediction is the last good lip minus the commanded descent.
    """
    eef = s.eef
    hint = np.array([eef[0] + float(os.environ.get("PAN_DX", "-0.044")),
                     eef[1] + float(os.environ.get("PAN_DY", "0.083"))])
    jaw = np.array([eef[0] + plan["jaw_dx"], eef[1] + plan["jaw_dy"]])
    rr = np.hypot(s.X - hint[0], s.Y - hint[1])
    dj = np.hypot(s.X - jaw[0], s.Y - jaw[1])
    best = None
    for zc in np.arange(zc_pred - half_z, zc_pred + half_z, 0.003):
        m = ((s.Z > zc - 0.004) & (s.Z < zc + 0.002) & (dj > 0.045) & (rr < 0.15))
        p = s.pts[m]
        if len(p) < 150:
            continue
        r = hough_rim(p, hint, half_c=0.05, step=0.004, Rr=(0.090, 0.108), dR=0.004)
        if not (0.091 < r["R"] < 0.107) or r["res"] > 0.005 or r["arc"] < 100:
            continue
        key = r["n"] / (1.0 + 100.0 * r["res"])
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        if not quiet:
            print("  lip_fit: no ring in %.3f..%.3f" % (zc_pred - half_z, zc_pred + half_z),
                  flush=True)
        return None
    r = best[1]
    out = {"centre": r["centre"][:2], "lip": float(r["zc"]), "R": r["R"], "tilt": r["tilt"],
           "arc": r["arc"], "res": r["res"], "base": float(r["zc"]) - float(plan["pan"]["h_pan"]),
           "d_xy": r["centre"][:2] - eef[:2]}
    if not quiet:
        print("  lip_fit c=%s R %.4f lip %.4f -> base %.4f tilt %.1f arc %.0f res %.1fmm" % (
            qz(out["centre"]), out["R"], out["lip"], out["base"], out["tilt"], out["arc"],
            out["res"] * 1000), flush=True)
    return out


def place3(ctx, plan=None):
    """Finish the placement from a jammed, slightly tilted carry: level, then lip-metrology
    descent with a press contact test.  Release only on contact or a measured base."""
    plan = plan or load_plan(("pan", "q_hold", "burner"))
    qh = np.array(plan["q_hold"])
    b = plan["burner"]
    support = float(b["z"])
    bc = np.array([b["cx"], b["cy"]])
    dxy = np.array([float(os.environ.get("PAN_DX", "-0.044")),
                    float(os.environ.get("PAN_DY", "0.083"))])
    p = rig.pose_of(rig.observe(ctx))
    e = float(np.linalg.norm(rig.rotvec_err(rig.quat_of(rig.observe(ctx)), qh)))
    print("== place3: eef %s rot_err %.3f support %.4f" % (qz(p), e, support), flush=True)
    if e > 0.04:
        pl, ok = rig.rotate_to(ctx, qh, p, gripper=1.0, max_rows=12, tol=0.06)
        print("  level: rot_err %.3f" % float(np.linalg.norm(
            rig.rotvec_err(rig.quat_of(pl), qh))), flush=True)
        rem(ctx, "p3-level")
    zpred = rig.pose_of(rig.observe(ctx))[2] + 0.0577
    g = None
    for it in range(5):
        p = rig.pose_of(rig.observe(ctx))
        g = lip_fit(rig.snapshot(ctx, tag="p3_%d" % it), plan, zpred)
        if g is None:
            print("  no lip fit - abort", flush=True)
            return None
        zpred = g["lip"]
        off = g["centre"] - bc
        print("  it %d: base %.4f (d support %+.4f) centre off %s |.| %.4f tilt %.1f "
              "rem %d" % (it, g["base"], g["base"] - support, qz(off),
                          float(np.hypot(*off)), g["tilt"],
                          int(json.loads(ctx.call("libero_status")
                                         ["content"][0]["text"])["episode"]
                              ["remaining_episode_steps"])), flush=True)
        if g["base"] - support < 0.003 or (g["tilt"] < 8 and np.max(np.abs(off)) < 0.02 and
                                          g["base"] - support < 0.008):
            break
        # xy correction (one goto) when it is worth it
        if np.max(np.abs(off)) > 0.020:
            want = bc - g["d_xy"]
            pl, ok = go(ctx, np.array([want[0], want[1], p[2]]), qh, 1.0, tol=0.005,
                        max_rows=18)
            print("   xy -> %s ok=%s" % (qz(rig.pose_of(pl)), ok), flush=True)
            rem(ctx, "p3-xy")
            p = rig.pose_of(pl)
        dz = g["base"] - support - 0.002
        if dz > 0.002:
            z_t = p[2] - min(dz, 0.05)
            pl, ok = go(ctx, np.array([p[0], p[1], z_t]), qh, 1.0, tol=0.004, max_rows=30)
            print("   descend to %.4f -> %.4f ok=%s" % (z_t, rig.pose_of(pl)[2], ok),
                  flush=True)
            rem(ctx, "p3-d")
    # ---- press test
    p = rig.pose_of(rig.observe(ctx))
    pl, _ = rig.goto(ctx, np.array([p[0], p[1], p[2] - 0.0018]), qh, 1.0, tol=0.0005,
                     max_rows=3)
    p2 = rig.pose_of(pl)
    g2 = lip_fit(rig.snapshot(ctx, tag="p3_press"), plan, g["lip"] - 0.006)
    contact = False
    if g2 is not None:
        d_eef = p[2] - p2[2]
        d_lip = g["lip"] - g2["lip"]
        contact = (d_eef > 0.0005) and (d_lip < 0.6 * d_eef)
        print("  PRESS d_eef %+.4f d_lip %+.4f -> %s | base %.4f (d %+.4f) tilt %.1f" % (
            -d_eef, -d_lip, "CONTACT" if contact else "free", g2["base"],
            g2["base"] - support, g2["tilt"]), flush=True)
        g = g2
        p = p2
    rem(ctx, "p3-press")
    off = g["centre"] - bc
    ok_rel = (contact or (g["base"] - support) < 0.004) and \
        np.max(np.abs(off)) < 0.022 and g["tilt"] < 8.0
    print("  RELEASE? %s  base %+.4f  off %s  tilt %.1f" % (
        bool(ok_rel), g["base"] - support, qz(off), g["tilt"]), flush=True)
    if not ok_rel:
        print("  NOT releasing", flush=True)
        return None
    pl = rig.hold(ctx, 14, gripper=-1.0)
    print("  opened: %s gq %.4f" % (qz(rig.pose_of(pl)), float(abs(rig.gq_of(pl)))), flush=True)
    q = rig.pose_of(pl)
    rig.goto(ctx, np.array([q[0], q[1], q[2] + 0.15]), qh, -1.0, tol=0.008, max_rows=20)
    print("  retreat: %s" % qz(rig.pose_of(rig.observe(ctx))), flush=True)
    rem(ctx, "p3-placed")
    return plan


def place4(ctx, plan=None):
    """FINISH from the jammed carry: rise (fold the elbow), level, cross, descend on a
    dead-reckoned base with the tilt correction, verify with the lip ring, release."""
    plan = plan or load_plan(("pan", "q_hold", "burner"))
    qh = np.array(plan["q_hold"])
    b = plan["burner"]
    support = float(b["z"])
    bc = np.array([b["cx"], b["cy"]])
    dxy = np.array([float(os.environ.get("PAN_DX", "-0.044")),
                    float(os.environ.get("PAN_DY", "0.083"))])
    base_dz = float(os.environ.get("BASE_DZ", "0.013"))     # base - eef, three readings
    Rb = float(plan["pan"]["R_base"])
    want = bc - dxy
    p = rig.pose_of(rig.observe(ctx))
    q = rig.quat_of(rig.observe(ctx))
    ang = 2.0 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(q, qh))))))
    print("== place4 eef %s tilt(tool vs q_hold) %.1f deg  want xy %s" % (
        qz(p), ang, qz(want)), flush=True)
    j = rig.observe(ctx).get("joint_pos")
    print("   joints %s" % np.round(j, 2), flush=True)
    # 1) level in place
    if ang > 3.0:
        pl, ok = rig.rotate_to(ctx, qh, p, gripper=1.0, max_rows=14, tol=0.05)
        p = rig.pose_of(pl)
        ang = 2.0 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(rig.quat_of(pl), qh))))))
        print("  levelled: rot_err %.3f tilt %.1f eef %s" % (
            float(np.linalg.norm(rig.rotvec_err(rig.quat_of(pl), qh))), ang, qz(p)), flush=True)
        rem(ctx, "p4-level")
    # 2) rise to fold the elbow, then cross
    pl, ok = go(ctx, np.array([p[0], p[1], 1.42]), qh, 1.0, tol=0.010, max_rows=22)
    p = rig.pose_of(pl)
    print("  risen: %s ok=%s" % (qz(p), ok), flush=True)
    j = rig.observe(ctx).get("joint_pos")
    print("   joints %s" % np.round(j, 2), flush=True)
    rem(ctx, "p4-rise")
    for k in range(2):
        pl, ok = go(ctx, np.array([want[0], want[1], p[2]]), qh, 1.0, tol=0.006, max_rows=26)
        p = rig.pose_of(pl)
        print("  across %d: %s ok=%s err %s" % (k, qz(p), ok, qz(want - p[:2])), flush=True)
        if np.max(np.abs(want - p[:2])) < 0.010:
            break
    rem(ctx, "p4-across")
    # 3) descend: coarse, then a lip check, then fine
    z_rel = float(os.environ.get("Z_REL", "0.0"))
    z_t = support + 0.001 + Rb * np.sin(np.radians(ang)) - base_dz - z_rel
    print("  descend target eef_z %.4f (tilt %.1f -> low edge %.4f)" % (
        z_t, ang, z_t + base_dz - Rb * np.sin(np.radians(ang))), flush=True)
    pl, ok = go(ctx, np.array([p[0], p[1], z_t + 0.030]), qh, 1.0, tol=0.006, max_rows=60)
    p = rig.pose_of(pl)
    print("  coarse: %s ok=%s" % (qz(p), ok), flush=True)
    rem(ctx, "p4-coarse")
    pl, ok = go(ctx, np.array([p[0], p[1], z_t]), qh, 1.0, tol=0.003, max_rows=14)
    p = rig.pose_of(pl)
    print("  at: %s ok=%s err %s" % (qz(p), ok, qz(np.array([p[0], p[1], z_t]) - p)), flush=True)
    rem(ctx, "p4-down")
    off = p[:2] + dxy - bc
    ok_rel = np.max(np.abs(off)) < 0.025
    print("  RELEASE? %s  pan centre ~%s off %s |.| %.4f" % (
        bool(ok_rel), qz(p[:2] + dxy), qz(off), float(np.hypot(*off))), flush=True)
    if not ok_rel:
        print("  NOT releasing (too far off the region)", flush=True)
        return None
    pl = rig.hold(ctx, 14, gripper=-1.0)
    print("  opened: %s gq %.4f" % (qz(rig.pose_of(pl)), float(abs(rig.gq_of(pl)))), flush=True)
    q2 = rig.pose_of(pl)
    rig.goto(ctx, np.array([q2[0], q2[1], q2[2] + 0.16]), qh, -1.0, tol=0.008, max_rows=20)
    print("  retreat: %s" % qz(rig.pose_of(rig.observe(ctx))), flush=True)
    rem(ctx, "p4-placed")
    return plan


def score(ctx, plan=None):
    """The goal predicates, from depth, with the arm clear."""
    plan = plan or load_plan(("pan", "burner"))
    b = plan["burner"]
    support = float(b["z"])
    bc = np.array([b["cx"], b["cy"]])
    s = rig.snapshot(ctx, tag="score")
    g = pan_state(s, plan, bc, (support + 0.026, support + 0.050))
    ok_on = False
    if g is not None:
        off = g["centre"] - bc
        ok_c = bool(np.max(np.abs(off)) < 0.02)
        ok_z = bool(g["base"] is not None and abs(g["base"] - support) < 0.004)
        ok_on = ok_c and ok_z
        print("== score: pan c=%s off burner %s (|.| %.4f)  base %s (support %.4f)  tilt %.1f"
              % (qz(g["centre"]), qz(off), float(np.hypot(*off)),
                 None if g["base"] is None else round(g["base"], 4), support, g["tilt"]),
              flush=True)
        print("   on(pan, cook_region): centre<2cm %s  base on support %s" % (ok_c, ok_z),
              flush=True)
    else:
        print("== score: NO pan found at the burner", flush=True)
    path = rig.save_frame("score")
    print("   red px now %s (occluded once the pan lands - not a verdict)" % rig.red_pixels(path),
          flush=True)
    ph = plan.get("pan_hint")
    if ph is not None:
        m = ((np.abs(s.X - ph[0]) < 0.10) & (np.abs(s.Y - ph[1]) < 0.10)
             & (s.Z > plan["table"] + 0.02))
        print("   source spot: %d px above table+2cm (want ~0)" % int(m.sum()), flush=True)
    print("   TURNON: fin az_final %s turned %s red_final %s" % (
        plan["knob"].get("az_final"), plan["knob"].get("az_turned"), plan.get("red_final")),
        flush=True)
    rem(ctx, "score")
    return ok_on


def look(ctx, plan=None):
    plan = plan or load_plan(("pan", "burner"))
    s = rig.snapshot(ctx, tag="look")
    k = plan["knob"]
    fa = rig.flat_face_az(s, np.array([k["ax"], k["ay"]]), k["fin_top"], quiet=False)
    path = rig.save_frame("look")
    print("  fin az %s  red %s px" % (None if fa is None else fa["az"],
                                      rig.red_pixels(path)), flush=True)
    print("  eef %s" % qz(rig.pose_of(rig.observe(ctx))), flush=True)


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    print("### ENTRY=%s" % entry, flush=True)
    if entry == "probe":
        probe(ctx)
    elif entry == "knob":
        knob(ctx)
    elif entry == "kpush":
        plan = load_plan(("knob",))
        pp = knob_pose(plan, float(os.environ.get("AZ", plan["knob"]["az0"])))
        print(" pose eef %s jaw %s q %s" % (qz(pp["eef"]), qz(pp["jaw"]), qz(pp["q"])),
              flush=True)
        qd = np.array(plan["q_down"])
        p = rig.pose_of(rig.observe(ctx))
        rig.rotate_to(ctx, pp["q"], p, gripper=-1.0, max_rows=10, tol=0.06)
        rig.goto(ctx, np.array([pp["eef"][0], pp["eef"][1], pp["z"] + 0.10]), pp["q"], -1.0,
                 tol=0.006, max_rows=20)
        pl, ok = rig.goto(ctx, pp["eef"], pp["q"], -1.0, tol=0.004, max_rows=16)
        start = rig.pose_of(pl)
        push_dir(ctx, start, pp["n"], float(os.environ.get("PUSH", "0.030")), pp["q"])
        rem(ctx, "kpush")
    elif entry == "pre":
        pre(ctx)
    elif entry == "grasp":
        grasp(ctx)
    elif entry == "fix":
        fix(ctx)
    elif entry == "carry":
        carry(ctx)
    elif entry == "place":
        place(ctx)
    elif entry == "place3":
        place3(ctx)
    elif entry == "place4":
        place4(ctx)
    elif entry == "run":
        probe(ctx)
        knob(ctx)
        pre(ctx)
        grasp(ctx)
        fix(ctx)
        carry(ctx)
        plan = place4(ctx)
        score(ctx)
    elif entry == "score":
        score(ctx)
    elif entry == "look":
        look(ctx)
    elif entry == "dry":
        plan = load_plan(("knob", "pan", "burner"))
        pp = knob_pose(plan, plan["knob"]["az0"])
        print(" knob pose eef %s jaw %s q %s" % (qz(pp["eef"]), qz(pp["jaw"]), qz(pp["q"])),
              flush=True)
        pan = plan["pan"]
        t = rig.rim_grasp_target({"centre": np.array([pan["cx"], pan["cy"]]),
                                  "R": pan["R"], "rim_z": pan["zc"]},
                                 jaw_dx=plan["jaw_dx"], jaw_dy=plan["jaw_dy"],
                                 dz=-0.055, side=-1.0)
        print(" pan grasp eef %s jaw %s" % (qz(t["eef"]), qz(t["jaw"])), flush=True)
        print(" burner %s" % plan["burner"], flush=True)
        print(" table %.4f plate %.4f" % (plan["table"], plan["plate"]["z"]), flush=True)
        rem(ctx, "dry")
    else:
        raise SystemExit("unknown ENTRY %r" % entry)
    return {"entry": entry}
