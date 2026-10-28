"""rig.py - LIBERO rig helpers. Verified on libero_90 tasks 1, 3, 4, 5 and 6
(510 / 159 / 241 / 407 / 931 / 632 rows).  The 632-row task 6 pass used **no demo file**:
`Q_HOOK`, `hook_press_z`, `probe_pull_biased` and `finger_top_offset` are what replaced it.

Everything here talks only through the six controller tools (status, libero_status,
libero_observe, libero_depth, libero_step, libero_hold).  Every analysis function takes
plain numpy arrays, so a saved snapshot can be re-analysed offline with zero rows.

    import sys; sys.path.insert(0, "code/lib")
    import rig

    s = rig.snapshot(ctx, tag="probe0")     # one libero_depth + frame.jpg; writes probe0.npz
    print(s.planes())                       # scene faces, arm pixels excluded
    print(s.pads())                         # pad bottom, as a z offset from eef_pos
    print(s.jaw())                          # finger-pair centre vs eef_pos  <-- read this
    print(s.hand_lead())                    # what REALLY pushes: 0.111 m ahead of eef_pos
    print(s.objects_on_table())             # coloured objects on the table, sorted by x
    print(s.rim_circle(s.bowl_mask()))      # rim centre / radius / top z
    print(s.top_face_front_y())             # counter/rim plane's front edge = drawer pos
    print(rig.panel_face(s), rig.panel_home(s, ref_y))     # 1 mm drawer-panel verdict

    print(rig.column_max_z(s, x, y))               # "did it land in the box"  (0 rows)
    print(rig.box_free_columns(s, -0.13, 0.07, -0.24, -0.11))   # free mouth of a drawer

    # closing a drawer: the verdict must be 1 mm, the STOP must be the eef's stall, and the
    # HAND's face (0.111 m ahead of eef_pos) is what must be aimed at the handle BAR
    ref = rig.closed_reference_y(s, band_z=1.135)   # the sibling CLOSED drawer's panel
    rig.push_to_stop(ctx, q_tgt, z_target=1.045, x_target=-0.0035,
                     panel_y=-0.077, body_y=-0.228, ref_y=ref[0])

    # --- put an object ON a surface (task 18, PASSED, 202 rows): a payload wider than the
    # jaws is only pinchable at its lip, and it HANGS TILTED (28 deg on a 20 cm pan)
    r = rig.carried_black_ring(s, jaw_dx, jaw_dy, Rlo=0.085, Rhi=0.115)   # carried BLACK object
    q_hold, deg = rig.level_quat(r["normal"], s.quat)   # 28.3 -> 3.7 deg in 11 rows
    rig.rotate_to(ctx, q_hold, s.eef)                   # ...then q_hold on EVERY later row
    print(rig.base_low(r["zc"] - h_pan, 0.075, r["tilt"]))   # the LOWEST point: aim the descent

    # --- a CARRIED object, and the ending that works for a stack (task 17 #5, PASSED) ---
    m = rig.measure_carried(s, (-0.020, 0.038), jaw_dx, jaw_dy, h_bowl, R_ref,
                            z_lo=z_rim + 0.004)     # base / zc / tilt / d_xy / d_z, no colour
    payload, ok = rig.rotate_to(ctx, q_hold, s.eef)  # goto CANNOT rotate (rot_err gate)
    payload, info = rig.descend_to_base(ctx, z_rim, h_bowl, q_hold, measure, aim_xy=aim)
    print(rig.eef_for_base(base, rel_z, h_bowl))     # base + h - rel_z  <- the one formula
    print(rig.contact_test(ctx, z_rim, h_bowl, q_hold, measure))   # press: did the base hold?

    # --- task 6 (open / close the bottom drawer) mechanics, measured end to end ---
    payload, done = rig.goto(ctx, target, q_tgt, -1, max_rows=20)   # prints residual + STALL
    off, log = rig.pitch_offset(s)                 # pitched finger offset, table-guarded
    t = rig.hook_targets(bottom_bar)               # pre-hook + DIAGONAL press targets
    rig.pull_burst(ctx, q_tgt, x=t["x"], z=t["z_press"], rows=8)    # dy .85 / dz -.35
    rig.push_tips_to_stop(ctx, q_tgt, x, z, y_stop=closed_panel_y, panel_fn=lambda: ...)
    print(rig.bar_front_from_panel(open_panel_y))  # the bar's front on a MOVED drawer

Numbers marked (rig) were re-measured on several poses and repeats.  Numbers marked
(placement) belong to one draw of the scene and must be re-measured every episode -
`snapshot` + `planes` is what re-measures them.

Do not call register / finish from a controller.
"""
import base64
import json
import os
import urllib.request

import numpy as np

FRAME_URL = "http://127.0.0.1:8766/frame.jpg"

# camera_to_base's rotation is right, its translation is not (rig).
T_RIG = np.array([0.637, -0.012, 1.659])

# ---- gripper geometry (rig, measured from depth with the wrist down, gripper open)
PAD_DZ = 0.0431        # pad bottom / lowest arm pixel = eef_z + PAD_DZ (readings .0406-.0470)
PAD_TALL = 0.022       # pad gripping band is about this tall: eef_z+0.043 .. eef_z+0.065
JAW_DX = -0.0140       # finger-pair centre = eef_pos + (JAW_DX, JAW_DY); eef_pos is NOT
JAW_DY = -0.0151       # the centre of the jaws.  Re-measure if the wrist pitch changes.
CLOSE_RATE = 0.0023    # |gripper_qpos| per closing row at 20 Hz  -> ~16 rows from full open
GQ_EMPTY = 0.002       # jaws closed on nothing; a 3-4 mm rim also reads 0.004, so gq alone
                       # can never confirm a pinch.

# ---- the HAND BLOCK leads the pads (rig, libero_90 task 4, measured two ways)
# Any horizontal push is made by this face, not by the pads: it is 4.2 cm deeper and ~4 cm
# higher than the pad band.  `Scene.hand_lead()` re-measures it from one depth call.
HAND_LEAD = 0.111      # hand's front face = eef_y - HAND_LEAD (0.1117 at two poses)
FINGER_LEAD = 0.069    # front finger's outer face = eef_y - FINGER_LEAD (0.063-0.076)
HAND_Z0 = 0.082        # that face spans eef_z + HAND_Z0 .. eef_z + HAND_Z1 (0.082-0.088 at
HAND_Z1 = 0.150        # two poses; 0.082 is the conservative low edge to aim with)

# ---- motion response (rig): metres per step at |action| = 1
STRIDE = {"x": 0.0125, "y": 0.0115, "z": 0.0110}


# =============================================================== tool wrappers
def observe(ctx):
    return json.loads(ctx.call("libero_observe")["content"][0]["text"])


def pose_of(payload):
    """libero_observe puts eef_pos on the payload, libero_step nests it."""
    if "observation" in payload:
        return np.array(payload["observation"]["eef_pos"], dtype=np.float64)
    return np.array(payload["eef_pos"], dtype=np.float64)


def quat_of(payload):
    if "observation" in payload:
        return np.array(payload["observation"]["eef_quat_xyzw"], dtype=np.float64)
    return np.array(payload["eef_quat_xyzw"], dtype=np.float64)


def gq_of(payload):
    """Positive jaw opening (|gripper_qpos[0]|)."""
    obs = payload.get("observation", payload)
    return float(abs(obs["gripper_qpos"][0]))


def step(ctx, rows, verbose=True, tag=""):
    """Run rows. ALWAYS pass a list of rows, never one flat row of 7 floats."""
    payload = json.loads(ctx.call("libero_step", actions=rows)["content"][0]["text"])
    if verbose:
        print("  %sstep %s eef=%s q=%s gq=%.4f" % (
            tag, payload.get("step_id"), np.round(pose_of(payload), 4),
            np.round(quat_of(payload), 3), gq_of(payload)), flush=True)
    return payload, False


def hold(ctx, rows, gripper=1.0, verbose=False):
    """Hold the pose for `rows` rows with the jaws commanded to `gripper`.

    A zero row of action is NOT a hold: its gripper field is 0 = mid-range, which drops
    whatever you are carrying.  Use this instead.
    """
    return step(ctx, [[0, 0, 0, 0, 0, 0, float(gripper)]] * int(rows), verbose=verbose)


def save_frame(tag, outdir="experiments/data"):
    """Fetch the live 512x512 agentview JPEG (plain HTTP, no rows). Returns the path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "frame_%s.jpg" % tag)
    with urllib.request.urlopen(FRAME_URL, timeout=10) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return path


def decode_jpeg(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


# =============================================================== quaternion servo
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def rotvec_err(q_cur, q_tgt):
    """World-frame rotation vector from q_cur to q_tgt (xyzw)."""
    qe = qmul(q_tgt, np.array([-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]]))
    qe /= np.linalg.norm(qe) + 1e-12
    if qe[3] < 0:
        qe = -qe
    s = np.linalg.norm(qe[:3])
    return np.zeros(3) if s < 1e-9 else qe[:3] / s * (2.0 * np.arctan2(s, qe[3]))


def mat_to_quat(R):
    """xyzw quaternion of a rotation matrix whose COLUMNS are the world images of x, y, z."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_from_fingers(finger_axis, jaw_axis):
    """q (xyzw) with tool z (the FINGER axis) -> `finger_axis`, tool y (the JAW axis) -> jaw.

    Build it from two measured world directions and you never recall a wrist orientation:
    z = finger axis, y = jaw axis made perpendicular to z, x = y x z.  For a handle bar the
    finger axis is forward-and-down (~32 deg below horizontal, e.g. a 1:-3:-2 ratio) and the
    """
    z = np.asarray(finger_axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    j = np.asarray(jaw_axis, dtype=np.float64)
    j = j - np.dot(j, z) * z
    j = j / np.linalg.norm(j)
    x = np.cross(j, z)
    x = x / np.linalg.norm(x)
    return mat_to_quat(np.column_stack([x, j, z]))


def pitch_quat(q_cur, deg=-90.0, axis="x"):
    """World-frame wrist rotation applied to `q_cur`.

    **Rotate IN PLACE first.**  In the same episode a 90 deg pitch commanded while also
    translating stalled at a 0.44 rad orientation error and the error then GREW back to 0.49 as
    the position error stayed large; driven alone at `rotvec_err/0.30` (~0.09 rad/row, ~18 rows
    for 90 deg) it converged, and the translation afterwards converged to < 2 mm.
    """
    h = 0.5 * np.deg2rad(float(deg))
    v = np.zeros(3)
    v["xyz".index(axis)] = np.sin(h)
    return qmul(np.array([v[0], v[1], v[2], np.cos(h)]), np.asarray(q_cur, dtype=np.float64))


def servo(ctx, target, q_tgt, gripper, tol=0.005, max_rows=30,
          rate=0.011, clip=1.0, zclip=1.0, verbose=True):
    """Bang-bang per-axis servo to `target` while servoing the quaternion every row.

    Returns (payload, complete).  CHECK THE PRINTED RESIDUAL: when max_rows runs out the
    loop ends with the arm short of the target and nothing else complains.
    """
    target = np.asarray(target, dtype=np.float64)
    payload = observe(ctx)
    complete = False
    for _ in range(int(max_rows)):
        p = pose_of(payload)
        err = target - p
        if np.max(np.abs(err)) < tol:
            if verbose:
                print("  servo converged err=%s" % np.round(err, 4), flush=True)
            return payload, complete
        a = np.zeros(7)
        a[:3] = np.clip(err / rate, -clip, clip)
        a[2] = np.clip(err[2] / rate, -zclip, zclip)
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = gripper
        payload, _complete = step(ctx, [a.tolist()], verbose=verbose)
    if verbose:
        print("  servo hit max_rows, residual=%s" % np.round(target - pose_of(payload), 4),
              flush=True)
    return payload, complete


def push(ctx, axis, sign, q_tgt, gripper, rows=6, hold_xyz=None, verbose=True):
    """Saturated push along `axis` (0/1/2) in direction `sign`, holding the others.

    hold_xyz: dict {axis_index: value} for the axes that are only corrected.
    """
    hold_xyz = hold_xyz or {}
    payload = observe(ctx)
    complete = False
    for _ in range(int(rows)):
        p = pose_of(payload)
        a = np.zeros(7)
        a[axis] = float(np.sign(sign))
        for k, val in hold_xyz.items():
            if k != axis:
                a[k] = float(np.clip((val - p[k]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = gripper
        payload, complete = step(ctx, [a.tolist()], verbose=verbose)
    return payload, complete


def close_grip(ctx, max_rows=24, settle=0.0008, verbose=True):
    """Close the jaws until they STOP MOVING, and report where they stopped.

    THE SINGLE MOST EXPENSIVE TRAP ON THIS RIG: `gripper = +1` closes at only
    ~0.0023 m/row, so 3-5 close rows leave the jaws 6 cm open.  Lifting then drags the
    pads past the object and the object stays behind, which looks exactly like "the
    pinch missed by a centimetre".  Hold the pose until the jaws settle.
    """
    payload = observe(ctx)
    complete = False
    prev, stall = None, 0
    for i in range(int(max_rows)):
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, 1.0]], verbose=False)
        gq = gq_of(payload)
        if prev is not None and abs(gq - prev) < settle:
            stall += 1
        else:
            stall = 0
        if verbose:
            print("  close row %d gq=%.4f%s" % (i, gq, "  (settled)" if stall else ""),
                  flush=True)
        prev = gq
        if stall >= 2:
            break
    return payload, complete, gq_of(payload)


# =============================================================== task 6 (open/close a drawer)
# The 50 demo traces that produced these are being removed from tasks/, so they are recorded
# here as measured asset geometry.  SKILL §10.1 and tasks/open-drawer.md §2-3 are the prose.
Q_HOOK = quat_from_fingers((0.260, -0.855, -0.455), (0.933, 0.10, 0.34))
HOOK_PRESS_DZ = -0.045   # press eef_z = bar_z0 + HOOK_PRESS_DZ  (0.9949 -> 0.9499)
HOOK_PRESS_DY = -0.015   # press eef_y = bar_front + HOOK_PRESS_DY (the pocket is only ~1.3 cm)
FINGER_TOP_DZ = 0.0197   # at Q_HOOK the finger's TOP surface is eef_z + 0.0197 (ascent test)


def hook_press_z(bar, dz=HOOK_PRESS_DZ):
    """The ONE press height that bites a drawer bar: `bar["z0"] + dz` -> 0.9499 on this asset.

    Measured on libero_90 task 6: at eef_z 0.955 / 0.965 / 0.971 / 0.975 the diagonal press
    converged in y with a <=1 mm residual (nothing touched) and the pull moved the panel
    0.000 m; at 0.9499 it stalled with a 10-12 mm y residual and the first 8-row burst moved
    the panel 4.4 cm.  Five millimetres is the whole margin, and it fails UPWARD only - so a
    failed press is retried 5 mm lower, never 1 cm higher (a +/-1 cm sweep costs ~180 rows).
    Note this is ~1.7 cm below the bar's centre and below anything `pads()` suggests: at this
    pitch `pads()` reads the PALM (+0.0586), not the fingers (see `finger_top_offset`).
    """
    return float(bar["z0"]) + float(dz)


def probe_pull_biased(ctx, q_tgt, x, z, band_z, xybox=None, rows=6, dy=0.85, dz=-0.35,
                      min_move=0.005, tag="probe_pull_biased", min_pixels=150, S=1.0):
    """Engagement verdict for a hooked drawer bar, using the PULL YOU WILL ACTUALLY RUN.

    **`probe_pull` (pure +y, z servoed) is a FALSE NEGATIVE on a handle bar.** Measured on
    libero_90 task 6, seconds apart on the same pose: the z-held probe returned move = 0.000 m
    at eef_z 0.950 / 0.955 / 0.965 / 0.971 - including the pose from which a biased burst moved
    the panel 4.4 cm.  The -z component is what makes the fingers hold the bar.
    """
    s0 = snapshot(ctx, tag=tag + "_a")
    f0 = front_face_y(s0, band_z=band_z, xybox=xybox, min_pixels=int(min_pixels))
    p0 = np.array(observe(ctx)["eef_pos"], dtype=np.float64)
    payload = pull_burst(ctx, q_tgt, x=x, z=z, rows=int(rows), dy=dy, dz=dz)
    p1 = pose_of(payload)
    s1 = snapshot(ctx, tag=tag + "_b")
    f1 = front_face_y(s1, band_z=band_z, xybox=xybox, min_pixels=int(min_pixels))
    y0 = None if f0 is None else float(f0[0])
    y1 = None if f1 is None else float(f1[0])
    mv = None if (y0 is None or y1 is None) else y1 - y0
    mv_u = None if mv is None else float(S) * float(mv)
    dz_actual = float(p1[2] - p0[2])
    return {"y_before": y0, "y_after": y1, "move": mv, "move_u": mv_u,
            "dz_actual": dz_actual,
            "blocked_z": bool(abs(dz_actual) < 0.004),
            "engaged": bool(mv_u is not None and mv_u >= float(min_move))}


def finger_top_offset(ctx, q_tgt, bar, x, z_press=None, y_off=HOOK_PRESS_DY, rows=12):
    """The finger's TOP surface at a pitched pose, measured with the arm (~5 rows, 0 depth).

    `pads()` cannot give this: at the hook pitch it reads the palm's underside, and on task 6
    that reading was a *flat* +0.0586 across boxes 0.03/0.04/0.045 (so it passes every guard in
    `pitch_offset`) while the real finger surface is 3.5 cm lower.  Instead: press into the
    handle recess, then command pure +z with x servoed and read where the arm stops.  Measured:
    rise 1.04 cm, stall at eef_z 0.9752 against the bar's underside at 0.9949 -> 0.0197.
    """
    goto(ctx, [x, float(bar["front_y"]) + 0.16, 1.00], q_tgt, -1.0, tol=0.008, max_rows=14)
    z = hook_press_z(bar) if z_press is None else float(z_press)
    goto(ctx, [x, float(bar["front_y"]) + float(y_off), z], q_tgt, -1.0, tol=0.004,
         max_rows=26, stall_rows=5)
    payload = observe(ctx)
    p = pose_of(payload)
    dead = 0
    for _ in range(int(rows)):
        a = np.zeros(7)
        a[2] = 1.0
        a[0] = float(np.clip((float(x) - p[0]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        q = pose_of(payload)
        dead = dead + 1 if float(np.max(np.abs(q - p))) < 0.0015 else 0
        p = q
        if dead >= 3:
            break
    off = float(bar["z0"]) - float(p[2])
    print("  finger_top_offset: +z stalled at eef_z %.4f against bar underside %.4f -> %.4f"
          % (p[2], float(bar["z0"]), off), flush=True)
    return off if dead >= 3 else None


# =============================================================== drawer hook / pull
def goto(ctx, target, q_tgt, gripper, tol=0.006, max_rows=24, rate=0.011, kick=0.0,
         stall_rows=6, verbose=True, tag=""):
    """Bang-bang goto that ALSO reports a stall and prints its residual.  Returns (payload, done).

    `servo` above is the same loop without the two things that cost rows on this rig:
    """
    target = np.asarray(target, dtype=np.float64)
    payload = observe(ctx)
    dead = 0
    for i in range(int(max_rows)):
        p = pose_of(payload)
        err = target - p
        if np.max(np.abs(err)) < tol:
            if verbose:
                print("  %sgoto converged err=%s after %d rows" % (tag, np.round(err, 4), i),
                      flush=True)
            return payload, True
        a = np.zeros(7)
        a[:3] = np.clip(err / rate, -1, 1)
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        if kick:
            a[3] = float(np.clip(a[3] + (kick if i % 2 == 0 else -kick), -1, 1))
        a[6] = gripper
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        q = pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if verbose and (i % 4 == 0 or dead >= stall_rows - 1):
            print("  %sgoto %2d eef=%s err=%s moved=%.4f dead=%d" % (
                tag, i, np.round(q, 4), np.round(target - q, 4), moved, dead), flush=True)
        if dead >= int(stall_rows):
            print("  %sgoto STALLED at eef=%s err=%s rot_err=%.3f" % (
                tag, np.round(q, 4), np.round(target - q, 4),
                float(np.linalg.norm(rotvec_err(quat_of(payload), q_tgt)))), flush=True)
            return payload, False
    if verbose:
        print("  %sgoto hit max_rows, residual=%s" % (tag, np.round(target - pose_of(payload), 4)),
              flush=True)
    return payload, False


def pitch_offset(scene, min_clearance=0.12, boxes=(0.030, 0.040, 0.045), tol=0.006,
                 diagnostic=(0.060,)):
    """The finger-surface offset at a PITCHED wrist, with the two traps of task 6 removed.

    Returns (offset, log) or (None, log).  Both guards are measurements, not theory:

    * **the table.**  `Scene.pads()` keeps every `arm` pixel inside a box around the eef, and
      `mean > 70` matches pale wood (rgb 197/184/180, chroma 17).  At eef_z 0.9877 with the
      table at 0.9498 it returned 0.9678 -> offset **-0.0199**, and the insert derived from it
      sat the fingers 5.5 cm above a drawer bar; the following pull moved the panel 0.000 m.
      So a negative offset, or a reading whose box-size spread exceeds `tol`, is rejected;
    * **the palm.**  Once the wrist is pitched the fingertips leave the box and the lowest
      pixel can be the palm, ~3 cm up the finger axis.  The flat-across-boxes test does NOT
      always catch it: on libero_90 task 6 at the hook orientation it returned a flat
      **+0.0606 / +0.0586 / +0.0586** (boxes 0.03 / 0.04 / 0.045, spread 2 mm) and that was the
      palm, 3.5 cm above the real finger surface.  So treat ANY pitched reading as a starting
      height and get the finger's real top from the arm: `finger_top_offset` (a pure +z ascent
      until the bar stalls it) gave eef_z + 0.0197.

    Uses `min_clearance` as the required gap between eef_z and the table (12 cm worked: the
    same call at eef_z 1.1058 returned **+0.0487 / +0.0467 / +0.0332** for boxes 0.03 / 0.045 /
    0.06).  Note which box is the outlier: the box has to be ±0.03-0.045 to stay on the
    fingertips, so `boxes` defaults to 0.030/0.040/0.045 and the WIDER boxes are only
    diagnostics -- 0.06 caught the palm 1.4 cm lower on that same pose and would have vetoed a
    good reading.  The log names which guard fired.
    """
    log = {}
    tz = scene.table_z()
    if scene.eef is None:
        return None, {"reason": "no eef in this snapshot"}
    if tz is not None and (scene.eef[2] - tz) < float(min_clearance):
        return None, {"reason": "eef only %.3f m above the table (need %.2f): the table will "
                                "enter the pads() box" % (scene.eef[2] - tz, min_clearance)}
    vals = []
    for h in tuple(boxes) + tuple(diagnostic):
        pd = scene.pads(half=h)
        log[round(float(h), 3)] = None if pd is None else round(float(pd["offset"]), 4)
        if pd is not None and h in tuple(boxes):
            vals.append(float(pd["offset"]))
    if len(vals) < 2:
        log["reason"] = "fewer than two box sizes returned an arm cluster"
        return None, log
    spread = max(vals) - min(vals)
    log["spread"] = round(spread, 4)
    if min(vals) < 0.0:
        log["reason"] = "negative offset: the table (or the palm), not the fingertips"
        return None, log
    if spread > float(tol):
        log["reason"] = "reading not flat across box sizes (spread %.4f > %.4f)" % (spread, tol)
        return None, log
    return float(np.median(vals)), log


DZ_BAR = 0.0322          # bar front - panel front, measured on libero_90 task 6 (0.030-0.034)
PITCH_BAND = 0.030       # at the hook pitch the gripping band is ~3 cm ABOVE eef_z (not +0.043)


def bar_front_from_panel(panel_y, dz=DZ_BAR):
    """The handle bar's front face on a MOVED drawer: `panel_y + DZ_BAR`.  0 rows.

    The bars move only in y, so once a drawer is out this is the only honest way to find its
    bar.  Do NOT ask `handle_bars(panel_y=<the open drawer's plane>)`: the bars are achromatic,
    the window then opens in front of the open panel, and the open GRIPPER is what it finds
    (task 6: it returned a "bar" at y +0.0759 with a 0.14 m x span, and the close pre-pose went
    0.28 m away from the drawer).  Keep the bar's x span and z band from the closed probe --
    neither moves.
    """
    return float(panel_y) + float(dz)


def hook_targets(bar, dz_bar=DZ_BAR, pre_ahead=0.16, press_ahead=None, press_z=None,
                 pre_z=1.000):
    """{pre, press, z_press, bar_front} for the diagonal press that hooks a drawer bar.

    `bar` is a `Scene.handle_bars` dict (closed drawer).  The aim that was measured to bite on
    task 6 (no demo file, 632 rows):
    """
    bar_front = float(bar["front_y"])
    x = 0.5 * (float(bar["x0"]) + float(bar["x1"]))
    ahead = HOOK_PRESS_DY if press_ahead is None else float(press_ahead)
    z = hook_press_z(bar) if press_z is None else float(press_z)
    return {"bar_front": bar_front, "x": x,
            "pre": np.array([x, bar_front + float(pre_ahead), float(pre_z)]),
            "press": np.array([x, bar_front + ahead, z]),
            "z_press": float(z), "y_press": bar_front + ahead}


def pull_burst(ctx, q_tgt, x, z=None, rows=8, dy=0.85, dz=-0.35, gripper=-1.0):
    """`rows` of saturated-ish +y with a small -z bias, x servoed, quaternion servoed.

    This is the pull profile the task-6 demos use (dy +0.4..0.9, dz -0.25..-0.75, jaws open).
    Keep the -z bias only while the bar is being dragged: with no contact it walks the tips
    DOWN the panel's face (task 6: z 0.974 -> 0.929) and the drawer does not move.  Pair it with
    a `front_face_y` check in the drawer's own band after every burst.
    """
    payload = observe(ctx)
    for _ in range(int(rows)):
        p = pose_of(payload)
        a = np.zeros(7)
        a[1] = float(dy)
        if z is not None:
            a[2] = float(dz)
        a[0] = float(np.clip((float(x) - p[0]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = float(gripper)
        payload, _ = step(ctx, [a.tolist()], verbose=False)
    return payload


def push_tips_to_stop(ctx, q_tgt, x, z, y_stop, panel_fn, rows=40, check_every=6,
                      tol=0.002, stall_rows=3, gripper=-1.0, verbose=True):
    """Shut a LOW drawer with the finger tips: saturated -y at `z`, verdict = `panel_fn()`.
    `S = +1` only; mirror the direction with `S` on a -y-facing chest.

    The §7 close recipe aims the HAND's face at the bar (`eef_z <= bar_top - 0.088`); on the
    lowest drawer of this chest that asks for eef_z 0.917 with the table at 0.9498, i.e. 3 cm
    under the table, so it cannot be used.  With the hook pitch the fingertips are ~3 cm above
    eef_z, so `z = bar_zc - PITCH_BAND` puts them in the bar band, and a saturated -y push takes
    the drawer home: measured -0.0839 -> -0.2447 (1.5 mm past the closed plane) in 24 rows.
    """
    payload = observe(ctx)
    dead, log = 0, []
    for i in range(int(rows)):
        p = pose_of(payload)
        a = np.zeros(7)
        a[1] = -1.0
        a[0] = float(np.clip((float(x) - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((float(z) - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = float(gripper)
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        q = pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if i % int(check_every) == 0 or dead >= int(stall_rows):
            y = panel_fn()
            log.append((i, round(float(q[0]), 4), round(float(q[1]), 4), y))
            if verbose:
                print("  push %2d eef=%s panel y=%s" % (i, np.round(q, 4), y), flush=True)
            if y is not None and abs(float(y) - float(y_stop)) <= float(tol):
                return payload, True, log
        if dead >= int(stall_rows):
            break
    y = panel_fn()
    ok = bool(y is not None and abs(float(y) - float(y_stop)) <= max(float(tol), 0.004))
    return payload, ok, log


def chest_window(scene, pad=0.03, drop=0.15, rise=0.01, ahead=0.22):
    """(xybox, band_z, counter_z, counter_y) of the chest's FRONTS, derived from depth.  0 rows.

    Every question about a drawer ("where is this drawer's front plane", "did it move") wants
    the same window, and the placement is redrawn every session, so derive it instead of
    recalling it: the counter's front edge (`Scene.top_face_front_y`) sits within ~2 mm of the
    drawer fronts' plane, so it is a free anchor for BOTH coordinates --
    """
    tf = scene.top_face_front_y()
    sgn = +1.0
    if tf is not None:
        cz, cy = float(tf[0]), float(tf[1])
        anchor = "counter"
    else:
        # NO counter plane in `top_face_front_y`'s box.  The old fallback here was
        # `nanmax(Z)` / -0.24, which on KITCHEN_SCENE4 silently produced band z 1.27..1.41
        # (a metre above the drawers) and a y window 0.15 m out of the chest -> every
        # `front_face_y` in that window returned the wrong plane, with no error anywhere.
        # Anchor on the MEASURED fronts plane instead (one extra 0-row pass), and say so.
        fr = drawer_axis(scene)
        if fr is None:
            print("!! chest_window: no counter plane AND no vertical fronts - falling back"
                  " to the raw scene bounds; CHECK this window before using it", flush=True)
            cz, cy = float(np.nanmax(scene.Z)), -0.24
            anchor = "NONE"
        else:
            cz, cy, sgn = float(fr["front_z"][1]), float(fr["front"]), float(fr["S"])
            anchor = "fronts"
            if fr["axis"] != "y":
                print("!! chest_window: this cabinet's fronts are %s-facing, not y - this"
                      " window (and every y-face call it feeds) is not meaningful; use"
                      " drawer_axis() + front_face_on() instead" % fr["axis"], flush=True)
    if anchor != "counter":
        print("  chest_window anchor=%s cz=%.4f cy=%.4f S=%+d" % (anchor, cz, cy, sgn),
              flush=True)
    band = (cz - float(drop), cz - float(rise))
    if anchor == "fronts":
        x0, x1 = float(fr["xwin"][0]), float(fr["xwin"][1])
    else:
        xs = []
        for p in scene.planes(axes="y", zrange=band, min_pixels=100):
            if abs(p[1] - cy) < 0.05 and -0.70 < p[3][0] < 0.45 and -0.70 < p[3][1] < 0.45:
                xs += [p[3][0], p[3][1]]
        x0, x1 = (min(xs), max(xs)) if xs else (-0.25, 0.20)
    y0 = min(cy, cy + sgn * float(ahead)) - 0.06
    y1 = max(cy, cy + sgn * float(ahead))
    return ((x0 - pad, x1 + pad), (y0, y1)), band, cz, cy


def bar_zs(bar):
    """(z0, z1) of a bar dict, from EITHER key convention.  0 rows.

    `Scene.handle_bars` / `rig.handle_bars_on` return the band as `bar["z"] = (lo, hi)` while
    part of the prose (and every hand-rolled `find_bar`) uses `bar["z0"] / bar["z1"]`.  Mixing
    them raises a bare `KeyError: 'z0'` at the END of an otherwise fine probe, i.e. after the
    """
    if "z0" in bar and "z1" in bar:
        return float(bar["z0"]), float(bar["z1"])
    z = bar["z"]
    return float(z[0]), float(z[1])


def drawer_band_for_bar(bar, drop=0.045, rise=0.006):
    """The z band of the drawer that OWNS `bar`, derived from the bar's own z band.  0 rows.

    All the chest's fronts are one flush plane, so a drawer's identity is its bar and its
    front panel occupies roughly `bar_z0 - drop .. bar_z1 + rise` (measured on one draw:
    bar 0.9949..1.0047 -> band 0.953..1.010, which holds that drawer's panel and excludes
    the next one up, whose panel starts ~0.07 higher).  Use the result as `band_z` for every
    "where is THIS drawer's front plane" question: `front_face_y`, `y_faces_mm`,
    `probe_pull`, and the final verdict.  Pass `drop`/`rise` only if the measured bar band
    is unusual (e.g. a bar whose top is cut off by the arm).
    """
    z0, z1 = bar_zs(bar)
    return (z0 - float(drop), z1 + float(rise))


def probe_pull(ctx, q_tgt, hold_xyz, band_z, rows=4, min_move=0.005, tag="probe_pull",
               xybox=None, min_pixels=150):
    """Cheap ENGAGEMENT verdict for a hooked drawer: a few saturated +y rows, then is it out?

    **NOT FOR A HANDLE BAR: this z-held probe is a FALSE NEGATIVE there.** On libero_90 task 6
    it returned move = 0.000 m at eef_z 0.950 / 0.955 / 0.965 / 0.971, including the pose from
    which a biased burst (dy +0.85 / dz -0.35) pulled the panel 4.4 cm seconds later - the -z
    component is what makes the fingers hold a bar.  Use `probe_pull_biased` for bars; this one
    is for a rim or a lip that a pure pull really would drag.
    """
    s0 = snapshot(ctx, tag=tag + "_a")
    f0 = front_face_y(s0, band_z=band_z, min_pixels=int(min_pixels), xybox=xybox)
    push(ctx, 1, +1, q_tgt, -1.0, rows=int(rows), hold_xyz=hold_xyz, verbose=False)
    s1 = snapshot(ctx, tag=tag + "_b")
    f1 = front_face_y(s1, band_z=band_z, min_pixels=int(min_pixels), xybox=xybox)
    yb = None if f0 is None else float(f0[0])
    ya = None if f1 is None else float(f1[0])
    mv = None if (yb is None or ya is None) else ya - yb
    return {"y_before": yb, "y_after": ya, "move": mv,
            "engaged": bool(mv is not None and mv >= float(min_move))}


# =============================================================== scene snapshot
class Scene:
    """One depth render + one RGB frame, plus the unprojection.  Costs 0 rows."""

    def __init__(self, z, pts, rot, rgb, eef=None, quat=None, gq=None):
        self.z, self.pts, self.rot, self.rgb = z, np.asarray(pts), rot, rgb
        self.eef = None if eef is None else np.asarray(eef, dtype=np.float64)
        self.quat = quat
        self.gq = gq
        self.X, self.Y, self.Z = (self.pts[:, :, 0], self.pts[:, :, 1], self.pts[:, :, 2])
        self._arm = None
        self._planes = None

    # -------------------------------------------------- construction
    @classmethod
    def from_ctx(cls, ctx, tag=None, outdir="experiments/data"):
        """libero_depth + frame.jpg. Saves <tag>.npz / <tag>.jpg when tag is given."""
        payload = json.loads(ctx.call("libero_depth")["content"][0]["text"])
        mm = np.frombuffer(base64.b64decode(payload["depth_mm"]), dtype="<u2")
        mm = mm.reshape(payload["height"], payload["width"])
        z = mm.astype(np.float64) * payload["depth_mm_scale"]
        cam = np.array(payload["camera_to_base"], dtype=np.float64).reshape(4, 4)
        rot = cam[:3, :3]
        fx, fy = payload["fx"], payload["fy"]
        cx, cy = payload["cx"], payload["cy"]
        h, w = z.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        p_cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
        pts = p_cam @ rot.T + T_RIG
        obs = observe(ctx)
        rgb = np.zeros(z.shape + (3,))
        if tag:
            try:
                rgb = decode_jpeg(save_frame(tag, outdir))
            except Exception as exc:              # no frame is not fatal for plane work
                print("  frame fetch failed:", exc, flush=True)
        else:
            try:
                rgb = decode_jpeg(save_frame("tmp", outdir))
            except Exception:
                pass
        o = obs.get("observation", obs)
        return cls(z, pts, rot, rgb, o["eef_pos"], o["eef_quat_xyzw"], o["gripper_qpos"])

    @classmethod
    def load(cls, path):
        d = np.load(path)
        k = d.files
        rgb = d["rgb"] if "rgb" in k else np.zeros(d["depth"].shape + (3,))
        rot = d["rot"] if "rot" in k else np.eye(3)
        return cls(d["depth"], d["pts"], rot, rgb,
                   d["eef"] if "eef" in k else None,
                   d["quat"] if "quat" in k else None,
                   d["gq"] if "gq" in k else None)

    def save(self, path):
        np.savez_compressed(path, depth=self.z, pts=self.pts, rot=self.rot,
                            rgb=self.rgb, eef=self.eef, quat=self.quat, gq=self.gq)

    # -------------------------------------------------- pixel classes (rig)
    @property
    def mean(self):
        return self.rgb.mean(2)

    @property
    def arm(self):
        """Achromatic and bright enough to keep the DARK pads (mean ~90-110).

        mean > 120 silently drops the fingers and leaves the wrist's underside ~4 cm
        higher.  This mask also matches a grey speckled bowl and grey handle bars - that
        is why the plane code excludes it and why a lifted-object test must not rely on
        """
        if self._arm is None:
            mx, mn = self.rgb.max(2), self.rgb.min(2)
            self._arm = ((mx - mn) < 25) & (self.mean > 70)
        return self._arm

    def table_z(self):
        """Median z of the flat surface under the arm's reach (the scene's height origin)."""
        m = (self.Z > 0.85) & (self.Z < 1.05) & (self.mean > 120)
        return float(np.median(self.Z[m])) if m.sum() > 500 else None

    # -------------------------------------------------- planes
    def planes(self, axes="xyz", zrange=(0.96, 1.35), xybox=None,
               min_pixels=100, exclude_arm=True, tol=0.006, top=5):
        """Bucketed surface planes: [(axis, offset, n, (lo,hi), (lo,hi)), ...].

        A face is identified by its NORMAL, not by its apparent size or darkness; that is
        what settles a side wall masquerading as a front panel.  The depth contains the
        robot, so arm pixels are excluded by default - otherwise the wrist plate, or the
        """
        pts = self.pts
        du = np.zeros_like(pts)
        dv = np.zeros_like(pts)
        du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
        dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
        n = np.cross(du, dv)
        n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
        region = (self.Z > zrange[0]) & (self.Z < zrange[1])
        if xybox is not None:
            (x0, x1), (y0, y1) = xybox
            region &= (self.X > x0) & (self.X < x1) & (self.Y > y0) & (self.Y < y1)
        if exclude_arm:
            region &= ~self.arm
        out = []
        for ax in axes:
            a = "xyz".index(ax)
            m = region & (np.abs(n[:, :, a]) > 0.85)
            if m.sum() < min_pixels:
                continue
            vals = np.round(self.pts[:, :, a][m] / 0.005) * 0.005
            uniq, cnt = np.unique(vals, return_counts=True)
            for o in np.argsort(cnt)[::-1][:top]:
                if cnt[o] < min_pixels:
                    break
                sel = m & (np.abs(self.pts[:, :, a] - uniq[o]) < tol)
                q = self.pts[sel]
                other = [i for i in range(3) if i != a]
                out.append((ax, float(uniq[o]), int(cnt[o]),
                            (float(q[:, other[0]].min()), float(q[:, other[0]].max())),
                            (float(q[:, other[1]].min()), float(q[:, other[1]].max()))))
        return out

    def faces_along(self, axis="y", **kw):
        return [p for p in self.planes(axes=axis, **kw)]

    def front_face(self, axis="y", **kw):
        """Most negative offset along `axis` (the face nearest the open side)."""
        ps = self.faces_along(axis, **kw)
        return min(ps, key=lambda p: p[1]) if ps else None

    # -------------------------------------------------- own gripper
    def pads(self, half=0.045, min_pixels=40):
        """Pad bottom as a z OFFSET from eef_pos, and its absolute z.

        Keep the jaws over open space (or at least clear of the scene at this height) or
        the box catches a drawer panel or the object.  Shrink `half` until the answer
        stops changing: 0.03 clips the pad's corner and reads ~6 mm high.
        """
        if self.eef is None:
            return None
        box = ((np.abs(self.X - self.eef[0]) < half) & (np.abs(self.Y - self.eef[1]) < half)
               & (self.Z > self.eef[2] - 0.02))
        m = self.arm & box
        if m.sum() < min_pixels:
            return None
        z = float(self.pts[m][:, 2].min())
        return {"pad_bottom": z, "offset": z - self.eef[2], "n": int(m.sum())}

    def jaw(self, half=0.12, span=0.030, min_pixels=30):
        """The two fingertip clusters: where the jaws really are, in x and y.

        eef_pos is NOT the centre of the finger pair (measured offset about
        (-0.014, -0.015) with the wrist down).  Any plan that says "put the eef on the
        object" is therefore 1.5 cm off, which is most of a bowl rim's wall thickness.
        """
        if self.eef is None:
            return None
        pd = self.pads()
        if pd is None:
            return None
        top = pd["pad_bottom"] + span
        m = (self.arm & (np.abs(self.X - self.eef[0]) < half)
             & (np.abs(self.Y - self.eef[1]) < half)
             & (self.Z > pd["pad_bottom"] - 0.004) & (self.Z < top))
        if m.sum() < 2 * min_pixels:
            return None
        p = self.pts[m]
        ym = np.median(p[:, 1])
        lo, hi = p[p[:, 1] <= ym], p[p[:, 1] > ym]
        if len(lo) < min_pixels or len(hi) < min_pixels:
            return None
        cy = 0.5 * (lo[:, 1].mean() + hi[:, 1].mean())
        cx = 0.5 * (lo[:, 0].mean() + hi[:, 0].mean())
        out = {"centre": np.array([cx, cy]),
               "centre_dx": cx - self.eef[0], "centre_dy": cy - self.eef[1],
               "gap": abs(hi[:, 1].mean() - lo[:, 1].mean()),
               "finger_y": (float(lo[:, 1].mean()), float(hi[:, 1].mean())),
               "pad_bottom": pd["pad_bottom"], "n": int(m.sum())}
        if self.gq is not None:
            # sanity: the jaw gap must track 2*gripper_qpos.  Two poses measured
            # centre_dx = -0.014 / -0.020 and centre_dy = -0.015 / -0.0155, so treat the
            # pair centre as good to ~5 mm - which is more than a rim wall is thick.
            out["gap_expected"] = 2.0 * float(abs(self.gq[0]))
        return out

    def pad_band(self, offset=None):
        """Absolute (z_lo, z_hi) the pads can grip in, from the current eef_z."""
        if self.eef is None:
            return None
        off = PAD_DZ if offset is None else offset
        z0 = self.eef[2] + off
        return (z0, z0 + PAD_TALL)

    def arm_yz(self, half_x=0.08, zr=(0.96, 1.30), yr=(-0.32, 0.18), dz=0.02, dy=0.02,
               min_pixels=15):
        """[(z, y, n)] of the arm silhouette in a thin x slab through the eef.  0 rows.

        The tool that actually located a blocked descent on libero_90 task 23.  A *table* is a
        flat plane, so it shows up as a single spike at table z; the HAND is a tilted body and
        shows up as a ridge that walks in y as z rises.  That ridge is what a vertical descent
        drags through a table object - and on that draw the ridge's front edge was 0.19 m
        BEHIND the eef, so every "is anything within 4 cm of the eef" test came back empty while
        the arm was standing on a leaning cutting board.
        """
        if self.eef is None:
            return None
        x0 = float(self.eef[0])
        m = self.arm & (np.abs(self.X - x0) < float(half_x))
        out = []
        for z0 in np.arange(float(zr[0]), float(zr[1]) + 1e-9, float(dz)):
            for y0 in np.arange(float(yr[0]), float(yr[1]) + 1e-9, float(dy)):
                c = (m & (self.Z >= z0) & (self.Z < z0 + float(dz))
                     & (np.abs(self.Y - y0) < 0.5 * float(dy)))
                if int(c.sum()) >= int(min_pixels):
                    out.append((round(float(z0), 3), round(float(y0), 3), int(c.sum())))
        return out

    def hand_lead(self, half_x=0.06, reach=0.25, zlo=0.040, zhi=0.200, band=0.004):
        """Where the PUSHING surface is: the hand block, not the pads.  0 rows.

        Measured on libero_90 task 4, two independent ways: the hand's front face is
        **0.111 m** ahead of `eef_pos` in -y and spans `eef_z+0.082 .. eef_z+0.15`; the
        front finger reaches only 0.069 m.  So a horizontal push is made by the hand and the
        pads never touch the target -> put the HAND's face on what you are pushing.
        """
        if self.eef is None:
            return None
        m = (self.arm & (np.abs(self.X - self.eef[0]) < half_x)
             & (np.abs(self.Y - self.eef[1]) < reach)
             & (self.Z > self.eef[2] + zlo) & (self.Z < self.eef[2] + zhi))
        if m.sum() < 200:
            return None
        ymin = float(self.Y[m].min())
        q = self.pts[m & (self.Y < ymin + band)]
        out = {"lead": float(self.eef[1] - ymin), "n": int(m.sum()),
               "z0": float(q[:, 2].min() - self.eef[2]),
               "z1": float(q[:, 2].max() - self.eef[2])}
        finger = m & (self.Z > self.eef[2] + PAD_DZ) & (self.Z < self.eef[2] + 0.080)
        if finger.sum() > 40:
            out["finger_lead"] = float(self.eef[1] - self.Y[finger].min())
        return out

    # -------------------------------------------------- objects
    def top_face_front_y(self, xbox=(-0.20, 0.15), ybox=(-0.45, 0.06), zrange=(1.05, 1.35),
                         min_pixels=3000, tol=0.006):
        """(z_plane, y_max, n) of the big horizontal plane at the cabinet top - 0 rows.

        That plane is the counter top AND, while the drawer is out, the drawer box's own rim,
        so its front edge tracks the drawer position with ~10000 pixels behind it:
        libero_90 task 4 measured **y -0.078 with the top drawer open and -0.229 once it was
        home**.  Far steadier than any 1 mm vertical-face bucket, and it needs no masks.
        """
        ps = self.planes(axes="z", zrange=zrange, xybox=(xbox, ybox),
                         min_pixels=min_pixels, top=3)
        if not ps:
            return None
        p = max(ps, key=lambda q: q[2])
        m = (~self.arm & (np.abs(self.Z - p[1]) < tol)
             & (self.X > xbox[0]) & (self.X < xbox[1])
             & (self.Y > ybox[0]) & (self.Y < ybox[1]))
        if m.sum() < min_pixels:
            return None
        # 99th percentile, not max: 26 dark pixels of the hand's own underside (achromatic
        # mean 61, so outside `arm`) sat at y -0.03 and made a shut drawer read as open.
        y = self.Y[m]
        return float(p[1]), float(np.percentile(y, 99)), int(m.sum())

    def objects_on_table(self, xy=None, half=0.09, xbox=(-0.30, 0.16), ybox=(-0.08, 0.30),
                         zlo=None, zhi=1.03, min_pixels=80, lo=20, hi=150, sat_min=35):
        """Coloured objects standing on the table, sorted by x (+x = "in front").

        The wrapped butters are chromatic (max-min ~55-90) and mid-bright; the cabinet faces
        are dark and achromatic, the wood table is bright (mean > 150) and the bowl is
        achromatic speckle, so `hi=150` is what keeps this honest *after* an object is lifted
        """
        mx, mn, mean = self.rgb.max(2), self.rgb.min(2), self.rgb.mean(2)
        zlo = (self.table_z() or 0.9498) + 0.0008 if zlo is None else zlo
        m = ((self.Z > zlo) & (self.Z < zhi) & (mean > lo) & (mean < hi)
             & ((mx - mn) > sat_min) & ~self.arm)
        if xy is not None:
            m &= (np.abs(self.X - xy[0]) < half) & (np.abs(self.Y - xy[1]) < half)
        else:
            m &= (self.X > xbox[0]) & (self.X < xbox[1]) & (self.Y > ybox[0]) & (self.Y < ybox[1])
        out, work = [], m.copy()
        while work.sum() >= min_pixels:
            keep, info = self.blob(work, min_pixels=min_pixels)
            if keep is None:
                break
            work &= ~keep
            p = self.pts[keep]
            top = p[p[:, 2] > p[:, 2].max() - 0.002]
            info["top_xy"] = (float(top[:, 0].mean()), float(top[:, 1].mean()))
            info["mask"] = keep
            out.append(info)
        return sorted(out, key=lambda i: i["centre"][0])

    def rim_circle(self, mask=None, pct=88):
        """Fit a circle to the top-decile rim pixels of an object blob.

        Returns centre (x, y), radius, rim-top z (= max z), and the +y outer edge.  The
        blob's bounding-box y-max over-reads the rim by ~4 mm; the fitted ring does not.
        """
        if mask is None:
            mask = self.bowl_mask()
        if mask is None or mask.sum() < 100:
            return None
        p = self.pts[mask]
        top = mask & (self.Z > np.percentile(p[:, 2], pct))
        rp = self.pts[top]
        if len(rp) < 20:
            return None
        A = np.c_[2 * rp[:, 0], 2 * rp[:, 1], np.ones(len(rp))]
        b = rp[:, 0] ** 2 + rp[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0], sol[1]
        R = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
        return {"centre": np.array([cx, cy]), "R": R, "rim_z": float(rp[:, 2].max()),
                "rim_z_mean": float(rp[:, 2].mean()),
                "outer_y": float(cy + R), "n": int(len(rp))}

    def bowl_mask(self, xy=None, zrange=(0.94, 1.04), half=0.12, lo=75, hi=178):
        """Speckled grey bowl on the table.

        The bowl is achromatic and land in the same brightness class as the arm, so the
        band is restricted in z and in x/y; `xy` is worth passing on later calls.
        """
        m = (self.Z > zrange[0]) & (self.Z < zrange[1]) & (self.mean > lo) & (self.mean < hi)
        if xy is None:
            m &= (self.Y > -0.12) & (self.Y < 0.12) & (self.X > -0.24) & (self.X < -0.02)
        else:
            m &= (np.abs(self.X - xy[0]) < half) & (np.abs(self.Y - xy[1]) < half)
        return m

    def blob(self, mask, min_pixels=60):
        """(kept_mask, info) for the largest connected patch of `mask`."""
        if mask.sum() < min_pixels:
            return None, {"n": int(mask.sum())}
        seen = np.zeros(mask.shape, bool)
        best = []
        h, w = mask.shape
        idx = np.argwhere(mask)
        for i0, j0 in idx:
            if seen[i0, j0]:
                continue
            stack = [(i0, j0)]
            seen[i0, j0] = True
            pix = []
            while stack:
                a, b = stack.pop()
                pix.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        na, nb = a + da, b + db
                        if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True
                            stack.append((na, nb))
            if len(pix) > len(best):
                best = pix
        if len(best) < min_pixels:
            return None, {"n": len(best)}
        keep = np.zeros(mask.shape, bool)
        for a, b in best:
            keep[a, b] = True
        p = self.pts[keep]
        info = {"n": int(keep.sum()),
                "x": (float(p[:, 0].min()), float(p[:, 0].max())),
                "y": (float(p[:, 1].min()), float(p[:, 1].max())),
                "z": (float(p[:, 2].min()), float(p[:, 2].max())),
                "centre": np.array([float(0.5 * (p[:, 0].min() + p[:, 0].max())),
                                    float(0.5 * (p[:, 1].min() + p[:, 1].max()))]),
                "zmax": float(p[:, 2].max())}
        return keep, info

    # -------------------------------------------------- handle bars (open/close a drawer)
    def handle_bars(self, panel_y=None, zrange=(0.96, 1.20), xbox=(-0.22, 0.12),
                    min_px=100, dz=0.004, reach=0.09, drop_arm=True):
        """The drawer handles: one dict per bar, sorted by z.  0 rows, no motion.

        A cabinet's handles are the only scene pixels **in front of** the drawer panels, so the
        test is geometric, not colour: pixels with `y > panel_y + 0.006`, bucketed in z.  They
        ARE achromatic (mean ~70-140), i.e. inside `arm`, hence the optional distance-from-eef
        """
        if panel_y is None:
            rows = [r for r in y_faces_mm(self, band_z=zrange, min_pixels=200)]
            panel_y = min(r[0] for r in rows) if rows else -0.24
        mx, mn, mean = self.rgb.max(2), self.rgb.min(2), self.rgb.mean(2)
        m = (((mx - mn) < 25) & (mean > 70)
             & (self.Y > panel_y + 0.006) & (self.Y < panel_y + reach)
             & (self.Z > zrange[0]) & (self.Z < zrange[1])
             & (self.X > xbox[0]) & (self.X < xbox[1]))
        if drop_arm and self.eef is not None:
            m &= (np.linalg.norm(self.pts - self.eef, axis=2) > 0.12)
        if m.sum() < min_px:
            return []
        edges = np.arange(zrange[0], zrange[1] + 1e-9, dz)
        h, e = np.histogram(self.Z[m], edges)
        bars, i = [], 0
        while i < len(h):
            if h[i] < 10:
                i += 1
                continue
            j = i
            while j + 1 < len(h) and h[j + 1] >= 10:
                j += 1
            sel = m & (self.Z >= e[i]) & (self.Z < e[j + 1])
            i = j + 1
            if sel.sum() < min_px:
                continue
            p = self.pts[sel]
            bars.append({"n": int(sel.sum()),
                         "front_y": float(np.percentile(p[:, 1], 99)),
                         "z0": float(p[:, 2].min()), "z1": float(p[:, 2].max()),
                         "zc": float(0.5 * (p[:, 2].min() + p[:, 2].max())),
                         "x0": float(p[:, 0].min()), "x1": float(p[:, 0].max())})
        return bars

    # -------------------------------------------------- the hand in ANY wrist pose
    def horizontal_arm_faces(self, half_x=0.06, dy=0.16, dz=0.14, nmin=0.75, band=0.005,
                             min_px=60):
        """[(dz, n, dy_lo, dy_hi)] of horizontal arm surfaces near eef_pos."""
        if self.eef is None:
            return []
        n = _normal(self.pts)
        rel = self.pts - self.eef
        m = (self.arm & (np.abs(rel[:, :, 0]) < half_x) & (np.abs(rel[:, :, 1]) < dy)
             & (np.abs(rel[:, :, 2]) < dz) & (np.abs(n[:, :, 2]) > nmin))
        out = []
        if m.sum() < min_px:
            return out
        for z0 in np.arange(-dz, dz - 1e-9, band):
            s = m & (rel[:, :, 2] >= z0) & (rel[:, :, 2] < z0 + band)
            if s.sum() < min_px:
                continue
            p = rel[s]
            out.append((round(float(z0), 4), int(s.sum()),
                        round(float(p[:, 1].min()), 4), round(float(p[:, 1].max()), 4)))
        return out

    def arm_envelope(self, half_x=0.05, dy=0.25, dz=0.22, reach=0.15):
        """(dy_lo, dy_hi, dz_lo, dz_hi, n): hand extent around eef_pos in the current pose."""
        if self.eef is None:
            return None
        rel = self.pts - self.eef
        m = (self.arm & (np.abs(rel[:, :, 0]) < half_x) & (np.abs(rel[:, :, 1]) < dy)
             & (np.abs(rel[:, :, 2]) < dz)
             & (np.linalg.norm(rel, axis=2) < reach))
        if m.sum() < 200:
            return None
        p = rel[m]
        return (round(float(p[:, 1].min()), 4), round(float(p[:, 1].max()), 4),
                round(float(p[:, 2].min()), 4), round(float(p[:, 2].max()), 4), int(m.sum()))

    def arm_low_map(self, box=0.28, step=0.03, z_floor=None, min_pixels=8):
        """[(x, y, zmin)] - the LOWEST arm silhouette in each (x, y) cell around the eef.  0 rows.

        This is the 0-row measurement that finds what a stalled descent is standing on.  The
        answer is almost never under the FINGERS: at wrist-down the pads are the lowest part,
        but at a pitched pose (`yaw(Q_HOOK,180)`) the hand block swings out and DOWN *behind*
        the eef - measured on libero_90 task 23: the low ridge sat ~0.19 m behind and ~0.015 m
        below eef_pos, entirely outside both the `pads()` and the `hand_lead` boxes.  That is
        why a descent can stop 10-14 cm above the target with *nothing* within 4 cm of the eef
        in the depth, `rot_err` climbing and the eef drifting along one horizontal axis (the
        hand sliding along a slanted table object).  On that draw the blocker was a leaning
        """
        if self.eef is None:
            return None
        tz = self.table_z()
        if z_floor is None and tz is not None:
            z_floor = float(tz) + 0.015
        rel = self.pts - self.eef
        m = self.arm
        if z_floor is not None:
            m = m & (self.Z > float(z_floor))
        out = []
        for dx in np.arange(-float(box), float(box) + 1e-9, float(step)):
            for dy in np.arange(-float(box), float(box) + 1e-9, float(step)):
                c = (np.abs(rel[:, :, 0] - dx) < 0.5 * float(step)) \
                    & (np.abs(rel[:, :, 1] - dy) < 0.5 * float(step)) & m
                if int(c.sum()) < int(min_pixels):
                    continue
                out.append((round(float(self.eef[0] + dx), 3),
                            round(float(self.eef[1] + dy), 3),
                            round(float(self.Z[c].min()), 4)))
        return sorted(out, key=lambda t: t[2])

    def arm_low(self, box=0.28, step=0.03, z_floor=None, min_pixels=8, top=6):
        """Scalar summary of `arm_low_map`: the lowest cells, the extent, and the ridge.  0 rows.

        `lowest` is a LIST of the lowest `top` cells `(x, y, zmin)` - a real surface, not one
        pixel - and `ridge` is the lowest of them.  Compare `ridge[2]` with the scene cells at
        the same (x, y) (`rig.above_table_map`) to name the object the hand is resting on.
        """
        cells = self.arm_low_map(box=box, step=step, z_floor=z_floor, min_pixels=min_pixels)
        if not cells:
            return None
        rel = self.pts - self.eef
        m = self.arm & (np.abs(rel[:, :, 0]) < float(box)) & (np.abs(rel[:, :, 1]) < float(box))
        p = rel[m]
        return {"n": int(m.sum()),
                "extent": tuple(round(float(v), 4) for v in
                                (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max(),
                                 p[:, 2].min(), p[:, 2].max())),
                "lowest": cells[:int(top)],
                "ridge": cells[0]}


def snapshot(ctx, tag=None, outdir="experiments/data"):
    """One libero_depth + one frame.jpg, 0 rows.  With `tag`, also saves <tag>.npz.

    Save it every time: a depth call costs ~1 s of wall clock and every later re-analysis of
    the same snapshot is free, offline, and re-runnable after the episode is gone.
    """
    s = Scene.from_ctx(ctx, tag=tag, outdir=outdir)
    if tag:
        os.makedirs(outdir, exist_ok=True)
        s.save(os.path.join(outdir, "%s.npz" % tag))
    return s


# =============================================================== plans
def above_table_map(scene, xr=(-0.30, 0.20), yr=(-0.50, 0.16), cell=0.04, dz=0.006,
                    min_pixels=12, arm_reach=0.45):
    """[(x, y, zmax, n)] above-table cells around the working area.  0 rows.

    The inventory of things that can block a low approach: on one draw of libero_90 task 23 the
    table in front of the bottom drawer carried two leaning cutting boards, a bottle and a
    speckled bowl, and the *only* cost of the episode that was not method was discovering them
    one stall at a time.  Colour-agnostic on purpose (`objects_on_table` wants a chromatic
    mid-bright blob and misses a black bottle and a grey board) - this is "what is taller than
    the table, and where", which is what a swept hand needs to know.
    """
    tz = scene.table_z()
    if tz is None:
        return []
    keep = np.ones(scene.Z.shape, dtype=bool)
    if scene.eef is not None:
        keep = ~(scene.arm
                 & (np.linalg.norm(scene.pts - scene.eef, axis=2) < float(arm_reach)))
    out = []
    for x0 in np.arange(float(xr[0]), float(xr[1]) + 1e-9, float(cell)):
        for y0 in np.arange(float(yr[0]), float(yr[1]) + 1e-9, float(cell)):
            m = (keep & (np.abs(scene.X - x0) < 0.5 * float(cell))
                 & (np.abs(scene.Y - y0) < 0.5 * float(cell))
                 & (scene.Z > float(tz) + float(dz)) & (scene.Z < 1.45))
            if int(m.sum()) < int(min_pixels):
                continue
            zmax = float(scene.Z[m].max())
            out.append((round(float(x0), 3), round(float(y0), 3), round(zmax, 4), int(m.sum())))
    return sorted(out, key=lambda t: -t[2])


def drawer_bands(y_faces_bands, panel_y, tol=0.004):
    """{name: dy} for fronts whose y has left `panel_y` by more than `tol`."""
    moved = {}
    for name, y in y_faces_bands.items():
        if y is None:
            continue
        dy = float(y) - float(panel_y)
        if abs(dy) > tol:
            moved[name] = round(dy, 4)
    return moved


def hand_mask_diff(scene, ref, thresh=0.004, box=None, signed=False):
    """Bool mask of pixels whose depth changed by more than `thresh` between two shots.

    `box` is ((x0, x1), (y0, y1), (z0, z1)).  `signed=True` keeps pixels nearer than `ref`.
    """
    d = np.asarray(ref.z) - np.asarray(scene.z)   # > 0: the current surface is NEARER
    m = (d > float(thresh)) if signed else (np.abs(d) > float(thresh))
    if box is not None:
        (x0, x1), (y0, y1), (z0, z1) = box
        m &= ((scene.X > x0) & (scene.X < x1) & (scene.Y > y0) & (scene.Y < y1)
              & (scene.Z > z0) & (scene.Z < z1))
    return m


def drawer_closed(scene, panel_y, body_y, band_z=None, xybox=((-0.16, 0.07), (-0.42, 0.02)),
                  min_pixels=60, top=8, ref_y=None, tol=0.0015):
    """(faces as [(offset, count)], closed) for a drawer on this cabinet.

    **THIS IS A COARSE TEST. It says "the panel has met the body face", not "the joint is
    home", and the two differ by a few millimetres.** On libero_90 task 4 it returned True
    with the panel at -0.227 while the closed reference (the sibling drawer's own panel,
    `closed_reference_y`) was -0.230 and the eef was still advancing 11 mm/row. The
    episode's `Close` predicate failed. Pass `ref_y` to get the 1 mm-accurate answer, and
    use `push_drawer_closed` (which stops on the eef's stall) to drive the close.
    """
    zlo = (band_z - 0.02) if band_z is not None else 1.065
    zhi = (band_z + 0.11) if band_z is not None else 1.195
    pls = scene.planes(axes="y", zrange=(zlo, zhi), xybox=xybox, min_pixels=min_pixels, top=top)
    panel = [p for p in pls if abs(p[1] - panel_y) < 0.012 and p[2] >= 250]
    bodyish = [p for p in pls if p[1] <= body_y + 0.018 and p[2] >= 400]
    mid = [p[1] for p in pls if body_y + 0.025 < p[1] < panel_y - 0.02 and p[2] >= 250]
    closed = (not panel) and (not mid) and bool(bodyish)
    if ref_y is not None:
        _y, _n, home = panel_home(scene, ref_y, band_z=(zlo, zhi), xybox=xybox, tol=tol)
        closed = bool(closed) and bool(home)
    return [(p[1], p[2]) for p in pls], bool(closed)


def column_max_z(scene, x, y, half=0.012, zrange=(0.90, 1.40), arm_free=False):
    """Highest scene point in a small vertical column - the "is it in the box" test.

    On the drawer floor the reading is `floor_z + object_height` (task 3: 1.113 + 0.0186 =
    1.1316); a value at the floor is an empty box; a value near the rim means the object is
    perched on a wall.  Costs 0 rows and is the only verdict available once the drawer shuts.
    """
    m = ((np.abs(scene.X - x) < half) & (np.abs(scene.Y - y) < half)
         & (scene.Z > zrange[0]) & (scene.Z < zrange[1]))
    if arm_free and scene.rgb.any():
        m = m & ~scene.arm
    return float(scene.Z[m].max()) if m.sum() else None


def box_free_columns(scene, x0, x1, y0, y1, step=0.02, half=0.015, zrange=(0.90, 1.40)):
    """[(x, y, max_z)] over a drawer's mouth: a column reading the floor is free.

    A column that reads the RIM is a wall; one that reads the TABLE z means the sight line
    passed under the box, not that the box is absent.  A near side wall hides part of the
    floor, so trust this grid over the floor plane's own x extent.
    """
    out = []
    for y in np.arange(y0, y1 + 1e-9, step):
        for x in np.arange(x0, x1 + 1e-9, step):
            out.append((round(float(x), 3), round(float(y), 3),
                        column_max_z(scene, x, y, half=half, zrange=zrange)))
    return out


def _normal(pts):
    """Per-pixel unit surface normal, from the u/v tangents of the unprojected grid."""
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(du, dv)
    return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)


def y_faces_mm(scene, band_z=(1.105, 1.19), xybox=((-0.16, 0.07), (-0.42, 0.02)),
               min_pixels=40, nmin=0.9, bin_mm=1.0):
    """[(y_offset, count)] of the y-facing, arm-excluded faces in a z band, 1 mm bins.

    `Scene.planes()` quantises offsets to 5 mm, which is coarser than the difference between
    "the panel has met the body face" and "the drawer joint is home" (task 4: the panel sat
    3 mm proud of the cabinet front and both landed in one 5 mm bucket, so `Close` failed while
    the plane list looked clean).  Keep 1 mm.  Now a thin call into `face_bins`, which is the
    same test on any axis; `~arm` drops the handle bar as well as the gripper, which is what
    you want when the question is where the *panel* is.
    """
    return face_bins(scene, 1, band_z, xybox, min_pixels=min_pixels, bin_mm=bin_mm, nmin=nmin)


def front_face_y(scene, band_z=(1.105, 1.19), xybox=((-0.16, 0.07), (-0.42, 0.02)),
                 min_pixels=60):
    """(y, n) of the FRONT-MOST (+y) face plane in the band - the drawer's panel.

    `S = +1` only: on a mirrored chest (KITCHEN_SCENE4, drawers open toward -y) "front-most" is
    the MINIMUM y, and this helper returns the *closed* plane - a home verdict on an open
    drawer, with no error.  Use `front_face_on(scene, fr, band_z)` there.  `~arm` also removes
    the handle bar, so a closed drawer's front-most face is its panel.
    """
    return front_face(scene, 1, +1.0, band_z, xybox, min_pixels)


def panel_face(scene, band_z=(1.105, 1.176), xybox=((-0.16, 0.07), (-0.42, 0.02)),
               min_pixels=300, min_cover=0.5):
    """(y, n, zspan) of the front-most 1 mm y-face whose z span covers the panel band.
    `S = +1` only; `panel_home_on` is the sign-agnostic form.

    `front_face_y`'s "max y" rule returns whatever 1 mm bin is nearest the camera.  On a real
    close that was a **124-pixel noise bin at -0.2285 while the drawer's own panel was a
    1591-pixel bin at -0.2305** — the same 1 mm the whole episode turns on.  A drawer's panel
    is a tall face, so require the bin to span `min_cover` of the band before believing it.
    """
    lo, hi = float(band_z[0]), float(band_z[1])
    n_all = _normal(scene.pts)
    rows = []
    for y, n in y_faces_mm(scene, band_z=(lo, hi), xybox=xybox, min_pixels=min_pixels,
                           bin_mm=1.0):
        m = (np.abs(scene.Y - y) < 0.0008) & (scene.Z > lo) & (scene.Z < hi) & ~scene.arm \
            & (scene.X > xybox[0][0]) & (scene.X < xybox[0][1]) \
            & (scene.Y > xybox[1][0]) & (scene.Y < xybox[1][1])
        m &= np.abs(n_all[:, :, 1]) > 0.85
        if m.sum() < min_pixels:
            continue
        zspan = float(scene.Z[m].max() - scene.Z[m].min())
        if zspan >= min_cover * (hi - lo):
            rows.append((y, n, zspan))
    if not rows:
        return None, 0, 0.0
    rows.sort(key=lambda t: -t[0])
    y, n, zspan = rows[0]
    return y, n, zspan


def closed_reference_y(scene, band_z, drop=0.14, half=0.045,
                       xybox=((-0.16, 0.07), (-0.42, 0.02))):
    """The plane a drawer's front panel occupies when it is HOME, from a closed sibling.
    `S = +1` only: on a mirrored chest the reference is the closed sibling on the
    other side, so use `panel_home_on(scene, fr, ref_off, band_z)`.

    This cabinet ships with the other drawer shut, so its panel gives the fully-closed
    plane for free, at 1 mm resolution, in the same depth call as the open drawer you are
    about to push.  Measured on libero_90 task 4: -0.2294 for the closed lower drawer, while
    the pushed drawer's panel came to -0.2270 (failed attempt) / -0.2305 (the one that
    passed).  Falling back to `body_y` from `planes()` (-0.228) is 2 mm too coarse to see that.
    """
    lo = band_z - float(drop) - half
    hi = band_z - float(drop) + half
    return front_face_y(scene, band_z=(lo, hi), xybox=xybox, min_pixels=80)


def panel_home(scene, ref_y, band_z=(1.105, 1.176), xybox=((-0.16, 0.07), (-0.42, 0.02)),
               tol=0.0015, min_pixels=300):
    """(y_face, n, home) - is the pushed drawer's panel at the closed reference plane?

    `tol` is 1.5 mm on purpose: 3 mm of remaining travel is invisible in the 5 mm buckets
    and was the entire difference between success and failure on libero_90 task 4.  The face
    is chosen by `panel_face` (by its z span), not by "front-most".
    """
    y, n, _zspan = panel_face(scene, band_z=band_z, xybox=xybox, min_pixels=min_pixels)
    if y is None:
        return None, 0, False
    return y, n, bool(abs(y - float(ref_y)) <= tol)


def floor_plane_gone(scene, z_floor, tol=0.005, min_pixels=500, zrange=(0.96, 1.35),
                     xybox=None):
    """(gone, n_pixels): is the drawer's FLOOR plane absent from this depth call?

    While a drawer is out its floor is a huge horizontal plane (libero_90 tasks 4 and 5:
    z 1.113-1.115 with 5 600-5 700 pixels).  Once it is shut the interior is invisible and
    that plane **disappears** - which is the hardest signal in the whole close verdict,
    because no 1 mm panel-plane ambiguity can fake it.  Use it whenever `panel_home` is
    fighting the other two signals.
    """
    ps = scene.planes(axes="z", zrange=zrange, xybox=xybox, min_pixels=min_pixels, top=12)
    n = int(sum(p[2] for p in ps if abs(p[1] - float(z_floor)) < tol))
    return n == 0, n


# =============================================================== the close that finishes
def push_to_stop(ctx, q_tgt, z_target, x_target, panel_y, body_y, ref_y=None,
                 band=(1.105, 1.176), max_rows=34, stall_rows=3, stall_tol=0.0015,
                 gripper=-1.0, tag="pushhome", verbose=True, floor_z=None):
    """Push -y until the JOINT is home, then take a 1 mm verdict.  Returns (payload, ok, log).

    THE TRAP THIS FIXES (libero_90 task 4, both directions measured).  `push_drawer_closed`
    stops on the eef's stall too, but it aims by `band_z` = where the *pads* go, and the
    pads are not what pushes: the hand block's front face is 0.111 m ahead of `eef_pos` and
    spans `eef_z+0.082 .. eef_z+0.15` (see `HAND_LEAD`, `Scene.hand_lead`).  Aiming the pads
    at the panel band (`z_target` 1.09) put the hand's face entirely ABOVE the panel's top
    edge; it rode the counter's front edge and stalled with the panel at -0.2270, 2 mm proud
    -> the episode failed on `Close`.  Aiming the hand's face at the protruding handle bar
    (**z_target 1.045** = bar_bottom - HAND_Z0, with x on the bar's centre) pushed the same
    """
    payload = observe(ctx)
    log = []
    dead, stalled = 0, False
    for _ in range(int(max_rows)):
        p = pose_of(payload)
        a = np.zeros(7)
        a[1] = -1.0
        a[0] = float(np.clip((x_target - p[0]) / 0.010, -0.6, 0.6))
        a[2] = float(np.clip((z_target - p[2]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = gripper
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        q = pose_of(payload)
        moved = float(np.hypot(q[0] - p[0], q[1] - p[1]))
        dead = dead + 1 if moved < stall_tol else 0
        if verbose:
            print("  push row eef=%s moved=%.4f dead=%d" % (np.round(q, 4), moved, dead),
                  flush=True)
        if dead >= stall_rows:
            stalled = True
            break

    froze = None
    if stalled:
        p = pose_of(payload)
        payload, _ = step(ctx, [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, gripper]], verbose=False)
        rose = float(pose_of(payload)[2] - p[2])
        froze = bool(abs(rose) < 0.002)
        if verbose:
            print("  stall check: +z row rose %.4f m -> %s%s" % (
                rose, "FROZEN, not the drawer" if froze else "real contact",
                "" if rose > 0.004 else "  (weak: could be a corner jam - trust the face "
                "verdict)"), flush=True)
        if not froze:                      # put the pads back on the bar for the depth call
            p = pose_of(payload)
            step(ctx, [[0.0, 0.0, float(np.clip((z_target - p[2]) / 0.010, -1, 1)),
                        0.0, 0.0, 0.0, gripper]], verbose=False)

    s = snapshot(ctx, tag=tag)
    faces, coarse = drawer_closed(s, panel_y, body_y, band_z=float(band[0]) + 0.02)
    y_face, n_face, home = (panel_home(s, ref_y, band_z=band) if ref_y is not None
                            else (None, 0, None))
    top = s.top_face_front_y()
    counter_in = None if top is None else bool(top[1] < panel_y - 0.10)
    # signal 4: the drawer's floor plane is only visible while the drawer is OUT
    floor_gone, floor_n = (floor_plane_gone(s, floor_z) if floor_z is not None
                           else (None, None))
    stalled = bool(stalled) and not bool(froze)
    # `home` is measured against the OTHER drawer's panel, and the two can differ by ~1.6 mm
    # (libero_90 task 5: -0.2305 home vs -0.2289 sibling -> home=False on a closed drawer).
    # So home=False is only a veto when the floor plane is still visible as well.
    ok = (bool(stalled) and bool(coarse) and (counter_in is not False)
          and (home is not False or floor_gone is True))
    if verbose:
        print("  faces(y,n)=%s coarse=%s" % (faces, coarse), flush=True)
        print("  panel face y=%s n=%d  ref_y=%s  home=%s" % (
            y_face, n_face, None if ref_y is None else round(float(ref_y), 4), home),
            flush=True)
        print("  top plane %s front y=%s -> drawer in=%s  stalled=%s  froze=%s" % (
            None if top is None else round(top[0], 3),
            None if top is None else round(top[1], 4), counter_in, stalled, froze),
            flush=True)
        if floor_gone is not None:
            print("  drawer-floor plane (z %.3f): %s n=%s" % (
                float(floor_z), "GONE -> interior invisible" if floor_gone
                else "STILL VISIBLE -> the drawer is still out", floor_n), flush=True)
        if home is False and floor_gone is True and counter_in:
            print("  note: home=False but the floor plane vanished and the rim plane moved "
                  "home -> the sibling's panel is the outlier, not the push.", flush=True)
        print("  VERDICT: %s" % ("HOME" if ok else "NOT HOME - do not just push harder; "
              "check the hand's face is on the BAR (eef_z <= bar_bottom - 0.082), then "
              "retreat ~10 mm and push again"), flush=True)
    log.append({"faces": faces, "coarse": coarse, "y_face": y_face, "ref_y": ref_y,
                "home": home, "top": top, "counter_in": counter_in, "stalled": stalled,
                "froze": froze, "floor_gone": floor_gone, "floor_n": floor_n,
                "ok": ok, "eef": pose_of(payload)})
    return payload, ok, log


def push_drawer_closed(ctx, q_tgt, panel_y, body_y, band_z, x_target, gripper=-1.0,
                       rows_per_batch=5, max_batches=8, ref_y=None, stall_rows=3,
                       stall_tol=0.0015, verify_z=True, verbose=True):
    """SUPERSEDED by `push_to_stop` - kept so older notes still run.

    This one aims z at `band_z`, i.e. at the PAD band, and the pads are not what pushes
    (`HAND_LEAD`: the hand block's face is 0.111 m ahead and 4 cm higher).  On libero_90
    task 4 that variant stalled with the panel 2-4 mm proud of the closed reference and the
    `Close` predicate failed; the same drawer went home at `z_target` 1.045, i.e. 4.5 cm
    lower, where the hand's face covers the handle bar.  Forwarded, so callers get the
    stall rule, the `+z` freeze proof and the 1 mm verdict of `push_to_stop`.
    """
    return push_to_stop(ctx, q_tgt, band_z, x_target, panel_y, body_y, ref_y=ref_y,
                        max_rows=int(rows_per_batch) * int(max_batches),
                        stall_rows=stall_rows, stall_tol=stall_tol, gripper=gripper,
                        verbose=verbose)


# =============================================================== rim grasp
def rim_grasp_target(rim, jaw_dx=None, jaw_dy=None, dz=-0.055, wall_inset=0.004, side=1.0):
    """eef target that puts the JAW PAIR on a rim's outer wall (pure, offline-testable).

    rim: a dict with "centre" (x, y), "R" and "rim_z" (the rim's plane - `rim_plane`'s zc).
    dz: pad bottom relative to the lip; -0.055 puts it ~12 mm below the lip, where a rim
    pinch holds.  `side` picks the wall to pinch: **+1 is the +y rim** (the historical
    default, unchanged) and **-1 is the -y rim**, which is what the stacking tasks want -
    the bowl then hangs on the eef's +y side, keeping the destination's far side and the
    reachable region on the -y side of the arm and out of the stretched-arm -x dead zone
    (SKILL §2).  A rim dict from `Scene.rim_circle` works for side=+1; for side=-1 the
    caller's "centre"/"R" are all that is used.
    """
    jx = JAW_DX if jaw_dx is None else jaw_dx
    jy = JAW_DY if jaw_dy is None else jaw_dy
    cy = float(rim["centre"][1])
    jaw_y = cy + (float(rim["R"]) - wall_inset) * (1.0 if side >= 0 else -1.0)
    jaw = np.array([float(rim["centre"][0]), jaw_y])
    return {"eef": np.array([jaw[0] - jx, jaw[1] - jy, float(rim["rim_z"]) + dz]),
            "jaw": jaw, "pad_bottom": float(rim["rim_z"]) + dz + PAD_DZ, "side": side}


def grasp_rim(ctx, rim, q_tgt, jaw_dx=None, jaw_dy=None, dz=-0.055,
              max_rows=26, lift=0.13, verbose=True, side=1.0):
    """Descend onto `rim`, close until the jaws settle, lift. Returns (payload, complete, gq).

    Does NOT verify the grasp - that needs the picture (`save_frame`) because a brightness
    mask over the old spot also matches the table grain.  See `object_gone`.
    """
    t = rim_grasp_target(rim, jaw_dx, jaw_dy, dz, side=side)
    if verbose:
        print("  rim grasp: jaw -> %s, eef -> %s, pad_bottom %.4f (rim top %.4f)" % (
            np.round(t["jaw"], 4), np.round(t["eef"], 4), t["pad_bottom"],
            float(rim["rim_z"])), flush=True)
    payload, _complete = servo(ctx, t["eef"], q_tgt, -1.0, max_rows=max_rows,
                              rate=0.011, zclip=0.8, tol=0.004, verbose=verbose)
    payload, complete, gq = close_grip(ctx, verbose=verbose)
    p = pose_of(payload)
    payload, complete = servo(ctx, [p[0], p[1], p[2] + lift], q_tgt, 1.0,
                              max_rows=int(lift / 0.010) + 4, tol=0.008, verbose=verbose)
    return payload, complete, gq


def object_gone(scene, xy, table_z=None, half=0.10, min_pixels=60, dz=0.035):
    """Did the thing at `xy` leave the table?  (zmax collapse, not just pixel count.)

    A grey-pixel mask over the old spot also catches the wood table grain (mean 75-178),
    so "the blob is still there" is a false positive after a *correct* grasp.  The bowl's
    rim is ~5 cm above the table; if the blob's zmax has fallen to within `dz` of the
    """
    if not scene.rgb.any():
        return None, {"n": 0, "note": "no rgb in this snapshot - cannot tell"}
    tz = table_z if table_z is not None else scene.table_z()
    keep, info = scene.blob(scene.bowl_mask(xy=xy, half=half), min_pixels=min_pixels)
    if info.get("n", 0) < min_pixels:
        return True, info
    gone = tz is not None and info["zmax"] < tz + dz
    return bool(gone), info


# =============================================================== stack metrology
# Measured on libero_90 task 17 (KITCHEN_SCENE2, three bowls on one table), 0 rows each.
def rim_plane(scene, mask, bands=(0.002, 0.003, 0.004), min_ring=40):
    """Circle AND PLANE fit to an object's rim ring -> centre, R, zc, tilt, arc, residual.

    zc is the ring's plane evaluated at the ring's CENTRE, i.e. the rim plane at the object's
    own axis, so `zc - h_bowl` is the object's base plane even when it is tilted;
    `rim_z` (the max) over-reads a tilted base by rr*sin(tilt) (1.6 mm/deg for this bowl).
    """
    if mask is None or mask.sum() < 100:
        return None
    p = scene.pts[mask]
    zmax = float(p[:, 2].max())
    rp, band = None, None
    for b in bands:
        ring = mask & (scene.Z > zmax - b)
        if int(ring.sum()) >= min_ring:
            rp, band = scene.pts[ring], float(b)
            break
    if rp is None:
        return None
    A = np.c_[2 * rp[:, 0], 2 * rp[:, 1], np.ones(len(rp))]
    rhs = rp[:, 0] ** 2 + rp[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    cx, cy = float(sol[0]), float(sol[1])
    R = float(np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-9)))
    Ap = np.c_[rp[:, 0], rp[:, 1], np.ones(len(rp))]
    (pa, pb, pc), *_ = np.linalg.lstsq(Ap, rp[:, 2], rcond=None)
    zc = float(pa * cx + pb * cy + pc)
    ang = np.sort(np.mod(np.degrees(np.arctan2(rp[:, 1] - cy, rp[:, 0] - cx)), 360.0))
    arc = float(360.0 - np.diff(np.r_[ang, ang[0] + 360.0]).max())
    res = float(np.sqrt(np.mean((np.hypot(rp[:, 0] - cx, rp[:, 1] - cy) - R) ** 2)))
    return {"centre": np.array([cx, cy]), "R": R, "rim_z": zmax, "zc": zc,
            "rim_zc": zc, "tilt": float(np.degrees(np.arctan(np.hypot(pa, pb)))),
            "arc": arc, "res": res, "band": band, "n_ring": int(len(rp)),
            "plane": (float(pa), float(pb), float(pc))}


def bowl_height(scene, masks, table_z=None, bands=(0.002, 0.003, 0.004)):
    """h = rim plane - base plane, calibrated on the bowls that are standing on the table.

    Several identical bowls on one flat table are three measurements of the same number:
    each one's base plane IS the table, so `zc - table_z` is the height the stack test needs,
    WITH `rim_plane`'s bias included.  Scoring `upper_zc - h` against the destination's `zc`
    then cancels that bias between the two bowls.  Measured (task 17): 0.0497-0.0499, spread
    0.2 mm.
    """
    tz = scene.table_z() if table_z is None else table_z
    hs = []
    for m in masks:
        r = rim_plane(scene, m, bands=bands)
        if r is not None and 0.040 < r["R"] < 0.070:
            hs.append(r["zc"] - tz)
    if not hs:
        return None, []
    return float(np.median(hs)), hs


def place_on_rim(ctx, q_tgt, z_rim, rel_z, h_bowl, gripper=-1.0, seat=0.004,
                 sink=0.00015, open_rows=12, retreat=0.15, zrate=0.10, tol=0.0015,
                 verbose=True):
    """SET a carried object down on a rim, then let go - do not drop it.

    DEPRECATED AS AN ENDING on the task-17 bowls: see the note at the bottom.  Kept because
    the descent half of it (coarse, then ~1 mm rows) is the right shape.
    """
    payload = observe(ctx)
    z_eef = eef_for_base(float(z_rim) + float(seat), rel_z, h_bowl)
    p = pose_of(payload)
    if verbose:
        print("  place_on_rim: rim %.4f, rel_z %+.4f, h %.4f -> eef %.4f (base %.4f = "
              "rim %+.4f); from eef %.4f" % (
                  z_rim, rel_z, h_bowl, z_eef, z_eef + rel_z - h_bowl,
                  z_eef + rel_z - h_bowl - z_rim, p[2]), flush=True)
    xy0 = p[:2].copy()                             # hold x/y: the caller aims, this seats
    if p[2] > z_eef + 0.020:                       # coarse: leave the last 20 mm to the rows
        payload, _ = goto(ctx, [p[0], p[1], z_eef + 0.020], q_tgt, 1.0, tol=0.004,
                          max_rows=int((p[2] - z_eef) / 0.010) + 8, verbose=verbose,
                          tag="[seat]")
    rows = 0
    for i in range(60):                            # fine: ~1 mm rows, never overshoot
        p = pose_of(payload)
        err = z_eef - p[2]
        if err > -1e-4:
            break
        a = np.zeros(7)
        a[0] = float(np.clip((xy0[0] - p[0]) / 0.011, -0.5, 0.5))
        a[1] = float(np.clip((xy0[1] - p[1]) / 0.011, -0.5, 0.5))
        a[2] = float(np.clip(err / 0.010, -zrate, 0.0))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = 1.0
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        rows += 1
        if verbose and (i % 4 == 0):
            print("  seat %2d eef %.4f (base %.4f, rim %+.4f)" % (
                i, pose_of(payload)[2], pose_of(payload)[2] + rel_z - h_bowl,
                pose_of(payload)[2] + rel_z - h_bowl - z_rim), flush=True)
    if verbose:
        print("  at seat: eef %.4f base %.4f = rim %+.4f after %d rows, gq %.4f" % (
            pose_of(payload)[2], pose_of(payload)[2] + rel_z - h_bowl,
            pose_of(payload)[2] + rel_z - h_bowl - z_rim, rows, gq_of(payload)), flush=True)
    for i in range(int(open_rows)):                # open WHILE sinking: contact, not a fall
        a = np.zeros(7)
        a[2] = float(-abs(sink) / 0.011)
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = float(gripper)
        payload, _ = step(ctx, [a.tolist()], verbose=False)
    p = pose_of(payload)
    payload, _ = goto(ctx, [p[0], p[1], p[2] + float(retreat)], q_tgt, float(gripper),
                      tol=0.006, max_rows=int(retreat / 0.010) + 8, verbose=verbose,
                      tag="[retreat]")
    if verbose:
        print("  released and retreated to %s (gq %.4f)" % (
            np.round(pose_of(payload), 4), gq_of(payload)), flush=True)
    return payload


# =============================================================== carried-object placement
# Extracted verbatim from the task-17 run that PASSED (libero_90 task 17, 272 rows, ending
# HELD, `finish` -> task_complete True): code/stack17f/controller.py.  Everything here exists
# to enforce two rules, and both of them cost an episode to learn:
#
#   carried rim plane  zc = eef_z + rel_z      carried BASE = zc - h_bowl
#   eef_z for a wanted base B:  B + h_bowl - rel_z                       <- eef_for_base
#
# `rel_z` is the MEASURED rim-plane offset (rim - eef), the quantity every measurement in this
# file returns.  It DRIFTS inside one carry (0.053 -> 0.074 on task 17), so every descent target
# is recomputed from a fresh measurement - that is what descend_to_base is for.  The other,
# silent half of the same bug: writing `base - rel_z` buries the object by exactly h_bowl
# (49.5 mm for these bowls) and looks like a normal descent the whole way down.
def eef_for_base(base, rel_z, h_bowl):
    """eef_z that puts the carried object's BASE plane at `base`."""
    return float(base) + float(h_bowl) - float(rel_z)


def plane_normal(p):
    """Least-squares plane normal of a point set (SVD), unit length, +z-ish."""
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    return n / (np.linalg.norm(n) + 1e-12)


def circle_in_plane(rp, n):
    """Circle fit of 3-D points in a plane of normal n -> (centre, R, zc, arc, res).

    `zc` is the plane's z AT the fitted centre, so `zc - h` is the object's base plane even
    when it is tilted; `arc` is the ring's angular coverage (the trust figure) and `res` the
    radial residual in metres.
    """
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


def robust_ring(scene, mask, Rlo=0.040, Rhi=0.070, tilt_max=33.0, dz=0.003, naz=24):
    """Rim circle+plane fit that does NOT need a good seed - for a TILTED carried object.

    `rim_plane`'s 2 mm world-z band only works on a level object, and SKILL 4b's
    `fit_ring3d` coarse-to-fine schedule seeds at n = +z: on a bowl hanging at a pivot tilt
    that seed takes only the near arc, the plane fit returns a 50-70 deg normal and the whole
    schedule follows it.  Measured on task 17: true R 0.0542 / tilt 12.6 / arc 181 /
    res 0.7 mm, while the +z seed returned R 0.016 / tilt 41 deg.
    """
    p = scene.pts[mask]
    if len(p) < 120:
        return None
    best = None
    for t in np.arange(0.0, float(tilt_max) + 1e-9, 3.0):
        tr = np.radians(t)
        for k in range(int(naz)):
            az = 2.0 * np.pi * k / float(naz)
            n = np.array([np.sin(tr) * np.cos(az), np.sin(tr) * np.sin(az), np.cos(tr)])
            rp = p[(p @ n) > (p @ n).max() - dz]
            if len(rp) < 80:
                continue
            _c, R, _zc, _arc, res = circle_in_plane(rp, n)
            if Rlo < R < Rhi and res < 0.0015 and (best is None or len(rp) > best[0]):
                best = (len(rp), n)
    if best is None:
        return None
    n, out = best[1], None
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


def measure_carried(scene, hint, jaw_dx, jaw_dy, h_bowl, R_ref, z_lo=None, quiet=False):
    """Measure the object in the jaws by GEOMETRY (SKILL 4b) - the speckled bowls are
    achromatic, mean 145-148, i.e. inside Scene.arm's own class, so colour cannot do it.
    """
    eef = scene.eef
    jaw = np.array([eef[0] + jaw_dx, eef[1] + jaw_dy])
    hxy = np.array([eef[0] + hint[0], eef[1] + hint[1]])
    lo_abs = -1e9 if z_lo is None else float(z_lo)
    cands = []
    for (dn, up) in ((-0.020, 0.070), (-0.020, 0.085), (-0.020, 0.100), (-0.020, 0.115),
                     (0.0, 0.060)):
        zlo, zhi = max(lo_abs, eef[2] + dn), eef[2] + up
        m = ((np.abs(scene.X - hxy[0]) < 0.085) & (np.abs(scene.Y - hxy[1]) < 0.085)
             & (scene.Z > zlo) & (scene.Z < zhi)
             & (np.hypot(scene.X - jaw[0], scene.Y - jaw[1]) > 0.036))
        keep, info = scene.blob(m, min_pixels=150)
        if keep is None:
            continue
        r = robust_ring(scene, keep, Rlo=max(0.040, R_ref - 0.010),
                        Rhi=min(0.070, R_ref + 0.010))
        if r is None:
            continue
        r["n"] = int(info["n"])
        r["win"] = (round(zlo, 4), round(zhi, 4))
        cands.append(r)
    good = [c for c in cands if abs(c["R"] - R_ref) < 0.0025 and c["res"] < 0.0013
            and c["arc"] > 110 and np.linalg.norm(c["centre"][:2] - hxy) < 0.045]
    if not good:
        if not quiet:
            print("  carried: no usable ring fit (%d candidates %s)" % (
                len(cands), [(round(c["R"], 3), round(c["zc"], 3)) for c in cands]),
                flush=True)
        return None
    good.sort(key=lambda c: -c["n_ring"])
    r = good[0]
    r["base"] = r["zc"] - float(h_bowl)
    r["d_xy"] = r["centre"][:2] - eef[:2]
    r["d_z"] = r["zc"] - eef[2]
    if not quiet:
        print("  carried: c=%s R=%.4f zc=%.4f base=%.4f tilt=%.1f arc=%.0f res=%.1fmm "
              "win=%s | rel centre-eef=%s rel zc-eef=%+.4f" % (
                  np.round(r["centre"], 4), r["R"], r["zc"], r["base"], r["tilt"],
                  r["arc"], r["res"] * 1000.0, r["win"], np.round(r["d_xy"], 4), r["d_z"]),
              flush=True)
    return r


def descend_rows(ctx, z_target, q_tgt, gripper, step=0.0012, max_rows=40, floor_z=None,
                 hold_xy=None):
    """~`step` m downward rows to z_target, never past floor_z.  Returns (payload, rows)."""
    payload = observe(ctx)
    z_target = float(z_target)
    rows = 0
    while rows < int(max_rows):
        p = pose_of(payload)
        err = z_target - p[2]
        if err > -1e-5:
            break
        if floor_z is not None and p[2] <= float(floor_z) + 1e-6:
            print("  descend: at the hard floor eef_z %.4f" % p[2], flush=True)
            break
        a = np.zeros(7)
        if hold_xy is not None:
            a[0] = float(np.clip((hold_xy[0] - p[0]) / 0.011, -0.35, 0.35))
            a[1] = float(np.clip((hold_xy[1] - p[1]) / 0.011, -0.35, 0.35))
        a[2] = float(np.clip(err / 0.011, -step / 0.011, 0.0))
        a[3:6] = np.clip(rotvec_err(quat_of(payload), q_tgt) / 0.5, -1, 1)
        a[6] = float(gripper)
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        rows += 1
    return payload, rows


def rotate_to(ctx, q_tgt, hold_xyz, gripper=1.0, max_rows=16, tol=0.05, tag="[rot]",
              verbose=True):
    """Rotate IN PLACE until the ORIENTATION error is under `tol`.

    `goto` and `servo` both gate on the position error only, so "hold this pose and turn the
    wrist" returns on row 0 (`converged err=[0 0 0] after 0 rows`) and rotates NOTHING.  That
    silent no-op left a carried bowl at a 14 deg pivot tilt for a whole carry on task 17 and
    put the aimed lateral offset 3 mm out.  Rotate with this, then re-read eef_pos and
    recompute every later target from the MEASURED pose (a rotation-only row walks the arm
    4-8 cm, SKILL 2).
    """
    payload = observe(ctx)
    hold = np.asarray(hold_xyz, dtype=np.float64)
    for i in range(int(max_rows)):
        err = rotvec_err(quat_of(payload), q_tgt)
        if np.linalg.norm(err) < tol:
            if verbose:
                print("  %s converged rot_err=%.3f after %d rows" % (
                    tag, float(np.linalg.norm(err)), i), flush=True)
            return payload, True
        p = pose_of(payload)
        a = np.zeros(7)
        a[:3] = np.clip((hold - p) / 0.011, -0.5, 0.5)
        a[3:6] = np.clip(err / 0.30, -1, 1)
        a[6] = float(gripper)
        payload, _ = step(ctx, [a.tolist()], verbose=False)
        if verbose and i % 3 == 0:
            print("  %s %2d rot_err=%.3f eef=%s" % (tag, i, float(np.linalg.norm(err)),
                                                    np.round(pose_of(payload), 4)),
                  flush=True)
    if verbose:
        print("  %s hit max_rows rot_err=%.3f" % (
            tag, float(np.linalg.norm(rotvec_err(quat_of(payload), q_tgt)))), flush=True)
    return payload, False


def contact_test(ctx, z_rim, h_bowl, q_hold, measure, press=0.0012, gripper=1.0,
                 xy_hold=None, threshold=0.0006, verbose=True):
    """Press `press` m and compare the carried object's base plane with the arm's motion.

    THE in-loop contact sensor for a HELD object.  The eef's own stall is not one (a lip
    pinch slips: task 17's arm never stalled while the bowl rested on the destination), and
    `gripper_qpos` is not one either (it decayed to 0.0021, the "closed on nothing" band,
    with a bowl hanging safely in the jaws for a whole descent).
    """
    payload = observe(ctx)
    r0 = measure()
    if r0 is None:
        return None
    p0 = pose_of(payload)
    floor = eef_for_base(float(z_rim) - 0.003, r0["d_z"], h_bowl)
    payload, rows = descend_rows(ctx, p0[2] - float(press), q_hold, gripper, step=0.0004,
                                 max_rows=6, floor_z=floor, hold_xy=xy_hold)
    eef1 = pose_of(payload)
    r1 = measure()
    if r1 is None:
        return None
    d_eef = float(eef1[2] - p0[2])
    d_base = float(r1["base"] - r0["base"])
    out = {"contact": bool(d_base - d_eef > float(threshold)), "d_eef": d_eef,
           "d_base": d_base, "base": float(r1["base"]), "rel_z": float(r1["d_z"]),
           "tilt": float(r1["tilt"]), "rows": int(rows)}
    if verbose:
        print("  contact_test: deef %+.2f mm vs dbase %+.2f mm -> %s (base %.4f = rim "
              "%+.1f mm, tilt %.1f)" % (
                  1000.0 * d_eef, 1000.0 * d_base,
                  "CONTACT" if out["contact"] else "FREE (no contact)",
                  out["base"], 1000.0 * (out["base"] - float(z_rim)), out["tilt"]),
              flush=True)
    return out


def descend_to_base(ctx, z_rim, h_bowl, q_hold, measure, aim_xy=None, gripper=1.0,
                    step=0.0025, tol=0.0010, max_steps=20, press=0.0012, verbose=True):
    """Closed-loop descent of a CARRIED object onto a rim plane, decided on the MEASURED base.

    `z_rim`  the destination rim plane (rim_plane's zc)
    `measure()` must take its OWN depth call and return {"base", "d_z", "d_xy", "centre",
    "tilt"} for the CURRENT pose (or None) - `rig.measure_carried` behind a lambda is enough.
    """
    payload = observe(ctx)
    info = {"rim": float(z_rim), "contact": None}
    prev = None
    for i in range(int(max_steps)):
        r = measure()
        if r is None:
            print("  descend_to_base: object unmeasurable at step %d" % i, flush=True)
            return payload, info
        base, rel_z = float(r["base"]), float(r["d_z"])
        lat = (None if aim_xy is None
               else float(np.linalg.norm(np.asarray(r["centre"][:2]) - np.asarray(aim_xy))))
        info["lateral"] = lat
        if verbose:
            print("  descend %2d: base %.4f rim %.4f -> %+.1f mm, tilt %.1f deg%s" % (
                i, base, z_rim, 1000.0 * (base - z_rim), r["tilt"],
                "" if lat is None else ", lateral %.1f mm" % (1000.0 * lat)), flush=True)
        if base < float(z_rim) - 0.010:
            print("  descend_to_base: ABORT, base is %.1f mm below the rim"
                  % (1000.0 * (z_rim - base)), flush=True)
            return payload, info
        stalled = (prev is not None and (prev[0] - base) < 0.0006
                   and (prev[1] - pose_of(payload)[2]) > 0.0012
                   and base < float(z_rim) + 0.006)
        if base <= float(z_rim) + float(tol) or stalled:
            ct = contact_test(ctx, z_rim, h_bowl, q_hold, measure, press=press,
                              gripper=gripper, verbose=verbose)
            if ct is not None:
                info.update({"base": ct["base"], "delta": ct["base"] - float(z_rim),
                             "tilt": ct["tilt"], "contact": ct["contact"]})
            else:
                info.update({"base": base, "delta": base - float(z_rim),
                             "tilt": r["tilt"]})
            return payload, info
        target = max(float(z_rim) + 0.0003, base - float(step))
        z_t = eef_for_base(target, rel_z, h_bowl)
        p = pose_of(payload)
        if aim_xy is not None:
            eef_xy = np.asarray(aim_xy) - np.asarray(r["d_xy"])
            if abs(eef_xy[0] - p[0]) > 0.004 or abs(eef_xy[1] - p[1]) > 0.004:
                payload, _ = goto(ctx, [eef_xy[0], eef_xy[1], p[2]], q_hold, gripper,
                                  tol=0.0015, max_rows=10, tag="[aim]")
                p = pose_of(payload)
        payload, _ = descend_rows(ctx, z_t, q_hold, gripper, step=0.0010, max_rows=4,
                                  hold_xy=None if aim_xy is None else
                                  (np.asarray(aim_xy) - np.asarray(r["d_xy"])))
        prev = (base, pose_of(payload)[2])
    print("  descend_to_base: out of steps", flush=True)
    return payload, info


# =============================================================== put an object ON a surface
# Measured on libero_90 task 18 (KITCHEN_SCENE3: the 20 cm frying pan onto the square stove),
# 202 rows, PASS.  SKILL §2 ("a payload much bigger than the jaws hangs at a tilt") and §4c
# ("the opposite case: a genuinely BLACK carried object") are the prose.
def level_quat(n_cur, q_cur):
    """Tool quaternion that takes a carried object's measured plane normal to +z.

    The pan pinch is a CLAMP: the payload follows the wrist, so a payload that settled at 28 deg
    while hanging hangs level after this rotation (task 18: 28.3 -> 3.7 deg in 11 rows).  Build
    it as a WORLD-FRAME left-multiply on the current quat (`qmul(q_c, q_cur)`, not `qmul(q_cur,
    q_c)`), then drive it with `rotate_to` (which gates on rot_err) and pass the result as
    `q_tgt` to EVERY later row - a later `goto` with `Q_DOWN` re-tilts the payload in one move.
    """
    n = np.asarray(n_cur, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    if n[2] < 0:
        n = -n
    axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
    ax = np.linalg.norm(axis)
    ang = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    if ax < 1e-6:
        return np.asarray(q_cur, dtype=np.float64), 0.0
    axis = axis / ax
    s = np.sin(0.5 * ang)
    qc = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * ang)])
    return qmul(qc, np.asarray(q_cur, dtype=np.float64)), float(np.degrees(ang))


def base_low(base, R_base, tilt_deg):
    """The LOWEST point of a tilted carried object: `base - R_base*sin(tilt)`.

    `base` is the object's base plane at its own axis (zc - h).  Aim a descent at THIS, not at
    `base`: a 20 cm pan at 28 deg droops 0.075*sin 28 = 35 mm below its base plane, and the rim
    then lands in the target.  Release at base_low = support + 1 mm: task 18 measured base_low
    0.9801 against a raised burner at 0.9800 and the payload settled level (0.4 deg).
    """
    return float(base) - float(R_base) * np.sin(np.radians(float(tilt_deg)))


def carried_black_ring(scene, jaw_dx=None, jaw_dy=None, centre=None, box=0.20,
                       Rlo=0.085, Rhi=0.115, zlo=-0.085, zhi=0.080, z_abs=None,
                       pad=0.033, tilt_max=45.0, min_pixels=400, quiet=False):
    """Rim circle+plane of a BLACK object - carried, released, or standing on the table.

    Colour DOES separate a black payload (rgb ~ (44,46,43)) - what it cannot separate is the
    payload from its own TABLE SHADOW (at table_z) and from the WRIST's dark blocks (rgb 32-55,
    same class), and the depth contains all three.  One z window fixes it: [eef_z+zlo,
    eef_z+zhi] with a generous box about the eef.  Measured on task 18's first lift, a
    colour-only mask returned an 8263-px blob over z 0.949-1.159 (pan + shadow + hand) and NO
    ring fit survived; with the window the pan fitted R 0.0985 against a true 0.0981,
    res 1.3-1.4 mm, arc 245-321 deg, and repeated to ~1 mm through a lift, a 28 deg levelling
    rotation, a 0.5 m carry and a 16 cm descent.
    """
    jx = JAW_DX if jaw_dx is None else jaw_dx
    jy = JAW_DY if jaw_dy is None else jaw_dy
    eef = scene.eef
    if eef is None:
        return None
    jaw = np.array([eef[0] + jx, eef[1] + jy])
    cx, cy = (eef[0], eef[1]) if centre is None else (float(centre[0]), float(centre[1]))
    mean = scene.mean
    chroma = scene.rgb.max(2) - scene.rgb.min(2)
    m = (mean < 85) & (chroma < 30) & (scene.z < 1.6)      # black, not the far background
    m &= (np.abs(scene.X - cx) < box) & (np.abs(scene.Y - cy) < box)
    if z_abs is None:
        m &= (scene.Z > eef[2] + zlo) & (scene.Z < eef[2] + zhi)
    else:
        m &= (scene.Z > float(z_abs[0])) & (scene.Z < float(z_abs[1]))
    m &= np.hypot(scene.X - jaw[0], scene.Y - jaw[1]) > pad   # the finger pads
    keep, info = scene.blob(m, min_pixels=min_pixels)
    if keep is None:
        if not quiet:
            print("  carried_black_ring: no blob (%d px)" % info.get("n", 0), flush=True)
        return None
    r = robust_ring(scene, keep, Rlo=Rlo, Rhi=Rhi, tilt_max=tilt_max, dz=0.004)
    if r is None:
        if not quiet:
            print("  carried_black_ring: no ring fit in a %d-px blob" % info["n"], flush=True)
        return None
    r["n"] = int(info["n"])
    r["d_xy"] = r["centre"][:2] - eef[:2]
    r["d_z"] = r["zc"] - eef[2]
    r["ok"] = bool(Rlo + 0.003 < r["R"] < Rhi - 0.005 and r["res"] < 0.004
                   and r["arc"] > 90 and r["tilt"] < tilt_max)
    if not quiet:
        print("  carried_black: c=%s R=%.4f zc=%.4f tilt=%.1f arc=%.0f res=%.1fmm n=%d "
              "| centre-eef=%s zc-eef=%+.4f %s" % (
                  np.round(r["centre"], 4), r["R"], r["zc"], r["tilt"], r["arc"],
                  r["res"] * 1000.0, r["n"], np.round(r["d_xy"], 4), r["d_z"],
                  "" if r["ok"] else "<-- SUSPECT"), flush=True)
    return r


# =============================================================== turning a knob (task 21)
# The one task where the thing to move is NOT picked up: a stove's knob is a vertical-axis
# joint with a raised fin on top, and the state to reach is an ANGLE.  Measured on libero_90
# task 21 (KITCHEN_SCENE3, 938 rows, `turnon` PASSED, the pan clause failed): the fin is
# 7.8 x 2.6 cm with a flat top face at z 1.0089 on a hub of r 0.043 whose top is 0.972.
def q_yaw(deg):
    """Quaternion (xyzw) for a world-frame +z rotation of `deg` degrees."""
    a = np.radians(float(deg)) * 0.5
    return np.array([0.0, 0.0, np.sin(a), np.cos(a)])


def yaw_quat(q_cur, deg):
    """`q_cur` turned by `deg` about the WORLD +z axis (left-multiply, like `level_quat`).

    With the tool pointing down this is the wrist's own yaw; it is also the rotation that keeps
    a levelled payload level (its normal is already +z), which is the free parameter to use when
    the arm cannot hold the levelling quaternion at the place pose.
    """
    return qmul(q_yaw(deg), np.asarray(q_cur, dtype=np.float64))


def flat_face_az(scene, centre, z_top, half=0.070, dz=0.004, nz=0.85,
                 black=95, chroma=35, min_pixels=60, quiet=True):
    """Azimuth (deg, mod 180 into (-90, 90]) of a FLAT dark top face about a vertical axis.

    This is the STATE SENSOR for a knob: its fin's azimuth IS the joint angle, and it is
    readable from one depth call even while the fingers straddle the fin (0 rows).  Three
    filters, each paid for on task 21:
    """
    c = np.asarray(centre, dtype=np.float64)
    mx, mn = scene.rgb.max(2), scene.rgb.min(2)
    blk = (scene.mean < black) & ((mx - mn) < chroma)
    r = np.hypot(scene.X - c[0], scene.Y - c[1])
    pts = scene.pts
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(du, dv)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    m = blk & (np.abs(n[:, :, 2]) > nz) & (r < half) \
        & (scene.Z > float(z_top) - dz) & (scene.Z < float(z_top) + 0.25 * dz)
    keep, info = scene.blob(m, min_pixels=min_pixels)
    if keep is None:
        if not quiet:
            print("  flat_face_az: no flat dark face in the band (best %d px)"
                  % info.get("n", 0), flush=True)
        return None
    pb = scene.pts[keep]
    d = pb[:, :2] - pb[:, :2].mean(0)
    w, v = np.linalg.eigh(d.T @ d / len(d))
    u = v[:, int(np.argmax(w))]
    az = float(np.degrees(np.arctan2(u[1], u[0])))
    while az > 90.0:
        az -= 180.0
    while az <= -90.0:
        az += 180.0
    out = {"az": az, "n": int(keep.sum()), "z": float(np.percentile(pb[:, 2], 90)),
           "half_len": float(2.0 * np.sqrt(max(w[1], 1e-9))),
           "half_w": float(2.0 * np.sqrt(max(w[0], 1e-9))),
           "centroid": pb[:, :2].mean(0)}
    if not quiet:
        print("  flat_face: az %+.1f deg  z %.4f  half-len %.3f half-w %.3f  n=%d" % (
            az, out["z"], out["half_len"], out["half_w"], out["n"]), flush=True)
    return out


def az_delta(a1, a0):
    """Signed change of an AXIS azimuth (mod 180), in (-90, 90]: how far a knob turned."""
    return (float(a1) - float(a0) + 90.0) % 180.0 - 90.0


def red_pixels(path, box=None, thr=110, delta=50):
    """Count strongly RED pixels in a saved frame -- an asset's ON-state light, for 0 rows.

    task 21's flat stove lights its burner rings when the knob is turned (0 px -> ~1400 px at
    ~22 deg of knob rotation), which makes `turnon` verifiable before `finish` without knowing
    the predicate.  `box` is (v0, v1, u0, u1) in agentview pixels: task 21's burner sits at
    """
    try:
        rgb = decode_jpeg(path).astype(float)
    except Exception:
        return None
    if box is not None:
        rgb = rgb[box[0]:box[1], box[2]:box[3]]
    m = (rgb[:, :, 0] > thr) & (rgb[:, :, 0] > rgb[:, :, 1] + delta) \
        & (rgb[:, :, 0] > rgb[:, :, 2] + delta)
    return int(m.sum())


def plane_mode(scene, xybox, zrange=(0.960, 0.990), bin_mm=2.0, tol=0.0025, mask=None):
    """The MODE of z over a footprint -- the right way to get a bevelled plate's top plane.

    A median reads 3 mm low on a bevelled plate (libero_90 task 21: median 0.9715 against the
    true 0.9740 while the rim bevel is 6 mm lower), and then every face up to 3 mm above that
    median looks "raised" -- which silently turns the plate into the support surface.  Returns
    {"z", "pts", "x", "y", "n"}: `pts` are the pixels within `tol` of the mode, so the footprint
    (the plate's own extent) comes with it.
    """
    (x0, x1), (y0, y1) = xybox
    m = (scene.Z > zrange[0]) & (scene.Z < zrange[1]) & (scene.X > x0) & (scene.X < x1) \
        & (scene.Y > y0) & (scene.Y < y1)
    if mask is not None:
        m &= mask
    p = scene.pts[m]
    if len(p) < 200:
        return None
    vals = np.round(p[:, 2] / (bin_mm / 1000.0)) * (bin_mm / 1000.0)
    uq, ct = np.unique(vals, return_counts=True)
    z = float(uq[int(np.argmax(ct))])
    top = p[np.abs(p[:, 2] - z) <= tol]
    return {"z": z, "pts": top, "n": int(len(top)),
            "x": (float(np.percentile(top[:, 0], 1)), float(np.percentile(top[:, 0], 99))),
            "y": (float(np.percentile(top[:, 1], 1)), float(np.percentile(top[:, 1], 99)))}


# =============================================================== §7b which way does this
# chest face, and which way do its drawers come out?
#
# EVERYTHING in this file that talks about "the drawer's front plane", "the handle bar in
# front of the panel", "the panel comes home" or "push the drawer shut" assumes the chest of
# KITCHEN_SCENE2, where the fronts are at y ~ -0.23 and the drawers come OUT toward +y.  The
# chest is placed at a different yaw in other scenes and then those helpers do not fail: they
# return *plausible* numbers 0.15-0.30 m out.  Measured on libero_90 task 22
# (KITCHEN_SCENE4): `handle_bars(panel_y=-0.24)` returned a 22 cm-wide, 7 cm-tall "bar" that
# was the drawer BOX behind the panel, and the travel read 0.31 m against a true 0.149 m.
#
# So: measure the axis and the OUT sign first, then use the `_on`/`_u` variants below, which
# take (axis, S) and are sign-agnostic.  One depth call, 0 rows.
def drawer_axis(scene, zrange=None, xbox=None, ybox=None, min_px=300):
    """The drawer bank's axis, its OUT sign and its fronts' plane.  0 rows.

    A cabinet's drawers all slide along ONE horizontal axis, their fronts are one flush plane,
    and they open toward the CAMERA side of that plane (the camera is at `T_RIG`).  Two
    measured draws, and they are mirrors of each other:
    """
    tz = scene.table_z()
    zlo = (float(tz) + 0.020) if (zrange is None and tz is not None) else (
        zrange[0] if zrange else 0.96)
    zhi = 1.32 if zrange is None else zrange[1]
    (x0, x1) = (-0.45, 0.28) if xbox is None else xbox
    (y0, y1) = (T_RIG[1] - 0.45, T_RIG[1] + 0.45) if ybox is None else ybox
    nrm = _normal(scene.pts)
    region = ((scene.Z > zlo) & (scene.Z < zhi) & (scene.X > x0) & (scene.X < x1)
              & (scene.Y > y0) & (scene.Y < y1) & ~scene.arm)
    best = None
    for i in (1, 0):
        m = region & (np.abs(nrm[:, :, i]) > 0.9)
        if int(m.sum()) < min_px:
            continue
        v = scene.pts[:, :, i][m]
        step = 0.005
        bins = np.arange(v.min() - step, v.max() + 2 * step, step)
        h, e = np.histogram(v, bins)
        o = int(np.argmax(h))
        if best is None or int(h[o]) > best["n"]:
            off = float(e[o] + step / 2)
            sel = m & (np.abs(scene.pts[:, :, i] - off) < 0.006)
            p = scene.pts[sel]
            S = +1 if T_RIG[i] > off else -1
            best = {"axis": "xy"[i], "i": int(i), "S": int(S), "front": off,
                    "n": int(h[o]), "anchor": "fronts",
                    "xwin": (float(p[:, 0].min()) - 0.04, float(p[:, 0].max()) + 0.03),
                    "ywin": (float(p[:, 1].min()) - 0.04, float(p[:, 1].max()) + 0.04),
                    "zwin": (float(zlo), float(zhi)),
                    "front_z": (float(p[:, 2].min()), float(p[:, 2].max()))}
            # the search window along the drawer's own axis must reach a PULLED-OUT panel
            # (0.15-0.20 m out), and the closed fronts are 1-2 mm behind the frame's face
            lo_a = min(off, off + S * 0.26) - 0.03
            hi_a = max(off, off + S * 0.26) + 0.06
            best["win"] = {"x": tuple(best["xwin"]), "y": tuple(best["ywin"])}
            best["win"]["xy"[i]] = (float(lo_a), float(hi_a))
            best["win"] = (best["win"]["x"], best["win"]["y"])
    return best


def face_bins(scene, i, band_z, xybox, min_pixels=60, bin_mm=1.0, nmin=0.9):
    """[(offset, count)] of the faces perpendicular to axis `i`, 1 mm bins.  Sign-agnostic.

    `y_faces_mm` with the axis as an argument, and **no** "front-most" assumption: the caller
    picks by `S * offset` (see `front_face`).  Keep the 1 mm bin: 3 mm of remaining drawer
    travel is the whole difference between a passing `Close` and a failing one.
    """
    pts = scene.pts
    n = _normal(pts)
    (x0, x1), (y0, y1) = xybox
    m = ((np.abs(n[:, :, i]) > nmin) & (pts[:, :, 2] > band_z[0]) & (pts[:, :, 2] < band_z[1])
         & ~scene.arm & (pts[:, :, 0] > x0) & (pts[:, :, 0] < x1)
         & (pts[:, :, 1] > y0) & (pts[:, :, 1] < y1))
    if int(m.sum()) < min_pixels:
        return []
    v = pts[:, :, i][m]
    step = 0.001 * float(bin_mm)
    bins = np.arange(v.min() - step, v.max() + 2 * step, step)
    h, e = np.histogram(v, bins)
    return sorted([(round(float(0.5 * (e[k] + e[k + 1])), 4), int(h[k]))
                   for k in range(len(h)) if h[k] >= min_pixels], key=lambda t: -t[1])


def front_face(scene, i, S, band_z, xybox, min_pixels=150):
    """(offset, n) of the OUT-most face along axis i, i.e. the drawer's own panel.

    "out" is `S * offset` maximum - max y on the S=+1 chest, MIN y on the S=-1 one.  Arm
    pixels are excluded, which also removes the handle bar, so this is the panel and not the
    bar standing proud of it.  This is the plane a closing push must move, and the plane whose
    """
    rows = face_bins(scene, i, band_z, xybox, min_pixels=min_pixels)
    if not rows:
        return None
    return max(rows, key=lambda t: S * t[0])


def front_face_on(scene, fr, band_z, xybox=None, min_pixels=150):
    """`front_face` with the (i, S) taken from `drawer_axis`; `xybox` defaults to `fr["win"]`."""
    return front_face(scene, fr["i"], fr["S"], band_z,
                      fr["win"] if xybox is None else xybox, min_pixels=min_pixels)


def handle_bars_on(scene, fr, panel_off, zrange, min_px=100, reach=0.09, drop_arm=True):
    """[one dict per handle bar] standing OUT of the panel at `panel_off`.  Sign-agnostic.

    A bar is the only scene feature ~3 cm proud of the fronts, ~1 cm tall, and achromatic
    (mean ~95, i.e. inside `Scene.arm`) - so this is a colour+geometry test.  `front_u` is the
    bar's front face in the OUT direction (`S * coord`), `x`/`y` its span, `zc` its centre.
    One bar per drawer is the per-drawer identity (SKILL §4); the LOWEST bar is the bottom
    drawer.
    """
    i, S = int(fr["i"]), float(fr["S"])
    co = scene.pts[:, :, i]
    rel = S * (co - float(panel_off))
    mx, mn, mean = scene.rgb.max(2), scene.rgb.min(2), scene.rgb.mean(2)
    m = (((mx - mn) < 25) & (mean > 70) & (mean < 170)
         & (rel > 0.006) & (rel < float(reach))
         & (scene.Z > float(zrange[0])) & (scene.Z < float(zrange[1]))
         & (scene.X > fr["win"][0][0]) & (scene.X < fr["win"][0][1])
         & (scene.Y > fr["win"][1][0]) & (scene.Y < fr["win"][1][1]))
    if drop_arm and scene.eef is not None:
        m &= (np.linalg.norm(scene.pts - scene.eef, axis=2) > 0.12)
    if int(m.sum()) < min_px:
        return []
    edges = np.arange(float(zrange[0]), float(zrange[1]) + 1e-9, 0.004)
    h, e = np.histogram(scene.Z[m], edges)
    bars, k = [], 0
    while k < len(h):
        if h[k] < 8:
            k += 1
            continue
        j = k
        while j + 1 < len(h) and h[j + 1] >= 8:
            j += 1
        sel = m & (scene.Z >= e[k]) & (scene.Z < e[j + 1])
        k = j + 1
        if int(sel.sum()) < min_px:
            continue
        p = scene.pts[sel]
        fu = float(np.percentile(S * p[:, i], 98))
        bars.append({"n": int(sel.sum()), "front_u": fu,
                     "front_off": float(S * fu),
                     "z": (float(p[:, 2].min()), float(p[:, 2].max())),
                     "zc": float(0.5 * (p[:, 2].min() + p[:, 2].max())),
                     # p1/p99, not min/max: on a real draw the raw span picked up 5 stray
                     # pixels and reported a 19.4 cm "bar" (mid 5 cm off the true 8.9 cm one)
                     "x": (float(np.percentile(p[:, 0], 1.0)),
                           float(np.percentile(p[:, 0], 99.0))),
                     "y": (float(np.percentile(p[:, 1], 1.0)),
                           float(np.percentile(p[:, 1], 99.0))),
                     "mean": float(mean[sel].mean())})
    return sorted(bars, key=lambda b: b["zc"])


def panel_home_on(scene, fr, ref_off, band_z, xybox=None, tol=0.0015, past_tol=0.006,
                  min_pixels=300, min_cover=0.5):
    """(offset, n, home) - is the pushed panel at the closed sibling's plane?  Sign-agnostic.

    The bin is chosen by its **z span** (`panel_face`'s rule), not by "front-most": on a real
    close the max-y bin was 124 px of noise while the drawer's own panel was a 1591 px bin
    1 mm deeper.  `home=False` is a veto only when another signal also disagrees - the two
    drawers of one cabinet do not share a home plane (task 5: 1.6 mm apart, task 22: 2.0 mm),
    and on task 22 the pushed panel came home 2 mm PAST the sibling, which is the correct
    state.
    """
    i, S = int(fr["i"]), float(fr["S"])
    xybox = fr["win"] if xybox is None else xybox
    lo, hi = float(band_z[0]), float(band_z[1])
    nrm = _normal(scene.pts)
    (x0, x1), (y0, y1) = xybox
    rows = []
    for off, n in face_bins(scene, i, (lo, hi), xybox, min_pixels=min_pixels, bin_mm=1.0):
        m = ((np.abs(scene.pts[:, :, i] - off) < 0.0015) & (scene.Z > lo) & (scene.Z < hi)
             & ~scene.arm & (scene.X > x0) & (scene.X < x1)
             & (scene.Y > y0) & (scene.Y < y1) & (np.abs(nrm[:, :, i]) > 0.85))
        if int(m.sum()) < min_pixels:
            continue
        zspan = float(scene.Z[m].max() - scene.Z[m].min())
        if zspan >= float(min_cover) * (hi - lo):
            rows.append((off, int(n), zspan))
    if not rows:
        return None, 0, False
    off, n, _z = max(rows, key=lambda t: S * t[0])
    du = float(S) * (off - float(ref_off))
    return off, n, bool(-float(past_tol) <= du <= float(tol))


def hook_quat_toward(direction):
    """`Q_HOOK` yawed about world z until its FINGERS lead along `direction`.  0 rows.

    The hook pitch is measured (finger (0.26, -0.855, -0.455), jaw (0.933, 0.10, 0.34)); its
    fingers lead toward **-y**.  A yaw about z preserves the vertical geometry (so the press /
    tips heights in SKILL §3, §10.1 stay valid) and simply turns the pusher to the scene:
    deg = 0 for -y, 180 for +y, +90 for +x, -90 for -x.  The hand block always ends up behind
    and above the fingertips, which is what makes this the recipe for a LOW bar (the
    hand-face recipe needs eef_z = bar_z0 - 0.094, impossible under the table).
    """
    d = np.asarray(direction, dtype=np.float64)
    d = np.array([d[0], d[1], 0.0])
    if np.linalg.norm(d) < 1e-9:
        return np.asarray(Q_HOOK, dtype=np.float64).copy()
    d = d / np.linalg.norm(d)
    deg = float(np.degrees(np.arctan2(d[1], d[0])))      # -y -> -90, +y -> +90, +x -> 0
    return yaw_quat(Q_HOOK, deg + 90.0)


def colour_class_count(scene, rgb, xybox, zrange, tol=12, mask=None):
    """Pixels in a window that still look like `rgb` - the "is the box interior visible" test.

    While a drawer is out you can see INTO the box; once it is shut the interior is gone.  On
    an asset whose interior is bright this is the single hardest signal in the close verdict
    (task 22: 6169 px out -> 0 px shut) and it survives two things that fool the plane tests:
    """
    (x0, x1), (y0, y1) = xybox
    m = ((scene.X > x0) & (scene.X < x1) & (scene.Y > y0) & (scene.Y < y1)
         & (scene.Z > zrange[0]) & (scene.Z < zrange[1]))
    if mask is not None:
        m &= mask
    if int(m.sum()) < 50:
        return 0
    d = np.abs(scene.rgb - np.asarray(rgb, dtype=np.float64)).mean(2)
    return int((m & (d < float(tol))).sum())


# =============================================================== task 24: what transfers
# libero_90 task 24 (KITCHEN_SCENE4, put the black bowl in the bottom drawer) cost 677 rows, and
# ~200 of them were one conceptual miss: the hand block is a ~0.2 m PLATE centred on eef_pos, so
# a rim grasp needs the same clearance a push does - on ALL FOUR sides - and which rim you can
# pinch is a measurement, not a preference.  Everything below was measured on that episode and
# is re-runnable offline on its snapshots.
Q_DOWN = np.array([0.9995, -0.010, -0.029, 0.002])   # measured wrist-down (SKILL 2)


def hand_slab(scene, half_x=0.06, yreach=0.22, zlo=0.05, dz=0.02, min_pixels=15):
    """The WRIST-DOWN hand as a slab around eef_pos: how far it reaches each way, how low.

    SKILL 3 gives the front face (0.111 m ahead of the eef) and the underside (eef_z+0.082).
    A GRASP needs the other end too, because the block is a plate *centred* on the eef, not a
    finger sticking out forward.  Measured at the stall pose on libero_90 task 24
    (eef_y -0.1124, eef_z 0.9923): the arm's lowest z per 2 cm y-cell was
        y -0.220 -> 1.0962 | -0.200 -> 1.0785 | -0.180 -> 1.0357 (pads)
        y -0.080 -> 1.0349 | -0.060 -> 1.0503 | -0.040 -> 1.0717 | -0.020 -> 1.0717
    i.e. lead 0.111 m at -y, 0.09 m at +y, and the plate's underside at eef_z+0.079 behind the
    pads.  So a wrist-down pinch needs ~11 cm of clear space on the finger side AND ~9 cm on
    """
    if scene.eef is None:
        return None
    eef = scene.eef
    m = (scene.arm & (np.abs(scene.X - eef[0]) < float(half_x))
         & (np.abs(scene.Y - eef[1]) < float(yreach)) & (scene.Z > eef[2] + float(zlo)))
    if int(m.sum()) < 200:
        return None
    cells = []
    for y0 in np.arange(eef[1] - yreach, eef[1] + yreach, dz):
        c = m & (np.abs(scene.Y - y0) < 0.5 * float(dz))
        if int(c.sum()) >= int(min_pixels):
            cells.append((float(y0), float(scene.Z[c].min())))
    if len(cells) < 4:
        return None
    ys = np.array([c[0] for c in cells])
    zs = np.array([c[1] for c in cells])
    front = zs[ys < eef[1] - 0.09]
    back = zs[ys > eef[1] + 0.05]
    # MIN, not mean: the question is what the LOWEST arm pixel does at each end, because that
    # is what a scene face stops (measured on the stall pose: eef+0.086 front, eef+0.058 back,
    # against the home pose's eef+0.082 - it moves with the wrist pose, so read it per pose).
    return {"lead": float(eef[1] - ys.min()), "back": float(ys.max() - eef[1]),
            "z_low": float(zs.min()) - eef[2],
            "z_low_front": (float(front.min()) - eef[2]) if len(front) else None,
            "z_low_back": (float(back.min()) - eef[2]) if len(back) else None,
            "y_cells": [(round(float(a), 3), round(float(b), 4)) for a, b in cells],
            "n": int(m.sum())}


def rim_grasp_axis(rim, rim_dir=(0.0, -1.0, 0.0), hand_out=True, inset=0.004, dz=-0.055,
                   jaw_dx=None, jaw_dy=None, q_down=None):
    """eef + tool QUATERNION that pinches the rim facing `rim_dir` (+-x or +-y).

    `rim_grasp_target` only does the +-y rims, and its `side` cannot express the +-x ones
    because those need the wrist YAWED 90 deg and the jaw offset rotated with it.  Two facts
    make the general form short:
    """
    jx = JAW_DX if jaw_dx is None else jaw_dx
    jy = JAW_DY if jaw_dy is None else jaw_dy
    d = np.asarray(rim_dir, dtype=np.float64)[:2]
    d = d / (np.linalg.norm(d) + 1e-12)
    ang = float(np.degrees(np.arctan2(d[1], d[0])))
    yaw = ang + 90.0 if hand_out else ang - 90.0
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    jd = np.array([[c, -s], [s, c]]) @ np.array([jx, jy], dtype=np.float64)
    jaw = np.asarray(rim["centre"], dtype=np.float64)[:2] + (float(rim["R"]) - float(inset)) * d
    eef = np.array([jaw[0] - jd[0], jaw[1] - jd[1], float(rim["rim_z"]) + float(dz)])
    q0 = Q_DOWN if q_down is None else np.asarray(q_down, dtype=np.float64)
    return {"eef": eef, "jaw": jaw, "quat": yaw_quat(q0, yaw), "yaw_deg": yaw,
            "jd_world": jd, "hint": np.asarray(rim["centre"], dtype=np.float64)[:2] - eef[:2],
            "pad_bottom": float(rim["rim_z"]) + float(dz) + PAD_DZ}


def measure_held(scene, hint, jd_world, h, R_ref=0.055, pad_r=0.030, half=0.10,
                 Rlo=0.042, Rhi=0.068, quiet=False):
    """The object hanging in the jaws, fitted so a RIM pinch is not thrown away.

    `measure_carried` cuts a 36 mm cylinder about the jaw centre and gates |centre - hint| <
    45 mm.  With a rim pinch the object's centre is one RADIUS from the jaw centre, so that
    cut removes the near arc, and a partial-arc fit then wanders: on task 24 a good pinch
    (gq 0.0040, the bowl off the table, `d_xy` repeating to 0.2 mm across a 20 cm lift) came
    back "no usable ring fit" (candidates R 0.0534, zc 1.052, tilt 5.3, arc 114, res 0.8 mm)
    and the caller's retry DROPPED the bowl.  So: pad cut 30 mm, no centre gate, and gates on
    R / res / arc only; the caller must then cross-check `d_xy` against the rigid model
    (`|centre - jaw| ~ R`) and use the MEASURED offset, not the model.
    """
    if scene.eef is None:
        return None
    eef = scene.eef
    jd = np.asarray(jd_world, dtype=np.float64)
    jaw = eef[:2] + jd
    hxy = eef[:2] + np.asarray(hint, dtype=np.float64)
    best = None
    for dn, up in ((-0.02, 0.070), (-0.02, 0.090), (-0.02, 0.110), (0.01, 0.080)):
        m = ((np.abs(scene.X - hxy[0]) < float(half)) & (np.abs(scene.Y - hxy[1]) < float(half))
             & (scene.Z > eef[2] + dn) & (scene.Z < eef[2] + up)
             & (np.hypot(scene.X - jaw[0], scene.Y - jaw[1]) > float(pad_r)))
        keep, info = scene.blob(m, min_pixels=150)
        if keep is None:
            continue
        r = robust_ring(scene, keep, Rlo=Rlo, Rhi=Rhi)
        if r is None:
            if not quiet:
                print("  held win(%+.2f,%+.2f) n=%d -> no ring" % (dn, up, info["n"]),
                      flush=True)
            continue
        if not (Rlo <= r["R"] <= Rhi and r["res"] < 0.0020 and r["arc"] > 80):
            if not quiet:
                print("  held win(%+.2f,%+.2f) -> R %.4f res %.1fmm arc %.0f (rejected)"
                      % (dn, up, r["R"], 1000 * r["res"], r["arc"]), flush=True)
            continue
        if best is None or r["arc"] > best["arc"]:
            best = r
    if best is None:
        return None
    r = best
    r["base"] = r["zc"] - float(h)
    r["d_xy"] = r["centre"][:2] - eef[:2]
    r["d_z"] = r["zc"] - eef[2]
    if not quiet:
        print("  HELD: centre %s R %.4f zc %.4f base %.4f tilt %.1f arc %.0f | "
              "centre-eef %s zc-eef %+.4f | |centre-jaw| %.4f (R %.4f)"
              % (np.round(r["centre"], 4), r["R"], r["zc"], r["base"], r["tilt"],
                 r["arc"], np.round(r["d_xy"], 4), r["d_z"],
                 float(np.linalg.norm(r["centre"][:2] - jaw)), r["R"]), flush=True)
    return r


def container_top_scan(scene, xybox, floor_z, h_obj, step=0.015, inset=0.015, tol=0.010,
                       robot_r=0.12, zmax_extra=0.090):
    """Is an object standing on a container's FLOOR?  Column grid, not a ring fit.  0 rows.

    The verdict for "put X in the drawer/bowl/box".  A rim-ring fit inside a bright wooden
    drawer merges with the interior into one 4 000-pixel blob (task 24: fitted R 0.075,
    res 10 mm - junk); a column scan cannot: a cell whose max z is `floor + h_obj` (+/-`tol`)
    is the object's top, a cell at the floor is the floor, and a cell above means a wall or a
    neighbouring object.  Same scan as `box_free_columns`, run AFTER the drop with the gripper
    OUT of the columns (`robot_r`).
    """
    (x0, x1), (y0, y1) = xybox
    want = float(floor_z) + float(h_obj)
    near = (np.linalg.norm(scene.pts - scene.eef, axis=2) > float(robot_r)
            if scene.eef is not None else True)
    cells, above, n = [], [], 0
    xs = np.arange(x0 + inset, x1 - inset + 1e-9, step)
    ys = np.arange(y0 + inset, y1 - inset + 1e-9, step)
    for x in xs:
        for y in ys:
            c = ((np.abs(scene.X - x) < 0.5 * step) & (np.abs(scene.Y - y) < 0.5 * step)
                 & (scene.Z > floor_z + 0.004) & (scene.Z < floor_z + float(zmax_extra)) & near)
            if int(c.sum()) < 8:
                continue
            n += 1
            zmax = float(scene.Z[c].max())
            if abs(zmax - want) < float(tol):
                cells.append((float(x), float(y), zmax))
            elif zmax > want + float(tol):
                above.append((round(float(x), 3), round(float(y), 3), round(zmax, 4)))
    out = {"cells": cells, "want": want, "above": above, "n_cells": n, "centre": None,
           "ok": False}
    if len(cells) >= 4:
        cen = np.array([[c[0], c[1]] for c in cells]).mean(0)
        out["centre"] = cen
        out["ok"] = bool(x0 < cen[0] < x1 and y0 < cen[1] < y1)
    return out


def mouth_walls(scene, i, zband, xybox, min_pixels=60, bin_mm=2.0, nmin=0.85,
                exclude_arm=False, top=6):
    """The INNER walls of an open box: vertical faces perpendicular to axis `i`, 1-2 mm bins.

    `face_bins` always excludes `Scene.arm`, and a drawer's interior is *bright* on
    KITCHEN_SCENE4, i.e. inside that class - so on this asset the mouth's own walls are
    invisible to it.  Measured on task 24 (z 0.980...1.030, no arm exclusion):
        x = +0.0805 n=1935 (the NEAR wall, fully visible, y 0.070...0.222)
        x = -0.1315 n= 256 (the far wall, mostly occluded)
    while the visible FLOOR only spanned x -0.130...+0.022.  The mouth is therefore 21 cm
    wide, not 15: the near wall hides 5.8 cm of floor from a 38 deg camera, so a mouth taken
    from the floor plane's extent is a CONSERVATIVE (inner) bound - fine for aiming at the
    """
    (x0, x1), (y0, y1) = xybox
    nrm = _normal(scene.pts)
    m = ((np.abs(nrm[:, :, i]) > nmin) & (scene.Z > zband[0]) & (scene.Z < zband[1])
         & (scene.X > x0) & (scene.X < x1) & (scene.Y > y0) & (scene.Y < y1))
    if exclude_arm:
        m &= ~scene.arm
    if int(m.sum()) < min_pixels:
        return []
    v = scene.pts[:, :, i][m]
    step = 0.001 * float(bin_mm)
    h, e = np.histogram(v, np.arange(v.min() - step, v.max() + 2 * step, step))
    rows = []
    for k in range(len(h)):
        if h[k] < min_pixels:
            continue
        off = float(0.5 * (e[k] + e[k + 1]))
        sel = m & (np.abs(scene.pts[:, :, i] - off) < 0.75 * step)
        p = scene.pts[sel]
        other = 1 - int(i)
        rows.append({"off": off, "n": int(h[k]),
                     "z": (float(p[:, 2].min()), float(p[:, 2].max())),
                     "span": (float(p[:, other].min()), float(p[:, other].max()))})
    return sorted(rows, key=lambda r: -r["n"])[:int(top)]


# =============================================================== task 26: a lying cylinder
# libero_90 task 26 (a wine bottle lying on its side) cost a whole episode to two geometric facts
# that are the same on any scene: a cylinder resting on a table has its AXIS over its top ridge
# (not over its blob centroid), and a parallel-jaw squeeze whose band is centred BELOW that axis
# pushes the cylinder UP out of the pinch.  Both are measurable from one depth call.
def axis_from_ridge(scene, mask=None, min_pixels=60, ridge_dz=0.002, cell=0.01, xy=None,
                    half=0.24, zhi=1.40):
    """(point, dir, t, ridge_z) of a lying cylinder's AXIS, from its top ridge.  0 rows.

    A resting cylinder's highest points lie directly over its axis, so the ridge's projection IS
    the axis.  The blob's centroid is not: only the upper band is visible and a colour mask clips
    one side, which measured 1 cm of error on a 4.5 cm bottle - most of a jaw's aim.  `mask` is the
    object's own class (a chroma/brightness test); without it the scene's coloured pixels are used,
    which is only sensible if the cylinder is the only coloured thing in `xy`/`half`.
    """
    if mask is None:
        mx, mn = scene.rgb.max(2), scene.rgb.min(2)
        mask = (mx - mn) > 30
        if xy is None:
            mask &= (scene.X > -0.70) & (scene.X < 0.20) & (scene.Y > -0.30) & (scene.Y < 0.30)
        else:
            mask &= (np.abs(scene.X - xy[0]) < half) & (np.abs(scene.Y - xy[1]) < half)
        mask &= (scene.Z > 0.85) & (scene.Z < zhi)
    keep, info = scene.blob(mask, min_pixels=min_pixels)
    if keep is None or info["n"] < min_pixels:
        return None
    p = scene.pts[keep]
    c = p[:, :2].mean(0)
    u, sv, vt = np.linalg.svd(p[:, :2] - c, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d[0] < 0:
        d = -d
    t = (p[:, :2] - c) @ d
    ridge = np.zeros(len(p), bool)
    for a in np.arange(np.floor(t.min() / cell) * cell, t.max(), cell):
        q = (t >= a) & (t < a + cell)
        if q.sum() < 4:
            continue
        ridge |= q & (p[:, 2] > p[q, 2].max() - ridge_dz)
    if ridge.sum() >= 12:
        A = np.c_[p[ridge, 0], np.ones(int(ridge.sum()))]
        sol, *_ = np.linalg.lstsq(A, p[ridge, 1], rcond=None)
        ax = np.array([1.0, sol[0]])
        ax /= np.linalg.norm(ax)
        if ax[0] < 0:
            ax = -ax
        perp = np.array([-ax[1], ax[0]])
        point = c + float(np.mean((p[ridge, :2] - c) @ perp)) * perp
        d = ax
        ridge_z = float(np.median(p[ridge, 2]))
    else:                        # a short/rounded object: fall back to the blob axis
        point = c
        ridge_z = float(np.percentile(p[:, 2], 95))
    tt = (p[:, :2] - point) @ d
    top = float(np.percentile(p[:, 2], 95))
    return {"point": point, "dir": d, "t": (float(tt.min()), float(tt.max())),
            "len": float(tt.max() - tt.min()), "ridge_z": ridge_z,
            "angle": float(np.degrees(np.arctan2(d[1], d[0]))),
            "centre": point + 0.5 * (tt.min() + tt.max()) * d,
            "n": int(keep.sum()), "top": top,
            "axis_z": top - 0.021,   # top of the visible surface minus the local half-width
            "mask": keep}


def band_for_axis(axis_z, dz_from_axis=0.006, pad_dz=None, pad_tall=None):
    """The eef_z whose pad BAND is centred `dz_from_axis` above a cylinder's axis.  0 rows.

    `eef_z = axis_z + dz_from_axis - (PAD_DZ + PAD_TALL/2)`.  Positive `dz_from_axis` is what
    makes a free cylinder stay in the pinch: below the axis the contact normals point up and the
    object is squeezed out (measured: `gq` closed to 0.0028 on nothing and the bottle popped into
    the fingers' hooks), above it they press it into the table.
    """
    pd = PAD_DZ if pad_dz is None else float(pad_dz)
    pt = PAD_TALL if pad_tall is None else float(pad_tall)
    return float(axis_z) + float(dz_from_axis) - (pd + 0.5 * pt)


def hand_stack(scene, bands=((0.040, 0.070), (0.070, 0.100), (0.100, 0.140),
                             (0.135, 0.200)), half_x=0.06, reach=0.16):
    """The wrist-down hand's own footprint per z band, relative to `eef_pos`.  0 rows.

    This is the shape a container, an overhang or a table object has to accept.  Measured on
    libero_90 task 26 (probe0, home pose): pads +0.043..0.065, a 1 cm-wide bar +0.075..0.095, a
    3 cm block +0.10..0.14, a 7.6 x 20 cm box +0.135..0.20 - i.e. the hand is **15.7 cm tall** and
    **20 cm long along the jaw axis** at every level above the pads, which is why a drawer whose
    free air above the mouth is only 11.5 cm tall cannot be entered deeper than the window allows.
    Returns [(dz_lo, dz_hi, n, x_extent, y_extent)].
    """
    if scene.eef is None:
        return None
    e = scene.eef
    out = []
    for lo, hi in bands:
        m = (scene.arm & (np.abs(scene.X - e[0]) < float(half_x))
             & (np.abs(scene.Y - e[1]) < float(reach))
             & (scene.Z >= e[2] + float(lo)) & (scene.Z < e[2] + float(hi)))
        if int(m.sum()) < 8:
            out.append((float(lo), float(hi), int(m.sum()), 0.0, 0.0))
            continue
        out.append((float(lo), float(hi), int(m.sum()),
                    float(np.ptp(scene.X[m])), float(np.ptp(scene.Y[m]))))
    return out
