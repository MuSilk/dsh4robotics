"""LEGACY - superseded by `code/open_drawer2/controller.py` (632 rows, passed, NO demo file).

Kept as the worked close-and-reopen: its `replay` stage needs `tasks/<task>_demo.hdf5`, which is
being removed, so on a future episode only its own-code stages are runnable.  Its `probe_pull`
now delegates to `rig.probe_pull_biased`: the z-held probe it used to run is a false negative on
a handle bar (SKILL §2).

Controller for LIBERO-90 task 6: OPEN THE BOTTOM DRAWER of the kitchen chest.

The drawer starts CLOSED. This controller opens it with its own measurements and its own
motion - no demo trace is replayed. Stages, dispatched by ENTRY (experiments/run_controller.py):

    ENTRY=probe   0 rows: snapshot + chest_window / front_face_y / handle_bars / pad+hand rig
    ENTRY=hook    pre-hook wrist-down, rotate the wrist in place to the hook pose, measure the
                  finger offset AT THAT PITCH, insert into the handle recess
    ENTRY=probe_pull  4 saturated +y rows + 2 depth calls: did THIS drawer's front plane move?
    ENTRY=pull    bursts of saturated +y rows, comparing the drawer's own panel plane
    ENTRY=verify  retreat clear, then the three plane signals
    ENTRY=run     probe -> hook -> probe_pull -> pull -> verify (height sweep on a failed probe)

Numbers marked (rig) are physical constants from code/lib/rig.py (pad/hand geometry, motion
response).  Every placement number is read off the current draw in `probe` and cached in
experiments/data/meas.json; the drawer's fronts/bars move 1-30 mm between draws.

The reusable parts of this file now live in the library, so a new task does not have to copy
them: `rig.goto` (residual + stall reporting), `rig.pitch_offset` (the pitched finger
offset, table- and palm-guarded), `rig.hook_targets` (pre-hook + the diagonal press),
`rig.pull_burst` (dy +.85 / dz -.35), `rig.push_tips_to_stop` (shut a low drawer with the finger
tips, verdict on the panel plane) and `rig.bar_front_from_panel`.  This file keeps the
episode-specific sequence: probe -> replay (learning) -> close -> reopen -> pull to the stop,
which is the cycle tasks/open-drawer.md §7 requires before `finish`.
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

# wrist-down quaternion (rig): w near 0 with a vector near 1 is this pose
QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])

# finger axis of the hook: forward (-y), down (-z), a little +x -> ~32 deg below horizontal;
# the jaw axis is the bar's long axis (world x).  Derived, and checked against the measured
# wrist-down quaternion in `probe` (SKILL tasks/open-drawer.md §2).
FINGER_AXIS = (1.0, -3.0, -2.0)
JAW_AXIS = (1.0, 0.0, 0.0)
Q_HOOK = rig.quat_from_fingers(FINGER_AXIS, JAW_AXIS)

DZ_STEP = 0.010          # the probe-pull height sweep step


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


# ------------------------------------------------------------------ motion
def goto(ctx, target, q_tgt, gripper, tol=0.006, max_rows=34, rate=0.011, kick=0.0,
         stall_rows=6, verbose=True):
    """rig.servo plus optional row-alternating roll kick, with a stall report.

    The kick is added ON TOP of the quaternion servo (which pulls the orientation back every
    row) and alternates sign, so it cannot wind the wrist.  A stall is reported rather than
    ending silently short.
    """
    target = np.asarray(target, dtype=np.float64)
    payload = rig.observe(ctx)
    dead = 0
    for i in range(int(max_rows)):
        p = rig.pose_of(payload)
        err = target - p
        if np.max(np.abs(err)) < tol:
            if verbose:
                _log("  goto converged err=%s" % np.round(err, 4))
            return payload, True
        a = np.zeros(7)
        a[:3] = np.clip(err / rate, -1, 1)
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), q_tgt) / 0.5, -1, 1)
        if kick:
            a[3] = float(np.clip(a[3] + (kick if i % 2 == 0 else -kick), -1, 1))
        a[6] = gripper
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if verbose and (i % 4 == 0 or dead >= stall_rows - 1):
            _log("  goto %2d eef=%s err=%s moved=%.4f dead=%d" % (
                i, np.round(q, 4), np.round(target - q, 4), moved, dead))
        if dead >= stall_rows:
            _log("  goto STALLED at eef=%s err=%s (rot_err=%.3f)" % (
                np.round(q, 4), np.round(target - q, 4),
                float(np.linalg.norm(rig.rotvec_err(rig.quat_of(payload), q_tgt)))))
            return payload, False
    _log("  goto hit max_rows residual=%s" % np.round(target - rig.pose_of(payload), 4))
    return payload, False


def rotate_in_place(ctx, q_tgt, gripper=-1.0, max_rows=30, tol=0.06, verbose=True):
    """Rotation-only rows (SKILL §2): a big pitch commanded while translating stalls, and the
    orientation error then GROWS.  Rotate first, translate after."""
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
    _log("episode eef %s quat %s gq %s" % (np.round(s.eef, 4), np.round(s.quat, 4),
                                           np.round(s.gq, 4)))
    _log("table_z %s" % (s.table_z(),))
    _log("quat check: quat_from_fingers(finger=(0,0,-1), jaw=(0,-1,0)) vs measured wrist-down"
         " = %.1f deg" % _quat_deg(rig.quat_from_fingers((0, 0, -1), (0, -1, 0)), s.quat))

    xybox, band_z, counter_z, counter_y = rig.chest_window(s)
    _log("chest_window xybox=%s band_z=%s counter_z=%.4f counter_y=%.4f" % (
        np.round(xybox, 3), np.round(band_z, 4), counter_z, counter_y))
    tf = s.top_face_front_y()
    _log("top_face_front_y %s" % (None if tf is None else tuple(np.round(tf, 4)),))
    closed_front = rig.front_face_y(s, band_z=band_z, xybox=xybox, min_pixels=150)
    _log("closed_front (fronts' flush plane) %s" % (None if closed_front is None
                                                    else tuple(np.round(closed_front, 4)),))
    _log("--- 1 mm y-face histogram in the front band")
    for y, n in rig.y_faces_mm(s, band_z=band_z, xybox=xybox, min_pixels=40)[:8]:
        _log("   y=%+.4f n=%d" % (y, n))

    bars = s.handle_bars(panel_y=closed_front[0], zrange=(s.table_z() + 0.005, counter_z - 0.01),
                         xbox=xybox[0])
    _log("--- handle bars (one per drawer, lowest = BOTTOM drawer): %d" % len(bars))
    for i, b in enumerate(bars):
        _log("  [%d] n=%4d front_y=%.4f zc=%.4f z %.4f..%.4f  x %.4f..%.4f (mid %.4f)" % (
            i, b["n"], b["front_y"], b["zc"], b["z0"], b["z1"], b["x0"], b["x1"],
            0.5 * (b["x0"] + b["x1"])))
    if not bars:
        _log("NO BARS FOUND - abort probe")
        return False
    bottom = min(bars, key=lambda b: b["zc"])
    drawer_band = rig.drawer_band_for_bar(bottom)
    own = rig.front_face_y(s, band_z=drawer_band, xybox=xybox, min_pixels=150)
    _log("bottom bar zc=%.4f -> drawer_band=%s; own panel front y=%s" % (
        bottom["zc"], np.round(drawer_band, 4),
        None if own is None else round(own[0], 4)))
    _log("--- gripper rig (measured this draw, at the start pose)")
    for half in (0.03, 0.045, 0.06):
        _log("   pads(half=%.3f) %s" % (half, s.pads(half=half)))
    _log("   jaw %s" % (s.jaw(),))
    _log("   hand_lead %s" % (s.hand_lead(),))

    m = dict(
        xybox=[list(xybox[0]), list(xybox[1])], band_z=list(band_z),
        counter_z=float(counter_z), counter_y=float(counter_y),
        closed_front=float(closed_front[0]),
        own_ref=float(own[0]) if own else float(closed_front[0]),
        bars=[{k: float(v) for k, v in b.items() if k != "n"} for b in bars],
        bar_n=[int(b["n"]) for b in bars],
        bottom=[float(bottom[k]) for k in ("z0", "z1", "zc", "front_y", "x0", "x1")],
        drawer_band=[float(drawer_band[0]), float(drawer_band[1])],
    )
    m["bar_x_mid"] = 0.5 * (m["bottom"][4] + m["bottom"][5])
    _save(m)
    _log("saved %s" % MEAS)
    return True


# ------------------------------------------------------------------ hook
def hook(ctx):
    """Pre-hook wrist-down in free space, rotate in place, measure the finger offset AT THIS
    PITCH, insert into the recess.  Returns the insert pose (x, z) as [x, y, z]."""
    m = _load()
    z0, z1, zc, front_y, x0, x1 = m["bottom"]
    x_mid = m["bar_x_mid"]
    _log("bar: front y=%.4f z %.4f..%.4f (zc %.4f) x mid %.4f" % (front_y, z0, z1, zc, x_mid))
    # 1. pre-hook: 16 cm in front of and 10 cm above the bar, wrist down, in free space
    pre = np.array([x_mid, front_y + 0.16, z1 + 0.10])
    _log("pre-hook %s" % np.round(pre, 4))
    goto(ctx, pre, QUAT_DOWN, -1.0, tol=0.008, max_rows=34, kick=0.0)
    s0 = rig.snapshot(ctx, tag="prehook", outdir=DATA)
    _log("pre-hook reached eef=%s quat=%s" % (np.round(s0.eef, 4), np.round(s0.quat, 4)))
    _log("  pads at pre-hook (wrist down) %s" % (s0.pads(half=0.045),))

    # 2. rotate IN PLACE to the hook orientation
    _log("rotate in place -> q_hook %s (%.0f deg from wrist-down)" % (
        np.round(Q_HOOK, 4), _quat_deg(Q_HOOK, s0.quat)))
    payload, ok = rotate_in_place(ctx, Q_HOOK, -1.0, max_rows=32)
    if not ok:
        _log("rotation did not converge; continuing with what we have")

    # 3. measure the finger-surface offset AT THIS PITCH, in free space (0 rows), then derive
    #    the insert height: the fingers' lowest surface must sit just under the bar's bottom,
    #    because a hook catches the bar from BELOW (the demos of this task all hook at
    #    eef_z 0.95-0.97 against a bar centre of 1.000).
    off = pitch_offset(ctx)
    if off is None:
        _log("  no pad reading at this pitch - abort")
        return None
    eef_z = float(z0 - off - 0.002)
    _log("insert height: eef_z = bar_z0 %.4f - finger offset %.4f - 0.002 = %.4f"
         % (z0, off, eef_z))

    # 4. insert
    p, ok = insert(ctx, eef_z)
    m2 = _load()
    m2["eef_z_nominal"] = eef_z
    _save(m2)
    return list(p)


def sweep(ctx, zs=None):
    """Insert at a height, run the 4-row probe pull, move on if the drawer's panel is still."""
    m = _load()
    if zs is None:
        nom = m.get("eef_z_nominal")
        if nom is None:
            off = m.get("pitch_offset")
            nom = float(m["bottom"][0] - off - 0.002)
        zs = (nom, nom - DZ_STEP, nom + DZ_STEP)
    _log("height sweep: %s" % [round(z, 4) for z in zs])
    for z in zs:
        _log("--- insert at eef_z %.4f" % z)
        insert(ctx, z)
        res = probe_pull(ctx)
        if res["engaged"]:
            _log("ENGAGED at eef_z %.4f (panel moved %.4f m)" % (z, res["move"]))
            return True
        _log("not engaged at eef_z %.4f (panel move %s)" % (z, res["move"]))
    return False


def hook_sweep(ctx):
    """For a run that is ALREADY pitched at Q_HOOK: re-measure the pitch offset and sweep.

    `hook` itself is for a fresh arm (pre-hook, rotate); this stage exists so a failed probe
    pull costs a re-run, not an episode (SKILL §0.3).
    """
    q = rig.quat_of(rig.observe(ctx))
    _log("current wrist is %.1f deg from q_hook" % _quat_deg(q, Q_HOOK))
    off = pitch_offset(ctx)
    if off is None:
        _log("no finger offset at this pitch")
        return False
    m = _load()
    nom = float(m["bottom"][0] - off - 0.002)
    m["eef_z_nominal"] = nom
    _save(m)
    _log("nominal insert eef_z = bar_z0 %.4f - offset %.4f - 0.002 = %.4f" % (
        m["bottom"][0], off, nom))
    return sweep(ctx)


# ------------------------------------------------------------------ ram + pull (demo recipe)
def ram(ctx, z=0.955, rows=18, dead_stop=3):
    """Descend in front of the drawer, then push -y with saturated rows until the arm STALLS.

    The 50 demo traces of this task all do this and none of them stops at a pre-computed y:
    they drive -y (and -z) until the arm stops moving, i.e. the fingers meet the drawer front,
    and only then pull +y.  Measured on demo_43: the eef stalls at y -0.2149 / z 0.9505 while
    -y is still commanded, then the +y pull opens the drawer.  A `goto` to a y 1 cm short of
    the front face (my first two attempts: -0.2004 / -0.205 with a 0.5 mm residual) never
    touches anything, so the pull runs its 4 rows against air.
    """
    m = _load()
    x, front = m["bar_x_mid"], m["bottom"][3]
    goto(ctx, [x, front + 0.16, 1.10], Q_HOOK, -1.0, tol=0.008, max_rows=20)
    goto(ctx, [x, front + 0.16, z], Q_HOOK, -1.0, tol=0.006, max_rows=16)
    payload = rig.observe(ctx)
    dead = 0
    for i in range(int(rows)):
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
        if i % 3 == 0 or dead:
            _log("  ram %2d eef=%s moved=%.4f dead=%d" % (i, np.round(q, 4), moved, dead))
        if dead >= int(dead_stop):
            _log("  ram STALLED at eef=%s (contact: %d dead rows)" % (np.round(q, 4), dead))
            break
    p = rig.pose_of(payload)
    m["insert"] = [float(v) for v in p]
    m["eef_z_insert"] = float(p[2])
    _save(m)
    _log("ram end eef=%s (bar front %.4f, bar z %.4f..%.4f zc %.4f)" % (
        np.round(p, 4), front, m["bottom"][0], m["bottom"][1], m["bottom"][2]))
    return p


def ram_sweep(ctx, zs=(0.955, 0.945, 0.965, 0.935)):
    """ram + 4-row probe pull at each height until the drawer's own panel moves."""
    for z in zs:
        _log("--- ram at eef_z %.4f" % z)
        ram(ctx, z=z)
        res = probe_pull(ctx)
        if res["engaged"]:
            _log("ENGAGED at eef_z %.4f (panel moved %.4f m)" % (z, res["move"]))
            return True
    return False


# ------------------------------------------------------------------ insert at a height
def pitch_offset(ctx):
    """The finger-surface offset at THIS pitch, measured in FREE SPACE (0 rows).

    SKILL §3's caveat is the whole reason this is a separate call: the offset does not rotate
    with the tool, and `Scene.pads()`' box is fixed in x/y around the eef, so at a pitched pose
    it reads whatever arm surface is lowest inside that box.  Measured at `pitch0` on this
    episode (eef_z 0.9877, 4 cm above the table) it returned the BRIGHT TABLE at 0.9678 --
    offset -0.0199 -- and an insert derived from it sat the fingers 5.5 cm above the handle
    bar and pulled nothing (probe_pull move = 0.000).  So measure it high enough that the
    table cannot enter the box, and cross-check it against the demo traces of this task (all
    50 demos hook at eef_z 0.95-0.97 with a bar centre of 1.000, i.e. an offset of +0.03..+0.05).
    """
    m = _load()
    x, front = m["bar_x_mid"], m["bottom"][3]
    payload, ok = goto(ctx, [x, front + 0.16, 1.10], Q_HOOK, -1.0, tol=0.008, max_rows=20)
    s = rig.snapshot(ctx, tag="pitch_clear", outdir=DATA)
    p = rig.pose_of(payload)
    out = {}
    for half in (0.030, 0.045, 0.060):
        pd = s.pads(half=half)
        out[half] = None if pd is None else pd["offset"]
    _log("pitch offset at eef=%s : %s (box half -> offset)" % (
        np.round(s.eef, 4), {k: None if v is None else round(v, 4) for k, v in out.items()}))
    _log("jaw at clear pitch %s" % (s.jaw(),))
    off = out.get(0.045)
    if off is None:
        return None
    m["pitch_offset"] = float(off)
    m["pitch_offset_box"] = {str(k): (None if v is None else float(v)) for k, v in out.items()}
    _save(m)
    return float(off)


def insert(ctx, eef_z):
    """Clear out in +y, set the height, then push -y into the handle recess."""
    m = _load()
    x, front = m["bar_x_mid"], m["bottom"][3]
    goto(ctx, [x, front + 0.16, max(1.10, eef_z + 0.02)], Q_HOOK, -1.0, tol=0.008,
         max_rows=20)
    goto(ctx, [x, front + 0.16, eef_z], Q_HOOK, -1.0, tol=0.006, max_rows=14)
    payload, ok = goto(ctx, [x, front + 0.010, eef_z], Q_HOOK, -1.0, tol=0.005,
                       max_rows=16)
    p = rig.pose_of(payload)
    _log("insert at eef_z %.4f -> eef=%s residual=%s (bar front %.4f, bar z %.4f..%.4f)" % (
        eef_z, np.round(p, 4), np.round(np.array([x, front + 0.010, eef_z]) - p, 4),
        front, m["bottom"][0], m["bottom"][1]))
    m["insert"] = [float(v) for v in p]
    m["eef_z_insert"] = float(p[2])
    _save(m)
    return p, ok


# ------------------------------------------------------------------ probe pull
def probe_pull(ctx, dz=0.0):
    """4 saturated +y rows; did the drawer's OWN front plane move? (SKILL §2: a pull is
    silent - the eef's motion cannot tell a hook from a miss.)"""
    m = _load()
    band = m["drawer_band"]
    xybox = (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    p = np.array(m["insert"], dtype=np.float64)
    # make sure we are at the measured insert pose (re-entrant stage)
    cur = rig.pose_of(rig.observe(ctx))
    if np.max(np.abs(cur - p)) > 0.010:
        _log("re-insert: eef=%s -> %s" % (np.round(cur, 4), np.round(p, 4)))
        goto(ctx, p, Q_HOOK, -1.0, tol=0.006, max_rows=20)
        p = rig.pose_of(rig.observe(ctx))
    # biased profile (dy +0.85 / dz -0.35): a z-held +y probe is a false negative on a bar
    res = rig.probe_pull_biased(ctx, Q_HOOK, float(p[0]), float(p[2]),
                                band_z=band, xybox=xybox, rows=6, tag="probe_pull")
    _log("probe_pull: y_before=%s y_after=%s move=%s dz_actual=%+.4f engaged=%s" % (
        None if res["y_before"] is None else round(res["y_before"], 4),
        None if res["y_after"] is None else round(res["y_after"], 4),
        None if res["move"] is None else round(res["move"], 4), res["dz_actual"],
        res["engaged"]))
    m["probe_pull"] = {k: (None if v is None else float(v)) for k, v in res.items()}
    m["panel_open_y"] = res["y_after"]
    _save(m)
    return res


# ------------------------------------------------------------------ pull
def pull(ctx, z=None, batches=4, rows=8):
    """Pull +y in bursts, holding x/z; stop when the drawer's own panel stops advancing."""
    m = _load()
    band = m["drawer_band"]
    xybox = (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    if z is None:
        z = m.get("eef_z_insert", m["insert"][2])
    cur = rig.pose_of(rig.observe(ctx))
    x = float(cur[0]) if abs(cur[0] - m["insert"][0]) > 0.02 else float(m["insert"][0])
    prev = rig.front_face_y(rig.snapshot(ctx, tag="pull_a", outdir=DATA), band_z=band,
                            xybox=xybox, min_pixels=150)
    prev_y = None if prev is None else float(prev[0])
    _log("pull start: eef=%s panel front y=%s" % (
        np.round(cur, 4), None if prev_y is None else round(prev_y, 4)))
    total = 0
    for i in range(int(batches)):
        rig.push(ctx, 1, +1, Q_HOOK, -1.0, rows=int(rows), hold_xyz={0: x, 2: float(z)},
                 verbose=False)
        s = rig.snapshot(ctx, tag="pull_%d" % i, outdir=DATA)
        f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
        y = None if f is None else float(f[0])
        e = rig.pose_of(rig.observe(ctx))
        d = None if (y is None or prev_y is None) else y - prev_y
        _log("  burst %d: eef=%s panel front y=%s (moved %s)" % (
            i, np.round(e, 4), None if y is None else round(y, 4),
            None if d is None else round(d, 4)))
        if d is not None:
            total += d
        if d is not None and d < 0.003:
            _log("  panel stopped advancing -> at the stop")
            break
        prev_y = y if y is not None else prev_y
    m["panel_open_y"] = prev_y
    _save(m)
    _log("drawer travel measured: %.4f m from own_ref %.4f" % (
        total, m["own_ref"]))
    return prev_y


# ------------------------------------------------------------------ verify
def verify(ctx, tag="verify0"):
    """Retreat clear (+z 0.10 then +y 0.12), then the three plane signals of tasks/open-drawer.md §3.7."""
    m = _load()
    p = rig.pose_of(rig.observe(ctx))
    _log("retreat from eef=%s" % np.round(p, 4))
    goto(ctx, [p[0], p[1], p[2] + 0.10], Q_HOOK, -1.0, tol=0.008, max_rows=14)
    p = rig.pose_of(rig.observe(ctx))
    goto(ctx, [p[0], p[1] + 0.12, p[2]], Q_HOOK, -1.0, tol=0.010, max_rows=16)
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    xybox = (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    band = m["drawer_band"]
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    y_open = None if f is None else float(f[0])
    moved = None if y_open is None else y_open - m["own_ref"]
    _log("SIGNAL 1: this drawer's own front plane y=%s vs own_ref %.4f -> moved %s" % (
        None if y_open is None else round(y_open, 4), m["own_ref"],
        None if moved is None else round(moved, 4)))
    bars_now = s.handle_bars(panel_y=m["own_ref"], zrange=(s.table_z() + 0.005,
                                                          m["counter_z"] - 0.01),
                             xbox=xybox[0], reach=0.30)
    _log("SIGNAL 2: bars with the closed plane as reference (reach 0.30): %d" % len(bars_now))
    for b in bars_now:
        _log("   front_y=%.4f zc=%.4f z %.4f..%.4f" % (b["front_y"], b["zc"], b["z0"], b["z1"]))
    front_bar = [b for b in bars_now if b["front_y"] > m["own_ref"] + 0.10]
    _log("   bar in front of own_ref+0.10: %s" % (
        None if not front_bar else [(round(b["front_y"], 4), round(b["zc"], 4))
                                    for b in front_bar]))
    closed_bars = s.handle_bars(panel_y=m["own_ref"], zrange=(s.table_z() + 0.005,
                                                             m["counter_z"] - 0.01),
                                xbox=xybox[0], reach=0.09)
    _log("SIGNAL 3: bars in the closed window (reach 0.09): %d (was %d)" % (
        len(closed_bars), len(m["bars"])))
    tf = s.top_face_front_y()
    _log("rim/counter front y=%s (closed reference of that plane: %.4f)" % (
        None if tf is None else round(tf[1], 4), m["counter_y"]))
    rig.save_frame(tag, outdir=DATA)
    # the drawer's slide joint has 0.1605 m of travel (demo states[:, 26]) and its full-out
    # panel plane is own_ref + 0.159, so "open" means >= 0.14 m out, not 0.10
    ok = (moved is not None and moved >= 0.14)
    _log("VERDICT: %s (drawer's own panel at %s, own_ref %.4f, moved %s)" % (
        "OPEN" if ok else "NOT OPEN", None if y_open is None else round(y_open, 4),
        m["own_ref"], None if moved is None else round(moved, 4)))
    m["verify"] = {"y_open": y_open, "moved": moved, "open": bool(ok)}
    _save(m)
    return bool(ok)


# ------------------------------------------------------------------ run
def run(ctx):
    if not os.path.exists(MEAS):
        _log("=== PROBE ===")
        if not probe(ctx):
            return False
    _log("=== HOOK ===")
    if hook(ctx) is None:
        _log("hook failed to measure/insert")
        return False
    _log("=== SWEEP: insert + 4-row probe pull per height ===")
    if not sweep(ctx):
        _log("probe pull never engaged - stopping so a new height can be planned")
        return False
    _log("=== PULL ===")
    pull(ctx)
    _log("=== VERIFY ===")
    ok = verify(ctx)
    return ok


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


def look(ctx):
    """0-row snapshot + frame under a tag from $TAG (for offline geometry work)."""
    tag = os.environ.get("TAG", "look")
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    _log("snapshot %s eef=%s quat=%s gq=%s" % (tag, np.round(s.eef, 4), np.round(s.quat, 4),
                                               np.round(s.gq, 4)))
    return tag


# ------------------------------------------------------------------ demo learning
DEMO_H5 = os.path.join(ROOT, "tasks",
                       "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet_demo.hdf5")


def _demo(name):
    import h5py
    g = h5py.File(DEMO_H5, "r")["data"][name]
    return (g["actions"][()], g["obs/ee_pos"][()], g["obs/ee_ori"][()])


def _quat_from_rv(rv):
    rv = np.asarray(rv, dtype=np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.concatenate([rv / th * np.sin(th / 2), [np.cos(th / 2)]])


def replay(ctx, name=None, start=0, upto=None):
    """Replay a demo trace (learning only - the finish must be reached by my own code).

    Used to learn the hook pose that actually works on this chest: the 50 demos are the only
    ground truth for the finger-to-bar relation, which the depth cannot resolve because the
    handle bars are achromatic and therefore inside `Scene.arm`.  Prints the eef error against
    the trace every 10 rows, so a draw mismatch shows up in the first 20 rows.
    """
    name = name or os.environ.get("DEMO", "demo_43")
    start = int(os.environ.get("START", start))
    upto = int(os.environ.get("UPTO", 0)) or upto or 999
    A, P, O = _demo(name)
    upto = min(upto, len(A))
    _log("replay %s rows %d..%d (wrist-down start at %s)" % (name, start, upto,
                                                             np.round(P[0], 4)))
    if start == 0:
        goto(ctx, P[0], _quat_from_rv(O[0]), -1.0, tol=0.008, max_rows=34)
    for i in range(start, upto):
        payload, _ = rig.step(ctx, [A[i].tolist()], verbose=False)
        q = rig.pose_of(payload)
        if i % 10 == 0 or i == upto - 1:
            _log("  row %3d eef=%s trace=%s err=%s" % (
                i, np.round(q, 4), np.round(P[i], 4), np.round(q - P[i], 4)))
    p = rig.pose_of(rig.observe(ctx))
    _log("replay end eef=%s (trace last %s)" % (np.round(p, 4), np.round(P[upto - 1], 4)))
    try:
        m = _load()
        s = rig.snapshot(ctx, tag="replay_end", outdir=DATA)
        xybox = (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
        f = rig.front_face_y(s, band_z=m["drawer_band"], xybox=xybox, min_pixels=150)
        _log("drawer's own front plane now y=%s (own_ref %.4f) -> moved %s" % (
            None if f is None else round(f[0], 4), m["own_ref"],
            None if f is None else round(f[0] - m["own_ref"], 4)))
        tf = s.top_face_front_y()
        _log("counter/rim front y=%s (was %.4f)" % (
            None if tf is None else round(tf[1], 4), m["counter_y"]))
        m["replay"] = {"name": name, "upto": upto,
                       "eef_end": [float(v) for v in p],
                       "panel_y": None if f is None else float(f[0])}
        _save(m)
    except Exception as exc:
        _log("post-replay check failed: %s" % exc)
    return p


# ------------------------------------------------------------------ close (own code)
def _open_geometry(s, m):
    """Panel plane of the open bottom drawer + its bar (own code, from one depth call)."""
    xybox = (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    f = rig.front_face_y(s, band_z=m["drawer_band"], xybox=xybox, min_pixels=150)
    panel_y = None if f is None else float(f[0])
    bars = s.handle_bars(panel_y=panel_y if panel_y is not None else m["closed_front"],
                         zrange=(s.table_z() + 0.005, m["counter_z"] - 0.01),
                         xbox=xybox[0], reach=0.16)
    bottom = min(bars, key=lambda b: b["zc"]) if bars else None
    return panel_y, bars, bottom


def close_push(ctx, max_rows=48, check_every=6, dz_bar=0.0322, z=None):
    """Push the open bottom drawer shut with the PITCHED fingers, and prove it by planes.

    The hand's front face (eef_y - 0.111, spanning eef_z + 0.082..) cannot be aimed at this
    drawer's bar band without putting the eef under the table (bar top 1.005 -> eef_z 0.917),
    so the push is made by the finger tips, which in the hook pose are the front-most and
    lowest part of the gripper (measured here: rel (dy -0.030, dz +0.030) at the stall).
    The bar's front is derived from the panel plane (`dz_bar` = 0.0322 measured this episode:
    closed panel -0.2432, closed bar front -0.2110), NOT from `handle_bars`, whose grey mask
    swallowed the open gripper itself on the first attempt (it read a bar front of +0.0759).

    A push has no in-loop force signal, so the ONLY stop/verdict is the drawer's own front
    plane (`front_face_y` in the drawer's own band), checked every `check_every` rows, plus
    the eef's stall as a backstop.
    """
    m = _load()
    band, xybox = m["drawer_band"], (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    s = rig.snapshot(ctx, tag="before_close", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    panel_now = None if f is None else float(f[0])
    if panel_now is None:
        _log("close: cannot see the drawer's own front plane")
        return False
    x = float(m["bar_x_mid"])
    z = float(m["bottom"][2] - 0.031) if z is None else float(z)
    _log("close: panel y=%.4f (bar front %.4f) ; x=%.4f z=%.4f fingertips ~%.4f, bar %.4f..%.4f"
         % (panel_now, panel_now + dz_bar, x, z, z + 0.030, m["bottom"][0], m["bottom"][1]))
    goto(ctx, [x, panel_now + 0.10, z + 0.09], Q_HOOK, -1.0, tol=0.008, max_rows=24)
    goto(ctx, [x, panel_now + 0.10, z], Q_HOOK, -1.0, tol=0.006, max_rows=16)
    payload = rig.observe(ctx)
    dead = 0
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
            _log("  push STALLED at eef=%s (panel off home %s)" % (
                np.round(q, 4),
                None if y is None else round(y - m["own_ref"], 4)))
            break
    s = rig.snapshot(ctx, tag="closed", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    panel_final = None if f is None else float(f[0])
    bars_now = s.handle_bars(panel_y=m["closed_front"],
                             zrange=(s.table_z() + 0.005, m["counter_z"] - 0.01),
                             xbox=xybox[0], reach=0.09)
    bar_front_now = (min(bars_now, key=lambda b: b["zc"])["front_y"] if bars_now else None)
    ok = panel_final is not None and abs(panel_final - m["own_ref"]) <= 0.002
    _log("CLOSE VERDICT: panel y=%s vs closed plane %.4f -> home=%s ; bars in the closed "
         "window %d (was %d), lowest bar front %s vs %.4f" % (
             None if panel_final is None else round(panel_final, 4), m["own_ref"], ok,
             len(bars_now), len(m["bars"]),
             None if bar_front_now is None else round(bar_front_now, 4), m["bottom"][3]))
    rig.save_frame("closed", outdir=DATA)
    m["closed_panel_y"] = panel_final
    m["closed_bar_front"] = None if bar_front_now is None else float(bar_front_now)
    _save(m)
    return bool(ok)


def press_hook(ctx, x, front, z, rows=14, down=-0.30, dead_stop=3):
    """Press -y while commanding -z (the trace descends onto the bar instead of sliding under
    it), until the arm stalls.  Returns the stalled pose."""
    payload = rig.observe(ctx)
    dead = 0
    q = rig.pose_of(payload)
    for i in range(int(rows)):
        p = rig.pose_of(payload)
        a = np.zeros(7)
        a[1] = -1.0
        a[2] = float(down)
        a[0] = float(np.clip((x - p[0]) / 0.010, -0.6, 0.6))
        a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), Q_HOOK) / 0.5, -1, 1)
        a[6] = -1.0
        payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        q = rig.pose_of(payload)
        moved = float(np.max(np.abs(q - p)))
        dead = dead + 1 if moved < 0.0015 else 0
        if i % 3 == 0 or dead:
            _log("  press %2d eef=%s moved=%.4f dead=%d" % (i, np.round(q, 4), moved, dead))
        if dead >= int(dead_stop):
            _log("  press STALLED at eef=%s" % np.round(q, 4))
            break
    return q


def pull_biased(ctx, x, batches=6, rows=8, down=-0.30, y_cmd=1.0, target_out=0.10):
    """Pull +y in bursts while pressing down, comparing the drawer's OWN panel plane."""
    m = _load()
    band, xybox = m["drawer_band"], (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    f0 = rig.front_face_y(rig.snapshot(ctx, tag="reopen_a", outdir=DATA), band_z=band,
                          xybox=xybox, min_pixels=150)
    prev = None if f0 is None else float(f0[0])
    _log("pull start: panel y=%s (closed plane %.4f)" % (
        None if prev is None else round(prev, 4), m["own_ref"]))
    for i in range(int(batches)):
        payload = rig.observe(ctx)
        for _ in range(int(rows)):
            p = rig.pose_of(payload)
            a = np.zeros(7)
            a[1] = float(y_cmd)
            a[2] = float(down)
            a[0] = float(np.clip((x - p[0]) / 0.010, -0.6, 0.6))
            a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), Q_HOOK) / 0.5, -1, 1)
            a[6] = -1.0
            payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        s = rig.snapshot(ctx, tag="reopen_%d" % i, outdir=DATA)
        f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
        y = None if f is None else float(f[0])
        e = rig.pose_of(rig.observe(ctx))
        d = None if (y is None or prev is None) else y - prev
        _log("  pull burst %d eef=%s panel y=%s moved %s (from closed %s)" % (
            i, np.round(e, 4), None if y is None else round(y, 4),
            None if d is None else round(d, 4),
            None if y is None else round(y - m["own_ref"], 4)))
        if y is not None and (y - m["own_ref"]) >= float(target_out):
            _log("  drawer out by %.4f m - enough" % (y - m["own_ref"]))
            break
        if d is not None and d < 0.002:
            _log("  panel stopped advancing - the fingers are out of the handle")
            break
        prev = y if y is not None else prev
    return prev


def reopen2(ctx, z=None):
    """Own-code reopen, second recipe: insert at a CONSTANT height with a stall-tolerant goto
    (the replay's working contact was eef y -0.1902 at eef_z 0.9739, i.e. the fingers resting on
    the bar's top-front while -y is still commanded), then pull +y with a small -z bias.

    The first attempt drove -z while pressing (the trace's own action pattern) and dived 5 cm
    under the bar: the eef reached z 0.9188 with the fingers at table level, so the pull had
    nothing to catch.
    """
    m = _load()
    x = float(m["bar_x_mid"])
    front = float(m["bottom"][3])
    z = float(m["bottom"][2] - 0.026) if z is None else float(z)
    _log("reopen2: x=%.4f bar front %.4f hook eef_z %.4f" % (x, front, z))
    goto(ctx, [x, front + 0.16, z + 0.09], Q_HOOK, -1.0, tol=0.008, max_rows=22)
    goto(ctx, [x, front + 0.16, z], Q_HOOK, -1.0, tol=0.006, max_rows=16)
    tgt = np.array([x, front + 0.010, z])
    payload, ok = goto(ctx, tgt, Q_HOOK, -1.0, tol=0.004, max_rows=18)
    p = rig.pose_of(payload)
    _log("insert end eef=%s target=%s residual=%s (a y residual > 3 mm is CONTACT)"
         % (np.round(p, 4), np.round(tgt, 4), np.round(tgt - p, 4)))
    m = _load()
    m["reopen_hook"] = [float(v) for v in p]
    _save(m)
    pull_biased(ctx, x, down=-0.30)
    return verify(ctx, tag="reopened")


# the orientation the working demo settles into at the hook (measured from the trace's ee_ori
# at rows 85-105: finger axis (0.26,-0.855,-0.455), jaw axis (0.933,0.10,0.34)); it is 15 deg
# away from the idealised FINGER_AXIS/JAW_AXIS build, and that is enough to decide contact.
Q_HOOK2 = rig.quat_from_fingers((0.260, -0.855, -0.455), (0.933, 0.10, 0.34))


def verify_quick(ctx, tag="quick"):
    """Cheap plane verdict: clear the rim (+z 0.10) and the panel (+y 0.12), then one depth."""
    m = _load()
    band, xybox = m["drawer_band"], (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    p = rig.pose_of(rig.observe(ctx))
    goto(ctx, [p[0], p[1], p[2] + 0.10], Q_HOOK2, -1.0, tol=0.014, max_rows=8)
    p = rig.pose_of(rig.observe(ctx))
    goto(ctx, [p[0], p[1] + 0.13, p[2]], Q_HOOK2, -1.0, tol=0.016, max_rows=10)
    s = rig.snapshot(ctx, tag=tag, outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    y = None if f is None else float(f[0])
    bars = s.handle_bars(panel_y=m["closed_front"],
                         zrange=(s.table_z() + 0.005, m["counter_z"] - 0.01),
                         xbox=xybox[0], reach=0.30)
    low = [b for b in bars if b["zc"] < m["bottom"][2] + 0.02]
    moved = None if y is None else y - m["own_ref"]
    _log("VERDICT: panel y=%s -> out %s m (own_ref %.4f) ; lowest bar front %s ; bars found %d"
         % (None if y is None else round(y, 4), None if moved is None else round(moved, 4),
            m["own_ref"], None if not low else round(low[0]["front_y"], 4), len(bars)))
    rig.save_frame(tag, outdir=DATA)
    m["final_panel_y"] = y
    m["final_out"] = moved
    _save(m)
    return bool(moved is not None and moved >= 0.14)


def pull_more3(ctx, z_ins=0.950, dz_bar=0.0322):
    """Re-hook the part-open drawer and pull it to the stop (the task's joint threshold wants
    the drawer essentially fully out: the demos run the joint to -0.1605 of its ~0.1605 range).
    """
    m = _load()
    band, xybox = m["drawer_band"], (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    s = rig.snapshot(ctx, tag="more3_a", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    panel_y = None if f is None else float(f[0])
    if panel_y is None:
        return False
    bar_front = panel_y + float(dz_bar)
    x = float(m["bar_x_mid"])
    _log("pull_more3: panel %.4f -> bar front %.4f ; x=%.4f" % (panel_y, bar_front, x))
    goto(ctx, [x, bar_front + 0.10, 1.00], Q_HOOK2, -1.0, tol=0.008, max_rows=8)
    tgt = np.array([x, bar_front - 0.005, float(z_ins)])
    payload, ok = goto(ctx, tgt, Q_HOOK2, -1.0, tol=0.004, max_rows=22)
    p = rig.pose_of(payload)
    _log("press end eef=%s residual=%s" % (np.round(p, 4), np.round(tgt - p, 4)))
    for i in range(2):
        payload = rig.observe(ctx)
        for _ in range(8):
            q = rig.pose_of(payload)
            a = np.zeros(7)
            a[1] = 0.85
            a[2] = -0.35
            a[0] = float(np.clip((x - q[0]) / 0.010, -0.6, 0.6))
            a[3:6] = np.clip(rig.rotvec_err(rig.quat_of(payload), Q_HOOK2) / 0.5, -1, 1)
            a[6] = -1.0
            payload, _ = rig.step(ctx, [a.tolist()], verbose=False)
        s = rig.snapshot(ctx, tag="more3_%d" % i, outdir=DATA)
        f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
        y = None if f is None else float(f[0])
        e = rig.pose_of(rig.observe(ctx))
        _log("  burst %d eef=%s panel y=%s out %s" % (
            i, np.round(e, 4), None if y is None else round(y, 4),
            None if y is None else round(y - m["own_ref"], 4)))
        if y is not None and (y - m["own_ref"]) >= 0.150:
            break
    return verify_quick(ctx, tag="more3_final")


def pull_more(ctx, z_ins=0.950, dz_bar=0.0322, target_out=0.150):
    """Hook again on the part-open drawer and pull it to its stop.

    The task's predicate is a drawer-joint threshold, so 'a bit out' is not enough: this
    re-measures where the drawer's own front panel is now, re-hooks on the bar at
    `panel + dz_bar`, and pulls in bursts until the panel stops advancing (its full-out
    plane on this chest is own_ref + 0.159).
    """
    m = _load()
    band, xybox = m["drawer_band"], (tuple(m["xybox"][0]), tuple(m["xybox"][1]))
    s = rig.snapshot(ctx, tag="more_a", outdir=DATA)
    f = rig.front_face_y(s, band_z=band, xybox=xybox, min_pixels=150)
    panel_y = None if f is None else float(f[0])
    _log("pull_more: panel now y=%s (out %s m)" % (
        None if panel_y is None else round(panel_y, 4),
        None if panel_y is None else round(panel_y - m["own_ref"], 4)))
    if panel_y is None:
        return False
    bar_front = panel_y + float(dz_bar)
    x = float(m["bar_x_mid"])
    goto(ctx, [x, bar_front + 0.16, 1.00], Q_HOOK2, -1.0, tol=0.008, max_rows=12)
    tgt = np.array([x, bar_front - 0.005, float(z_ins)])
    payload, ok = goto(ctx, tgt, Q_HOOK2, -1.0, tol=0.004, max_rows=14)
    p = rig.pose_of(payload)
    _log("press end eef=%s residual=%s" % (np.round(p, 4), np.round(tgt - p, 4)))
    pull_biased(ctx, x, y_cmd=0.85, down=-0.35, batches=2, rows=8,
                target_out=float(target_out))
    return verify(ctx, tag="more_verify")


def reopen3(ctx):
    """Own-code reopen: rotate to the MEASURED hook orientation, press diagonally onto the
    bar's top-front (goto `y` past the bar and `z` under it, so the arm stalls on contact --
    the replay stalled at eef y -0.189 / z 0.9735 there), then pull +y with a -z bias."""
    m = _load()
    x = float(m["bar_x_mid"])
    front = float(m["bottom"][3])
    _log("reopen3: x=%.4f front=%.4f q_hook2=%s (%.0f deg from q_hook)" % (
        x, front, np.round(Q_HOOK2, 3), _quat_deg(Q_HOOK2, Q_HOOK)))
    rotate_in_place(ctx, Q_HOOK2, -1.0, max_rows=8, tol=0.10)
    goto(ctx, [x, front + 0.16, 1.00], Q_HOOK2, -1.0, tol=0.008, max_rows=12)
    tgt = np.array([x, front - 0.005, 0.950])
    payload, ok = goto(ctx, tgt, Q_HOOK2, -1.0, tol=0.004, max_rows=16)
    p = rig.pose_of(payload)
    _log("press end eef=%s target=%s residual=%s" % (np.round(p, 4), np.round(tgt, 4),
                                                     np.round(tgt - p, 4)))
    m = _load()
    m["reopen_hook"] = [float(v) for v in p]
    _save(m)
    pull_biased(ctx, x, y_cmd=0.85, down=-0.35, batches=3, rows=8)
    return verify(ctx, tag="reopened")


def reopen(ctx, z=None):
    """My own hook-and-pull: press the fingers onto the bar, then pull +y while pressing down."""
    m = _load()
    x = float(m["bar_x_mid"])
    front = float(m["bottom"][3])
    z = float(m["bottom"][2] - 0.026) if z is None else float(z)
    _log("reopen: x=%.4f bar front %.4f hook eef_z %.4f" % (x, front, z))
    goto(ctx, [x, front + 0.16, z + 0.10], Q_HOOK, -1.0, tol=0.008, max_rows=22)
    goto(ctx, [x, front + 0.16, z], Q_HOOK, -1.0, tol=0.006, max_rows=16)
    p = press_hook(ctx, x, front, z)
    m = _load()
    m["reopen_hook"] = [float(v) for v in p]
    _save(m)
    pull_biased(ctx, x)
    return verify(ctx, tag="reopened")
