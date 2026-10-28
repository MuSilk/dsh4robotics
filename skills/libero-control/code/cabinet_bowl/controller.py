"""Controller for LIBERO-90 task 1: close the top drawer, then put the black bowl on top.

Entry points are dispatched by the ENTRY env var from experiments/run_controller.py.
Each stage is runnable on its own on the same open episode so a failure costs a re-run.
"""
import base64
import json
import os
import time
import urllib.request

import numpy as np

FRAME_URL = "http://127.0.0.1:8766/frame.jpg"
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "data")

T_RIG = np.array([0.637, -0.012, 1.659])


# ---------------------------------------------------------------- rig helpers
def depth_payload(ctx):
    payload = json.loads(ctx.call("libero_depth")["content"][0]["text"])
    mm = np.frombuffer(base64.b64decode(payload["depth_mm"]), dtype="<u2").reshape(payload["height"], payload["width"])
    z = mm.astype(np.float64) * payload["depth_mm_scale"]
    cam = np.array(payload["camera_to_base"], dtype=np.float64).reshape(4, 4)
    rot = cam[:3, :3]
    fx, fy, cx, cy = payload["fx"], payload["fy"], payload["cx"], payload["cy"]
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)
    pts = pc @ rot.T + T_RIG
    return payload, z, pts, rot


def fetch_frame(path):
    try:
        with urllib.request.urlopen(FRAME_URL, timeout=10) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as exc:  # pragma: no cover
        print("frame fetch failed:", exc)
        return False


def decode_jpeg(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def step(ctx, rows, verbose=True):
    """Execute rows. The second return is unused."""
    payload = json.loads(ctx.call("libero_step", actions=rows)["content"][0]["text"])
    if verbose:
        pose = payload["observation"]["eef_pos"]
        print("  step %s eef=%s q=%s gq=%s" % (
            payload.get("step_id"), np.round(pose, 4),
            np.round(payload["observation"]["eef_quat_xyzw"], 3),
            np.round(payload["observation"]["gripper_qpos"], 3)))
    return payload, False


def observe(ctx):
    return json.loads(ctx.call("libero_observe")["content"][0]["text"])


def pose_of(payload):
    if "observation" in payload:
        return np.array(payload["observation"]["eef_pos"], dtype=np.float64)
    return np.array(payload["eef_pos"], dtype=np.float64)


def quat_of(payload):
    if "observation" in payload:
        return np.array(payload["observation"]["eef_quat_xyzw"], dtype=np.float64)
    return np.array(payload["eef_quat_xyzw"], dtype=np.float64)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def rotvec_err(q_cur, q_tgt):
    qe = qmul(q_tgt, np.array([-q_cur[0], -q_cur[1], -q_cur[2], q_cur[3]]))
    qe /= np.linalg.norm(qe) + 1e-12
    if qe[3] < 0:
        qe = -qe
    s = np.linalg.norm(qe[:3])
    return np.zeros(3) if s < 1e-9 else qe[:3] / s * (2.0 * np.arctan2(s, qe[3]))


def servo(ctx, target, q_tgt, gripper, tol=0.006, max_rows=60, rate=0.010, clip=1.0,
          zclip=1.0, log_every=0, complete_check=True):
    """Bang-bang per-axis servo to `target` with quaternion servoing every row."""
    target = np.asarray(target, dtype=np.float64)
    payload = observe(ctx)
    complete = False
    for i in range(max_rows):
        p = pose_of(payload)
        q = quat_of(payload)
        err = target - p
        if np.max(np.abs(err)) < tol:
            print("  servo converged err=%s" % np.round(err, 4))
            return payload, complete
        d = np.clip(err / rate, -clip, clip)
        d[2] = np.clip(err[2] / rate, -zclip, zclip)
        rv = rotvec_err(q, q_tgt)
        a = np.zeros(7)
        a[:3] = d
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = gripper
        payload, complete = step(ctx, [a.tolist()], verbose=(log_every == 0 or i % log_every == 0))
        if complete and complete_check:
            return payload, True
    print("  servo hit max_rows, err=%s" % np.round(target - pose_of(payload), 4))
    return payload, complete


def goto_xyz(ctx, target, q_tgt, gripper, **kw):
    return servo(ctx, target, q_tgt, gripper, **kw)


def push_axis(ctx, axis, sign, q_tgt, gripper, rows=20, other=None, tol=0.006, zclip=1.0,
              z_target=None, x_target=None, log_every=1):
    """Saturated push along `axis` (index 0/1/2) in direction sign, small corrections on others."""
    complete = False
    payload = observe(ctx)
    for i in range(rows):
        p = pose_of(payload)
        q = quat_of(payload)
        a = np.zeros(7)
        a[axis] = sign * 1.0
        if axis != 2 and z_target is not None:
            a[2] = np.clip((z_target - p[2]) / 0.010, -0.6, 0.6)
        if axis != 0 and x_target is not None:
            a[0] = np.clip((x_target - p[0]) / 0.010, -0.6, 0.6)
        if other is not None:
            for k, val in other.items():
                a[k] = np.clip((val - p[k]) / 0.010, -0.6, 0.6)
        rv = rotvec_err(q, q_tgt)
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = gripper
        payload, complete = step(ctx, [a.tolist()])
        if complete:
            return payload, True
    return payload, complete


# ---------------------------------------------------------------- entries
def probe(ctx):
    """Zero-step measurement: save depth + rgb raw material, print decisive numbers."""
    os.makedirs(DATA, exist_ok=True)
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    q = np.array(obs["eef_quat_xyzw"], dtype=np.float64)
    print("eef", np.round(eef, 4), "quat", np.round(q, 4), "grip", np.round(obs.get("gripper_qpos", []), 4))

    payload, z, pts, rot = depth_payload(ctx)
    frame_path = os.path.join(DATA, "frame0.jpg")
    got = fetch_frame(frame_path)
    rgb = decode_jpeg(frame_path) if got else np.zeros(z.shape + (3,))
    np.savez_compressed(os.path.join(DATA, "probe0.npz"), depth=z, pts=pts, rgb=rgb,
                        eef=eef, quat=q, rot=rot,
                        fx=payload["fx"], fy=payload["fy"], cx=payload["cx"], cy=payload["cy"])
    print("saved probe0.npz z-range", np.round([np.nanmin(z), np.nanmax(z)], 3), "rgb", rgb.shape)

    # pad offset, arm-coloured pixels in a small box around the eef
    mx, mn, mean = rgb.max(2), rgb.min(2), rgb.mean(2)
    arm = ((mx - mn) < 25) & (mean > 70)
    for half in (0.03, 0.04, 0.06):
        box = ((np.abs(pts[:, :, 0] - eef[0]) < half) & (np.abs(pts[:, :, 1] - eef[1]) < half)
               & (pts[:, :, 2] > 1.05))
        sel = arm & box
        if sel.sum():
            print("pad offset half=%.2f n=%d pad_bottom=%.4f offset=%+.4f" % (
                half, sel.sum(), pts[sel][:, 2].min(), pts[sel][:, 2].min() - eef[2]))
        else:
            print("pad offset half=%.2f no arm pixels" % half)

    # table plane check
    tbl = (np.abs(pts[:, :, 2] - 0.95) < 0.02) & (pts[:, :, 2] > 0.9)
    print("table pixels near z=0.95:", int(tbl.sum()))


def measure(ctx, tag, verbose=True):
    """Depth call at the current pose: prints scene planes + arm geometry."""
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    mm = pts
    du = np.zeros_like(pts)
    dv = np.zeros_like(pts)
    du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(du, dv)
    n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    info = {"eef": eef}
    if verbose:
        print("-- measure %s eef=%s" % (tag, np.round(eef, 4)))
    frame_tmp = os.path.join(DATA, "frame_tmp.jpg")
    rgb_here = decode_jpeg(frame_tmp) if fetch_frame(frame_tmp) else np.zeros(Z.shape + (3,))
    mx, mn, mean = rgb_here.max(2), rgb_here.min(2), rgb_here.mean(2)
    arm = ((mx - mn) < 25) & (mean > 70)
    # vertical faces facing +y / -y inside the drawer region (arm pixels excluded)
    sel = (np.abs(n[:, :, 1]) > 0.85) & (Z > 1.09) & (Z < 1.20) & (X > -0.16) & (X < 0.07) & ~arm
    vals = np.round(Y[sel] / 0.005) * 0.005
    uniq, cnt = np.unique(vals, return_counts=True)
    faces = []
    for o in np.argsort(cnt)[::-1]:
        if cnt[o] < 60:
            break
        sel2 = sel & (np.abs(Y - uniq[o]) < 0.006)
        faces.append((float(uniq[o]), int(cnt[o]), float(X[sel2].min()), float(X[sel2].max())))
        if verbose:
            print("   y-face %+.3f n=%d x[%.3f..%.3f]" % (uniq[o], cnt[o], X[sel2].min(), X[sel2].max()))
    info["faces"] = faces
    # horizontal planes near the cabinet
    selh = (np.abs(n[:, :, 2]) > 0.9) & (X > -0.17) & (X < 0.12) & (Y > -0.42) & (Y < 0.0)
    vals = np.round(Z[selh] / 0.005) * 0.005
    uniq, cnt = np.unique(vals, return_counts=True)
    hs = []
    for o in np.argsort(cnt)[::-1]:
        if cnt[o] < 300:
            break
        hs.append((float(uniq[o]), int(cnt[o])))
        if verbose:
            print("   z-plane %+.3f n=%d" % (uniq[o], cnt[o]))
    info["hplanes"] = hs
    # pad offset (tight box, deep only)
    box = ((np.abs(X - eef[0]) < 0.04) & (np.abs(Y - eef[1]) < 0.04) & (Z > eef[2] - 0.02))
    if (arm & box).sum():
        pb = float(pts[arm & box][:, 2].min())
        info["pad_bottom"] = pb
        info["pad_offset"] = pb - eef[2]
        if verbose:
            print("   pad_bottom=%.4f offset=%+.4f" % (pb, pb - eef[2]))
    np.savez_compressed(os.path.join(DATA, "m_%s.npz" % tag), depth=z, pts=pts, eef=eef)
    return info


QUAT_DOWN = np.array([0.99967, 0.00217, -0.02575, -0.00077])
PRE = np.array([-0.020, 0.055, 1.080])


def align(ctx):
    """Stage: descend over the table and slide to the pre-push pose in front of the panel."""
    obs = observe(ctx)
    print("start eef", np.round(obs["eef_pos"], 4))
    servo(ctx, [obs["eef_pos"][0], 0.055, 1.080], QUAT_DOWN, -1.0, max_rows=20)
    servo(ctx, PRE, QUAT_DOWN, -1.0, max_rows=30)
    measure(ctx, "pre")


def push(ctx):
    """Stage: push -y in batches until the panel plane meets the body front."""
    info = measure(ctx, "pushstart")
    total = 0
    for b in range(5):
        payload, done = push_axis(ctx, 1, -1.0, QUAT_DOWN, -1.0, rows=6,
                                 z_target=1.080, x_target=-0.020)
        total += 6
        if done:
            return True
        info = measure(ctx, "push%d" % b)
        ys = [f[0] for f in info["faces"]]
        print("  after %d push rows: eef=%s faces=%s" % (total, np.round(pose_of(payload), 4), ys))
        mid = [y for y in ys if -0.205 < y < -0.03]
        front = min(ys) if ys else 1.0
        closed = (not mid) and front < -0.215
        print("  mid=%s front=%.3f" % (np.round(mid, 3), front))
        if closed:
            print("  CLOSED confirmed by depth")
            return False
    return False


def close_drawer(ctx):
    align(ctx)
    push(ctx)


def find_bowl(ctx, expect_xyz=None, verbose=True):
    """Depth: locate the bowl blob on the table. Returns dict or None."""
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    frame_tmp = os.path.join(DATA, "frame_tmp.jpg")
    rgb_here = decode_jpeg(frame_tmp) if fetch_frame(frame_tmp) else np.zeros(Z.shape + (3,))
    mean = rgb_here.mean(2)
    mx, mn = rgb_here.max(2), rgb_here.min(2)
    arm = ((mx - mn) < 25) & (mean > 70)
    box = ((np.abs(X - eef[0]) < 0.05) & (np.abs(Y - eef[1]) < 0.05) & (Z > eef[2]))
    above = arm & box
    if verbose:
        print("-- find_bowl eef=%s" % np.round(eef, 4))
    if above.sum():
        pb = float(pts[above][:, 2].min())
        if verbose:
            print("   pad_bottom=%.4f offset=%+.4f" % (pb, pb - eef[2]))
    # bowl pixels: on/near the table, dark-ish, not the arm
    m = (Z > 0.945) & (Z < 1.035) & (Y > -0.10) & (Y < 0.10) & (X > -0.32) & (X < -0.02) \
        & (mean > 75) & (mean < 178)
    if expect_xyz is not None:
        m &= (np.abs(X - expect_xyz[0]) < 0.09) & (np.abs(Y - expect_xyz[1]) < 0.09)
    if m.sum() < 100:
        if verbose:
            print("   bowl blob gone (n=%d)" % int(m.sum()))
        return None
    p = pts[m]
    top = p[:, 2] > np.percentile(p[:, 2], 90)
    info = {"n": int(m.sum()), "rim_z": float(p[top, 2].mean()), "zmax": float(p[:, 2].max()),
            "xmin": float(p[top, 0].min()), "xmax": float(p[top, 0].max()),
            "ymin": float(p[top, 1].min()), "ymax": float(p[top, 1].max()),
            "zmin": float(p[:, 2].min())}
    info["cx"] = 0.5 * (info["xmin"] + info["xmax"])
    info["cy"] = 0.5 * (info["ymin"] + info["ymax"])
    if verbose:
        print("   bowl n=%d rim_z=%.4f zmax=%.4f zmin=%.4f x[%.3f..%.3f] y[%.3f..%.3f] centre=(%.3f,%.3f)" % (
            info["n"], info["rim_z"], info["zmax"], info["zmin"], info["xmin"], info["xmax"],
            info["ymin"], info["ymax"], info["cx"], info["cy"]))
    return info


def bowl(ctx):
    """Stage: grasp the black bowl on its +y rim wall, lift, carry to the cabinet top, set down."""
    complete = False
    # 1. back out of the cabinet and rise
    servo(ctx, [-0.070, 0.105, 1.150], QUAT_DOWN, -1.0, max_rows=20)
    servo(ctx, [-0.103, 0.045, 1.150], QUAT_DOWN, -1.0, max_rows=20)
    info = find_bowl(ctx)
    if info is None:
        print("bowl not found")
        return
    ox, oy, rim_z = info["cx"], info["ymax"], info["rim_z"]
    grasp = np.array([ox, oy, rim_z - 0.044 - 0.020])
    print("grasp target", np.round(grasp, 4))
    # 2. descend onto the rim
    servo(ctx, grasp, QUAT_DOWN, -1.0, max_rows=20, rate=0.010, zclip=0.7, tol=0.005)
    # 3. close on the rim
    payload = observe(ctx)
    hold = [[0, 0, 0, 0, 0, 0, 1.0]] * 4
    payload, complete = step(ctx, hold)
    if complete:
        return True
    print("after close gq=", np.round(payload["observation"]["gripper_qpos"], 4))
    # 4. lift
    p = pose_of(payload)
    servo(ctx, [p[0], p[1], p[2] + 0.10], QUAT_DOWN, 1.0, max_rows=12, tol=0.006)
    # check the bowl came along: the blob at the old spot should be gone
    still = find_bowl(ctx, expect_xyz=(ox, oy))
    if still is not None:
        print("WARNING: bowl still on the table")
    else:
        print("bowl lifted (old spot empty)")
    # 5. carry above the cabinet top
    p = pose_of(observe(ctx))
    carry_z = 1.176 + 0.021 + 0.09
    servo(ctx, [p[0], p[1], carry_z], QUAT_DOWN, 1.0, max_rows=14)
    servo(ctx, [-0.040, -0.300, carry_z], QUAT_DOWN, 1.0, max_rows=30)
    # 6. set down: bowl bottom = eef_z + 0.021
    target = np.array([-0.040, -0.300, 1.176 - 0.021 + 0.004])
    print("setdown target", np.round(target, 4))
    payload = observe(ctx)
    for i in range(16):
        p = pose_of(payload)
        err = target[2] - p[2]
        a = np.zeros(7)
        a[2] = np.clip(err / 0.010, -1, 1)
        a[0] = np.clip((target[0] - p[0]) / 0.010, -0.4, 0.4)
        a[1] = np.clip((target[1] - p[1]) / 0.010, -0.4, 0.4)
        rv = rotvec_err(quat_of(payload), QUAT_DOWN)
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = 1.0
        payload, complete = step(ctx, [a.tolist()])
        if complete:
            return True
    # 7. release and hold
    print("releasing")
    for i in range(4):
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, -1.0]])
        if complete:
            return True
    return False


def bowlgeom(ctx):
    """Zero-step: cross-section of the bowl rim to size the pinch."""
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    frame_tmp = os.path.join(DATA, "frame_tmp.jpg")
    rgb_here = decode_jpeg(frame_tmp) if fetch_frame(frame_tmp) else np.zeros(Z.shape + (3,))
    mean = rgb_here.mean(2)
    m = (Z > 0.94) & (Z < 1.04) & (Y > -0.12) & (Y < 0.12) & (X > -0.24) & (X < -0.02) \
        & (mean > 75) & (mean < 178)
    p = pts[m]
    print("bowl n=%d  eef=%s" % (m.sum(), np.round(eef, 4)))
    print(" x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" % (
        p[:,0].min(),p[:,0].max(),p[:,1].min(),p[:,1].max(),p[:,2].min(),p[:,2].max()))
    cx = 0.5*(p[:,0].min()+p[:,0].max())
    slab = m & (np.abs(X-cx) < 0.012)
    q = pts[slab]
    print(" slab |x-%.3f|<0.012 n=%d" % (cx, slab.sum()))
    for zlo in np.arange(0.945, 1.005, 0.006):
        ssel = slab & (Z >= zlo) & (Z < zlo+0.006)
        if ssel.sum():
            w = pts[ssel]
            ys = np.sort(w[:,1])
            print("  z[%.3f..%.3f) n=%3d y %.3f..%.3f  (gaps: %s)" % (
                zlo, zlo+0.006, ssel.sum(), ys.min(), ys.max(),
                np.round(np.diff(ys)[np.diff(ys)>0.004], 4) if len(ys)>1 else "-"))
            print("      all y:", np.round(np.unique(np.round(ys,3)), 3))


def grasp(ctx):
    """Stage: return to the bowl, grasp its +y rim wall, lift, verify."""
    complete = False
    p0 = pose_of(observe(ctx))
    print("grasp start eef", np.round(p0, 4))
    servo(ctx, [p0[0], p0[1], 1.20], QUAT_DOWN, -1.0, max_rows=8)
    servo(ctx, [-0.104, 0.055, 1.185], QUAT_DOWN, -1.0, max_rows=40)
    for attempt in range(2):
        info = find_bowl(ctx)
        if info is None:
            print("bowl already gone")
            return True
        ox, rim_z = info["cx"], info["rim_z"]
        oy = info["ymax"] - 0.006 - 0.008 * attempt
        grasp_pt = np.array([ox, oy, rim_z - offset_est(ctx) - 0.020])
        print("attempt %d grasp target %s" % (attempt, np.round(grasp_pt, 4)))
        servo(ctx, grasp_pt, QUAT_DOWN, -1.0, max_rows=32, rate=0.011, zclip=0.8,
              tol=0.004, complete_check=True)
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, 1.0]] * 4)
        if complete:
            return True
        gq = payload["observation"]["gripper_qpos"]
        print("after close gq=", np.round(gq, 4))
        # lift and verify
        p = pose_of(payload)
        servo(ctx, [p[0], p[1], p[2] + 0.12], QUAT_DOWN, 1.0, max_rows=16, tol=0.006)
        still = find_bowl(ctx, expect_xyz=(ox, info["cy"]))
        if still is None:
            print("GRASP OK (old spot empty)")
            return True
        print("GRASP FAILED, bowl still at rim_z=%.4f centre=(%.3f,%.3f)" % (
            still["rim_z"], still["cx"], still["cy"]))
        p = pose_of(observe(ctx))
        servo(ctx, [p[0], p[1], 1.20], QUAT_DOWN, -1.0, max_rows=8)
    return False


def offset_est(ctx):
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    frame_tmp = os.path.join(DATA, "frame_tmp.jpg")
    rgb_here = decode_jpeg(frame_tmp) if fetch_frame(frame_tmp) else np.zeros(Z.shape + (3,))
    mean = rgb_here.mean(2)
    mx, mn = rgb_here.max(2), rgb_here.min(2)
    arm = ((mx - mn) < 25) & (mean > 70)
    box = ((np.abs(X - eef[0]) < 0.045) & (np.abs(Y - eef[1]) < 0.045) & (Z > eef[2] + 0.005))
    if (arm & box).sum() < 50:
        print("   offset_est: no pixels, using 0.042")
        return 0.042
    off = float(pts[arm & box][:, 2].min()) - eef[2]
    print("   offset_est = %+.4f (n=%d)" % (off, int((arm & box).sum())))
    return off


def place(ctx):
    """Stage: carry the bowl over the cabinet top and set it down."""
    complete = False
    p = pose_of(observe(ctx))
    print("place start eef", np.round(p, 4))
    carry_z = 1.176 + 0.021 + 0.09
    servo(ctx, [p[0], p[1], carry_z], QUAT_DOWN, 1.0, max_rows=16)
    servo(ctx, [-0.040, -0.300, carry_z], QUAT_DOWN, 1.0, max_rows=36)
    target = np.array([-0.040, -0.300, 1.176 - 0.021 + 0.004])
    print("setdown target", np.round(target, 4))
    payload = observe(ctx)
    for i in range(20):
        p = pose_of(payload)
        a = np.zeros(7)
        a[2] = np.clip((target[2] - p[2]) / 0.010, -1, 1)
        a[0] = np.clip((target[0] - p[0]) / 0.010, -0.4, 0.4)
        a[1] = np.clip((target[1] - p[1]) / 0.010, -0.4, 0.4)
        rv = rotvec_err(quat_of(payload), QUAT_DOWN)
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = 1.0
        payload, complete = step(ctx, [a.tolist()])
        if complete:
            return True
    print("releasing")
    for i in range(6):
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, -1.0]])
        if complete:
            return True
    return False


def gripper_geom(ctx):
    """Zero-step: vertical finger faces near the open gripper -> pad faces and their z span."""
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    print("eef", np.round(eef, 4), "gq", np.round(obs["gripper_qpos"], 4))
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    du = np.zeros_like(pts); dv = np.zeros_like(pts)
    du[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dv[1:-1, :] = pts[2:, :] - pts[:-2, :]
    n = np.cross(du, dv); n /= (np.linalg.norm(n, axis=2, keepdims=True) + 1e-9)
    near = (np.abs(X - eef[0]) < 0.09) & (np.abs(Y - eef[1]) < 0.09) & (Z > eef[2] - 0.01)
    for nm, mask in (("y", np.abs(n[:,:,1]) > 0.85), ("x", np.abs(n[:,:,0]) > 0.85), ("z", np.abs(n[:,:,2]) > 0.85)):
        a = "xyz".index(nm)
        m = mask & near
        if m.sum() < 20:
            print(" no %s faces" % nm); continue
        vals = np.round(pts[:,:,a][m] / 0.004) * 0.004
        uniq, cnt = np.unique(vals, return_counts=True)
        print(" %s-faces near eef:" % nm)
        for o in np.argsort(cnt)[::-1][:8]:
            if cnt[o] < 25: break
            sel = m & (np.abs(pts[:,:,a] - uniq[o]) < 0.005)
            q = pts[sel]
            other = [i for i in range(3) if i != a]
            print("   %s=%+.3f n=%d  %s %.3f..%.3f  %s %.3f..%.3f" % (
                nm, uniq[o], cnt[o], "xyz"[other[0]], q[:,other[0]].min(), q[:,other[0]].max(),
                "xyz"[other[1]], q[:,other[1]].min(), q[:,other[1]].max()))


def padgeom(ctx):
    """Zero-step: the dark pad material of the two fingers, and the bowl rim ring."""
    obs = observe(ctx)
    eef = np.array(obs["eef_pos"], dtype=np.float64)
    print("eef", np.round(eef, 4), "gq", np.round(obs["gripper_qpos"], 4))
    payload, z, pts, rot = depth_payload(ctx)
    X, Y, Z = pts[:, :, 0], pts[:, :, 1], pts[:, :, 2]
    frame_tmp = os.path.join(DATA, "frame_tmp.jpg")
    rgb_here = decode_jpeg(frame_tmp) if fetch_frame(frame_tmp) else np.zeros(Z.shape + (3,))
    mean = rgb_here.mean(2); mx = rgb_here.max(2); mn = rgb_here.min(2)
    near = (np.abs(X - eef[0]) < 0.10) & (np.abs(Y - eef[1]) < 0.10) & (Z > eef[2])
    pad = near & ((mx - mn) < 25) & (mean > 78) & (mean < 122)
    print("pad-material pixels n=%d" % pad.sum())
    zs = np.sort(pts[pad][:, 2])
    print(" pad z: min %.4f  p10 %.4f  p50 %.4f  max %.4f" % (zs.min(), np.percentile(zs,10), np.median(zs), zs.max()))
    low = pad & (Z < np.percentile(pts[pad][:,2], 25)) if pad.sum() else pad
    for name, mask in (("pad-low", low),):
        if mask.sum() < 10: continue
        q = pts[mask]
        print(" %s n=%d x %.3f..%.3f y %.3f..%.3f z %.3f..%.3f" % (
            name, mask.sum(), q[:,0].min(),q[:,0].max(),q[:,1].min(),q[:,1].max(),q[:,2].min(),q[:,2].max()))
        ys = np.sort(q[:,1])
        print("   y sorted:", np.round(ys, 3))
    # bowl rim ring: top decile of the bowl blob, circle fit
    bm = (Z > 0.94) & (Z < 1.04) & (Y > -0.12) & (Y < 0.12) & (X > -0.24) & (X < -0.02) \
        & (mean > 75) & (mean < 178) & ~pad
    bp = pts[bm]
    top = bm & (Z > np.percentile(bp[:,2], 88))
    rp = pts[top]
    A = np.c_[2*rp[:,0], 2*rp[:,1], np.ones(len(rp))]
    b = rp[:,0]**2 + rp[:,1]**2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    R = np.sqrt(sol[2] + cx**2 + cy**2)
    print("rim ring fit: centre (%.4f, %.4f) R=%.4f  n=%d  zmean %.4f" % (cx, cy, R, len(rp), rp[:,2].mean()))
    print(" rim y extents: min %.4f max %.4f | x min %.4f max %.4f" % (
        rp[:,1].min(), rp[:,1].max(), rp[:,0].min(), rp[:,0].max()))
    print(" predicted far/near rim outer y: %.4f / %.4f" % (cy-R, cy+R))


# measured at gq=0.029: finger-pair centre = eef + (-0.014, -0.015); tips 0.0434 above eef_z
JAW_DX = -0.0140
JAW_DY = -0.0151
PAD_Z = 0.0434


def grasp2(ctx):
    """Stage: rim pinch using the measured finger-pair offset from eef_pos."""
    p = pose_of(observe(ctx))
    print("grasp2 start eef", np.round(p, 4))
    servo(ctx, [-0.087, 0.050, 1.145], QUAT_DOWN, -1.0, max_rows=12)
    info = find_bowl(ctx)
    if info is None:
        print("bowl gone?")
        return True
    rim_z = info["rim_z"]
    for attempt in range(3):
        # rim +y wall: jaw centre on the wall's outer surface
        wall_y = info["ymax"] - 0.004 + 0.007 * attempt
        gx = info["cx"] - JAW_DX
        gy = wall_y - JAW_DY
        gz = rim_z - PAD_Z + 0.002
        target = np.array([gx, gy, gz])
        print("attempt %d: rim_z=%.4f ymax=%.4f wall_y=%.4f -> target %s" % (
            attempt, rim_z, info["ymax"], wall_y, np.round(target, 4)))
        servo(ctx, target, QUAT_DOWN, -1.0, max_rows=26, rate=0.011, zclip=0.8, tol=0.004)
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, 1.0]] * 4)
        if complete:
            return True
        gq = payload["observation"]["gripper_qpos"]
        print("after close gq=", np.round(gq, 4), "eef=", np.round(pose_of(payload), 4))
        p = pose_of(payload)
        servo(ctx, [p[0], p[1], p[2] + 0.14], QUAT_DOWN, 1.0, max_rows=16, tol=0.008)
        still = find_bowl(ctx, expect_xyz=(info["cx"], info["cy"]))
        if still is None:
            print("GRASP OK on attempt %d" % attempt)
            return True
        print("FAILED attempt %d, bowl still there" % attempt)
        p = pose_of(observe(ctx))
        servo(ctx, [p[0], p[1], 1.16], QUAT_DOWN, -1.0, max_rows=8)
        info = still
    return False


def finish_task(ctx):
    """Grasp (with a full gripper close), verify, carry to the cabinet top, set down."""
    p = pose_of(observe(ctx))
    print("finish_task start eef", np.round(p, 4))
    target = np.array([-0.0890, 0.0459, 0.9507])
    servo(ctx, target, QUAT_DOWN, -1.0, max_rows=28, rate=0.011, zclip=0.8, tol=0.004)
    # the gripper takes ~15 rows to close: hold the pose with gripper=+1
    payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, 1.0]] * 18)
    if complete:
        return True
    print("closed gq=", np.round(payload["observation"]["gripper_qpos"], 4))
    p = pose_of(payload)
    servo(ctx, [p[0], p[1], p[2] + 0.13], QUAT_DOWN, 1.0, max_rows=14, tol=0.008)
    still = find_bowl(ctx, expect_xyz=(-0.103, -0.016))
    if still is not None:
        print("GRASP FAILED: bowl still on the table")
        return False
    print("GRASP OK: lifting")
    servo(ctx, [-0.040, -0.300, 1.287], QUAT_DOWN, 1.0, max_rows=42)
    p = pose_of(observe(ctx))
    print("over cabinet top eef", np.round(p, 4))
    # set down: bowl bottom = eef_z + 0.021
    tgt = np.array([-0.040, -0.300, 1.159])
    payload = observe(ctx)
    for i in range(18):
        q = pose_of(payload)
        a = np.zeros(7)
        a[2] = np.clip((tgt[2] - q[2]) / 0.010, -1, 1)
        a[0] = np.clip((tgt[0] - q[0]) / 0.010, -0.4, 0.4)
        a[1] = np.clip((tgt[1] - q[1]) / 0.010, -0.4, 0.4)
        rv = rotvec_err(quat_of(payload), QUAT_DOWN)
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = 1.0
        payload, complete = step(ctx, [a.tolist()])
        if complete:
            return True
    print("releasing")
    for i in range(8):
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, -1.0]])
        if complete:
            return True
    return False


def place2(ctx):
    """Stage: bowl in hand -> carry over the cabinet top, set down, release."""
    complete = False
    p = pose_of(observe(ctx))
    print("place2 start eef", np.round(p, 4), "gq", np.round(observe(ctx)["gripper_qpos"], 4))
    servo(ctx, [-0.040, -0.300, 1.290], QUAT_DOWN, 1.0, max_rows=40)
    p = pose_of(observe(ctx))
    print("over top eef", np.round(p, 4))
    tgt = np.array([-0.040, -0.300, 1.163])
    payload = observe(ctx)
    for i in range(16):
        q = pose_of(payload)
        a = np.zeros(7)
        a[2] = np.clip((tgt[2] - q[2]) / 0.010, -1, 1)
        a[0] = np.clip((tgt[0] - q[0]) / 0.010, -0.4, 0.4)
        a[1] = np.clip((tgt[1] - q[1]) / 0.010, -0.4, 0.4)
        rv = rotvec_err(quat_of(payload), QUAT_DOWN)
        a[3:6] = np.clip(rv / 0.5, -1, 1)
        a[6] = 1.0
        payload, complete = step(ctx, [a.tolist()])
        if complete:
            return True
    print("releasing")
    for i in range(8):
        payload, complete = step(ctx, [[0, 0, 0, 0, 0, 0, -1.0]])
        if complete:
            return True
    print("after release eef", np.round(pose_of(payload), 4))
    # settle: rise a little and check
    payload, _complete = step(ctx, [[0, 0, 0.4, 0, 0, 0, -1.0]] * 3)
    return False


def main(ctx):
    entry = os.environ.get("ENTRY", "probe")
    print("ENTRY", entry, flush=True)
    t0 = time.time()
    globals()[entry](ctx)
    print("entry %s done in %.1fs" % (entry, time.time() - t0), flush=True)
