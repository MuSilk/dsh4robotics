#!/usr/bin/env python3
"""Depth-referenced perception + motion for the LIBERO-90 episodes.

The camera the depth comes from is NOT in the base frame the arm reports: the
rotation is right, the translation is offset by a constant DELTA per episode

    eef_pt = depth_pt + DELTA

Solve it once, first thing, with solve_delta(); every 3-D target you compute
after that is in the eef/base frame and can be fed straight to goto().

    import rig
    ctx  = rig.connect("ctl")
    delta, table_z, checks = rig.solve_delta(ctx)  # prints its own checks
    agent, wrist = rig.views(ctx, "home")          # home_agent.jpg / home_wrist.jpg
    objs, agent, wrist = rig.scene_objects(ctx, delta)   # clusters, eef frame
    rig.crop(agent, objs[0]["bbox"], "obj0.png", 4)
    rig.goto(ctx, [x, y, z], rig.OPEN)
    z, stalled = rig.descend(ctx, [x, y], 0.40)
    rig.close(ctx, 8)                              # jaw ~0.0197 == holding something
"""
import base64
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcp  # noqa: E402
from geom import components, quat_to_mat  # noqa: E402

OPEN, CLOSE = -1.0, 1.0
R_DOWN = np.diag([1.0, -1.0, -1.0])   # gripper down == eef_quat_xyzw ~ (1,0,0,0)
GAIN = 0.06                           # P gain: full scale for errors > 6 cm
FULL_SCALE = 0.010                    # m per step at action 1.0 (first step of a burst ~4x less)
JAW_OPEN = 0.078                      # finger gap at gripper_qpos 0.034
TABLE_EXPECT = (0.38, 0.48)           # this family of scenes puts the table here


def connect(name="ctl", path=None):
    return mcp.connect(path, name)


tool = mcp.tool


# ---------------------------------------------------------------- state
def _state(payload):
    """Flat observation dict -> pose/rotation/joints/jaw. libero_observe and
    libero_step both return the same flat layout (no 'observation' wrapper)."""
    o = payload.get("observation", payload)
    return {
        "p": np.array(o["eef_pos"], dtype=float),
        "R": quat_to_mat(o["eef_quat_xyzw"]),
        "quat": np.array(o["eef_quat_xyzw"], dtype=float),
        "q": np.array(o["joint_pos"], dtype=float),
        "jaw": float(abs(o["gripper_qpos"][0])),
        "step": o.get("step_id"),
        "remaining": o.get("remaining_episode_steps"),
    }


def observe(ctx):
    return _state(tool(ctx, "libero_observe"))


def step(ctx, rows, n=1):
    """Send 7-D rows (repeated n times) and return the new observe() dict."""
    return _state(tool(ctx, "libero_step", actions=[[float(v) for v in r] for r in rows] * n))


def push(ctx, row, n=1):
    """Repeat one row n times (libero_hold does the same, but this reads the pose back)."""
    return step(ctx, [row], n)


# ---------------------------------------------------------------- pixels
def views(ctx, tag="view"):
    """(agent rgb float HxWx3, wrist rgb float HxWx3); also saves tag_agent.jpg / tag_wrist.jpg."""
    cam = tool(ctx, "libero_cameras")
    agent = np.asarray(Image.open(io.BytesIO(base64.b64decode(cam["agentview_jpeg"]))).convert("RGB"), dtype=float)
    wrist = np.asarray(Image.open(io.BytesIO(base64.b64decode(cam["wrist_jpeg"]))).convert("RGB"), dtype=float)
    Image.fromarray(agent.astype("uint8")).save("%s_agent.jpg" % tag)
    Image.fromarray(wrist.astype("uint8")).save("%s_wrist.jpg" % tag)
    return agent, wrist


def depth(ctx):
    """(points HxWx3 in the depth frame, z metres, raw payload, R, t). Pixel-aligned with agentview."""
    dep = tool(ctx, "libero_depth")
    z = np.frombuffer(base64.b64decode(dep["depth_mm"]), dtype="<u2").reshape(
        dep["height"], dep["width"]).astype(float) * dep["depth_mm_scale"]
    M = np.array(dep["camera_to_base"], dtype=float).reshape(4, 4)
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(u - dep["cx"]) * z / dep["fx"], (v - dep["cy"]) * z / dep["fy"], z], -1)
    pts = rays @ M[:3, :3].T + M[:3, 3]
    return pts, z, dep, M[:3, :3], M[:3, 3]


def unproject(pts, r, c):
    return np.array(pts[int(r), int(c)], dtype=float)


def crop(rgb, bbox, out, scale=4, pad=10):
    """bbox = (x0, y0, x1, y1) in pixels; upscale so you can read the label."""
    x0, y0, x1, y1 = [int(v) for v in bbox]
    im = Image.fromarray(rgb.astype("uint8")).crop(
        (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, rgb.shape[1] - 1), min(y1 + pad, rgb.shape[0] - 1)))
    im.resize((im.width * scale, im.height * scale), Image.LANCZOS).save(out)
    return out


# ---------------------------------------------------------------- calibration
def table_plane(pts, z, rows=(360, 512), cols=(20, 480)):
    """Height of the table top in the DEPTH frame, as the mode of the near-table band."""
    valid = (z > 0.05) & (z < 4.0)
    m = np.zeros_like(z, bool)
    m[rows[0]:rows[1], cols[0]:cols[1]] = True
    sel = m & valid
    hist, edges = np.histogram(pts[..., 2][sel], bins=60)
    i = int(hist.argmax())
    return float(0.5 * (edges[i] + edges[i + 1]))


def gripper_tip(rgb, pts, z, min_px=800):
    """3-D point of the finger tips in the depth frame + the info needed to trust it.

    The arm is the only light-grey thing that hangs in from the top of the picture.
    The table is also close to the camera, so the near-depth mask alone is not enough;
    we require the blob to start at row 0, stay narrower than 250 px, and to end above
    row 300.
    """
    valid = (z > 0.05) & (z < 4.0)
    arm = (rgb.mean(-1) > 105) & (rgb.max(-1) - rgb.min(-1) < 28) & valid & (z > 0.25) & (z < 1.05)
    comps = [c for c in components(arm, min_px=min_px)
             if c[:, 0].min() <= 4 and c[:, 0].max() < 300 and (c[:, 1].max() - c[:, 1].min()) < 250]
    if not comps:
        raise RuntimeError("no gripper blob: arm out of frame? re-solve at a home-like pose")
    g = max(comps, key=len)
    rmax = g[:, 0].max()
    low = g[g[:, 0] >= rmax - 2]          # bottom 3 rows of the blob == the two finger tips
    tip = pts[low[:, 0], low[:, 1]].mean(0)
    mid = 0.5 * (low[:, 1].min() + low[:, 1].max())
    left = low[low[:, 1] < mid]
    right = low[low[:, 1] >= mid]
    sep = None
    if len(left) > 2 and len(right) > 2:
        pl = pts[left[:, 0], left[:, 1]].mean(0)
        pr = pts[right[:, 0], right[:, 1]].mean(0)
        sep = float(np.linalg.norm(pl - pr))
    return tip, {"blob_px": len(g), "bbox": (int(g[:, 1].min()), int(g[:, 0].min()),
                                            int(g[:, 1].max()), int(g[:, 0].max())),
                 "finger_sep": sep, "tip_depth": float(z[low[:, 0], low[:, 1]].mean())}


def solve_delta(ctx, verbose=True):
    """DELTA such that eef_pt = depth_pt + DELTA. Do this before any 3-D target.

    Validity of the result rests on one measurement: the two finger tips must be
    JAW_OPEN apart (they are, on this rig, within a couple of mm) and the table must
    land at a sane height once DELTA_z is added. Both are printed.
    """
    agent, _ = views(ctx, "calib")
    pts, z, dep, Rc, tc = depth(ctx)
    eef = observe(ctx)["p"]
    tip, info = gripper_tip(agent, pts, z)
    delta = eef - tip
    tz = table_plane(pts, z)
    checks = {
        "finger_sep": info["finger_sep"],
        "finger_sep_ok": info["finger_sep"] is not None and abs(info["finger_sep"] - JAW_OPEN) < 0.03,
        "table_eef": tz + delta[2],
        "table_ok": TABLE_EXPECT[0] <= tz + delta[2] <= TABLE_EXPECT[1],
        "delta_z_positive": delta[2] > 0.2,
    }
    if verbose:
        print("DELTA", np.round(delta, 3), "| tip", np.round(tip, 3), "| jaw gap measured",
              None if info["finger_sep"] is None else round(info["finger_sep"], 3),
              "| table(eef)", round(tz + delta[2], 3), "|", checks, flush=True)
    return delta, tz, checks


# ---------------------------------------------------------------- objects
def scene_objects(ctx, delta=None, table_z=None, min_px=150, zmax=0.30):
    """One record per object standing on the table, in the eef frame.

    Filters out the far wall / horizon blobs (they touch row 0 or span the picture).
    """
    agent, wrist = views(ctx, "objects")
    pts, z, dep, Rc, tc = depth(ctx)
    if table_z is None:
        table_z = table_plane(pts, z)
    if delta is None:
        delta, _, _ = solve_delta(ctx, verbose=False)
    e = pts + delta
    valid = (z > 0.05) & (z < 4.0)
    obj = valid & (pts[..., 2] > table_z + 0.012) & (pts[..., 2] < table_z + zmax)
    out = []
    for pix in components(obj, min_px=min_px):
        ys, xs = pix[:, 0], pix[:, 1]
        if ys.min() <= 1 and ys.max() > 100:          # horizon / wall
            continue
        if (xs.max() - xs.min()) > 320:               # spans the picture
            continue
        xyz = e[ys, xs]
        top = xyz[xyz[:, 2] > np.percentile(xyz[:, 2], 60)]
        out.append({"px": (float(xs.mean()), float(ys.mean())),
                    "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "n": int(len(pix)),
                    "xy": np.median(top[:, :2], axis=0),
                    "top": float(np.percentile(top[:, 2], 95)),
                    "pix": pix})
    out.sort(key=lambda r: -r["n"])
    return out, agent, wrist


def nearest(objs, px):
    """The object whose pixel centroid is closest to px=(col,row)."""
    px = np.asarray(px, dtype=float)
    return min(objs, key=lambda o: np.hypot(o["px"][0] - px[0], o["px"][1] - px[1]))


def floor_region(pts, z, table_z, delta, bbox=None, zlo=0.012, zhi=0.06, min_px=300):
    """Interior floor of a container: points above the table but below its rim.

    bbox=(x0,y0,x1,y1) in pixels restricts the search to one cluster (take it from
    scene_objects).  Returns {'xy','xr','yr','z','n','pix'} in the eef frame, or None.
    This is how you find where to lower a held object into a tray or basket: xy is a
    point safely inside, z is the floor, and xr/yr are the walls to stay between.
    """
    valid = (z > 0.05) & (z < 4.0)
    m = valid & (pts[..., 2] > table_z + zlo) & (pts[..., 2] < table_z + zhi)
    if bbox is not None:
        box = np.zeros_like(z, bool)
        box[int(bbox[1]):int(bbox[3]) + 1, int(bbox[0]):int(bbox[2]) + 1] = True
        m &= box
    comps = components(m, min_px=min_px)
    if not comps:
        return None
    p = max(comps, key=len)
    xyz = pts[p[:, 0], p[:, 1]] + delta
    return {"xy": np.median(xyz[:, :2], axis=0),
            "xr": (float(xyz[:, 0].min()), float(xyz[:, 0].max())),
            "yr": (float(xyz[:, 1].min()), float(xyz[:, 1].max())),
            "z": float(np.median(xyz[:, 2])), "n": int(len(p)), "pix": p}


def inside(xy, xr, yr):
    return xr[0] <= xy[0] <= xr[1] and yr[0] <= xy[1] <= yr[1]


# ---------------------------------------------------------------- motion
def goto(ctx, xyz, grip, tol=0.008, blocks=16, n=5, gain=GAIN, cap=0.9):
    """Gentle closed-loop move: never parks on a saturated command for long."""
    tgt = np.asarray(xyz, dtype=float)
    st = observe(ctx)
    p = st["p"]
    for _ in range(blocks):
        if max(abs(tgt - p)) < tol:
            return p
        row = np.zeros(7)
        row[:3] = np.clip((tgt - p) / gain, -cap, cap)
        row[6] = grip
        p = step(ctx, [row], n)["p"]
    print("goto stopped at", np.round(p, 3), "target", np.round(tgt, 3), flush=True)
    return p


def descend(ctx, xy, z_min, grip=OPEN, dz=0.008, max_blocks=25, tol=0.003):
    """Lower z until the arm stops following. Returns (z, stalled).

    A stall is the top of whatever is under the finger pads (table, tray rim, box).
    A 7x4 cm box straddled by an open jaw does NOT stall the descent - the pads pass
    either side of it, so 'no stall' is not 'missed the object'.
    """
    st = observe(ctx)
    prev = st["p"]
    for _ in range(max_blocks):
        tgt = np.array([xy[0], xy[1], max(prev[2] - dz, z_min)], dtype=float)
        p = goto(ctx, tgt, grip, tol=tol, blocks=1, n=3)
        if abs(p[2] - tgt[2]) > 0.004 and abs(p[2] - prev[2]) < 0.002:
            print("stall at z", round(p[2], 4), flush=True)
            return p[2], True
        prev = p
        if p[2] <= z_min + 0.002:
            break
    return p[2], False


def close(ctx, n=8):
    return push(ctx, [0, 0, 0, 0, 0, 0, CLOSE], n)["jaw"]


def open_(ctx, n=6):
    return push(ctx, [0, 0, 0, 0, 0, 0, OPEN], n)["jaw"]


def unwind(ctx, blocks=8, n=4):
    """Spin about the tool axis to walk q6 off its limit; also lines the jaws up."""
    st = None
    for _ in range(blocks):
        st = push(ctx, [0, 0, 0, 0, 0, 1.0, OPEN], n)
        if 0.2 < st["q"][5] < 3.4:
            break
    print("after unwind", np.round(st["p"], 3), "q", np.round(st["q"], 2), flush=True)
    return st


def spin_jaws(ctx, axis=(0, 1, 0), blocks=8, n=4):
    """Spin the hand about its approach axis until the jaw axis R[:,1] lines up with `axis`."""
    axis = np.asarray(axis, dtype=float)
    st = observe(ctx)
    for _ in range(blocks):
        if abs(abs(st["R"][:, 1] @ axis) - 1) < 0.08:
            break
        st = push(ctx, [0, 0, 0, 0, 0, 0.5, OPEN], n)
    print("jaw axis", np.round(st["R"][:, 1], 2), "q", np.round(st["q"], 2), flush=True)
    return st
