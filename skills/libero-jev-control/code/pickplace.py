#!/usr/bin/env python3
"""Pick-and-place recipes built on rig: the whole episode is usually three calls.

    import rig, pickplace as pp
    ctx = rig.connect("ctl")
    delta, table_z, checks = rig.solve_delta(ctx)

    pp.pick_at(ctx, target_xy, top_z=0.462)          # descend, grip, lift, reads the jaw
    pp.place_into(ctx, tray_xy, z_release=0.49)      # carry over, lower, release, retreat

Measured on task 56 (butter -> tray), 219 steps including two wasted calibrations:
pick 40 steps, carry+place 45 steps.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402

GOOD_GRASP = 0.010      # jaw above this after closing == holding something (0.005 == air)


def pick_at(ctx, xy, top_z, grasp_below_top=0.012, hover=0.14, lift=0.13,
            attempts=2, good=GOOD_GRASP):
    """Grasp the object whose top is at `top_z` and whose centre is at `xy`.

    Descends with the jaws open to top_z - grasp_below_top, closes, checks the jaw,
    lifts. If the jaw says air, opens, drops another centimetre and tries once more.
    Returns {'ok', 'jaw', 'z_grasp', 'stalled'}.
    """
    xy = np.asarray(xy, dtype=float)
    rig.goto(ctx, [xy[0], xy[1], top_z + hover], rig.OPEN)
    z_grasp = top_z - grasp_below_top
    stalled = False
    info = {"ok": False, "jaw": None, "z_grasp": z_grasp, "stalled": stalled}
    for attempt in range(attempts):
        z, stalled = rig.descend(ctx, xy, z_grasp)
        if stalled and z > z_grasp + 0.004:          # something taller than expected
            z_grasp = z - 0.010
            rig.goto(ctx, [xy[0], xy[1], z_grasp], rig.OPEN, tol=0.004, blocks=6)
        jaw = rig.close(ctx, 8)
        info.update(jaw=jaw, z_grasp=z_grasp, stalled=stalled)
        print("attempt", attempt, "z", round(z_grasp, 3), "jaw", round(jaw, 4), flush=True)
        if jaw >= good:
            info["ok"] = True
            break
        rig.open_(ctx, 5)
        z_grasp -= 0.010
        rig.goto(ctx, [xy[0], xy[1], z_grasp], rig.OPEN, tol=0.004, blocks=6)
    rig.goto(ctx, [xy[0], xy[1], z_grasp + lift], rig.CLOSE, blocks=10)
    info["jaw"] = rig.observe(ctx)["jaw"]
    info["ok"] = info["ok"] and info["jaw"] >= good
    return info


def place_into(ctx, xy, z_release, travel_z=0.58, retreat=0.14, hold_open=6):
    """Carry at travel_z, lower to z_release inside the container, open, retreat straight up.

    travel_z must clear the tallest thing on the way (tray rims here are 0.48-0.52).
    """
    xy = np.asarray(xy, dtype=float)
    rig.goto(ctx, [xy[0], xy[1], max(travel_z, rig.observe(ctx)["p"][2])], rig.CLOSE, blocks=16)
    rig.goto(ctx, [xy[0], xy[1], z_release], rig.CLOSE, tol=0.004, blocks=10)
    jaw = rig.open_(ctx, hold_open)
    p = rig.goto(ctx, [xy[0], xy[1], z_release + retreat], rig.OPEN, blocks=12)
    print("released at", np.round(p, 3), "jaw", round(jaw, 4), flush=True)
    return {"p": p, "jaw": jaw}


def lift_to(ctx, z, xy=None):
    """Straight-up move that keeps the current xy (use before travelling anywhere)."""
    p = rig.observe(ctx)["p"]
    xy = p[:2] if xy is None else np.asarray(xy)
    return rig.goto(ctx, [xy[0], xy[1], z], rig.CLOSE)
