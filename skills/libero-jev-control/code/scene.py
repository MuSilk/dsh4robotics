#!/usr/bin/env python3
"""DEPRECATED - kept only so old imports still resolve.

The original example in this file hardcoded the agentview camera position
(T_CAM = 0.637, -0.012, 1.659) and unprojected with it.  That was wrong for this
episode: the reported camera translation is offset from the arm frame by a constant
DELTA that has to be solved per episode.  Do not use T_CAM, and do not colour-match
for objects - use `rig` instead:

    import rig
    delta, table_z, ok = rig.solve_delta(ctx)   # eef_pt = depth_pt + delta
    objs, agent, wrist = rig.scene_objects(ctx, delta)
    rig.goto(ctx, [x, y, z], rig.OPEN)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp import SOCK, Ctx, connect, tool  # noqa: E402,F401

OPEN, CLOSE = -1.0, 1.0
