#!/usr/bin/env python3
"""Look at the scene before touching it: pose, DELTA + its checks, table height, and one
upscaled crop per object on the table so you can read the label yourself.

    python3 code/inspect.py

Writes obj00.png, obj01.png, ... (4x) into the cwd and prints, per object, its pixel
centroid, its centre and top in the eef/base frame, and its pixel bbox.
No motion, no vision model - just depth, pixels and your own eyes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig  # noqa: E402


def main():
    ctx = rig.connect("inspect")
    st = rig.observe(ctx)
    print("instruction:", rig.tool(ctx, "libero_status")["instruction"])
    print("eef", np.round(st["p"], 4), "jaw", round(st["jaw"], 4),
          "step", st["step"], "remaining", st["remaining"], flush=True)
    print("R[:,1] (jaw axis)", np.round(st["R"][:, 1], 3),
          "R[:,2] (approach)", np.round(st["R"][:, 2], 3), flush=True)

    delta, table_z, checks = rig.solve_delta(ctx)
    print("DELTA", np.round(delta, 3), "table_z(eef)", round(table_z + delta[2], 4),
          "checks", checks, flush=True)

    objs, agent, wrist = rig.scene_objects(ctx, delta, table_z)
    print("%d clusters on the table" % len(objs), flush=True)
    for i, o in enumerate(objs):
        name = "obj%02d.png" % i
        rig.crop(agent, o["bbox"], name, scale=4)
        print("%s n=%-6d px=(%4.0f,%4.0f) bbox=%s xy=%s top=%.3f"
              % (name, o["n"], o["px"][0], o["px"][1], o["bbox"],
                 np.round(o["xy"], 3), o["top"]), flush=True)
    rig.crop(agent, (0, 0, agent.shape[1] - 1, agent.shape[0] - 1), "scene.png", scale=1, pad=0)
    print("also wrote scene.png (full agentview)", flush=True)


if __name__ == "__main__":
    main()
