import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_close_drawer as P

env, obs = P.make_env(0)
m = env.sim.model
d = env.sim.data


def dump(tag):
    out = {}
    for b in ["gripper0_leftfinger", "gripper0_rightfinger", "gripper0_eef"]:
        out[b] = d.body_xpos[m.body_name2id(b)]
    print(tag, {k: np.round(v, 4) for k, v in out.items()})
    for g in m.geom_names:
        if g and "finger" in g:
            i = m.geom_name2id(g)
            print("   geom", g, "body", m.body_id2name(m.geom_bodyid[i]),
                  "pos", np.round(d.geom_xpos[i], 4), "size", np.round(m.geom_size[i], 4),
                  "type", m.geom_type[i])


dump("open")
for _ in range(10):
    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
dump("closed")
print("eef site", np.round(d.site_xpos[m.site_name2id("gripper0_grip_site")], 4))
for s in m.site_names:
    if "grip" in s or "tip" in s:
        print("site", s, np.round(d.site_xpos[m.site_name2id(s)], 4))
