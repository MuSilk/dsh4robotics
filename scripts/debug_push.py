import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_close_drawer import make_env, qpos, eef, move_to, finger_pos

env, obs = make_env(0)
m = env.sim.model
d = env.sim.data

print("initial q", qpos(env))
for _ in range(8):
    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
move_to(env, [0.0, 0.02, 1.30], 1.0, max_steps=30)
move_to(env, [0.0, 0.02, 1.10], 1.0, max_steps=30)
print("at approach eef", np.round(eef(env), 4), "q", round(qpos(env), 4))
print("fingertip1", np.round(d.site_xpos[m.site_name2id("gripper0_finger_joint1_tip")], 4) if "gripper0_finger_joint1_tip" in m.site_names else "no site")
print("leftfinger", np.round(d.body_xpos[m.body_name2id("gripper0_leftfinger")], 4))
print("rightfinger", np.round(d.body_xpos[m.body_name2id("gripper0_rightfinger")], 4))

# push slowly
for i in range(40):
    env.step(np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 1.0]))
    if i % 4 == 0:
        print(f"i={i} eef {np.round(eef(env),4)} q {qpos(env):.4f}")

print("final eef", np.round(eef(env), 4), "q", round(qpos(env), 4))
print("joints", np.round(env.robots[0]._joint_positions, 3))
# contacts involving cabinet_top
cab_body = m.body_name2id("wooden_cabinet_1_cabinet_top")
print("--- contacts involving top drawer ---")
seen = set()
for i in range(d.ncon):
    c = d.contact[i]
    b1 = m.geom_bodyid[c.geom1]
    b2 = m.geom_bodyid[c.geom2]
    names = (m.body_id2name(b1), m.body_id2name(b2))
    if cab_body in (b1, b2):
        key = (names, m.geom_id2name(c.geom1) if m.geom_id2name(c.geom1) else c.geom1,
               m.geom_id2name(c.geom2) if m.geom_id2name(c.geom2) else c.geom2)
        if key in seen:
            continue
        seen.add(key)
        print("  ", names, "geoms", m.geom_id2name(c.geom1), c.geom1, m.geom_id2name(c.geom2), c.geom2,
              "dist", round(c.dist, 5), "pos", np.round(c.pos, 4))
print("--- contact count", d.ncon)
