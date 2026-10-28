import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_close_drawer import make_env, qpos, eef

env, obs = make_env(0)
suite = __import__("plan_close_drawer").STATE["suite"]
m = env.sim.model
d = env.sim.data
states = suite.get_task_init_states(0)
jnt = m.joint_name2id("wooden_cabinet_1_top_level")
qadr = m.jnt_qposadr[jnt]
mid = m.body_name2id("wooden_cabinet_1_cabinet_top")
base = m.body_name2id("wooden_cabinet_1_base")
handle_site = m.site_name2id("wooden_cabinet_1_top_region")
print("id  cab_base(x,y,yaw)          drawer_q  eef0")
for i in range(len(states)):
    obs = env.set_init_state(states[i])
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    yaw = d.body_xmat[base].reshape(3, 3)
    yawz = np.arctan2(yaw[1, 0], yaw[0, 0])
    print(f"{i:2d}  {np.round(d.body_xpos[base][[0,1]],3)}  yaw={yawz:+.3f}  q={d.qpos[qadr]:+.4f}  "
          f"eef={np.round(obs['robot0_eef_pos'],3)}  reg={np.round(d.site_xpos[handle_site],3)}")
