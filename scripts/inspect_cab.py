import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numpy as np
import robosuite as rs
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

suite = benchmark.get_benchmark_dict()["libero_90"]()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = ControlEnv(
    bddl_file_name=bddl,
    robots=["Panda"],
    controller="OSC_POSE",
    use_camera_obs=False,
    has_renderer=False,
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=310,
    ignore_done=True,
)
env.seed(0)
env.reset()
states = suite.get_task_init_states(0)
obs = env.set_init_state(states[0])
for _ in range(5):
    obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))

u = env
print("eef", np.round(obs["robot0_eef_pos"], 4))
print("robot base", np.round(u.robots[0].base_pos, 4))
m = u.sim.model
d = u.sim.data
print("--- bodies with cabinet ---")
for i, b in enumerate(m.body_names):
    if "cabinet" in b:
        print("  ", b, np.round(d.body_xpos[i], 4))
print("--- joints with cabinet ---")
for i, j in enumerate(m.joint_names):
    if "cabinet" in j:
        print("  ", i, j, "qpos", np.round(d.qpos[m.jnt_qposadr[i]], 4), "range", m.jnt_range[i])
print("--- sites with top ---")
for i, s in enumerate(m.site_names):
    if "top" in s or "cabinet" in s:
        print("  ", s, np.round(d.site_xpos[i], 4))
print("--- top drawer geoms ---")
for i, g in enumerate(m.geom_names):
    b = m.body_id2name(m.geom_bodyid[i])
    if b and ("cabinet_top" in b):
        print("  ", i, g, "body", b, "pos", np.round(d.geom_xpos[i], 4), "size", np.round(m.geom_size[i], 4), "type", m.geom_type[i])
print("--- handle bodies ---")
for i, b in enumerate(m.body_names):
    if "handle" in b:
        print("  ", b, np.round(d.body_xpos[i], 4))

for i, g in enumerate(m.geom_names):
    if g and "cabinet" in g:
        print("  ", i, g, "body", m.body_id2name(m.geom_bodyid[i]), "pos", np.round(d.geom_xpos[i], 4),
              "size", np.round(m.geom_size[i], 4))
print("success", u.check_success())
print("obj_of_interest", u.obj_of_interest)
