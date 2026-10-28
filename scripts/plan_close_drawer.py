import os
import sys
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numpy as np
import robosuite as rs
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

STATE = {"suite": None, "env": None, "task": None}


def make_env(init_state_id=0):
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
    obs = env.set_init_state(states[init_state_id])
    for _ in range(5):
        obs, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
    STATE.update(suite=suite, env=env, task=task)
    return env, obs


def qpos(env):
    d = env.sim.data
    m = env.sim.model
    i = m.joint_name2id("wooden_cabinet_1_top_level")
    return float(d.qpos[m.jnt_qposadr[i]])


def eef(env):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])


def finger_pos(env, side):
    body = f"gripper0_{side}finger"
    return np.array(env.sim.data.get_body_xpos(body))


def move_to(env, target, gripper, max_steps=40, tol=0.004, scale=1.0, log=False):
    target = np.asarray(target, dtype=float)
    for i in range(max_steps):
        cur = eef(env)
        d = target - cur
        if np.linalg.norm(d) < tol:
            break
        a = np.clip(d / 0.05, -scale, scale)
        obs, _, _, _ = env.step(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, gripper]))
        if log:
            print(f"   step {i} eef {np.round(eef(env),4)} q {qpos(env):.4f}")
    return eef(env)


def run(push_z=1.09, approach_y=0.02, push_y=-0.21, close_first=8, verbose=False):
    env, obs = make_env(0)
    # close gripper
    for _ in range(close_first):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    if verbose:
        print("after close eef", np.round(eef(env), 4), "q", round(qpos(env), 4))
        print("finger R", np.round(finger_pos(env, "right"), 4), "L", np.round(finger_pos(env, "left"), 4))
    move_to(env, [0.0, approach_y, 1.30], 1.0, max_steps=30, log=verbose)
    move_to(env, [0.0, approach_y, push_z], 1.0, max_steps=30, log=verbose)
    move_to(env, [0.0, push_y, push_z], 1.0, max_steps=40, scale=0.8, log=verbose)
    # hold and press
    for _ in range(10):
        env.step(np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 1.0]))
    q = qpos(env)
    ok = env.check_success()
    if verbose:
        print("final eef", np.round(eef(env), 4), "q", round(q, 4), "success", ok)
    env.close()
    return q, ok


if __name__ == "__main__":
    q, ok = run(verbose=True)
    print("RESULT", q, ok)
