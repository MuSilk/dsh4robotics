import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_close_drawer as P

GRIP = 1.0


def eef(env):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])


def move_to(env, target, gripper, max_steps=30, tol=0.004, k=0.02):
    target = np.asarray(target, float)
    used = 0
    for _ in range(max_steps):
        d = target - eef(env)
        if np.linalg.norm(d) < tol:
            break
        a = np.clip(d / k, -1.0, 1.0)
        env.step(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, gripper]))
        used += 1
    return used


def push(env, dy, n_max):
    used = 0
    for _ in range(n_max):
        env.step(np.array([0.0, dy, 0.0, 0.0, 0.0, 0.0, GRIP]))
        used += 1
        if env.check_success():
            break
    return used


def policy(env, verbose=False):
    steps = 0
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP]))
        steps += 1
    steps += move_to(env, [0.020, 0.040, 1.28], GRIP)
    steps += move_to(env, [0.020, 0.040, 1.14], GRIP)
    steps += push(env, -0.5, 60)
    if env.check_success():
        return True, steps, P.qpos(env)
    # fallback: lower push
    steps += move_to(env, [0.020, 0.050, 1.04], GRIP)
    steps += push(env, -0.5, 60)
    if env.check_success():
        return True, steps, P.qpos(env)
    # fallback 2: higher push
    steps += move_to(env, [0.020, 0.050, 1.16], GRIP)
    steps += push(env, -0.5, 60)
    return env.check_success(), steps, P.qpos(env)


if __name__ == "__main__":
    for sid in [0, 47, 48, 49]:
        env, obs = P.make_env(sid)
        ok, n, q = policy(env, verbose=True)
        print(f"state {sid}: success={ok} steps={n} q={q:+.4f}")
        env.close()
