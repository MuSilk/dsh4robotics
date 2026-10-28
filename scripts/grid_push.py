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
    for _ in range(max_steps):
        d = target - eef(env)
        if np.linalg.norm(d) < tol:
            break
        a = np.clip(d / k, -1.0, 1.0)
        env.step(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, gripper]))


def trial(sid, x, z, speed, close_first=True, n_push=60, verbose=False):
    env, obs = P.make_env(sid)
    if close_first:
        for _ in range(8):
            env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIP]))
    move_to(env, [x, 0.04, 1.28], GRIP)
    move_to(env, [x, 0.04, z], GRIP)
    if verbose:
        print(f"  at approach x={x} z={z} eef={np.round(eef(env),4)} q={P.qpos(env):+.4f}")
    n = 0
    while n < n_push:
        env.step(np.array([0.0, -speed, 0.0, 0.0, 0.0, 0.0, GRIP]))
        n += 1
        if env.check_success():
            break
    q = P.qpos(env)
    ok = env.check_success()
    if verbose:
        print(f"  end eef={np.round(eef(env),4)} q={q:+.4f} ok={ok} steps={n}")
    env.close()
    return ok, q


if __name__ == "__main__":
    grid = [
        (0.020, 1.090, 0.5),
        (0.020, 1.090, 0.8),
        (0.020, 1.140, 0.5),
        (0.020, 1.040, 0.5),
        (0.000, 1.090, 0.5),
        (0.024, 1.090, 0.5),
    ]
    for x, z, sp in grid:
        ok, q = trial(48, x, z, sp, verbose=False)
        print(f"state48 x={x} z={z} speed={sp}: ok={ok} q={q:+.4f}")
