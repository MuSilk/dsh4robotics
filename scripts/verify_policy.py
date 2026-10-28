import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_close_drawer as P

CLOSE_GRIP = 1.0


def obs_eef(env):
    return np.array(env.sim.data.site_xpos[env.sim.model.site_name2id("gripper0_grip_site")])


def move_to(env, target, gripper, max_steps=25, tol=0.006):
    target = np.asarray(target, float)
    for _ in range(max_steps):
        cur = obs_eef(env)
        d = target - cur
        if np.linalg.norm(d) < tol:
            break
        a = np.clip(d / 0.03, -1.0, 1.0)
        _, _, _, _ = env.step(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, gripper]))


def policy(env, verbose=False):
    steps = 0
    # 1. close gripper
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, CLOSE_GRIP]))
        steps += 1
    # 2. go above, in front of the top drawer handle
    move_to(env, [0.0, 0.03, 1.26], CLOSE_GRIP)
    steps += 1
    # 3. descend to push height
    move_to(env, [0.0, 0.03, 1.10], CLOSE_GRIP)
    if verbose:
        print("  approach eef", np.round(obs_eef(env), 4), "q", round(P.qpos(env), 4))
    # 4. push in -y until success or budget
    rows = 0
    while rows < 60:
        _, _, done, _ = env.step(np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, CLOSE_GRIP]))
        rows += 1
        if env.check_success():
            return True, steps + rows
        if done:
            break
    if verbose:
        print("  after push1 eef", np.round(obs_eef(env), 4), "q", round(P.qpos(env), 4))
    # 5. backup: retry slightly lower
    move_to(env, [0.0, 0.05, 1.05], CLOSE_GRIP)
    rows2 = 0
    while rows2 < 60:
        env.step(np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, CLOSE_GRIP]))
        rows2 += 1
        if env.check_success():
            return True, steps + rows + rows2
    return False, steps + rows + rows2


for sid in [0, 47, 48, 49]:
    env, obs = P.make_env(sid)
    ok, n = policy(env, verbose=True)
    print(f"state {sid}: success={ok} steps~{n} q={P.qpos(env):+.4f}")
    env.close()
