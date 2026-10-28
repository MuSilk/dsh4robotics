import os, sys
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import plan_close_drawer as P
from policy_test import policy
res = []
for sid in range(50):
    env, obs = P.make_env(sid)
    ok, n, q = policy(env)
    res.append((sid, ok, n, q))
    print(f"state {sid}: success={ok} steps={n} q={q:+.4f}", flush=True)
    env.close()
print("FAILED:", [r for r in res if not r[1]])
