import os, sys, json
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import plan_close_drawer as P
import policy_test as PT

rows = []
orig_step = P.STATE["env"]

env, obs = P.make_env(0)
# wrap env.step to record
class Rec:
    def __init__(self, env):
        self.env = env
        self.rows = []
    def step(self, a):
        self.rows.append([float(x) for x in np.asarray(a).ravel()])
        return self.env.step(a)
    def __getattr__(self, k):
        return getattr(self.env, k)

rec = Rec(env)
ok, n, q = PT.policy(rec)
print("ok", ok, "steps", n, "q", q)
print(json.dumps(rec.rows))
