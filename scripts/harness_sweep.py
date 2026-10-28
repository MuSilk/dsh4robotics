import os, sys
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller_harness import run
bad = []
for sid in range(1, 25):
    ok, n, q = run(sid)
    print(f"controller state {sid}: success={ok} steps={n} q={q:+.4f}", flush=True)
    if not ok:
        bad.append(sid)
print("FAILED", bad)
