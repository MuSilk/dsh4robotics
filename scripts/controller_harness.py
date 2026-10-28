"""Run code/close_drawer/controller.py against the local (renderless) LIBERO env
using the exact MCP payload shapes the worker returns."""
import os
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import sys
import json
import importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan_close_drawer as P

CTRL = os.path.join(HERE, "..", "code", "close_drawer", "controller.py")


def load_module():
    spec = importlib.util.spec_from_file_location("close_drawer_controller", CTRL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LocalWorker:
    def __init__(self, init_state_id, hard_cap=300):
        self.env, obs = P.make_env(init_state_id)
        self.step_id = 0
        self.hard_cap = hard_cap
        self.task_complete = bool(self.env.check_success())
        self.episode_ended = self.task_complete
        self.obs = obs
        self.init_state_id = init_state_id

    def _observe(self):
        o = self.obs
        return {
            "step_id": self.step_id,
            "instruction": "close the top drawer of the cabinet",
            "eef_pos": [float(x) for x in o["robot0_eef_pos"]],
            "eef_quat_xyzw": [float(x) for x in o["robot0_eef_quat"]],
            "gripper_qpos": [float(x) for x in o["robot0_gripper_qpos"]],
            "joint_pos": [float(x) for x in o["robot0_joint_pos"]],
            "task_complete": self.task_complete,
            "episode_ended": self.episode_ended,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "init_state_id": self.init_state_id,
            "cameras": ["agentview", "robot0_eye_in_hand"],
            "frames": {"agentview": "frame.jpg", "robot0_eye_in_hand": "wrist.jpg"},
        }

    def call(self, name, **arguments):
        if name == "libero_observe":
            payload = self._observe()
        elif name == "libero_step":
            rows = arguments["actions"]
            executed = 0
            for row in rows:
                if self.step_id >= self.hard_cap or self.episode_ended:
                    break
                self.obs, _, done, _ = self.env.step(np.array(row, dtype=float))
                self.step_id += 1
                executed += 1
                self.task_complete = bool(self.env.check_success())
                if self.task_complete or done or self.step_id >= self.hard_cap:
                    self.episode_ended = True
                    break
            payload = {
                "steps_executed": executed,
                "step_id": self.step_id,
                "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
                "episode_ended": self.episode_ended,
                "task_complete": self.task_complete,
                "reason": "success" if self.task_complete else "",
                "observation": self._observe(),
            }
        else:
            raise ValueError(name)
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}


class Ctx:
    def __init__(self, worker):
        self.worker = worker

    def call(self, name, **arguments):
        return self.worker.call(name, **arguments)


def run(init_state_id):
    worker = LocalWorker(init_state_id)
    mod = load_module()
    mod.main(Ctx(worker))
    ok = worker.task_complete
    q = P.qpos(worker.env)
    worker.env.close()
    return ok, worker.step_id, q


if __name__ == "__main__":
    for sid in [0, 47, 48, 49]:
        ok, n, q = run(sid)
        print(f"controller state {sid}: success={ok} steps={n} q={q:+.4f}", flush=True)
