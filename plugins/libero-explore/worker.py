"""LIBERO exploration service.

The server opens the first round. The agent resets onto a random scene,
spends a shared step budget, and submits one controller. That program is
then rolled out on every initial scene, with no agent in the loop.

LIBERO_TASK_ID=all is the same loop over every LIBERO-90 task. reset samples
a task and a scene. The task index is hidden; the instruction is not.

Robot tools use LIBERO's official 7-D OSC_POSE action: libero_status,
libero_observe, libero_cameras, libero_depth, libero_step, libero_hold.
Agent-only tools: status, score, reset, run, submit.
"""
import ast
import base64
import hashlib
import json
import math
import os
import queue
import random
import shutil
import socket
import subprocess
import sys
import threading
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", os.environ.get("LIBERO_GPU", "0"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("LIBERO_GPU", "0"))
# The worker's HOME is the agent workspace, so ~/.libero is not the real config.
# Without this, the first status() stops on LIBERO's dataset-path prompt.
os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/changzou/.libero")

import cv2
import numpy as np

MAX_ROWS = 50
SETTLE_STEPS = 5
CONTROLLER_TOOLS = frozenset({
    "status", "libero_status", "libero_observe", "libero_cameras", "libero_depth",
    "libero_step", "libero_hold",
})
CONTROLLER_TIMEOUT = 900
DEPTH_CAMERA = "agentview"
CAMERA_SIZE = 512
ACTION_LAYOUT = {
    "dim": 7,
    "control": "20 Hz OSC_POSE, the official LIBERO / robosuite Panda action",
    "semantics": "normalized deltas, not absolute poses. A row of zeros holds the arm and commands the gripper to mid-range; it is not a grasp hold. libero_hold repeats the previous row.",
    "fields": [
        {"index": 0, "name": "dx", "frame": "robot base", "range": [-1, 1], "at_1": "0.05 m"},
        {"index": 1, "name": "dy", "frame": "robot base", "range": [-1, 1], "at_1": "0.05 m"},
        {"index": 2, "name": "dz", "frame": "robot base", "range": [-1, 1], "at_1": "0.05 m"},
        {"index": 3, "name": "droll", "range": [-1, 1], "at_1": "0.5 rad"},
        {"index": 4, "name": "dpitch", "range": [-1, 1], "at_1": "0.5 rad"},
        {"index": 5, "name": "dyaw", "range": [-1, 1], "at_1": "0.5 rad"},
        {"index": 6, "name": "gripper", "range": [-1, 1], "note": "negative opens, positive closes"},
    ],
}
STATE_LAYOUT = {
    "fields": [
        {"name": "eef_pos", "dim": 3, "unit": "m", "frame": "robot base"},
        {"name": "eef_quat_xyzw", "dim": 4, "note": "robosuite xyzw, not wxyz"},
        {"name": "gripper_qpos", "dim": 2},
        {"name": "joint_pos", "dim": 7, "unit": "rad"},
    ],
    "cameras": ["agentview", "robot0_eye_in_hand"],
}


def all_state_ids(count):
    if count < 1:
        raise RuntimeError(f"task has {count} init states")
    return list(range(count))


def parse_eval_inits(raw):
    """None means every initial scene. A positive int is how many per task."""
    text = str(raw if raw is not None else "all").strip().lower()
    if text in ("", "all"):
        return None
    if not text.isdigit() or int(text) < 1:
        raise RuntimeError("eval init limit must be 'all' or a positive integer")
    return int(text)


def all_episodes(counts, per_task=None):
    """(task, init state) pairs. per_task keeps the first N scenes of each task."""
    if per_task is not None and int(per_task) < 1:
        raise RuntimeError("eval init limit must be 'all' or a positive integer")
    episodes = []
    for task_id, count in enumerate(counts):
        if count < 1:
            raise RuntimeError(f"task {task_id} has {count} init states")
        kept = count if per_task is None else min(int(per_task), count)
        for state_id in range(kept):
            episodes.append((task_id, state_id))
    if not episodes:
        raise RuntimeError("suite has no episodes")
    return episodes


def episode_cap(horizon, budget_total, budget_used):
    remaining = budget_total - budget_used
    if remaining <= 0:
        raise RuntimeError("exploration budget is exhausted; submit the controller")
    return min(horizon, remaining)


AGENT_HIDDEN = {"init_state_id", "explore_init_states", "visited", "held_out_count"}
CONTROLLER_HIDDEN = AGENT_HIDDEN | {"budget", "budget_used", "budget_remaining", "budget_total"}


def redact(value, hidden):
    """Drop scene-index fields before a result reaches the model or a controller."""
    if isinstance(value, dict):
        return {
            key: redact(item, hidden)
            for key, item in value.items()
            if key not in hidden
        }
    if isinstance(value, list):
        return [redact(item, hidden) for item in value]
    return value


def _log_tail(path, limit=2000):
    path = Path(path)
    if not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", "replace")


def _bind_if_present(command, source, writable=False):
    if Path(source).exists():
        command.extend(["--bind" if writable else "--ro-bind", source, source])


def _imajev_socket(socket_path):
    parent = Path(socket_path).resolve().parent
    path = parent / "imajev.sock"
    if len(str(path)) < 108:
        return str(path)
    return str(parent / "ij.sock")


def controller_env():
    """Environment of the bwrap process. The sandbox child only keeps --setenv."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def controller_command(python, runtime, project, code_root, socket_path):
    """Run the controller where it cannot read the held-out init files.

    The process sees its code, the interpreter, and the robot socket.
    LIBERO's init-state directory is not mounted.
    """
    python_path = Path(python).resolve()
    env_root = python_path
    for parent in python_path.parents:
        if parent.name == "bin" and (parent.parent / "lib").is_dir():
            env_root = parent.parent
            break
    sock_dir = str(Path(socket_path).resolve().parent)
    command = [
        "bwrap",
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
    ]
    for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        _bind_if_present(command, source)
    resolv = Path("/etc/resolv.conf")
    if resolv.is_symlink():
        _bind_if_present(command, os.path.realpath(resolv))
    command.extend(["--ro-bind", str(env_root), str(env_root)])
    plugin = str(Path(runtime).resolve().parent.parent)
    command.extend([
        "--ro-bind", plugin, plugin,
        "--bind", code_root, code_root,
        "--bind", sock_dir, sock_dir,
        "--chdir", project,
        "--setenv", "PATH", f"{python_path.parent}:/usr/bin:/bin",
        "--setenv", "HOME", "/tmp",
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PYTHONUNBUFFERED", "1",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--setenv", "PYTHONPATH", code_root,
        "--setenv", "LIBERO_SOCKET", socket_path,
        "--setenv", "IMAJEV_SOCKET", _imajev_socket(socket_path),
        "--setenv", "MPLCONFIGDIR", "/tmp",
        "--unsetenv", "LIBERO_CONFIG_PATH",
        "--unsetenv", "LIBERO_RUN_DIR",
        "--unsetenv", "LIBERO_WORKSPACE",
        "--",
        python,
        runtime,
        "run",
        project,
        "controller.py",
    ])
    return command


def only(arguments, allowed):
    unknown = sorted(set(arguments) - set(allowed))
    if unknown:
        raise ValueError("unexpected arguments: " + ", ".join(unknown))


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vec(value):
    if value is None:
        return []
    return [round(float(item), 5) for item in np.asarray(value).reshape(-1)]


def metric_depth(sim, depth_buffer):
    """MuJoCo's [0, 1] depth buffer to metres along the camera optical axis."""
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    depth = np.clip(np.asarray(depth_buffer, dtype=np.float64), 0.0, 1.0)
    return near / (1.0 - depth * (1.0 - near / far))


def validate_actions(actions):
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ROWS:
        raise ValueError(f"actions must be a list of 1..{MAX_ROWS} rows")
    rows = []
    for row in actions:
        if not isinstance(row, list) or len(row) != 7 or not all(finite(item) for item in row):
            raise ValueError("each action row must have 7 finite numbers")
        if any(not -1.0 <= float(item) <= 1.0 for item in row):
            raise ValueError("action values must be inside [-1, 1]")
        rows.append([float(item) for item in row])
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_frame(path: Path, image) -> bool:
    if image is None:
        return False
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return False
    frame = np.ascontiguousarray(np.flipud(frame))
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        return False
    tmp = path.with_name(path.name + ".partial")
    tmp.write_bytes(encoded.tobytes())
    tmp.replace(path)
    return True


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    files = [path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts]
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def require_main(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
        raise ValueError("controller.py must define main(ctx)")


class Service:
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.run_dir = Path(config["run_dir"])
        self.workspace = Path(config["workspace"]).resolve()
        self.socket_path = config["socket"]
        self.hard_cap = int(config["hard_cap"])
        self.episode_horizon = self.hard_cap
        self.budget_total = int(os.environ.get("LIBERO_BUDGET", "2400"))
        if self.budget_total < 1 or self.episode_horizon < 1:
            raise RuntimeError("budget and episode horizon must be positive")
        self.budget_used = 0
        self.eval_ids = []
        self.visited = []
        self.rounds = 0
        self.busy = False
        self.eval_rows = []
        self.eval_index = 0
        self.eval_total = 0
        self.eval_success = 0
        self.eval_error = ""
        self.eval_done = False
        self.submitted_source = ""
        self.submitted_bundle = ""
        self.task_id = int(config["task_id"])
        self.all_tasks = bool(config.get("all_tasks"))
        self.eval_init_limit = config.get("eval_init_limit")
        self._state_cache = {}
        self.wall_seconds = int(config["wall_seconds"])
        self.bundles = {}
        self.episode_opened = False
        self.phase = "idle"
        self.env = None
        self.suite = None
        self.task = None
        self.init_states = None
        self.step_id = 0
        self.init_state_id = None
        self.last_obs = None
        self.last_action = None
        self.episode_ended = False
        self.task_complete = False
        self.end_reason = ""
        self.episode_kind = None
        self.episode_id = None
        self.last_outcome = None
        self.gl_queue = queue.Queue()
        self.gl_thread = threading.Thread(target=self._gl_loop, name="libero90-gl", daemon=True)
        self.gl_thread.start()
        self._load_session()
        # A frame from the previous process is not this episode. The old file was
        # also the cross-thread EGL noise, and the page treats any file as live.
        if self.env is None:
            for name in ("frame.jpg", "wrist.jpg"):
                stale = self.run_dir / name
                if stale.exists():
                    stale.unlink()
        self.monitor = self._monitor_payload()
        self._publish()

    def _gl_loop(self):
        while True:
            job = self.gl_queue.get()
            job()

    def _on_gl(self, fn):
        """MuJoCo's EGL context is valid on one thread. Every render runs there."""
        if threading.current_thread() is self.gl_thread:
            return fn()
        box = {}
        done = threading.Event()

        def job():
            try:
                box["value"] = fn()
            except BaseException as exc:
                box["error"] = exc
            finally:
                done.set()

        self.gl_queue.put(job)
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _activate_gl(self):
        env = self.env
        if env is None:
            return
        context = getattr(env.env.sim, "_render_context_offscreen", None)
        if context is not None:
            context.gl_ctx.make_current()

    def _session_path(self):
        return self.run_dir / "session.json"

    def _load_session(self):
        path = self._session_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.bundles = data.get("bundles") or {}
        self.episode_opened = bool(data.get("episode_opened"))
        if data.get("finished") or data.get("submitted_bundle") or data.get("exploration_closed"):
            self.phase = "closed"
            self.episode_opened = True

    def _persist(self):
        write_json(self._session_path(), {
            "bundles": self.bundles,
            "bundles": self.bundles,
            "episode_opened": self.episode_opened,
            "finished": self.phase == "closed",
        })

    def _publish(self):
        self.monitor = self._monitor_payload()
        write_json(self.run_dir / "status.json", self.monitor)

    def _monitor_payload(self):
        language = self.task["language"] if self.task else ""
        name = self.task["name"] if self.task else ""
        if self.env is not None and not self.episode_ended:
            if self.task_complete:
                state, reason, success = "success", "", True
            else:
                state, reason, success = "running", "", False
        elif self.episode_ended and self.task_complete:
            state, reason, success = "success", self.end_reason, True
        elif self.episode_ended:
            state, reason, success = "failed", self.end_reason, False
        elif self.last_outcome:
            state = "success" if self.last_outcome["success"] else "failed"
            reason = self.last_outcome["reason"]
            success = self.last_outcome["success"]
        else:
            state, reason, success = "idle", "", False
        return {
            "state": state,
            "phase": self.phase,
            "success": success,
            "reason": reason,
            "step": self.step_id if self.env is not None or self.episode_ended else (self.last_outcome or {}).get("step", 0),
            "max_steps": self.hard_cap,
            "reward": 0,
            "frame": (self.run_dir / "frame.jpg").exists(),
            "task_id": self.task_id,
            "name": name,
            "language": language,
            "error": "",
            "init_state_id": self.init_state_id,
            "episode_id": self.episode_id,
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "round": self.rounds,
            "eval_index": self.eval_index,
            "eval_total": self.eval_total,
            "eval_success": self.eval_success,
            "eval_done": self.eval_done,
            "eval_error": self.eval_error,
        }

    def snapshot(self):
        with self.lock:
            return dict(self.monitor)

    def public_status(self):
        """Status the agent can fetch. The success predicate stays hidden until the episode ends."""
        with self.lock:
            payload = dict(self.monitor)
            in_progress = self.env is not None and not self.episode_ended
        if in_progress:
            payload.pop("success", None)
            payload.pop("reason", None)
            payload.pop("reward", None)
            payload["state"] = "running"
        return payload

    def call(self, name, arguments, role):
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")
        if name == "start_episode" and role != "server":
            raise PermissionError("The first round is opened by the server. Use reset for another one")
        if role == "controller" and name not in CONTROLLER_TOOLS:
            raise PermissionError(f"{name} is not available inside a controller")
        if name == "finish":
            raise RuntimeError(
                "There is no finish. reset opens another random scene. "
                "submit freezes the controller and rolls it out on every initial scene"
                + (" of every task" if self.all_tasks else "")
            )
        if name == "reset":
            return self._reset(arguments, role)
        if name == "run":
            return self._run(arguments, role)
        if name == "submit":
            return self._submit(arguments, role)
        with self.lock:
            if self.phase == "closed" and name not in ("status", "libero_status"):
                raise RuntimeError("Exploration is closed")
            if self.phase in ("formal", "evaluating") and role != "controller" and name != "status":
                if self.phase == "evaluating":
                    raise RuntimeError("Evaluation is running")
                raise RuntimeError("A controller is running; wait for it to finish")
            return self._locked(name, arguments, role)

    def _locked(self, name, arguments, role):
        if name == "status":
            only(arguments, ())
            return self._status()
        if name == "score":
            only(arguments, ())
            return self._score_locked()
        if name == "start_episode":
            only(arguments, ("init_state_id",))
            held = self._start_episode(arguments.get("init_state_id"))
            if held is not None:
                return held
            observed = self._observe()
            return observed
        if name == "register":
            only(arguments, ("source",))
            return self._register(arguments.get("source"))
        if name == "libero_status":
            only(arguments, ())
            return self._libero_status()
        if name == "libero_observe":
            only(arguments, ())
            return self._observe()
        if name == "libero_cameras":
            only(arguments, ())
            return self._cameras()
        if name == "libero_depth":
            only(arguments, ())
            return self._depth()
        if name == "libero_step":
            only(arguments, ("actions",))
            return self._step(validate_actions(arguments.get("actions")))
        if name == "libero_hold":
            only(arguments, ("steps",))
            steps = arguments.get("steps")
            if type(steps) is not int or not 1 <= steps <= MAX_ROWS:
                raise ValueError(f"steps must be an integer 1..{MAX_ROWS}")
            action = self.last_action or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            return self._step([list(action) for _ in range(steps)])
        raise PermissionError(f"Unknown tool: {name}")

    def _load_suite(self):
        if self.suite is not None:
            return
        from libero.libero import benchmark
        self.suite = benchmark.get_benchmark_dict()["libero_90"]()

    def _states(self, task_id):
        self._load_suite()
        cached = self._state_cache.get(task_id)
        if cached is None:
            cached = list(self.suite.get_task_init_states(task_id))
            if not cached:
                raise RuntimeError(f"task {task_id} has no init states")
            self._state_cache[task_id] = cached
        return cached

    def _bind_task(self, task_id):
        """Point the session at one task. A different bddl drops the current sim."""
        self._load_suite()
        if task_id < 0 or task_id >= self.suite.n_tasks:
            raise RuntimeError(f"task_id {task_id} is outside 0..{self.suite.n_tasks - 1}")
        if self.task is not None and self.task.get("task_id") == task_id and self.init_states is not None:
            return
        if self.env is not None:
            self._close_env()
        from libero.libero import get_libero_path
        task = self.suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.task_id = task_id
        self.init_states = self._states(task_id)
        self.task = {
            "suite": "libero_90",
            "task_id": task_id,
            "name": task.name,
            "language": task.language,
            "bddl": bddl,
        }

    def _ensure_task(self):
        if self.task is not None and self.init_states is not None:
            return
        if self.all_tasks:
            raise RuntimeError("no task selected")
        self._bind_task(self.task_id)

    def _public_task(self):
        """What the agent may see. In a suite session the index and the file name stay hidden."""
        if not self.task:
            return None
        if not self.all_tasks:
            return self.task
        return {"suite": "libero_90", "language": self.task["language"]}

    def _status(self):
        bundles = {
            bundle: {
                "sha256": item["sha256"],
                "source": item["source"],
            }
            for bundle, item in self.bundles.items()
        }
        episode = self._episode_view()
        if self.phase == "evaluating" and episode:
            episode = {
                "step_id": episode["step_id"],
                "episode_step_limit": episode["episode_step_limit"],
                "remaining_episode_steps": episode["remaining_episode_steps"],
                "episode_ended": episode["episode_ended"],
            }
        return {
            "task": self._public_task(),
            "episode": episode,
            "episode_limits": {"max_steps": self.hard_cap, "horizon": self.episode_horizon},
            "budget": {
                "used": self.budget_used,
                "total": self.budget_total,
                "remaining": max(0, self.budget_total - self.budget_used),
            },
            "round": self.rounds,
            "bundles": bundles,
            "phase": self.phase,
        }

    def _episode_view(self):
        if self.init_state_id is None and not self.episode_ended:
            return None
        return {
            "step_id": self.step_id,
            "episode_step_limit": self.hard_cap,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "episode_id": self.episode_id,
            "episode_ended": self.episode_ended,
            "kind": self.episode_kind,
        }

    def _libero_status(self):
        self._ensure_task()
        return {
            "instruction": self.task["language"],
            "task": self._public_task(),
            "action_layout": ACTION_LAYOUT,
            "state_layout": STATE_LAYOUT,
            "max_action_rows": MAX_ROWS,
            "episode_limits": {"max_steps": self.hard_cap, "horizon": self.episode_horizon},
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "episode": self._episode_view(),
            "cameras": ["agentview", "robot0_eye_in_hand"],
        }

    def _require_episode(self):
        if self.env is None:
            raise RuntimeError("No round is open")

    def _observe(self):
        self._require_episode()
        obs = self.last_obs or {}
        return {
            "step_id": self.step_id,
            "instruction": self.task["language"] if self.task else "",
            "eef_pos": vec(obs.get("robot0_eef_pos")),
            "eef_quat_xyzw": vec(obs.get("robot0_eef_quat")),
            "gripper_qpos": vec(obs.get("robot0_gripper_qpos")),
            "joint_pos": vec(obs.get("robot0_joint_pos")),
            "episode_ended": self.episode_ended,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "episode_id": self.episode_id,
            "budget_used": self.budget_used,
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "round": self.rounds,
            "cameras": ["agentview", "robot0_eye_in_hand"],
        }

    def _jpeg_b64(self, filename):
        path = self.run_dir / filename
        if not path.is_file():
            raise RuntimeError(f"{filename} is not ready")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _cameras(self):
        """Both current JPEGs plus the pose fields. Same pixels the monitor shows."""
        view = self._observe()
        return {
            "height": CAMERA_SIZE,
            "width": CAMERA_SIZE,
            "step_id": view["step_id"],
            "episode_id": view["episode_id"],
            "instruction": view["instruction"],
            "eef_pos": view["eef_pos"],
            "eef_quat_xyzw": view["eef_quat_xyzw"],
            "gripper_qpos": view["gripper_qpos"],
            "joint_pos": view["joint_pos"],
            "agentview_jpeg": self._jpeg_b64("frame.jpg"),
            "wrist_jpeg": self._jpeg_b64("wrist.jpg"),
        }

    def _depth(self):
        if threading.current_thread() is not self.gl_thread:
            return self._on_gl(self._depth)
        self._activate_gl()
        self._require_episode()
        import robosuite.utils.transform_utils as T
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
        )

        env = self.env.env
        sim = env.sim
        robot = env.robots[0]
        _rgb, depth = sim.render(
            camera_name=DEPTH_CAMERA,
            width=CAMERA_SIZE,
            height=CAMERA_SIZE,
            depth=True,
        )
        # frame.jpg is the OpenGL buffer flipped upright. Depth uses that same grid.
        metres = np.flipud(metric_depth(sim, depth))
        millimetres = np.clip(np.rint(metres * 1000.0), 0, 65535).astype("<u2")
        intrinsic = get_camera_intrinsic_matrix(sim, DEPTH_CAMERA, CAMERA_SIZE, CAMERA_SIZE)
        camera_to_world = get_camera_extrinsic_matrix(sim, DEPTH_CAMERA)
        base_pose = T.make_pose(
            sim.data.get_body_xpos(robot.robot_model.root_body),
            sim.data.get_body_xmat(robot.robot_model.root_body).reshape(3, 3),
        )
        camera_to_base = T.pose_inv(base_pose) @ camera_to_world
        return {
            "camera": DEPTH_CAMERA,
            "height": CAMERA_SIZE,
            "width": CAMERA_SIZE,
            "step_id": self.step_id,
            "episode_id": self.episode_id,
            "fx": round(float(intrinsic[0, 0]), 5),
            "fy": round(float(intrinsic[1, 1]), 5),
            "cx": round(float(intrinsic[0, 2]), 5),
            "cy": round(float(intrinsic[1, 2]), 5),
            "camera_to_base": [round(float(item), 6) for item in camera_to_base.reshape(-1)],
            "depth_mm_dtype": "<u2",
            "depth_mm_scale": 0.001,
            "depth_mm": base64.b64encode(np.ascontiguousarray(millimetres).tobytes()).decode("ascii"),
        }

    def _remember(self, obs):
        self.last_obs = obs
        save_frame(self.run_dir / "frame.jpg", obs.get("agentview_image") if obs else None)
        save_frame(self.run_dir / "wrist.jpg", obs.get("robot0_eye_in_hand_image") if obs else None)

    def _close_env(self):
        if threading.current_thread() is not self.gl_thread:
            return self._on_gl(self._close_env)
        env = self.env
        self.env = None
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    def _open_env(self, init_state_id, kind, task_id=None):
        if threading.current_thread() is not self.gl_thread:
            return self._on_gl(lambda: self._open_env(init_state_id, kind, task_id))
        self._bind_task(self.task_id if task_id is None else task_id)
        if init_state_id < 0 or init_state_id >= len(self.init_states):
            raise RuntimeError("sampled scene is outside this task")
        # Rebuilding OffScreenRenderEnv recompiles the model and the EGL context.
        # The same task keeps one sim and only loads the next init state.
        # A different task has already closed it in _bind_task.
        if self.env is None:
            from libero.libero.envs import OffScreenRenderEnv
            self.env = OffScreenRenderEnv(
                bddl_file_name=self.task["bddl"],
                camera_heights=CAMERA_SIZE,
                camera_widths=CAMERA_SIZE,
                horizon=self.episode_horizon + SETTLE_STEPS + 5,
                ignore_done=True,
                hard_reset=False,
            )
        env = self.env
        try:
            env.seed(0)
            self._activate_gl()
            env.reset()
            obs = env.set_init_state(self.init_states[init_state_id])
            settle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for _ in range(SETTLE_STEPS):
                self._activate_gl()
                obs, _reward, _done, _info = env.step(settle)
        except Exception:
            self._close_env()
            raise
        self.last_outcome = None
        self.init_state_id = init_state_id
        self.episode_id = f"round_{uuid.uuid4().hex[:8]}"
        self.step_id = 0
        self.last_action = None
        self.episode_ended = False
        self.task_complete = bool(env.check_success())
        self.end_reason = ""
        self.episode_kind = kind
        self._remember(obs)
        self._publish()

    def _ensure_states(self):
        if self.eval_ids:
            return
        if self.all_tasks:
            self._load_suite()
            counts = [len(self._states(task_id)) for task_id in range(self.suite.n_tasks)]
            self.eval_ids = all_episodes(counts, self.eval_init_limit)
            return
        self._ensure_task()
        self.eval_ids = [
            (self.task_id, state_id)
            for _task_id, state_id in all_episodes([len(self.init_states)], self.eval_init_limit)
        ]

    def _pick_init(self):
        if self.all_tasks:
            self._load_suite()
            task_id = random.randrange(self.suite.n_tasks)
            return task_id, random.randrange(len(self._states(task_id)))
        self._ensure_states()
        return self.task_id, random.randrange(len(self.init_states))

    def _open_explore_episode(self, task_id, init_state_id):
        self.hard_cap = episode_cap(self.episode_horizon, self.budget_total, self.budget_used)
        self.phase = "explore"
        self._open_env(init_state_id, "explore", task_id)
        self.episode_opened = True
        self.rounds += 1
        self.visited.append([task_id, init_state_id] if self.all_tasks else init_state_id)
        self._persist()
        self._publish()

    def _start_episode(self, _init_state_id=None):
        if self.phase == "closed":
            raise RuntimeError("Exploration is closed")
        if self.env is not None:
            raise RuntimeError("A round is already open")
        task_id, init_state_id = self._pick_init()
        self._open_explore_episode(task_id, init_state_id)
        return None

    def _mark_end(self, reason):
        self.episode_ended = True
        self.end_reason = reason
        self._publish()

    def _step(self, rows):
        if threading.current_thread() is not self.gl_thread:
            return self._on_gl(lambda: self._step(rows))
        self._activate_gl()
        self._require_episode()
        if self.episode_ended:
            raise RuntimeError("episode already ended")
        executed = 0
        for row in rows:
            if self.step_id >= self.hard_cap:
                self.task_complete = bool(self.env.check_success())
                self._mark_end("max_steps")
                break
            self._activate_gl()
            obs, _reward, _done, _info = self.env.step(row)
            self.last_action = list(row)
            self.step_id += 1
            executed += 1
            if self.episode_kind == "explore":
                self.budget_used += 1
            self._remember(obs)
            self.task_complete = bool(self.env.check_success())
            if self.episode_kind == "eval" and self.task_complete:
                self._mark_end("success")
                break
            if self.episode_kind == "explore" and self.budget_used >= self.budget_total:
                self._mark_end("budget")
                break
            if self.step_id >= self.hard_cap:
                self._mark_end("max_steps")
                break
            self._publish()
        ended = self.episode_ended
        if ended and self.env is not None:
            self.last_outcome = {
                "success": bool(self.task_complete),
                "reason": self.end_reason,
                "step": self.step_id,
            }
        result = {
            "steps_executed": executed,
            "step_id": self.step_id,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "budget_used": self.budget_used,
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "episode_ended": ended,
            "session_episode_over": False,
            "observation": self._observe(),
        }
        if ended:
            result["task_complete"] = bool(self.task_complete)
            result["failure"] = not self.task_complete
            result["reason"] = self.end_reason
            if result["failure"]:
                result["failure_reason"] = self._failure_reason()
        return result

    def _source_dir(self, source):
        if not isinstance(source, str) or not source:
            raise ValueError("source is required")
        root = self.workspace
        src = (root / source).resolve()
        if src.is_file() and src.name == "controller.py":
            src = src.parent
        code = (root / "code").resolve()
        if code != src and code not in src.parents:
            raise ValueError("source must be a project under code/, for example code/close_drawer")
        if not (src / "controller.py").is_file():
            raise ValueError("controller.py is required")
        require_main(src / "controller.py")
        return src

    def _register(self, source):
        if self.phase == "closed":
            raise RuntimeError("Exploration is closed")
        src = self._source_dir(source)
        digest = tree_hash(src)
        bundle = digest[:16]
        dest = self.run_dir / "bundles" / bundle / "project"
        if bundle not in self.bundles:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            self.bundles[bundle] = {
                "sha256": digest,
                "source": source,
            }
            self._persist()
        return {"bundle": bundle, "sha256": digest, "project_directory": str(dest)}

    def _agent_only(self, role):
        if role != "agent":
            raise PermissionError("only the agent can call this")

    def _idle_for_agent(self):
        if self.busy or self.phase == "formal":
            raise RuntimeError("A controller is running; wait for it to finish")
        if self.phase == "evaluating":
            raise RuntimeError("Held-out evaluation is running")
        if self.phase == "closed":
            raise RuntimeError("Exploration is closed")

    def _score_locked(self):
        if self.env is None:
            raise RuntimeError("No round is open")
        complete = bool(self.env.check_success())
        self.task_complete = complete
        return {
            "task_complete": complete,
            "failure": not complete,
            "failure_reason": [] if complete else self._failure_reason(),
            "step_id": self.step_id,
            "episode_ended": self.episode_ended,
            "reason": self.end_reason,
            "round": self.rounds,
            "budget_used": self.budget_used,
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
        }

    def _reset(self, arguments, role):
        self._agent_only(role)
        only(arguments, ())
        with self.lock:
            self._idle_for_agent()
            episode_cap(self.episode_horizon, self.budget_total, self.budget_used)
            task_id, init_state_id = self._pick_init()
            previous = self._score_locked() if self.env is not None else None
            self.episode_ended = False
            self._open_explore_episode(task_id, init_state_id)
            observed = self._observe()
            if previous is not None:
                observed["previous"] = previous
            return observed

    def _snapshot(self, source):
        src = self._source_dir(source)
        code_root = (self.workspace / "code").resolve()
        digest = tree_hash(code_root)
        bundle = digest[:16]
        dest = self.run_dir / "bundles" / bundle / "code"
        if bundle not in self.bundles:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(code_root, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            self.bundles[bundle] = {"sha256": digest, "source": source}
            self._persist()
        project = dest / src.relative_to(code_root)
        if not (project / "controller.py").is_file():
            raise RuntimeError("snapshot is missing controller.py")
        return bundle, project, dest

    def _run(self, arguments, role):
        self._agent_only(role)
        only(arguments, ("source",))
        with self.lock:
            self._idle_for_agent()
            if self.phase != "explore" or self.env is None or self.episode_ended:
                raise RuntimeError("This round has ended. reset to open another, or submit the controller")
            src = self._source_dir(arguments.get("source"))
            self.busy = True
            self.phase = "formal"
            self._publish()
            project = str(src)
            code_root = str((self.workspace / "code").resolve())
        log_path = self.run_dir / "controller.log"
        code = -1
        error = ""
        try:
            code = self._run_controller(project, code_root, log_path)
        except Exception as exc:
            error = str(exc)
        finally:
            with self.lock:
                self.busy = False
                outcome = self._score_locked() if self.env is not None else {"task_complete": False, "failure": True}
                outcome["exit_code"] = code
                outcome["log_tail"] = _log_tail(log_path)
                if error:
                    outcome["error"] = error
                if self.phase == "formal":
                    self.phase = "explore" if self.env is not None else "idle"
                self._publish()
        return outcome

    def _submit(self, arguments, role):
        self._agent_only(role)
        only(arguments, ("source",))
        with self.lock:
            self._idle_for_agent()
            self._ensure_states()
            bundle, project, code_root = self._snapshot(arguments.get("source"))
            self.submitted_bundle = bundle
            self.submitted_source = arguments.get("source") or ""
            self.phase = "evaluating"
            self.busy = True
            self.eval_total = len(self.eval_ids)
            self.eval_index = 0
            self.eval_success = 0
            self.eval_rows = []
            self.eval_done = False
            self.eval_error = ""
            self.episode_ended = False
            self._persist()
            self._publish()
            held = list(self.eval_ids)
            project_text, root_text = str(project), str(code_root)
        threading.Thread(
            target=self._evaluate,
            args=(project_text, root_text, held),
            name="libero-explore-eval",
            daemon=True,
        ).start()
        return {
            "submitted": True,
            "bundle": bundle,
            "finished": True,
            "session_episode_over": True,
            "note": self._rollout_note(),
        }

    def _evaluate(self, project, code_root, held):
        error = ""
        try:
            for index, episode in enumerate(held):
                task_id, state_id = episode
                tag = f"t{task_id}-i{state_id}"
                log_path = self.run_dir / "eval_logs" / f"{tag}.log"
                scratch = self.run_dir / "eval_scratch" / tag
                try:
                    if scratch.exists():
                        shutil.rmtree(scratch)
                    shutil.copytree(code_root, scratch, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    rel = Path(project).resolve().relative_to(Path(code_root).resolve())
                    with self.lock:
                        self.eval_index = index
                        self.hard_cap = self.episode_horizon
                        self.phase = "evaluating"
                        self._open_env(state_id, "eval", task_id)
                        self._publish()
                    code = self._run_controller(str(scratch / rel), str(scratch), log_path)
                    with self.lock:
                        success = bool(self.env.check_success()) if self.env is not None else False
                        reason = self.end_reason or ("success" if success else "returned")
                        if success:
                            self.eval_success += 1
                        self.eval_rows.append({
                            "task_id": task_id,
                            "init_state_id": state_id,
                            "language": self.task["language"] if self.task else "",
                            "success": success,
                            "steps": self.step_id,
                            "reason": reason,
                            "exit_code": code,
                        })
                        self.episode_ended = False
                        self._publish()
                except Exception as exc:
                    error = str(exc)
                    with self.lock:
                        self.eval_rows.append({
                            "task_id": task_id,
                            "init_state_id": state_id,
                            "language": self.task["language"] if self.task else "",
                            "success": False,
                            "steps": self.step_id,
                            "reason": "error",
                            "exit_code": -1,
                            "error": error,
                        })
                        self._close_env()
                        self._publish()
                    break
                finally:
                    if scratch.exists():
                        shutil.rmtree(scratch, ignore_errors=True)
        finally:
            with self.lock:
                self._close_env()
                self.busy = False
                self.phase = "closed"
                self._write_report(error)
                self._publish()

    def _rollout_note(self):
        scope = "every initial scene of every LIBERO-90 task" if self.all_tasks else "every initial scene of this task"
        return f"The controller is being rolled out on {scope}. Further robot calls will fail."

    def _write_report(self, error):
        total = len(self.eval_rows)
        successes = sum(1 for row in self.eval_rows if row.get("success"))
        self.eval_success = successes
        self.eval_error = error
        self.eval_done = True
        report = {
            "all_tasks": self.all_tasks,
            "eval_inits": self.eval_init_limit,
            "task_id": None if self.all_tasks else self.task_id,
            "language": "" if self.all_tasks else (self.task["language"] if self.task else ""),
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
            "horizon": self.episode_horizon,
            "rounds": self.rounds,
            "visited": self.visited,
            "bundle": self.submitted_bundle,
            "source": self.submitted_source,
            "episodes": self.eval_rows,
            "successes": successes,
            "total": total,
            "rate": (successes / total) if total else 0,
            "error": error,
        }
        write_json(self.run_dir / "eval.json", report)
        self.last_outcome = {
            "success": total > 0 and successes == total and not error,
            "reason": f"{successes}/{total}",
            "step": self.budget_used,
        }
        self.end_reason = self.last_outcome["reason"]
        print(
            f"held-out {successes}/{total}" + (f" error={error}" if error else ""),
            flush=True,
        )

    def _run_controller(self, project, code_root, log_path):
        runtime = Path(__file__).resolve().parent / "api" / "runtime.py"
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = controller_command(
            python=sys.executable,
            runtime=str(runtime),
            project=str(Path(project).resolve()),
            code_root=str(Path(code_root).resolve()),
            socket_path=str(self.socket_path),
        )
        with log_path.open("ab", buffering=0) as log:
            log.write(f"\n--- {project}\n".encode())
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(Path(project).resolve()),
                    env=controller_env(),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=CONTROLLER_TIMEOUT,
                )
                return proc.returncode
            except subprocess.TimeoutExpired:
                log.write(b"\n[timeout]\n")
                return 124

    def _failure_reason(self):
        """Goal clauses that are false. Only for a finished or step-capped episode."""
        problem = getattr(self.env, "env", None)
        if problem is None or not hasattr(problem, "_eval_predicate"):
            return []
        failed = []
        for state in problem.parsed_problem.get("goal_state") or []:
            if not isinstance(state, (list, tuple)) or not state:
                continue
            text = _format_goal(state)
            try:
                satisfied = bool(problem._eval_predicate(list(state)))
            except Exception as exc:
                failed.append(f"{text} ({exc})")
                continue
            if not satisfied:
                failed.append(text)
        return failed


class Handler:
    def __init__(self, service):
        self.service = service

    def handle(self, conn):
        reader = conn.makefile("r", encoding="utf-8")
        writer = conn.makefile("w", encoding="utf-8")
        role = "agent"
        try:
            while True:
                line = reader.readline(32 * 1024 * 1024)
                if not line:
                    break
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    self._send(writer, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                    continue
                ident = request.get("id")
                try:
                    result, role, notify = self.dispatch(request, role)
                    if notify:
                        continue
                    self._send(writer, {"jsonrpc": "2.0", "id": ident, "result": result})
                except Exception as exc:
                    self._send(writer, {"jsonrpc": "2.0", "id": ident, "error": {"code": -32000, "message": str(exc)}})
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def dispatch(self, request, role):
        if request.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC")
        method = request.get("method")
        if method == "initialize":
            info = ((request.get("params") or {}).get("clientInfo") or {})
            if info.get("name") == "python-controller":
                role = "controller"
            elif info.get("name") == "server":
                role = "server"
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "libero-explore", "version": "1"},
            }
            return result, role, False
        if method == "notifications/initialized":
            return None, role, True
        if method == "tools/list":
            return {"tools": []}, role, False
        if method == "tools/call":
            params = request.get("params") or {}
            value = self.service.call(params.get("name"), params.get("arguments") or {}, role)
            if role == "controller":
                hidden = CONTROLLER_HIDDEN
            elif role == "agent":
                hidden = AGENT_HIDDEN
            else:
                hidden = frozenset()
            if role in ("agent", "controller") and self.service.all_tasks:
                hidden = hidden | {"task_id", "bddl"}
            if hidden:
                value = redact(value, hidden)
            text = json.dumps(value, allow_nan=False, ensure_ascii=False)
            return {"content": [{"type": "text", "text": text}], "isError": False}, role, False
        raise ValueError(f"Unknown method: {method}")

    def _send(self, writer, payload):
        writer.write(json.dumps(payload, allow_nan=False) + "\n")
        writer.flush()


def _format_goal(state):
    name = str(state[0])
    args = ", ".join(str(part) for part in state[1:])
    return f"{name}({args})" if args else name


def serve_http(service, port):
    class Monitor(BaseHTTPRequestHandler):
        def log_message(self, _fmt, *_args):
            return

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/status":
                body = json.dumps(service.public_status(), ensure_ascii=False).encode()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if path in ("/frame.jpg", "/wrist.jpg"):
                frame = service.run_dir / path.lstrip("/")
                if not frame.exists():
                    self.send_response(404)
                    self._cors()
                    self.end_headers()
                    return
                body = frame.read_bytes()
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self._cors()
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", port), Monitor)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def serve_socket(service, path):
    sock_path = Path(path)
    if sock_path.exists():
        sock_path.unlink()
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(sock_path))
    server.listen(16)
    handler = Handler(service)

    def accept():
        while True:
            try:
                conn, _addr = server.accept()
            except OSError:
                break
            threading.Thread(target=handler.handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return server


def config_from_env():
    run_dir = os.environ.get("LIBERO_RUN_DIR", "/data/changzou/deepseek4robotics/runs/libero90")
    task_raw = os.environ.get("LIBERO_TASK_ID", "0").strip().lower()
    return {
        "run_dir": run_dir,
        "workspace": os.environ.get(
            "LIBERO_WORKSPACE",
            "/data/changzou/deepseek4robotics/runs/libero90/agent",
        ),
        "socket": os.environ.get("LIBERO_SOCKET", str(Path(run_dir) / "sock" / "mcp.sock")),
        "port": int(os.environ.get("LIBERO_PORT", "8766")),
        "hard_cap": int(os.environ.get("LIBERO_HARD_CAP", "300")),
        "task_id": 0 if task_raw == "all" else int(task_raw),
        "all_tasks": task_raw == "all",
        "eval_init_limit": parse_eval_inits(os.environ.get("LIBERO_EVAL_INITS", "all")),
        "wall_seconds": int(os.environ.get("LIBERO_EPISODE_SECONDS", "1200")),
    }


def main():
    config = config_from_env()
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
    service = Service(config)
    serve_http(service, config["port"])
    serve_socket(service, config["socket"])
    print(f"libero-explore worker listening socket={config['socket']} http=127.0.0.1:{config['port']}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
