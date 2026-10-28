"""RoboTwin exploration service.

The server opens the first round. The agent resets onto a random seed, spends a
shared action budget, and submits one controller. That program is then rolled
out on held-out seeds, with no agent in the loop.

ROBOTWIN_TASK=all is the same loop over every RoboTwin task. reset samples a
task and a seed. The seed is never returned to the agent. Eval seeds start at
ROBOTWIN_SEED_START and are disjoint from the explore window.

Each robotwin_step row is one official take_action: an absolute dual-arm
target, planned and executed. It is not a 20 Hz servo tick. ee rows are 16
numbers (left xyz, left wxyz, left gripper, right xyz, right wxyz, right
gripper). qpos rows are left joints, left gripper, right joints, right
gripper. Gripper 0 is closed and 1 is open.
"""
import ast
import base64
import hashlib
import json
import math
import os
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

import cv2
import numpy as np

ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/data/changzou/deepseek4robotics/third_party/RoboTwin"))
MAX_ROWS = 8
EE_DIM = 16
SETTLE_ATTEMPTS = 40
CONTROLLER_TIMEOUT = int(os.environ.get("ROBOTWIN_CONTROLLER_TIMEOUT", "3600"))
CONTROLLER_TOOLS = frozenset({
    "status", "robotwin_status", "robotwin_observe", "robotwin_cameras", "robotwin_depth",
    "robotwin_step", "robotwin_hold",
})
AGENT_HIDDEN = {"seed", "visited", "explore_seed"}
CONTROLLER_HIDDEN = AGENT_HIDDEN | {"budget", "budget_used", "budget_remaining", "budget_total"}

_SIM_READY = False


class SeedRejected(Exception):
    """This seed is unstable, or the scripted expert cannot solve it."""


def load_task_names(path):
    names = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        name = stripped[2:].split("#", 1)[0].strip()
        if name:
            names.append(name)
    if not names:
        raise RuntimeError(f"no tasks in {path}")
    return names


def load_step_limits(path):
    limits = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        limits[name.strip()] = int(value.strip())
    if not limits:
        raise RuntimeError(f"no step limits in {path}")
    return limits


def explore_base(seed_start, eval_seeds):
    """First explore seed. Eval may scan eval_seeds * 50 seeds and must not overlap."""
    if int(seed_start) < 0 or int(eval_seeds) < 1:
        raise RuntimeError("seed start and eval seed count must be positive")
    return int(seed_start) + int(eval_seeds) * 50


def episode_cap(horizon, budget_total, budget_used):
    remaining = int(budget_total) - int(budget_used)
    if remaining <= 0:
        raise RuntimeError("exploration budget is exhausted; submit the controller")
    if int(horizon) < 1:
        raise RuntimeError("episode horizon must be positive")
    return min(int(horizon), remaining)


def normalize_action_type(value):
    text = str(value or "ee").strip().lower()
    if text in ("ee", "endpose", "pose"):
        return "ee"
    if text in ("qpos", "joint", "joints"):
        return "qpos"
    raise RuntimeError("action type must be ee or qpos")


def action_width(action_type, left_dim, right_dim):
    if action_type == "ee":
        return EE_DIM
    if int(left_dim) < 1 or int(right_dim) < 1:
        raise RuntimeError("joint action size is not known yet")
    return int(left_dim) + 1 + int(right_dim) + 1


def gripper_indexes(action_type, left_dim, right_dim):
    if action_type == "ee":
        return (7, 15)
    left_dim = int(left_dim)
    return (left_dim, left_dim + 1 + int(right_dim))


def arm_dims(root, task_config):
    """Joint counts from the embodiment yaml. Does not start the simulator."""
    import yaml
    root = Path(root)
    with (root / "env_cfg" / "task_config" / f"{task_config}.yml").open(encoding="utf-8") as handle:
        task = yaml.safe_load(handle)
    embodiment = task.get("embodiment")
    name = embodiment[0] if isinstance(embodiment, (list, tuple)) else embodiment
    with (root / "env_cfg" / "task_config" / "_embodiment_config.yml").open(encoding="utf-8") as handle:
        listing = yaml.safe_load(handle)
    robot_file = Path(listing[name]["file_path"])
    if not robot_file.is_absolute():
        robot_file = root / robot_file
    with (robot_file / "config.yml").open(encoding="utf-8") as handle:
        robot = yaml.safe_load(handle)
    joints = robot["arm_joints_name"]
    return int(len(joints[0])), int(len(joints[1])), str(name)


def instruction_text(task_name, episode_info, instruction_type, seed, description_dir):
    """Seen/unseen line when the expert filled placeholders, otherwise the task summary."""
    data = json.loads((Path(description_dir) / f"{task_name}.json").read_text(encoding="utf-8"))
    fallback = data.get("full_description") or str(task_name).replace("_", " ")
    info = {}
    if isinstance(episode_info, dict) and isinstance(episode_info.get("info"), dict):
        info = episode_info["info"]
    if not info:
        return fallback
    kind = instruction_type if instruction_type in ("seen", "unseen") else "seen"
    try:
        from generate_episode_instructions import generate_episode_descriptions
        results = generate_episode_descriptions(task_name, [info], 8, seed=seed)
    except BaseException:
        return fallback
    candidates = (results[0].get(kind) if results else None) or []
    if not candidates:
        return fallback
    return random.Random(f"{seed}:select").choice(candidates)


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


def validate_actions(actions, width, grippers):
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ROWS:
        raise ValueError(f"actions must be a list of 1..{MAX_ROWS} rows")
    rows = []
    for row in actions:
        if not isinstance(row, list) or len(row) != width or not all(finite(item) for item in row):
            raise ValueError(f"each action row must have {width} finite numbers")
        values = [float(item) for item in row]
        for index in grippers:
            if not 0.0 <= values[index] <= 1.0:
                raise ValueError("gripper values must be inside [0, 1], 0 closed and 1 open")
        if width == EE_DIM:
            for origin in (0, 8):
                if any(abs(values[origin + axis]) > 5 for axis in range(3)):
                    raise ValueError("end-effector xyz must be within 5 m of the origin")
        else:
            for index, value in enumerate(values):
                if index not in grippers and abs(value) > 10:
                    raise ValueError("joint targets must be within [-10, 10] rad")
        rows.append(values)
    return rows


def redact(value, hidden):
    if isinstance(value, dict):
        return {key: redact(item, hidden) for key, item in value.items() if key not in hidden}
    if isinstance(value, list):
        return [redact(item, hidden) for item in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_frame(path: Path, image) -> bool:
    if image is None:
        return False
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return False
    frame = np.ascontiguousarray(frame[:, :, :3])
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}


def controller_command(python, runtime, project, code_root, socket_path):
    """Run the controller where the RoboTwin tree and the seed log are not mounted."""
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
        "--setenv", "ROBOTWIN_SOCKET", socket_path,
        "--setenv", "LIBERO_SOCKET", socket_path,
        "--setenv", "IMAJEV_SOCKET", _imajev_socket(socket_path),
        "--setenv", "MPLCONFIGDIR", "/tmp",
        "--unsetenv", "ROBOTWIN_ROOT",
        "--unsetenv", "ROBOTWIN_RUN_DIR",
        "--unsetenv", "ROBOTWIN_WORKSPACE",
        "--",
        python,
        runtime,
        "run",
        project,
        "controller.py",
    ])
    return command


def _unstable(exc):
    return type(exc).__name__ == "UnStableError" or "unstable" in str(exc).lower()


def prepare_sim(root):
    """Put RoboTwin on the path and keep camera renders on the raster shader."""
    global _SIM_READY
    if _SIM_READY:
        return
    root = Path(root)
    os.chdir(root)
    for path in (root, root / "scripts", root / "description" / "utils"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import sapien.render as render
    if os.environ.get("ROBOTWIN_RASTER", "1") != "0":
        render.set_camera_shader_dir("default")
        render.set_ray_tracing_samples_per_pixel(1)
        original = render.set_camera_shader_dir

        def _set(shader_dir):
            if shader_dir == "rt":
                return original("default")
            return original(shader_dir)

        render.set_camera_shader_dir = _set
    _SIM_READY = True


def _tips(obs):
    end = (obs or {}).get("endpose") or {}
    return vec(end.get("left_endpose"))[:3], vec(end.get("right_endpose"))[:3]


def _moved(before, after):
    for left, right in zip(before, after):
        if len(left) == 3 and len(right) == 3:
            if math.dist(left, right) > 0.005:
                return True
    return False


class Service:
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.root = Path(config["root"])
        self.run_dir = Path(config["run_dir"])
        self.workspace = Path(config["workspace"]).resolve()
        self.socket_path = config["socket"]
        self.description_dir = self.root / "description" / "task_instruction"
        self.task_config = config["task_config"]
        self.action_type = normalize_action_type(config["action_type"])
        self.instruction_type = config["instruction_type"]
        self.expert_check = bool(config["expert_check"])
        self.all_tasks = bool(config["all_tasks"])
        self.task_names = list(config["task_names"])
        self.step_limits = dict(config["step_limits"])
        self.horizon_override = config["horizon_override"]
        self.budget_total = int(config["budget_total"])
        self.eval_seeds = int(config["eval_seeds"])
        self.seed_start = int(config["seed_start"])
        self.explore_seed = explore_base(self.seed_start, self.eval_seeds)
        self.left_dim, self.right_dim, self.embodiment_name = arm_dims(self.root, self.task_config)
        if self.budget_total < 1 or self.eval_seeds < 1:
            raise RuntimeError("budget and eval seed count must be positive")
        for name in self.task_names:
            if name not in self.step_limits:
                raise RuntimeError(f"{name} is not in the step-limit file")
        self.budget_used = 0
        self.visited = []
        self.visited_set = set()
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
        self.bundles = {}
        self.episode_opened = False
        self.phase = "idle"
        self.env = None
        self.bound_task = None
        self._args_cache = {}
        self.scene_open = False
        self.scene_opens = 0
        self.task_name = None
        self.seed = None
        self.step_id = 0
        self.hard_cap = self._horizon(self.task_names[0])
        self.last_obs = None
        self.last_action = None
        self.episode_ended = False
        self.task_complete = False
        self.end_reason = ""
        self.episode_kind = None
        self.episode_id = None
        self.last_outcome = None
        self._load_session()
        if not self.scene_open:
            for name in ("frame.jpg", "wrist.jpg", "left.jpg", "right.jpg"):
                stale = self.run_dir / name
                if stale.exists():
                    stale.unlink()
        self.monitor = self._monitor_payload()
        self._publish()

    def _horizon(self, task_name):
        official = int(self.step_limits[task_name])
        if self.horizon_override is None:
            return official
        return int(self.horizon_override)

    def _session_path(self):
        return self.run_dir / "session.json"

    def _load_session(self):
        path = self._session_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.bundles = data.get("bundles") or {}
        self.episode_opened = bool(data.get("episode_opened"))
        if data.get("finished") or data.get("submitted_bundle"):
            self.phase = "closed"
            self.episode_opened = True

    def _persist(self):
        write_json(self._session_path(), {
            "bundles": self.bundles,
            "episode_opened": self.episode_opened,
            "finished": self.phase == "closed",
            "submitted_bundle": self.submitted_bundle,
        })

    def _publish(self):
        self.monitor = self._monitor_payload()
        write_json(self.run_dir / "status.json", self.monitor)

    def _language(self):
        if self.env is not None and getattr(self.env, "instruction", None):
            return str(self.env.instruction)
        return ""

    def _monitor_payload(self):
        if self.scene_open and not self.episode_ended:
            state, reason, success = ("success", "", True) if self.task_complete else ("running", "", False)
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
            "step": self.step_id if self.scene_open or self.episode_ended else (self.last_outcome or {}).get("step", 0),
            "max_steps": self.hard_cap,
            "frame": (self.run_dir / "frame.jpg").exists(),
            "task_name": self.task_name or "",
            "language": self._language(),
            "error": "",
            "episode_id": self.episode_id,
            "budget_used": self.budget_used,
            "budget_total": self.budget_total,
            "round": self.rounds,
            "action_type": self.action_type,
            "embodiment": self.embodiment_name,
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
        with self.lock:
            payload = dict(self.monitor)
            in_progress = self.scene_open and not self.episode_ended
        if in_progress:
            payload.pop("success", None)
            payload.pop("reason", None)
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
                "There is no finish. reset opens another random seed. "
                "submit freezes the controller and rolls it out on held-out seeds"
            )
        if name == "reset":
            return self._reset(arguments, role)
        if name == "run":
            return self._run(arguments, role)
        if name == "submit":
            return self._submit(arguments, role)
        with self.lock:
            if self.phase == "closed" and name not in ("status", "robotwin_status"):
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
            only(arguments, ())
            self._open_explore()
            return self._observe()
        if name == "robotwin_status":
            only(arguments, ())
            return self._robotwin_status()
        if name == "robotwin_observe":
            only(arguments, ())
            return self._observe()
        if name == "robotwin_cameras":
            only(arguments, ())
            return self._cameras()
        if name == "robotwin_depth":
            only(arguments, ())
            return self._depth()
        if name == "robotwin_step":
            only(arguments, ("actions",))
            width = action_width(self.action_type, self.left_dim, self.right_dim)
            grippers = gripper_indexes(self.action_type, self.left_dim, self.right_dim)
            return self._step(validate_actions(arguments.get("actions"), width, grippers))
        if name == "robotwin_hold":
            only(arguments, ("steps",))
            steps = arguments.get("steps")
            if type(steps) is not int or not 1 <= steps <= MAX_ROWS:
                raise ValueError(f"steps must be an integer 1..{MAX_ROWS}")
            row = self._current_row()
            return self._step([list(row) for _ in range(steps)])
        raise PermissionError(f"Unknown tool: {name}")

    def _status(self):
        return {
            "task": self._public_task(),
            "episode": self._episode_view(),
            "episode_limits": {"max_steps": self.hard_cap},
            "budget": {
                "used": self.budget_used,
                "total": self.budget_total,
                "remaining": max(0, self.budget_total - self.budget_used),
            },
            "round": self.rounds,
            "phase": self.phase,
            "action_type": self.action_type,
            "action_layout": self._layout(),
            "embodiment": self.embodiment_name,
            "task_config": self.task_config,
        }

    def _public_task(self):
        if not self.task_name and not self._language():
            return None
        return {
            "task_name": self.task_name or "",
            "language": self._language(),
            "action_type": self.action_type,
        }

    def _episode_view(self):
        if not self.scene_open and not self.episode_ended:
            return None
        return {
            "step_id": self.step_id,
            "episode_step_limit": self.hard_cap,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "episode_id": self.episode_id,
            "episode_ended": self.episode_ended,
            "kind": self.episode_kind,
        }

    def _layout(self):
        if self.action_type == "ee":
            return {
                "action_type": "ee",
                "dim": EE_DIM,
                "control": "absolute dual-arm end-effector target; one official take_action per row",
                "quaternion": "wxyz",
                "gripper": "0 closed, 1 open",
                "frame": "same frame as left_endpose and right_endpose",
                "fields": [
                    {"index": [0, 1, 2], "name": "left_xyz", "unit": "m"},
                    {"index": [3, 4, 5, 6], "name": "left_quat_wxyz"},
                    {"index": 7, "name": "left_gripper", "range": [0, 1]},
                    {"index": [8, 9, 10], "name": "right_xyz", "unit": "m"},
                    {"index": [11, 12, 13, 14], "name": "right_quat_wxyz"},
                    {"index": 15, "name": "right_gripper", "range": [0, 1]},
                ],
            }
        return {
            "action_type": "qpos",
            "dim": action_width("qpos", self.left_dim, self.right_dim),
            "control": "absolute dual-arm joint target; one official take_action per row",
            "gripper": "0 closed, 1 open",
            "left_arm_dim": self.left_dim,
            "right_arm_dim": self.right_dim,
            "order": "left joints, left gripper, right joints, right gripper",
        }

    def _robotwin_status(self):
        return {
            "instruction": self._language(),
            "task": self._public_task(),
            "action_layout": self._layout(),
            "max_action_rows": MAX_ROWS,
            "episode_limits": {"max_steps": self.hard_cap},
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "episode": self._episode_view(),
            "cameras": ["head_camera", "left_camera", "right_camera"],
            "nominal_table_z": 0.74,
        }

    def _require_scene(self):
        if not self.scene_open or self.env is None:
            raise RuntimeError("No round is open")

    def _observe(self):
        self._require_scene()
        obs = self.last_obs or {}
        end = obs.get("endpose") or {}
        joints = obs.get("joint_action") or {}
        return {
            "step_id": self.step_id,
            "instruction": self._language(),
            "action_type": self.action_type,
            "action_dim": action_width(self.action_type, self.left_dim, self.right_dim),
            "left_endpose": vec(end.get("left_endpose")),
            "right_endpose": vec(end.get("right_endpose")),
            "left_gripper": round(float(end.get("left_gripper") or 0), 5),
            "right_gripper": round(float(end.get("right_gripper") or 0), 5),
            "left_joint_pos": vec(joints.get("left_arm")),
            "right_joint_pos": vec(joints.get("right_arm")),
            "episode_ended": self.episode_ended,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "episode_id": self.episode_id,
            "budget_used": self.budget_used,
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "round": self.rounds,
            "cameras": ["head_camera", "left_camera", "right_camera"],
        }

    def _current_row(self):
        if self.last_action is not None:
            return list(self.last_action)
        obs = self._observe()
        if self.action_type == "qpos":
            return obs["left_joint_pos"] + [obs["left_gripper"]] + obs["right_joint_pos"] + [obs["right_gripper"]]
        return obs["left_endpose"] + [obs["left_gripper"]] + obs["right_endpose"] + [obs["right_gripper"]]

    def _jpeg_b64(self, filename):
        path = self.run_dir / filename
        if not path.is_file():
            raise RuntimeError(f"{filename} is not ready")
        return base64.b64encode(path.read_bytes()).decode("ascii")

    def _remember(self, obs):
        self.last_obs = obs
        cameras = (obs or {}).get("observation") or {}
        save_frame(self.run_dir / "frame.jpg", (cameras.get("head_camera") or {}).get("rgb"))
        left = (cameras.get("left_camera") or {}).get("rgb")
        right = (cameras.get("right_camera") or {}).get("rgb")
        save_frame(self.run_dir / "left.jpg", left)
        save_frame(self.run_dir / "wrist.jpg", left)
        save_frame(self.run_dir / "right.jpg", right)

    def _cameras(self):
        self._require_scene()
        self._remember(self.env.get_obs())
        view = self._observe()
        image = np.asarray((self.last_obs.get("observation") or {})["head_camera"]["rgb"])
        view.update({
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "head_jpeg": self._jpeg_b64("frame.jpg"),
            "left_jpeg": self._jpeg_b64("left.jpg"),
            "right_jpeg": self._jpeg_b64("right.jpg"),
        })
        return view

    def _depth(self):
        self._require_scene()
        self.env._update_render()
        self.env.cameras.update_picture()
        depth = np.asarray(self.env.cameras.get_depth()["head_camera"]["depth"], dtype=np.float64)
        millimetres = np.clip(np.rint(depth), 0, 65535).astype("<u2")
        config = self.env.cameras.get_config()["head_camera"]
        matrix = np.asarray(config["intrinsic_cv"], dtype=np.float64)
        pose = np.asarray(config["cam2world_gl"], dtype=np.float64).reshape(-1)
        return {
            "camera": "head_camera",
            "height": int(millimetres.shape[0]),
            "width": int(millimetres.shape[1]),
            "step_id": self.step_id,
            "episode_id": self.episode_id,
            "fx": round(float(matrix[0, 0]), 5),
            "fy": round(float(matrix[1, 1]), 5),
            "cx": round(float(matrix[0, 2]), 5),
            "cy": round(float(matrix[1, 2]), 5),
            "cam2world_gl": [round(float(item), 6) for item in pose],
            "depth_mm_dtype": "<u2",
            "depth_mm_scale": 0.001,
            "depth_mm": base64.b64encode(np.ascontiguousarray(millimetres).tobytes()).decode("ascii"),
        }

    def _task_args(self, task_name):
        cached = self._args_cache.get(task_name)
        if cached is not None:
            return cached
        prepare_sim(self.root)
        from eval_policy_xpolicylab import load_task_args
        args, embodiment = load_task_args({
            "task_name": task_name,
            "task_config": self.task_config,
            "policy_name": "deepseek",
            "ckpt_setting": "explore",
        })
        args["eval_mode"] = True
        args["save_data"] = False
        args["render_freq"] = 0
        args["eval_video_save_dir"] = None
        data_type = args.setdefault("data_type", {})
        data_type["rgb"] = True
        data_type["qpos"] = True
        data_type["endpose"] = True
        data_type["depth"] = False
        self._args_cache[task_name] = args
        self.embodiment_name = str(embodiment)
        return args

    def _ensure_env(self, task_name):
        prepare_sim(self.root)
        if self.env is not None and self.bound_task == task_name:
            return self.env
        self._drop_env()
        import importlib
        module = importlib.import_module(f"envs.{task_name}")
        self.env = getattr(module, task_name)()
        self.bound_task = task_name
        return self.env

    def _release_scene(self):
        env = self.env
        self.scene_open = False
        if env is None or not hasattr(env, "scene"):
            return
        try:
            env.close_env(clear_cache=self.scene_opens > 0 and self.scene_opens % 5 == 0)
        except Exception:
            pass

    def _drop_env(self):
        self._release_scene()
        self.env = None
        self.bound_task = None

    def _setup(self, env, args, seed):
        kwargs = dict(args)
        kwargs["render_freq"] = 0
        kwargs["eval_mode"] = True
        kwargs["save_data"] = False
        kwargs["eval_video_save_dir"] = None
        env.setup_demo(now_ep_num=int(self.rounds), seed=int(seed), is_test=True, **kwargs)
        env.eval_video_path = None

    def _expert(self, env):
        try:
            episode_info = env.play_once() or {}
        except Exception as exc:
            if _unstable(exc):
                raise
            print(f"expert check error: {type(exc).__name__}: {exc}", flush=True)
            return None
        try:
            solved = bool(getattr(env, "plan_success", False) and env.check_success())
        except Exception as exc:
            print(f"expert check error: {type(exc).__name__}: {exc}", flush=True)
            return None
        if not solved:
            return None
        return episode_info if isinstance(episode_info, dict) else {}

    def _open_exact(self, task_name, seed, kind):
        if kind == "explore":
            self.hard_cap = episode_cap(self._horizon(task_name), self.budget_total, self.budget_used)
        else:
            self.hard_cap = self._horizon(task_name)
        self.phase = "opening"
        self.task_name = task_name
        self._publish()
        env = self._ensure_env(task_name)
        self._release_scene()
        args = self._task_args(task_name)
        info = {}
        try:
            self._setup(env, args, seed)
            if self.expert_check:
                info = self._expert(env)
                self._release_scene()
                if info is None:
                    raise SeedRejected(f"expert cannot solve seed {seed}")
                self._setup(env, args, seed)
            env.step_lim = int(self.hard_cap)
            env.eval_video_path = None
            instruction = instruction_text(task_name, info, self.instruction_type, seed, self.description_dir)
            env.set_instruction(instruction)
            obs = env.get_obs()
        except SeedRejected:
            self._release_scene()
            raise
        except Exception as exc:
            self._release_scene()
            if _unstable(exc):
                raise SeedRejected(str(exc)) from exc
            raise
        self.scene_open = True
        self.scene_opens += 1
        self.task_name = task_name
        self.seed = int(seed)
        self.step_id = int(getattr(env, "take_action_cnt", 0) or 0)
        self.last_action = None
        self.episode_ended = False
        self.task_complete = bool(env.check_success())
        self.end_reason = ""
        self.episode_kind = kind
        self.episode_id = f"round_{uuid.uuid4().hex[:8]}"
        self._remember(obs)
        write_json(self.run_dir / "operator.json", {
            "task_name": task_name,
            "seed": int(seed),
            "instruction": instruction,
            "kind": kind,
        })
        if kind == "explore":
            self.rounds += 1
            self.visited.append([task_name, int(seed)] if self.all_tasks else int(seed))
            self.visited_set.add((task_name, int(seed)))
            self.phase = "explore"
            self.episode_opened = True
        else:
            self.phase = "evaluating"
        self._persist()
        self._publish()
        print(
            f"round {self.rounds} task={task_name} seed={seed} cap={self.hard_cap} {instruction}",
            flush=True,
        )
        if self.task_complete:
            self._mark_end("success")

    def _pick_explore(self):
        task_name = random.choice(self.task_names) if self.all_tasks else self.task_names[0]
        span = 2000
        for _ in range(64):
            seed = random.randrange(self.explore_seed, self.explore_seed + span)
            if (task_name, seed) not in self.visited_set:
                return task_name, seed
        return task_name, random.randrange(self.explore_seed, self.explore_seed + span)

    def _open_explore(self):
        if self.phase == "closed":
            raise RuntimeError("Exploration is closed")
        episode_cap(self._horizon(self.task_names[0]), self.budget_total, self.budget_used)
        task_name, seed = self._pick_explore()
        last = None
        for offset in range(SETTLE_ATTEMPTS):
            try:
                self._open_exact(task_name, seed + offset, "explore")
                return
            except SeedRejected as exc:
                last = exc
        raise RuntimeError(f"no stable scene from seed {seed}: {last}")

    def _mark_end(self, reason):
        self.episode_ended = True
        self.end_reason = reason
        self._publish()

    def _step(self, rows):
        self._require_scene()
        if self.episode_ended:
            raise RuntimeError("episode already ended")
        executed = 0
        moved = False
        for row in rows:
            if self.step_id >= self.hard_cap or self.env.take_action_cnt >= self.env.step_lim:
                self.task_complete = bool(self.env.check_success())
                self._mark_end("max_steps")
                break
            if self.episode_kind == "explore" and self.budget_used >= self.budget_total:
                self._mark_end("budget")
                break
            before = _tips(self.last_obs)
            before_count = int(self.env.take_action_cnt)
            self.env.take_action(row, action_type=self.action_type)
            delta = int(self.env.take_action_cnt) - before_count
            if delta <= 0:
                self.task_complete = bool(self.env.eval_success or self.env.check_success())
                self._mark_end("success" if self.task_complete else "max_steps")
                break
            self.last_action = list(row)
            self.step_id = int(self.env.take_action_cnt)
            executed += delta
            if self.episode_kind == "explore":
                self.budget_used += delta
            obs = self.env.get_obs()
            moved = moved or _moved(before, _tips(obs))
            self._remember(obs)
            self.task_complete = bool(self.env.eval_success or self.env.check_success())
            if self.task_complete:
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
        if ended:
            self.last_outcome = {
                "success": bool(self.task_complete),
                "reason": self.end_reason,
                "step": self.step_id,
            }
        result = {
            "steps_executed": executed,
            "moved": moved,
            "step_id": self.step_id,
            "remaining_episode_steps": max(0, self.hard_cap - self.step_id),
            "budget_used": self.budget_used,
            "budget_remaining": max(0, self.budget_total - self.budget_used),
            "episode_ended": ended,
            "session_episode_over": False,
            "observation": self._observe() if self.scene_open else None,
        }
        if ended:
            result["task_complete"] = bool(self.task_complete)
            result["failure"] = not self.task_complete
            result["reason"] = self.end_reason
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
            raise ValueError("source must be a project under code/, for example code/place")
        if not (src / "controller.py").is_file():
            raise ValueError("controller.py is required")
        require_main(src / "controller.py")
        return src

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
        if not self.scene_open or self.env is None:
            raise RuntimeError("No round is open")
        complete = bool(self.env.check_success())
        self.task_complete = complete
        return {
            "task_complete": complete,
            "failure": not complete,
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
            previous = self._score_locked() if self.scene_open else None
            self.episode_ended = False
            self._open_explore()
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
            if self.phase != "explore" or not self.scene_open or self.episode_ended:
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
                outcome = self._score_locked() if self.scene_open else {"task_complete": False, "failure": True}
                outcome["exit_code"] = code
                outcome["log_tail"] = _log_tail(log_path)
                if error:
                    outcome["error"] = error
                if self.phase == "formal":
                    self.phase = "explore" if self.scene_open else "idle"
                self._publish()
        return outcome

    def _submit(self, arguments, role):
        self._agent_only(role)
        only(arguments, ("source",))
        with self.lock:
            self._idle_for_agent()
            bundle, project, code_root = self._snapshot(arguments.get("source"))
            self.submitted_bundle = bundle
            self.submitted_source = arguments.get("source") or ""
            self.phase = "evaluating"
            self.busy = True
            self.eval_total = self.eval_seeds * len(self.task_names)
            self.eval_index = 0
            self.eval_success = 0
            self.eval_rows = []
            self.eval_done = False
            self.eval_error = ""
            self.episode_ended = False
            self._persist()
            self._publish()
            project_text, root_text = str(project), str(code_root)
            tasks = list(self.task_names)
        threading.Thread(
            target=self._evaluate,
            args=(project_text, root_text, tasks),
            name="robotwin-explore-eval",
            daemon=True,
        ).start()
        scope = "every task" if self.all_tasks else self.task_names[0]
        return {
            "submitted": True,
            "bundle": bundle,
            "finished": True,
            "session_episode_over": True,
            "note": (
                f"The controller is being rolled out on {self.eval_seeds} held-out seeds "
                f"of {scope}. Further robot calls will fail."
            ),
        }

    def _evaluate(self, project, code_root, tasks):
        error = ""
        try:
            for task_name in tasks:
                found = 0
                seed = self.seed_start
                scanned = 0
                limit = self.eval_seeds * 50
                while found < self.eval_seeds and scanned < limit:
                    scanned += 1
                    tag = f"{task_name}-s{seed}"
                    log_path = self.run_dir / "eval_logs" / f"{tag}.log"
                    scratch = self.run_dir / "eval_scratch" / tag
                    try:
                        with self.lock:
                            self.eval_index = len(self.eval_rows)
                            self._open_exact(task_name, seed, "eval")
                            self._publish()
                        if scratch.exists():
                            shutil.rmtree(scratch)
                        shutil.copytree(code_root, scratch, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                        rel = Path(project).resolve().relative_to(Path(code_root).resolve())
                        code = self._run_controller(str(scratch / rel), str(scratch), log_path)
                        with self.lock:
                            success = bool(self.env.check_success()) if self.scene_open and self.env is not None else False
                            reason = self.end_reason or ("success" if success else "returned")
                            if success:
                                self.eval_success += 1
                            self.eval_rows.append({
                                "task_name": task_name,
                                "seed": seed,
                                "language": self._language(),
                                "success": success,
                                "steps": self.step_id,
                                "reason": reason,
                                "exit_code": code,
                            })
                            found += 1
                            self.episode_ended = False
                            self._publish()
                    except SeedRejected as exc:
                        print(f"skip {task_name} seed={seed}: {exc}", flush=True)
                    except Exception as exc:
                        error = str(exc)
                        with self.lock:
                            self.eval_rows.append({
                                "task_name": task_name,
                                "seed": seed,
                                "language": self._language(),
                                "success": False,
                                "steps": self.step_id,
                                "reason": "error",
                                "exit_code": -1,
                                "error": error,
                            })
                            self._release_scene()
                            self._publish()
                        break
                    finally:
                        if scratch.exists():
                            shutil.rmtree(scratch, ignore_errors=True)
                    if error:
                        break
                    seed += 1
                if error:
                    break
                if found < self.eval_seeds:
                    error = f"{task_name}: opened {found}/{self.eval_seeds} stable seeds"
                    break
        finally:
            with self.lock:
                self._release_scene()
                self.busy = False
                self.phase = "closed"
                self._write_report(error)
                self._publish()

    def _write_report(self, error):
        total = len(self.eval_rows)
        successes = sum(1 for row in self.eval_rows if row.get("success"))
        self.eval_success = successes
        self.eval_error = error
        self.eval_done = True
        report = {
            "all_tasks": self.all_tasks,
            "eval_seeds": self.eval_seeds,
            "seed_start": self.seed_start,
            "explore_seed": self.explore_seed,
            "expert_check": self.expert_check,
            "action_type": self.action_type,
            "task_config": self.task_config,
            "task_name": None if self.all_tasks else self.task_names[0],
            "budget_total": self.budget_total,
            "budget_used": self.budget_used,
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
        print(f"held-out {successes}/{total}" + (f" error={error}" if error else ""), flush=True)

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
            return {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "robotwin-explore", "version": "1"},
            }, role, False
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
                hidden = hidden | {"task_name"}
            if hidden:
                value = redact(value, hidden)
            text = json.dumps(value, allow_nan=False, ensure_ascii=False)
            return {"content": [{"type": "text", "text": text}], "isError": False}, role, False
        raise ValueError(f"Unknown method: {method}")

    def _send(self, writer, payload):
        writer.write(json.dumps(payload, allow_nan=False) + "\n")
        writer.flush()


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
            if path in ("/frame.jpg", "/wrist.jpg", "/left.jpg", "/right.jpg"):
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
    run_dir = os.environ.get("ROBOTWIN_RUN_DIR", "/data/changzou/deepseek4robotics/runs/robotwin-explore")
    task_raw = os.environ.get("ROBOTWIN_TASK", "adjust_bottle").strip()
    root = Path(os.environ.get("ROBOTWIN_ROOT", str(ROOT)))
    names = load_task_names(root / "env_cfg" / "eval" / "all_tasks.yml")
    all_tasks = task_raw.lower() == "all"
    if not all_tasks and task_raw not in names:
        raise RuntimeError(f"unknown task {task_raw}")
    horizon = os.environ.get("ROBOTWIN_HORIZON", "").strip()
    return {
        "root": str(root),
        "run_dir": run_dir,
        "workspace": os.environ.get("ROBOTWIN_WORKSPACE", str(Path(run_dir) / "agent")),
        "socket": os.environ.get("ROBOTWIN_SOCKET") or os.environ.get("LIBERO_SOCKET") or str(Path(run_dir) / "sock" / "mcp.sock"),
        "port": int(os.environ.get("ROBOTWIN_PORT") or os.environ.get("LIBERO_PORT") or "8976"),
        "task_names": names if all_tasks else [task_raw],
        "all_tasks": all_tasks,
        "task_config": os.environ.get("ROBOTWIN_TASK_CONFIG", "demo_clean"),
        "action_type": os.environ.get("ROBOTWIN_ACTION_TYPE", "ee"),
        "instruction_type": os.environ.get("ROBOTWIN_INSTRUCTION", "seen"),
        "expert_check": os.environ.get("ROBOTWIN_EXPERT_CHECK", "0") == "1",
        "budget_total": int(os.environ.get("ROBOTWIN_BUDGET", "1200")),
        "horizon_override": None if horizon in ("", "0", "official") else int(horizon),
        "eval_seeds": int(os.environ.get("ROBOTWIN_EVAL_SEEDS", "5")),
        "seed_start": int(os.environ.get("ROBOTWIN_SEED_START", "100000")),
        "step_limits": load_step_limits(root / "env_cfg" / "task_config" / "_eval_step_limit.yml"),
    }


def main():
    config = config_from_env()
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["socket"]).parent.mkdir(parents=True, exist_ok=True)
    service = Service(config)
    serve_http(service, config["port"])
    serve_socket(service, config["socket"])
    print(
        f"robotwin-explore worker listening socket={config['socket']} http=127.0.0.1:{config['port']}",
        flush=True,
    )
    threading.Event().wait()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
