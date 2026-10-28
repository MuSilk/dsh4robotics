"""LIBERO-90 robot service.

Lifecycle tools: status, register, finish.
The server opens the one episode before the agent starts.
Robot tools use LIBERO's official 7-D OSC_POSE action: libero_status,
libero_observe, libero_cameras, libero_depth, libero_step, libero_hold.

The simulator lives only in this process. Controllers reach it through
api.runtime.Context and LIBERO_SOCKET.
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
        self.task_id = int(config["task_id"])
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
            raise PermissionError("The episode is opened by the server")
        if role == "controller" and name not in CONTROLLER_TOOLS:
            raise PermissionError(f"{name} is not available inside a controller")
        if name == "finish":
            return self._finish(arguments, role)
        with self.lock:
            if self.phase == "formal" and role != "controller":
                raise RuntimeError("A controller is running; wait for it to finish")
            return self._locked(name, arguments, role)

    def _locked(self, name, arguments, role):
        if name == "status":
            only(arguments, ())
            return self._status()
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

    def _ensure_task(self):
        if self.suite is not None:
            return
        from libero.libero import benchmark, get_libero_path
        suite = benchmark.get_benchmark_dict()["libero_90"]()
        if self.task_id < 0 or self.task_id >= suite.n_tasks:
            raise RuntimeError(f"task_id {self.task_id} is outside 0..{suite.n_tasks - 1}")
        task = suite.get_task(self.task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        states = list(suite.get_task_init_states(self.task_id))
        if not states:
            raise RuntimeError("task has no init states")
        self.suite = suite
        self.init_states = states
        self.task = {
            "suite": "libero_90",
            "task_id": self.task_id,
            "name": task.name,
            "language": task.language,
            "bddl": bddl,
        }

    def _status(self):
        bundles = {
            bundle: {
                "sha256": item["sha256"],
                "source": item["source"],
            }
            for bundle, item in self.bundles.items()
        }
        return {
            "task": self.task,
            "episode": self._episode_view(),
            "episode_limits": {"max_steps": self.hard_cap},
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
            "init_state_id": self.init_state_id,
            "episode_id": self.episode_id,
            "episode_ended": self.episode_ended,
            "kind": self.episode_kind,
        }

    def _libero_status(self):
        self._ensure_task()
        return {
            "instruction": self.task["language"],
            "task": self.task,
            "action_layout": ACTION_LAYOUT,
            "state_layout": STATE_LAYOUT,
            "max_action_rows": MAX_ROWS,
            "episode_limits": {"max_steps": self.hard_cap},
            "episode": self._episode_view(),
            "cameras": ["agentview", "robot0_eye_in_hand"],
        }

    def _require_episode(self):
        if self.env is None:
            raise RuntimeError("No interactive episode")

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
            "init_state_id": self.init_state_id,
            "episode_id": self.episode_id,
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

    def _open_env(self, init_state_id, kind):
        if threading.current_thread() is not self.gl_thread:
            return self._on_gl(lambda: self._open_env(init_state_id, kind))
        self._ensure_task()
        if init_state_id < 0 or init_state_id >= len(self.init_states):
            raise RuntimeError(f"init_state_id {init_state_id} is outside 0..{len(self.init_states) - 1}")
        from libero.libero.envs import OffScreenRenderEnv
        self._close_env()
        env = OffScreenRenderEnv(
            bddl_file_name=self.task["bddl"],
            camera_heights=CAMERA_SIZE,
            camera_widths=CAMERA_SIZE,
            horizon=self.hard_cap + SETTLE_STEPS + 5,
            ignore_done=True,
        )
        try:
            env.seed(0)
            env.reset()
            self.env = env
            self._activate_gl()
            obs = env.set_init_state(self.init_states[init_state_id])
            settle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for _ in range(SETTLE_STEPS):
                self._activate_gl()
                obs, _reward, _done, _info = env.step(settle)
        except Exception:
            self.env = None
            try:
                env.close()
            except Exception:
                pass
            raise
        self.last_outcome = None
        self.init_state_id = init_state_id
        self.episode_id = f"libero90_{kind}_{init_state_id}_{uuid.uuid4().hex[:8]}"
        self.step_id = 0
        self.last_action = None
        self.episode_ended = False
        self.task_complete = bool(env.check_success())
        self.end_reason = ""
        self.episode_kind = kind
        self._remember(obs)
        self._publish()

    def _start_episode(self, init_state_id=None):
        if self.phase == "closed" or (self.episode_opened and self.env is None):
            raise RuntimeError("The episode is already finished")
        if self.env is not None and not self.episode_ended:
            raise RuntimeError("The episode is still open")
        if self.env is not None:
            raise RuntimeError("Call finish")
        self._ensure_task()
        if init_state_id is None:
            init_state_id = random.randrange(len(self.init_states))
        elif type(init_state_id) is not int:
            raise ValueError("init_state_id must be an integer")
        self._open_env(init_state_id, "interactive")
        self.episode_opened = True
        self.phase = "interactive"
        self._persist()
        self._publish()
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
            self._remember(obs)
            # LIBERO sets done from check_success(). That result stays on the
            # monitor only; it does not end the episode or go back to the agent.
            self.task_complete = bool(self.env.check_success())
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
            "episode_ended": ended,
            "session_episode_over": ended,
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

    def _finish(self, arguments, role):
        if role == "controller":
            raise PermissionError("finish is not available inside a controller")
        only(arguments, ())
        with self.lock:
            if self.phase == "closed":
                raise RuntimeError("The episode is already finished")
            if self.env is None:
                raise RuntimeError("No episode to finish")
            self.task_complete = bool(self.env.check_success())
            failure_reason = [] if self.task_complete else self._failure_reason()
            if not self.episode_ended:
                self._mark_end("success" if self.task_complete else "finish")
            self.last_outcome = {
                "success": bool(self.task_complete),
                "reason": self.end_reason,
                "step": self.step_id,
                "failure_reason": failure_reason,
            }
            self._close_env()
            self.episode_ended = False
            self.phase = "closed"
            self._persist()
            self._publish()
            value = self._status()
            value["finished"] = True
            value["session_episode_over"] = True
            value["task_complete"] = bool(self.last_outcome["success"])
            value["failure"] = not self.last_outcome["success"]
            value["reason"] = self.last_outcome["reason"]
            value["step_id"] = self.last_outcome["step"]
            if value["failure"]:
                value["failure_reason"] = self.last_outcome.get("failure_reason") or []
            return value

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
                "serverInfo": {"name": "libero90", "version": "1"},
            }
            return result, role, False
        if method == "notifications/initialized":
            return None, role, True
        if method == "tools/list":
            return {"tools": []}, role, False
        if method == "tools/call":
            params = request.get("params") or {}
            value = self.service.call(params.get("name"), params.get("arguments") or {}, role)
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
    return {
        "run_dir": run_dir,
        "workspace": os.environ.get(
            "LIBERO_WORKSPACE",
            "/data/changzou/deepseek4robotics/runs/libero90/agent",
        ),
        "socket": os.environ.get("LIBERO_SOCKET", str(Path(run_dir) / "sock" / "mcp.sock")),
        "port": int(os.environ.get("LIBERO_PORT", "8766")),
        "hard_cap": int(os.environ.get("LIBERO_HARD_CAP", "300")),
        "task_id": int(os.environ.get("LIBERO_TASK_ID", "0")),
        "wall_seconds": int(os.environ.get("LIBERO_EPISODE_SECONDS", "1200")),
    }


def main():
    config = config_from_env()
    Path(config["run_dir"]).mkdir(parents=True, exist_ok=True)
    service = Service(config)
    serve_http(service, config["port"])
    serve_socket(service, config["socket"])
    print(f"libero90 worker listening socket={config['socket']} http=127.0.0.1:{config['port']}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
