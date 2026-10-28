#!/usr/bin/env python3
"""Imajev-4B behind a unix socket. The DSH sandbox has no network, so the model stays on the host."""
import json
import os
import socket
import sys
import threading
import traceback
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

IMAJEV_ROOT = Path(os.environ.get("IMAJEV_ROOT", "/data/changzou/deepseek4robotics/third_party/imajev"))
sys.path.insert(0, str(IMAJEV_ROOT / "src"))
sys.path.insert(0, str(IMAJEV_ROOT / "scripts"))

ALIASES = {"agentview": "frame.jpg", "wrist": "wrist.jpg"}
STATE_LIMIT = 32 * 1024
LOAD_TIMEOUT = 210


class Engine:
    def __init__(self, root):
        self.root = Path(root)
        self.ready = threading.Event()
        self.error = None
        self.backend = None
        self.calibration = None
        self.lock = threading.Lock()

    def load(self):
        try:
            from playground.server import TorchBackend
            from vision_decision.calibration import TemperatureCalibrator
            bundle = self.root / "artifacts" / "model-qwen4b.json"
            adapter = self.root / "adapters" / "imajev-4b"
            calibration = adapter / "calibration.json"
            self.backend = TorchBackend(bundle=bundle, adapter=adapter, device="cuda", rotations=1)
            self.calibration = TemperatureCalibrator.load(calibration)
            print(
                f"imajev ready in {self.backend.load_seconds:.1f}s "
                f"codes={self.backend.readout_codes} device=cuda",
                flush=True,
            )
        except Exception as exc:
            self.error = exc
            traceback.print_exc()
        finally:
            self.ready.set()

    def wait_ready(self):
        if not self.ready.wait(LOAD_TIMEOUT):
            raise TimeoutError("imajev is still loading")
        if self.error is not None:
            raise RuntimeError(f"imajev failed to load: {self.error}")

    def decide(self, images, payload):
        from playground.server import compile_payload
        from vision_decision.jev_api import to_response
        self.wait_ready()
        request, plan = compile_payload(payload, self.backend.max_options)
        with self.lock:
            results, usage = self.backend.score(images, request)
        photo_only = bool(images) and not request.state
        results = [
            self.calibration.calibrate_result(
                result, field.type, len(result.scores) - 1,
                image=bool(images), photo_only=photo_only,
            )
            for field, result in zip(request.fields, results)
        ]
        body = to_response(request, results, model="imajev-4b", plan=plan)
        body["usage"] = usage
        return body


def _within(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_image(token, workspace, run_dir):
    text = str(token).strip()
    if not text:
        raise ValueError("image path is empty")
    if text in ALIASES:
        path = (run_dir / ALIASES[text]).resolve()
        if not path.is_file():
            raise ValueError(f"{text} is not available yet")
        return path
    raw = Path(text)
    path = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
    roots = [workspace.resolve(), run_dir.resolve()]
    if not any(_within(path, root) for root in roots):
        raise ValueError("image path must be agentview, wrist, or a file under the workspace")
    if not path.is_file():
        raise ValueError(f"image not found: {text}")
    return path


def load_paths(tokens, workspace, run_dir):
    from vision_decision.images import load_image
    if tokens is None:
        tokens = []
    if not isinstance(tokens, list) or any(not isinstance(item, str) for item in tokens):
        raise ValueError("images must be an array of strings")
    if len(tokens) > 2:
        raise ValueError("at most two images")
    loaded = []
    for token in tokens:
        image, _meta = load_image(resolve_image(token, workspace, run_dir))
        loaded.append(image)
    return loaded


def payload_from(arguments):
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be an object")
    questions = arguments.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    payload = {"questions": questions}
    if "state" in arguments and arguments["state"] is not None:
        encoded = json.dumps(arguments["state"], ensure_ascii=False)
        if len(encoded.encode("utf-8")) > STATE_LIMIT:
            raise ValueError("state exceeds 32 KiB")
        payload["state"] = arguments["state"]
    return payload


class Server:
    def __init__(self, engine, workspace, run_dir):
        self.engine = engine
        self.workspace = Path(workspace)
        self.run_dir = Path(run_dir)

    def handle(self, conn):
        reader = conn.makefile("r", encoding="utf-8")
        writer = conn.makefile("w", encoding="utf-8")
        try:
            while True:
                line = reader.readline(1024 * 1024)
                if not line:
                    break
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    self._send(writer, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                    continue
                ident = request.get("id")
                try:
                    result = self.dispatch(request)
                    if result is None:
                        continue
                    self._send(writer, {"jsonrpc": "2.0", "id": ident, "result": result})
                except Exception as exc:
                    self._send(writer, {"jsonrpc": "2.0", "id": ident, "error": {"code": -32000, "message": str(exc)}})
        finally:
            conn.close()

    def dispatch(self, request):
        if request.get("jsonrpc") != "2.0":
            raise ValueError("Invalid JSON-RPC")
        method = request.get("method")
        if method == "ready":
            if not self.engine.ready.is_set():
                return {"ready": False}
            if self.engine.error is not None:
                raise RuntimeError(f"imajev failed to load: {self.engine.error}")
            return {"ready": True}
        if method == "initialize":
            return {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "imajev", "version": "1"},
            }
        if method == "notifications/initialized":
            return None
        if method != "tools/call":
            raise ValueError(f"unknown method {method}")
        params = request.get("params") or {}
        if params.get("name") != "imajev_decide":
            raise ValueError(f"unknown tool {params.get('name')}")
        arguments = params.get("arguments") or {}
        payload = payload_from(arguments)
        images = load_paths(arguments.get("images"), self.workspace, self.run_dir)
        body = self.engine.decide(images, payload)
        return {"content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]}

    @staticmethod
    def _send(writer, message):
        writer.write(json.dumps(message, ensure_ascii=False) + "\n")
        writer.flush()


def serve(sock_path, server):
    path = Path(sock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(path))
    listener.listen(4)
    print(f"imajev listening {path}", flush=True)

    def accept():
        while True:
            try:
                conn, _addr = listener.accept()
            except OSError:
                break
            threading.Thread(target=server.handle, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    return listener


def main():
    workspace = os.environ.get("IMAJEV_WORKSPACE", "/data/changzou/deepseek4robotics/runs/libero90/agent")
    run_dir = os.environ.get("IMAJEV_RUN_DIR", "/data/changzou/deepseek4robotics/runs/libero90")
    sock_path = os.environ.get("IMAJEV_SOCKET", str(Path(run_dir) / "sock" / "imajev.sock"))
    engine = Engine(IMAJEV_ROOT)
    server = Server(engine, workspace, run_dir)
    serve(sock_path, server)
    threading.Thread(target=engine.load, daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
