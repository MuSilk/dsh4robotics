#!/usr/bin/env python3
"""Imajev-4B over its unix socket. It names what is in a picture. It does not return metres.

    import vlm
    body = vlm.decide(
        {"which": {"type": "choice",
                   "instructions": "Which picture shows the bottle upright?",
                   "criteria": {"first": "the upright bottle", "second": "something else"}}},
        images=["agentview"])
    label, unknown = vlm.answer(body, "which")

`agentview` is the head camera. `wrist` is the left wrist. A file path is also
accepted, at most two images. Read `abstained` and `unknown_probability`.
The first call of a session loads the model and can take a few minutes.
"""
import json
import os
import socket
from pathlib import Path

SOCK = os.environ.get("IMAJEV_SOCKET", "")
TIMEOUT = 240


class Unavailable(RuntimeError):
    pass


class Ctx:
    def __init__(self, sock):
        self.sock, self.buf, self.n = sock, b"", 0

    def _send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def _recv(self):
        while b"\n" not in self.buf:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise RuntimeError("socket closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line.decode())

    def call(self, name, **arguments):
        self.n += 1
        self._send({"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}})
        while True:
            msg = self._recv()
            if msg.get("id") == self.n:
                break
        if "error" in msg:
            raise RuntimeError("%s failed: %s" % (name, msg["error"]))
        return msg["result"]


def connect(path=None, name="ctl"):
    path = path or SOCK
    if not path:
        raise Unavailable("IMAJEV_SOCKET is not set")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(TIMEOUT)
    try:
        sock.connect(path)
    except OSError as exc:
        raise Unavailable(str(exc)) from exc
    ctx = Ctx(sock)
    ctx.n += 1
    ctx._send({"jsonrpc": "2.0", "id": ctx.n, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": name, "version": "1"}}})
    while True:
        msg = ctx._recv()
        if msg.get("id") == ctx.n:
            break
    ctx._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return ctx


_CACHE = {}


def available():
    if "ok" in _CACHE:
        return _CACHE["ok"]
    try:
        _CACHE["ctx"] = connect(name="probe")
        _CACHE["ok"] = True
    except Exception:
        _CACHE["ok"] = False
    return _CACHE["ok"]


def _image_token(token):
    text = str(token).strip()
    if text in ("agentview", "wrist") or not text:
        return text
    path = Path(text)
    if path.is_file():
        return str(path.resolve())
    return text


def decide(questions, images=None, state=None, ctx=None):
    ctx = ctx or _CACHE.get("ctx") or connect(name="ctl")
    args = {"questions": questions}
    if images:
        args["images"] = [_image_token(item) for item in list(images)[:2]]
    if state is not None:
        args["state"] = state
    return json.loads(ctx.call("imajev_decide", **args)["content"][0]["text"])


def answer(body, key):
    item = (body or {}).get(key) or {}
    unknown = item.get("unknown_probability")
    if item.get("abstained"):
        return None, unknown
    value = item.get("answer", item.get("label", item.get("value")))
    if isinstance(value, str) and value.strip().lower() in ("", "unknown", "none", "neither"):
        return None, unknown
    return value, unknown
