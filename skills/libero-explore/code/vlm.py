#!/usr/bin/env python3
"""Imajev-4B over its unix socket: ask which object is which (identity only).

The sandbox has no network; the model lives on the host behind $IMAJEV_SOCKET and
speaks the same JSON-RPC as the robot socket in mcp.py.

    import vlm
    if vlm.available():
        body = vlm.decide(
            {"which": {"type": "choice",
                       "instructions": "Which of these two crops shows a stick of butter?",
                       "criteria": {"first": "a butter pack", "second": "cream cheese",
                                    "neither": "something else"}}},
            images=["obj06.png", "obj05.png"])
        label, conf = vlm.answer(body, "which")   # ('first', 0.97) or (None, p) if it abstains

Rules that matter:
  * at most two images per call, each either "agentview", "wrist", or a path under
    the session workspace (a 4x crop written by rig.crop() is ideal);
  * one call = one model pass, seconds of wall clock, so batch the question;
  * always read `abstained` and `unknown_probability`: an abstention means the
    picture does not show it, which is a *different* fact from a wrong answer.
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
                raise RuntimeError("imajev socket closed")
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


def connect(path=None, name="ctl", timeout=TIMEOUT):
    path = path or SOCK
    if not path:
        raise Unavailable("IMAJEV_SOCKET is not set")
    if not os.path.exists(path):
        raise Unavailable("imajev is not listening at %s" % path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect(path)
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
    """True when the socket answers a connect. Never raises."""
    if "ok" in _CACHE:
        return _CACHE["ok"]
    try:
        _CACHE["ctx"] = connect(name="probe")
        _CACHE["ok"] = True
    except Exception:
        _CACHE["ok"] = False
    return _CACHE["ok"]


def _image_token(token):
    """agentview/wrist stay as names. A file that exists is sent as an absolute path.

    The imajev worker resolves a relative path against the session workspace, while
    a controller's crops are written in its own directory. An absolute path under
    the workspace or the run directory is accepted.
    """
    text = str(token).strip()
    if text in ("agentview", "wrist") or not text:
        return text
    path = Path(text)
    if path.is_file():
        return str(path.resolve())
    return text


def decide(questions, images=None, state=None, ctx=None):
    """One imajev_decide call. Returns the parsed JSON body (dict).

    `questions` is a Jev object: {key: {"type": "choice"|"score"|"multi"|"noul",
    "instructions": str, "criteria": {name: str}}}. `images` is a list of 0-2
    tokens: "agentview", "wrist", or an image path.
    """
    ctx = ctx or _CACHE.get("ctx") or connect(name="ctl")
    args = {"questions": questions}
    if images:
        args["images"] = [_image_token(item) for item in list(images)[:2]]
    if state is not None:
        args["state"] = state
    return json.loads(ctx.call("imajev_decide", **args)["content"][0]["text"])


def answer(body, key):
    """(label_or_score, unknown_probability) for one question, honouring abstention.

    Returns (None, p) when the model abstained or was unsure, so the caller can
    fall back to the geometric shortlist instead of trusting a coin flip.
    """
    item = (body or {}).get(key) or {}
    p = item.get("unknown_probability")
    if item.get("abstained"):
        return None, p
    val = item.get("answer", item.get("label", item.get("value")))
    if isinstance(val, str) and val.strip().lower() in ("", "unknown", "none", "neither"):
        return None, p
    return val, p
