"""Run a controller's main(ctx) against the LIVE episode over the session's MCP socket.

SHIPS WITH THE SKILL: this is the ~40-line wrapper SKILL.md §0.3 asks for, already written.
It costs no rows and it makes every stage re-runnable.

    ENTRY=probe python3 code/lib/run_controller.py 2>&1 | tee experiments/logs/probe.log

CONTROLLER=<path to a project dir under code/> chooses the controller (default code/open_drawer2).
The controller may only call status / libero_status / libero_observe / libero_depth /
libero_step / libero_hold.  clientInfo.name = "python-controller" puts this connection in the
controller role, so register/finish are refused on it and the episode stays open.
"""
import importlib.util
import json
import os
import socket
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # workspace
SOCK = os.environ.get("MCP_SOCK") or os.path.join(os.path.dirname(ROOT), "sock", "mcp.sock")


class Ctx:
    def __init__(self, sock):
        self.sock = sock
        self.buf = b""
        self.n = 0

    def _send(self, obj):
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def _recv_line(self):
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise RuntimeError("socket closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line.decode())

    def call(self, name, **arguments):
        self.n += 1
        rid = self.n
        self._send({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                    "params": {"name": name, "arguments": arguments}})
        while True:
            msg = self._recv_line()
            if msg.get("id") == rid:
                break
        if "error" in msg:
            raise RuntimeError("%s failed: %s" % (name, msg["error"]))
        return msg["result"]


def connect():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(SOCK)
    ctx = Ctx(s)
    ctx.n += 1
    ctx._send({"jsonrpc": "2.0", "id": ctx.n, "method": "initialize",
               "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                          "clientInfo": {"name": "python-controller", "version": "1.0"}}})
    while True:
        msg = ctx._recv_line()
        if msg.get("id") == ctx.n:
            break
    ctx._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    return ctx


def load(path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    f = os.path.join(path, "controller.py")
    spec = importlib.util.spec_from_file_location("controller", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    rel = os.environ.get("CONTROLLER", "code/open_drawer2")
    os.chdir(ROOT)
    mod = load(rel)
    ctx = connect()
    out = mod.main(ctx)
    print("RESULT %r" % (out,), flush=True)


if __name__ == "__main__":
    main()
