#!/usr/bin/env python3
"""JSON-RPC transport to the RoboTwin episode server.

    import mcp
    ctx = mcp.connect("ctl")
    mcp.tool(ctx, "robotwin_observe")

The socket is $ROBOTWIN_SOCKET, or $LIBERO_SOCKET when the session launcher
published the same path under that name. This module does not move the arms.
"""
import json
import os
import socket

SOCK = os.environ.get("ROBOTWIN_SOCKET") or os.environ.get("LIBERO_SOCKET") or os.environ.get("MCP_SOCK") or ""


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


def connect(path=None, name="ctl", timeout=900):
    path = path or SOCK
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


def tool(ctx, name, **arguments):
    """Call a tool and return its JSON payload."""
    return json.loads(ctx.call(name, **arguments)["content"][0]["text"])
