#!/usr/bin/env python3
"""JSON-RPC transport to the LIBERO episode server.

    import mcp
    ctx = mcp.connect("ctl")
    mcp.tool(ctx, "libero_observe")     # -> dict, flat (no "observation" wrapper)

SOCK is a unix socket named by $LIBERO_SOCKET (or $MCP_SOCKET); nothing here
touches pixels or the arm, it just forwards tool calls.
"""
import json
import os
import socket

SOCK = os.environ.get("LIBERO_SOCKET") or os.environ.get("MCP_SOCK") or ""


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


def connect(path=None, name="ctl", timeout=240):
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
    """Call a tool and return its JSON payload (the text content, parsed)."""
    return json.loads(ctx.call(name, **arguments)["content"][0]["text"])
