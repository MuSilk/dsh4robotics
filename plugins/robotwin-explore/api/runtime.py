"""Stdlib client for robot calls from Python.

`Context()` connects ordinary Python to the same tools the session calls directly.
`main(ctx)` is the only entry a registered controller may define. While an
episode is open the process is given a Context that can call `status`,
`robotwin_status`, `robotwin_observe`, `robotwin_cameras`, `robotwin_depth`,
`robotwin_step`, and `robotwin_hold`.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import socket


class Context:
    def __init__(self, reader=None, writer=None):
        """Connect ordinary Python to the robot tools, or use the given reader and writer."""
        self._socket = None
        self._request_lock = threading.RLock()
        if reader is None and writer is None:
            socket_path = os.environ.get("ROBOTWIN_SOCKET") or os.environ.get("LIBERO_SOCKET")
            if not socket_path:
                raise RuntimeError("ROBOTWIN_SOCKET is not set")
            self._socket = socket.socket(socket.AF_UNIX)
            try:
                self._socket.connect(socket_path)
                reader, writer = self._socket.makefile("r"), self._socket.makefile("w")
            except BaseException:
                self._socket.close()
                raise
        self._reader, self._writer, self._sequence = reader, writer, 0
        if self._socket:
            try:
                self.request("initialize", {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "python-controller", "version": "2"},
                })
            except BaseException:
                self.close()
                raise

    def close(self):
        if self._socket:
            self._writer.close()
            self._reader.close()
            self._socket.close()
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def request(self, method, params):
        with self._request_lock:
            return self._request(method, params)

    def _request(self, method, params):
        self._sequence += 1
        request = {"jsonrpc": "2.0", "id": self._sequence, "method": method, "params": params}
        self._writer.write(json.dumps(request, allow_nan=False) + "\n")
        self._writer.flush()
        line = self._reader.readline(32 * 1024 * 1024)
        if not line:
            raise RuntimeError("MCP request failed: server closed the connection")
        response = json.loads(line)
        if response.get("id") != self._sequence:
            raise RuntimeError("MCP request failed: identity mismatch")
        if "error" in response:
            raise RuntimeError("MCP tool failed: " + json.dumps({"error": response["error"]}, allow_nan=False))
        return response["result"]

    def call(self, tool, **arguments):
        result = self.request("tools/call", {"name": tool, "arguments": arguments})
        if result.get("isError"):
            text = next((block.get("text", "") for block in result.get("content", []) if block.get("type") == "text"), "")
            raise RuntimeError("MCP tool failed: " + text)
        return result


def main():
    mode, project, entrypoint = sys.argv[1:]
    if mode != "run":
        raise SystemExit("usage: runtime.py run PROJECT ENTRYPOINT")
    sys.path.insert(0, project)
    spec = importlib.util.spec_from_file_location("submitted_controller", Path(project) / entrypoint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with Context() as context:
        module.main(context)


if __name__ == "__main__":
    main()
