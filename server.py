#!/usr/bin/env python3
"""Start LIBERO outside the sandbox, then a DSH session inside it.

The worker process owns the simulator, the GPU, and the scene files. A second
process owns Imajev. bubblewrap gets a clean workspace, the plugins, and the
unix sockets. The page on port 8765 is the monitor; the DSH page is only the chat.
"""
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/data/changzou/deepseek4robotics")
RUN = ROOT / "runs" / "libero90"
AGENT = RUN / "agent"
RUN_DIR = RUN
SOCK_DIR = RUN / "sock"
SOCKET_PATH = SOCK_DIR / "mcp.sock"
SANDBOX_HOME = RUN / "dsh-home"
PROFILE = ROOT / ".dsh-home" / "profiles" / "web"
PLUGIN = ROOT / "plugins" / "libero90"
IMAJEV_PLUGIN = ROOT / "plugins" / "imajev"
IMAJEV_ROOT = ROOT / "third_party" / "imajev"
IMAJEV_PYTHON = Path("/data/changzou/miniconda3/envs/imajev/bin/python")
HOST_HOME = ROOT / ".dsh-home"
PYTHON = Path("/data/changzou/miniconda3/envs/ds4ro/bin/python")
NODE = Path("/home/changzou/.nvm/versions/node/v24.22.0")
DSH = NODE / "bin" / "dsh"
MONITOR_PORT = 8765
WORKER_PORT = 8766
DSH_PORT = 3080
# Physical GPU placed in CUDA_VISIBLE_DEVICES. Lanes that share a card all use
# the same value; the EGL index inside each process stays 0.
GPU_VISIBLE = "0"
SKILL_FILE = ROOT / "skills" / "libero-jev-control" / "SKILL.md"
LIBERO_TASK_ID = 56
# DSH rewrites this file on boot by truncating it. Keep a private copy per session.
PROFILE_ROOT = """# dsh profile root — an empty entry list. The tree is composed as patches:
# each bundle in package.json's dsh.profile.bundles, then cordis.patch.yml, then any
# --patch overlays. Edit cordis.patch.yml, not this file.
[]
"""
WEB_URL_RE = re.compile(r"https?://\S+")

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO-90</title>
<style>
  body { margin: 0; background: #161616; color: #f2f2f2; font: 15px/1.45 sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 8px; }
  p { margin: 0 0 16px; color: #bdbdbd; }
  img { width: min(100%, 512px); height: auto; background: #000; }
  dl { display: grid; grid-template-columns: 7em 1fr; gap: 6px 12px; margin: 16px 0 0; }
  dt { color: #9a9a9a; }
  dd { margin: 0; }
</style>
</head>
<body>
<main>
  <h1>LIBERO-90</h1>
  <p>这里只看仿真。对话在 DSH 页面里进行。</p>
  <img id="frame" alt="agentview">
  <dl>
    <dt>状态</dt><dd id="state">连接中</dd>
    <dt>阶段</dt><dd id="phase">-</dd>
    <dt>步数</dt><dd id="step">-</dd>
    <dt>指令</dt><dd id="language">-</dd>
    <dt>原因</dt><dd id="reason">-</dd>
    <dt>错误</dt><dd id="error">-</dd>
  </dl>
</main>
<script>
const stateText = {idle:"空闲", running:"执行中", success:"成功", failed:"失败", error:"出错", offline:"未连接"};
const phaseText = {idle:"空闲", interactive:"交互", formal:"正式评测", closed:"已提交"};
async function tick() {
  try {
    const response = await fetch("/status", {cache: "no-store"});
    const status = await response.json();
    document.getElementById("state").textContent = stateText[status.state] || status.state || "-";
    document.getElementById("phase").textContent = phaseText[status.phase] || status.phase || "-";
    document.getElementById("step").textContent = (status.step ?? 0) + " / " + (status.max_steps ?? "-");
    document.getElementById("language").textContent = status.language || "-";
    document.getElementById("reason").textContent = status.reason || "-";
    document.getElementById("error").textContent = status.error || "-";
    const image = document.getElementById("frame");
    if (status.frame) image.src = "/frame.jpg?t=" + Date.now();
    else image.removeAttribute("src");
  } catch (error) {
    document.getElementById("state").textContent = "未连接";
  }
}
tick();
setInterval(tick, 700);
</script>
</body>
</html>
"""


def port_open(port):
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def require_ports_free():
    busy = [port for port in (DSH_PORT, MONITOR_PORT, WORKER_PORT) if port_open(port)]
    if busy:
        names = ", ".join(str(port) for port in busy)
        raise SystemExit(f"端口 {names} 已被占用。先停掉原来的 dsh web 和仿真进程，再运行 server.py。")


def use_lane(monitor_port, worker_port, dsh_port, gpu="0"):
    """Point this process at one lane. Call it before prepare_workspace.

    Several lanes may name the same physical GPU. Each still needs its own
    monitor port, worker port, and DSH port. The interactive server keeps
    8765, 8766, and 3080 and does not call this.
    """
    global MONITOR_PORT, WORKER_PORT, DSH_PORT, GPU_VISIBLE
    ports = (int(monitor_port), int(worker_port), int(dsh_port))
    if len(set(ports)) != 3:
        raise SystemExit(f"车道端口重复：{list(ports)}")
    MONITOR_PORT, WORKER_PORT, DSH_PORT = ports
    GPU_VISIBLE = str(gpu)


def rewrite_camera_endpoints(workspace, worker_port):
    """Point frozen copies at this lane's worker. Leave SKILL.md unchanged.

    The trained skill text must not gain a port. REFERENCE.md and controllers
    in the workspace copy are the files that name 127.0.0.1:8766.
    """
    old = "127.0.0.1:8766"
    new = f"127.0.0.1:{int(worker_port)}"
    if old == new:
        return
    root = Path(workspace)
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SKILL.md":
            continue
        if "vendor" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix not in {".md", ".py", ".txt"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if old not in text:
            continue
        path.write_text(text.replace(old, new), encoding="utf-8")


def socket_location(session_root):
    """Unix socket paths must be shorter than 108 bytes.

    Interactive sessions fit. A training session nested under skillopt/ does not,
    so those sockets live in a short directory that is bind-mounted as itself.
    """
    natural_dir = Path(session_root) / "sock"
    natural = natural_dir / "mcp.sock"
    if len(str(natural)) < 108:
        return natural_dir, natural
    digest = hashlib.sha1(str(Path(session_root).resolve()).encode()).hexdigest()[:10]
    short_dir = ROOT / ".socks" / digest
    return short_dir, short_dir / "mcp.sock"


def prepare_workspace(session_root=None, skill_text=None):
    """A new session directory. Previous workspaces, chats, and episode files stay unmounted.

    session_root places the session outside runs/libero90/sessions, so a training
    rollout is not picked up by scripts/sync_skill.py. skill_text replaces only
    SKILL.md; the rest of the skill tree stays frozen.
    """
    global AGENT, RUN_DIR, SOCK_DIR, SOCKET_PATH, SANDBOX_HOME
    if session_root is None:
        session_id = uuid.uuid4().hex[:8]
        root = RUN / "sessions" / session_id
    else:
        root = Path(session_root)
    AGENT = root / "workspace"
    RUN_DIR = root / "run"
    SOCK_DIR, SOCKET_PATH = socket_location(root)
    SANDBOX_HOME = root / "dsh-home"
    for path in (AGENT / "code", RUN_DIR, SOCK_DIR, SANDBOX_HOME / "profiles"):
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SKILL_FILE, AGENT / "SKILL.md")
    for path in sorted(SKILL_FILE.parent.glob("*.md")):
        if path.name != "SKILL.md":
            shutil.copy2(path, AGENT / path.name)
    tasks = SKILL_FILE.parent / "tasks"
    if tasks.is_dir():
        shutil.copytree(tasks, AGENT / "tasks", dirs_exist_ok=True)
    figures = SKILL_FILE.parent / "figures"
    if figures.is_dir():
        shutil.copytree(figures, AGENT / "figures", dirs_exist_ok=True)
    starter = SKILL_FILE.parent / "code"
    if starter.is_dir():
        shutil.copytree(starter, AGENT / "code", dirs_exist_ok=True)
    # DSH rewrites cordis.yml on every boot by truncating it first. Parallel
    # lanes must not share that file, or one reads the empty window and exits.
    profile = SANDBOX_HOME / "profiles" / "web"
    if profile.is_symlink():
        profile.unlink()
    if not profile.exists():
        shutil.copytree(PROFILE, profile, symlinks=True)
    # The shared template can be mid-rewrite while an older lane is still running.
    (profile / "cordis.yml").write_text(PROFILE_ROOT, encoding="utf-8")
    for name in (".credentials.yaml", ".anonymous-user-id"):
        source = HOST_HOME / name
        if source.exists():
            destination = SANDBOX_HOME / name
            shutil.copy2(source, destination)
            os.chmod(destination, 0o600)
    if skill_text is not None:
        (AGENT / "SKILL.md").write_text(skill_text, encoding="utf-8")
    rewrite_camera_endpoints(AGENT, WORKER_PORT)
    print(f"工作区 {AGENT}", flush=True)


def die_with_parent():
    """SIGTERM this process when server.py is gone, even if its cleanup never runs."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGTERM), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl PR_SET_PDEATHSIG")
    if os.getppid() == 1:
        os._exit(0)


def worker_environment(task_id=None, hard_cap=None):
    """One visible GPU, so the EGL device inside the process is always 0."""
    return {
        "PATH": f"{PYTHON.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(AGENT),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(GPU_VISIBLE),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": "0",
        "LIBERO_GPU": "0",
        "LIBERO_SOCKET": str(SOCKET_PATH),
        "LIBERO_RUN_DIR": str(RUN_DIR),
        "LIBERO_WORKSPACE": str(AGENT),
        "LIBERO_HARD_CAP": str(1000 if hard_cap is None else int(hard_cap)),
        "LIBERO_TASK_ID": str(LIBERO_TASK_ID if task_id is None else int(task_id)),
        "LIBERO_PORT": str(WORKER_PORT),
        "LIBERO_CONFIG_PATH": "/home/changzou/.libero",
    }


def start_worker(task_id=None, hard_cap=None):
    log_path = RUN_DIR / "worker.log"
    log = open(log_path, "ab", buffering=0)
    env = worker_environment(task_id, hard_cap)
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    return subprocess.Popen(
        [str(PYTHON), str(PLUGIN / "worker.py")],
        cwd=str(RUN_DIR),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        preexec_fn=die_with_parent,
    )


def imajev_socket_path():
    """Stay under the 108-byte unix-socket limit. mcp.sock is the length check."""
    path = SOCK_DIR / "imajev.sock"
    if len(str(path)) < 108:
        return path
    return SOCK_DIR / "ij.sock"


def start_imajev():
    if not IMAJEV_PYTHON.exists():
        raise SystemExit(f"找不到 {IMAJEV_PYTHON}。")
    socket_path = imajev_socket_path()
    if socket_path.exists():
        socket_path.unlink()
    log_path = RUN_DIR / "imajev.log"
    log = open(log_path, "ab", buffering=0)
    env = {
        "PATH": f"{IMAJEV_PYTHON.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(RUN_DIR),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(GPU_VISIBLE),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(IMAJEV_ROOT / ".cache" / "huggingface"),
        "IMAJEV_ROOT": str(IMAJEV_ROOT),
        "IMAJEV_SOCKET": str(socket_path),
        "IMAJEV_WORKSPACE": str(AGENT),
        "IMAJEV_RUN_DIR": str(RUN_DIR),
        "PYTHONPATH": f"{IMAJEV_ROOT / 'src'}:{IMAJEV_ROOT / 'scripts'}",
    }
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    return subprocess.Popen(
        [str(IMAJEV_PYTHON), str(IMAJEV_PLUGIN / "worker.py")],
        cwd=str(IMAJEV_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        preexec_fn=die_with_parent,
    )


def wait_for_unix(process, path, log_path, timeout, label):
    deadline = time_now() + timeout
    while time_now() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"{label}进程退出了，退出码 {process.returncode}。日志在 {log_path}。")
        if Path(path).exists():
            with socket.socket(socket.AF_UNIX) as probe:
                try:
                    probe.connect(str(path))
                    return
                except OSError:
                    pass
        threading.Event().wait(0.3)
    raise SystemExit(f"等待 {path} 超时。日志在 {log_path}。")


def wait_for_socket(process, timeout=180):
    wait_for_unix(process, SOCKET_PATH, RUN_DIR / "worker.log", timeout, "仿真")


def wait_for_imajev(process, timeout=240):
    """Wait until the weights are on the GPU. The socket opens before that."""
    path = imajev_socket_path()
    log_path = RUN_DIR / "imajev.log"
    print("正在加载决策模型。", flush=True)
    wait_for_unix(process, path, log_path, 30, "决策模型")
    deadline = time_now() + timeout
    while time_now() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"决策模型进程退出了，退出码 {process.returncode}。日志在 {log_path}。")
        try:
            with socket.socket(socket.AF_UNIX) as sock:
                sock.settimeout(5)
                sock.connect(str(path))
                reader = sock.makefile("r", encoding="utf-8")
                writer = sock.makefile("w", encoding="utf-8")
                writer.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "ready",
                    "params": {},
                }) + "\n")
                writer.flush()
                line = reader.readline()
        except (OSError, json.JSONDecodeError):
            threading.Event().wait(0.5)
            continue
        if not line:
            threading.Event().wait(0.5)
            continue
        message = json.loads(line)
        if message.get("error"):
            detail = message["error"].get("message") if isinstance(message["error"], dict) else message["error"]
            raise SystemExit(f"决策模型没有加载成功：{detail}。日志在 {log_path}。")
        if (message.get("result") or {}).get("ready"):
            print("决策模型已加载。", flush=True)
            return
        threading.Event().wait(0.5)
    raise SystemExit(f"决策模型加载超时。日志在 {log_path}。")


def time_now():
    return __import__("time").monotonic()


def socket_rpc(reader, writer, method, params, ident):
    writer.write(json.dumps({
        "jsonrpc": "2.0",
        "id": ident,
        "method": method,
        "params": params,
    }) + "\n")
    writer.flush()
    line = reader.readline()
    if not line:
        raise RuntimeError("仿真套接字在打开任务时断开了")
    message = json.loads(line)
    if message.get("error"):
        detail = message["error"].get("message") if isinstance(message["error"], dict) else message["error"]
        raise RuntimeError(str(detail))
    return message.get("result")


def open_episode(init_state_id=None):
    """Open the one episode before the agent starts. The agent has no start tool.

    init_state_id pins the saved reset. Omit it and the worker draws one at random.
    """
    arguments = {}
    if init_state_id is not None:
        arguments["init_state_id"] = int(init_state_id)
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(300)
        sock.connect(str(SOCKET_PATH))
        reader = sock.makefile("r", encoding="utf-8")
        writer = sock.makefile("w", encoding="utf-8")
        socket_rpc(reader, writer, "initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "server", "version": "1"},
        }, 1)
        socket_rpc(reader, writer, "tools/call", {
            "name": "start_episode",
            "arguments": arguments,
        }, 2)
    status_path = RUN_DIR / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    language = status.get("language") or ""
    init_state = status.get("init_state_id")
    if not language:
        raise SystemExit("仿真没有返回任务指令。")
    task_id = status.get("task_id", LIBERO_TASK_ID)
    (AGENT / "TASK.txt").write_text(task_prompt(language, task_id), encoding="utf-8")
    print(f"任务已打开：{language}（init_state {init_state}）", flush=True)


def task_prompt(language, task_id):
    return f"Read SKILL.md and follow it. LIBERO-90 task {int(task_id)} is to {language}.\n"


def deepseek_api_key():
    """The key DSH resolves as DEEPSEEK_API_KEY. Never log the value."""
    current = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if current:
        return current
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() == "DEEPSEEK_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit("找不到 DEEPSEEK_API_KEY。把它写在项目 .env 或当前环境里。")


def bind_if_present(command, source, writable=False):
    if Path(source).exists():
        command.extend(["--bind" if writable else "--ro-bind", source, source])


def sandbox_command(program):
    command = [
        "bwrap",
        "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--die-with-parent", "--new-session", "--cap-drop", "ALL",
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--tmpfs", "/dev/shm",
    ]
    for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
        bind_if_present(command, source)
    resolv = Path("/etc/resolv.conf")
    if resolv.is_symlink():
        bind_if_present(command, os.path.realpath(resolv))
    bind_if_present(command, str(NODE))
    command.extend([
        "--ro-bind", str(PROFILE), str(PROFILE),
        "--bind", str(SANDBOX_HOME), str(SANDBOX_HOME),
        "--bind", str(AGENT), str(AGENT),
        "--bind", str(SOCK_DIR), str(SOCK_DIR),
        "--ro-bind", str(PLUGIN), str(PLUGIN),
        "--ro-bind", str(IMAJEV_PLUGIN), str(IMAJEV_PLUGIN),
        "--chdir", str(AGENT),
        "--setenv", "PATH", f"{NODE / 'bin'}:/usr/bin:/bin",
        "--setenv", "HOME", str(AGENT),
        "--setenv", "DSH_HOME", str(SANDBOX_HOME),
        "--setenv", "LIBERO_SOCKET", str(SOCKET_PATH),
        "--setenv", "IMAJEV_SOCKET", str(imajev_socket_path()),
        "--setenv", "PYTHONPATH", str(PLUGIN),
        "--setenv", "LANG", "C.UTF-8",
        "--setenv", "TERM", os.environ.get("TERM", "xterm-256color"),
    ])
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        if os.environ.get(name):
            command.extend(["--setenv", name, os.environ[name]])
    command.append("--")
    command.extend(program)
    return command


def start_dsh():
    if not DSH.exists():
        raise SystemExit(f"找不到 {DSH}。")
    env = {
        "PATH": f"{NODE / 'bin'}:/usr/bin:/bin",
        "HOME": str(AGENT),
        "DSH_HOME": str(SANDBOX_HOME),
        "LIBERO_SOCKET": str(SOCKET_PATH),
        "IMAJEV_SOCKET": str(imajev_socket_path()),
        "LANG": "C.UTF-8",
        "TERM": os.environ.get("TERM", "xterm-256color"),
        "DEEPSEEK_API_KEY": deepseek_api_key(),
    }
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    return subprocess.Popen(
        sandbox_command([str(DSH), "web", "--no-open", "--port", str(DSH_PORT)]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        preexec_fn=die_with_parent,
    )


def relay_dsh(process):
    """Forward DSH output and return the launch URL once it is printed."""
    found = threading.Event()
    holder = {}

    def relay():
        stream = process.stdout
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            sys.stdout.buffer.write(raw)
            sys.stdout.buffer.flush()
            if "url" in holder:
                continue
            match = WEB_URL_RE.search(raw.decode("utf-8", "replace"))
            if match and "dsh web:" in raw.decode("utf-8", "replace"):
                holder["url"] = match.group(0)
                found.set()

    threading.Thread(target=relay, daemon=True).start()
    deadline = time_now() + 180
    while not found.wait(0.5):
        if process.poll() is not None:
            raise SystemExit(f"DSH 已退出，退出码 {process.returncode}。")
        if time_now() >= deadline:
            raise SystemExit("DSH 没有打印出对话页面地址。")
    if process.poll() is not None:
        raise SystemExit(f"DSH 已退出，退出码 {process.returncode}。")
    return holder["url"]


def rpc(opener, origin, method, args):
    body = json.dumps({
        "type": "client-request",
        "rpcId": str(uuid.uuid4()),
        "method": method,
        "payload": {"args": args},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{origin}/api/{method}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} 返回 HTTP {exc.code}：{detail}") from exc
    result = payload.get("result") if isinstance(payload, dict) else None
    if not isinstance(result, dict) or not result.get("ok"):
        error = result.get("error") if isinstance(result, dict) else None
        message = error.get("message") if isinstance(error, dict) else "没有返回结果"
        raise RuntimeError(f"{method} 失败：{message}")
    value = result.get("value")
    return value if isinstance(value, dict) else {}


def start_chat(web_url):
    """Open one new DSH conversation and send the task prompt."""
    parsed = urllib.request.urlparse(web_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    with opener.open(web_url, timeout=30) as response:
        response.read()
    created = rpc(opener, origin, "session/create", {"request": {"cwd": str(AGENT)}})
    session_id = created.get("sessionId")
    if not session_id:
        raise RuntimeError("session/create 没有返回会话编号。")
    rpc(opener, origin, "session/prompt", {"request": {
        "requestId": str(uuid.uuid4()),
        "sessionId": session_id,
        "mode": "queue",
        "content": [{"type": "text", "text": (AGENT / "TASK.txt").read_text(encoding="utf-8")}],
        "clientTimeZone": "Asia/Shanghai",
    }})
    return session_id


def start_monitor():
    run_dir = RUN_DIR

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/status":
                file = run_dir / "status.json"
                if file.exists():
                    body = file.read_bytes()
                else:
                    body = b'{"state":"idle","phase":"idle","step":0,"max_steps":300,"frame":false,"language":"","reason":"","error":""}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/frame.jpg":
                file = run_dir / "frame.jpg"
                if not file.exists():
                    self.send_response(404)
                    self.end_headers()
                    return
                body = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", MONITOR_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def _interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    require_ports_free()
    prepare_workspace()
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    print("正在启动仿真和决策模型。", flush=True)
    worker = None
    imajev = None
    monitor = None
    dsh = None
    try:
        imajev = start_imajev()
        worker = start_worker()
        wait_for_imajev(imajev)
        wait_for_socket(worker)
        print("正在打开任务。", flush=True)
        open_episode()
        monitor = start_monitor()
        print(f"监控页面 http://127.0.0.1:{MONITOR_PORT}/", flush=True)
        print("正在沙箱里启动 DSH。", flush=True)
        dsh = start_dsh()
        web_url = relay_dsh(dsh)
        try:
            session_id = start_chat(web_url)
        except Exception as exc:
            print(f"自动开始对话失败：{exc}", flush=True)
            print("对话页面仍是上面那一行地址，可以手动新建对话。", flush=True)
        else:
            print(f"已新建对话 {session_id}，任务已发送。打开上面那一行地址即可看到。", flush=True)
        return dsh.wait()
    finally:
        stop(dsh)
        stop(imajev)
        stop(worker)
        if monitor is not None:
            monitor.shutdown()


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        sys.exit(130)
