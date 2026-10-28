#!/usr/bin/env python3
"""Exploration session for one RoboTwin task.

The agent resets its own rounds and spends at most --budget planned actions
(default 1200). Each reset samples a seed outside the eval scan. The submitted
controller is then rolled out, without the agent, on --eval-seeds held-out seeds.

--task all is the same session across every RoboTwin task. reset then samples
a task as well as a seed, and submit rolls the controller out on --eval-seeds
seeds of every task.

This process does not use plugins/libero90 or plugins/libero-explore.
Monitor 8975, worker 8976, chat 3380, so it can run beside the LIBERO servers.
The simulator and imajev share --gpu.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path("/data/changzou/deepseek4robotics")
sys.path.insert(0, str(ROOT))
import server


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

EXPLORE_PLUGIN = ROOT / "plugins" / "robotwin-explore"
SKILL_PATH = ROOT / "skills" / "robotwin-explore" / "SKILL.md"
ROBOTWIN_ROOT = ROOT / "third_party" / "RoboTwin"
RUN_ROOT = ROOT / "runs" / "robotwin-explore"
PYTHON = Path("/data/changzou/miniconda3/envs/robotwin/bin/python")
MONITOR_PORT = 8975
WORKER_PORT = 8976
DSH_PORT = 3380

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboTwin 探索</title>
<style>
  body { margin: 0; background: #161616; color: #f2f2f2; font: 15px/1.45 sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 8px; }
  p { margin: 0 0 16px; color: #bdbdbd; }
  .cams { display: flex; gap: 12px; flex-wrap: wrap; }
  figure { margin: 0; }
  figcaption { color: #9a9a9a; font-size: 13px; margin-top: 4px; }
  img { width: min(100%, 320px); height: auto; background: #000; }
  dl { display: grid; grid-template-columns: 8em 1fr; gap: 6px 12px; margin: 16px 0 0; }
  dt { color: #9a9a9a; }
  dd { margin: 0; }
</style>
</head>
<body>
<main>
  <h1>RoboTwin 探索</h1>
  <p>模型在动作预算内自己 reset，提交控制器后，在没练过的 seed 上评测。对话在 DSH 页面。</p>
  <div class="cams">
    <figure><img id="frame" alt="head"><figcaption>头顶</figcaption></figure>
    <figure><img id="left" alt="left wrist"><figcaption>左腕</figcaption></figure>
    <figure><img id="right" alt="right wrist"><figcaption>右腕</figcaption></figure>
  </div>
  <dl>
    <dt>状态</dt><dd id="state">连接中</dd>
    <dt>阶段</dt><dd id="phase">-</dd>
    <dt>本轮</dt><dd id="step">-</dd>
    <dt>预算</dt><dd id="budget">-</dd>
    <dt>轮次</dt><dd id="round">-</dd>
    <dt>任务</dt><dd id="task">-</dd>
    <dt>动作</dt><dd id="action">-</dd>
    <dt>泛化评测</dt><dd id="eval">-</dd>
    <dt>指令</dt><dd id="language">-</dd>
    <dt>原因</dt><dd id="reason">-</dd>
    <dt>错误</dt><dd id="error">-</dd>
  </dl>
</main>
<script>
const stateText = {idle:"空闲", running:"执行中", success:"成功", failed:"失败", error:"出错", offline:"未连接"};
const phaseText = {idle:"空闲", opening:"打开场景", explore:"探索", formal:"试跑控制器", evaluating:"泛化评测", closed:"评测结束"};
async function tick() {
  try {
    const response = await fetch("/status", {cache: "no-store"});
    const status = await response.json();
    document.getElementById("state").textContent = stateText[status.state] || status.state || "-";
    document.getElementById("phase").textContent = phaseText[status.phase] || status.phase || "-";
    document.getElementById("step").textContent = (status.step ?? 0) + " / " + (status.max_steps ?? "-");
    document.getElementById("budget").textContent = (status.budget_used ?? 0) + " / " + (status.budget_total ?? "-");
    document.getElementById("round").textContent = status.round ?? "-";
    document.getElementById("task").textContent = status.task_name || "-";
    document.getElementById("action").textContent = status.action_type || "-";
    const total = Number(status.eval_total || 0);
    document.getElementById("eval").textContent = total
      ? (status.eval_success ?? 0) + " / " + total + (status.eval_done ? " 完成" : " 进行中")
      : "-";
    document.getElementById("language").textContent = status.language || "-";
    document.getElementById("reason").textContent = status.reason || "-";
    document.getElementById("error").textContent = status.eval_error || status.error || "-";
    const stamp = Date.now();
    if (status.frame) {
      document.getElementById("frame").src = "/frame.jpg?t=" + stamp;
      document.getElementById("left").src = "/left.jpg?t=" + stamp;
      document.getElementById("right").src = "/right.jpg?t=" + stamp;
    }
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


def parse_args():
    names = load_task_names(ROBOTWIN_ROOT / "env_cfg" / "eval" / "all_tasks.yml")
    parser = argparse.ArgumentParser(description="RoboTwin 探索会话：预算内随机 reset，提交控制器，再在留出的 seed 上重放。")
    parser.add_argument("--task", default="adjust_bottle", help="任务名，或 all。默认 adjust_bottle")
    parser.add_argument("--task-config", default="demo_clean", help="env_cfg/task_config 里的配置，默认 demo_clean")
    parser.add_argument("--action-type", default="ee", choices=("ee", "qpos"), help="官方 take_action 的动作。ee 是绝对末端位姿，qpos 是绝对关节")
    parser.add_argument("--instruction", default="seen", choices=("seen", "unseen"), help="只在 --expert-check 时用来填模板")
    parser.add_argument("--budget", type=int, default=1200, help="探索阶段 planned action 上限，评测不计入")
    parser.add_argument("--horizon", default="official", help="单轮动作上限。official 用任务自己的 step limit")
    parser.add_argument("--eval-seeds", type=int, default=5, help="每个任务留出的评测 seed 数")
    parser.add_argument("--seed-start", type=int, default=100000, help="评测 seed 起点，和官方 eval 一样从 100000 起")
    parser.add_argument("--expert-check", action="store_true", help="丢掉脚本专家做不成的 seed。每个 seed 会先完整跑一遍专家")
    parser.add_argument("--raytrace", action="store_true", help="相机用 RoboTwin 默认光追。默认是光栅，方便交互")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if str(args.task).lower() == "all":
        args.task = "all"
    elif args.task not in names:
        parser.error("task 必须是任务名，或 all。例如 adjust_bottle")
    if args.budget < 1 or args.eval_seeds < 1 or args.seed_start < 0:
        parser.error("budget、eval-seeds 必须为正，seed-start 不能为负")
    if str(args.horizon).lower() in ("official", "0"):
        args.horizon = None
    else:
        try:
            args.horizon = int(args.horizon)
        except ValueError:
            parser.error("horizon 是正整数，或 official")
        if args.horizon < 1:
            parser.error("horizon 是正整数，或 official")
    config = ROBOTWIN_ROOT / "env_cfg" / "task_config" / f"{args.task_config}.yml"
    if not config.is_file():
        parser.error(f"找不到任务配置 {config.name}")
    return args


def start_worker(args):
    log_path = server.RUN_DIR / "worker.log"
    log = open(log_path, "ab", buffering=0)
    env = {
        "PATH": f"{PYTHON.parent}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": os.environ.get("HOME", "/home/changzou"),
        "LANG": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": str(server.GPU_VISIBLE),
        "ROBOTWIN_ROOT": str(ROBOTWIN_ROOT),
        "ROBOTWIN_SOCKET": str(server.SOCKET_PATH),
        "LIBERO_SOCKET": str(server.SOCKET_PATH),
        "ROBOTWIN_RUN_DIR": str(server.RUN_DIR),
        "ROBOTWIN_WORKSPACE": str(server.AGENT),
        "ROBOTWIN_PORT": str(WORKER_PORT),
        "ROBOTWIN_TASK": str(args.task),
        "ROBOTWIN_TASK_CONFIG": args.task_config,
        "ROBOTWIN_ACTION_TYPE": args.action_type,
        "ROBOTWIN_INSTRUCTION": args.instruction,
        "ROBOTWIN_BUDGET": str(args.budget),
        "ROBOTWIN_HORIZON": "official" if args.horizon is None else str(args.horizon),
        "ROBOTWIN_EVAL_SEEDS": str(args.eval_seeds),
        "ROBOTWIN_SEED_START": str(args.seed_start),
        "ROBOTWIN_EXPERT_CHECK": "1" if args.expert_check else "0",
        "ROBOTWIN_RASTER": "0" if args.raytrace else "1",
    }
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    return subprocess.Popen(
        [str(PYTHON), str(EXPLORE_PLUGIN / "worker.py")],
        cwd=str(ROBOTWIN_ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        start_new_session=True,
        preexec_fn=server.die_with_parent,
    )


def install_bundle(profile, name, source):
    dest = profile / "node_modules" / "@local" / name
    if dest.is_symlink():
        dest.unlink()
    elif dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def install_plugin():
    """This session's DSH profile: the RoboTwin tools and imajev."""
    profile = server.SANDBOX_HOME / "profiles" / "web"
    manifest_path = profile / "package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependencies"] = {
        "@local/robotwin-explore": f"file:{EXPLORE_PLUGIN}",
        "@local/imajev": f"file:{server.IMAJEV_PLUGIN}",
    }
    manifest.setdefault("dsh", {}).setdefault("profile", {})["bundles"] = [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        "@local/robotwin-explore",
        "@local/imajev",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    install_bundle(profile, "robotwin-explore", EXPLORE_PLUGIN)
    install_bundle(profile, "imajev", server.IMAJEV_PLUGIN)


def open_episode():
    """Open the first round. Planner startup can take several minutes."""
    with server.socket.socket(server.socket.AF_UNIX) as sock:
        sock.settimeout(900)
        sock.connect(str(server.SOCKET_PATH))
        reader = sock.makefile("r", encoding="utf-8")
        writer = sock.makefile("w", encoding="utf-8")
        server.socket_rpc(reader, writer, "initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "server", "version": "1"},
        }, 1)
        server.socket_rpc(reader, writer, "tools/call", {
            "name": "start_episode",
            "arguments": {},
        }, 2)


def write_task(args):
    status = json.loads((server.RUN_DIR / "status.json").read_text(encoding="utf-8"))
    if not status.get("language"):
        raise SystemExit("仿真没有返回任务指令。")
    if args.task == "all":
        scope = (
            f"This session is every RoboTwin task, not one task.\n"
            f"The round already open says: {status.get('language')}.\n"
            f"Round {status.get('round')} is already open. reset samples another task and another seed. "
            f"You are not told which. Read the instruction again.\n"
        )
        rollout = f"After submit the same program is rolled out on {args.eval_seeds} held-out seeds of every task.\n"
    else:
        scope = (
            f"RoboTwin task {status.get('task_name')}: {status.get('language')}.\n"
            f"Round {status.get('round')} is already open. reset samples the next seed at random.\n"
        )
        rollout = f"After submit the same program is rolled out on {args.eval_seeds} held-out seeds of this task.\n"
    horizon = "the task's official step limit" if args.horizon is None else str(args.horizon)
    text = (
        f"Read SKILL.md and follow it.\n"
        f"{scope}"
        f"Action type: {args.action_type}. Task config: {args.task_config}.\n"
        f"Exploration budget: {status.get('budget_used', 0)} / {status.get('budget_total')} planned actions.\n"
        f"This round's action cap: {status.get('max_steps')} ({horizon} per round, clipped by the budget).\n"
        f"Write code/<name>/controller.py, trial it with run, reset, and submit.\n"
        f"{rollout}"
    )
    (server.AGENT / "TASK.txt").write_text(text, encoding="utf-8")
    return status


def start_monitor():
    run_dir = server.RUN_DIR

    class Handler(server.BaseHTTPRequestHandler):
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
                    body = b'{"state":"idle","phase":"idle","step":0,"max_steps":400,"frame":false,"language":"","reason":"","error":""}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in ("/frame.jpg", "/left.jpg", "/right.jpg", "/wrist.jpg"):
                file = run_dir / path.lstrip("/")
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

    monitor = server.ThreadingHTTPServer(("127.0.0.1", MONITOR_PORT), Handler)
    threading.Thread(target=monitor.serve_forever, daemon=True).start()
    return monitor


def watch_eval(process):
    announced = False
    while process.poll() is None and not announced:
        path = server.RUN_DIR / "eval.json"
        if path.is_file():
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                threading.Event().wait(1)
                continue
            successes = report.get("successes", 0)
            total = report.get("total", 0)
            rate = float(report.get("rate") or 0)
            scope = "全部任务" if report.get("all_tasks") else "当前任务"
            print(f"{scope}的留出 seed 评测 {successes}/{total} = {rate:.1%}", flush=True)
            if report.get("error"):
                print(f"评测中断：{report['error']}", flush=True)
            print(f"明细 {path}", flush=True)
            announced = True
        else:
            threading.Event().wait(2)


def main():
    if not EXPLORE_PLUGIN.joinpath("worker.py").is_file():
        raise SystemExit(f"找不到 {EXPLORE_PLUGIN}")
    if not SKILL_PATH.is_file():
        raise SystemExit(f"找不到 {SKILL_PATH}")
    if not PYTHON.is_file():
        raise SystemExit(f"找不到 {PYTHON}")
    if shutil.which("bwrap") is None:
        raise SystemExit("找不到 bwrap，控制器无法在隔离环境里评测。")
    args = parse_args()
    server.use_lane(MONITOR_PORT, WORKER_PORT, DSH_PORT, args.gpu)
    server.require_ports_free()
    server.PLUGIN = EXPLORE_PLUGIN
    server.SKILL_FILE = SKILL_PATH
    session_id = __import__("uuid").uuid4().hex[:8]
    session_root = RUN_ROOT / "sessions" / session_id
    skill_text = SKILL_PATH.read_text(encoding="utf-8")
    server.prepare_workspace(session_root=session_root, skill_text=skill_text)
    for path in server.AGENT.glob("*.md"):
        if path.name != "SKILL.md":
            path.unlink()
    install_plugin()
    signal.signal(signal.SIGTERM, server._interrupt)
    signal.signal(signal.SIGHUP, server._interrupt)
    horizon = "各任务官方上限" if args.horizon is None else f"单轮上限 {args.horizon}"
    scope = "全部任务" if args.task == "all" else f"任务 {args.task}"
    print(
        f"探索预算 {args.budget} 次规划动作，{horizon}，{scope}，动作 {args.action_type}，配置 {args.task_config}。"
        f"reset 换一个探索 seed，提交后每个任务评测 {args.eval_seeds} 个留出 seed（从 {args.seed_start} 起）。",
        flush=True,
    )
    if args.expert_check:
        print("打开场景前会先跑脚本专家，做不成的 seed 会被丢掉。", flush=True)
    else:
        print("不稳定的 seed 会跳过。脚本专家默认不跑；要和官方评测同一批 seed，加上 --expert-check。", flush=True)
    print("评测步数不计入预算。LIBERO 的 server 和插件没有改。", flush=True)
    worker = None
    imajev = None
    monitor = None
    dsh = None
    try:
        imajev = server.start_imajev()
        worker = start_worker(args)
        print(f"imajev {server.imajev_socket_path()}", flush=True)
        server.wait_for_imajev(imajev)
        server.wait_for_socket(worker)
        print("正在打开第一轮。规划器第一次加载可能要几分钟。", flush=True)
        open_episode()
        status = write_task(args)
        operator_path = server.RUN_DIR / "operator.json"
        operator = json.loads(operator_path.read_text(encoding="utf-8")) if operator_path.is_file() else {}
        print(
            f"任务：{status.get('language')}（{operator.get('task_name', status.get('task_name'))} seed {operator.get('seed', '-')}）",
            flush=True,
        )
        monitor = start_monitor()
        print(f"监控页面 http://127.0.0.1:{MONITOR_PORT}/", flush=True)
        print("正在沙箱里启动 DSH。", flush=True)
        dsh = server.start_dsh()
        web_url = server.relay_dsh(dsh)
        try:
            chat_id = server.start_chat(web_url)
        except Exception as exc:
            print(f"自动开始对话失败：{exc}", flush=True)
            print("对话页面仍是上面那一行地址，可以手动新建对话。", flush=True)
        else:
            print(f"已新建对话 {chat_id}，任务已发送。打开上面那一行地址即可看到。", flush=True)
        threading.Thread(target=watch_eval, args=(dsh,), daemon=True).start()
        return dsh.wait()
    finally:
        server.stop(dsh)
        server.stop(imajev)
        server.stop(worker)
        if monitor is not None:
            monitor.shutdown()


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except KeyboardInterrupt:
        sys.exit(130)
