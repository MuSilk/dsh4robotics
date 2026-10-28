#!/usr/bin/env python3
"""Exploration session for one LIBERO-90 task.

The agent resets its own rounds and spends at most --budget environment steps
(default 2400). Each reset samples a scene at random. The submitted controller
is then rolled out, without the agent, on every initial scene of the task.

--task all is the same session across every LIBERO-90 task. reset then samples
a task as well as a scene, and submit rolls the controller out on every scene
of every task.

This process does not use plugins/libero90. The interactive server.py is unchanged.
Monitor 8865, worker 8866, chat 3180, so it can run beside the interactive server.
Two simulators on one GPU will contend for memory.
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

EXPLORE_PLUGIN = ROOT / "plugins" / "libero-explore"
SKILL_PATH = ROOT / "skills" / "libero-explore" / "SKILL.md"
RUN_ROOT = ROOT / "runs" / "libero-explore"
MONITOR_PORT = 8865
WORKER_PORT = 8866
DSH_PORT = 3180

PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO 探索</title>
<style>
  body { margin: 0; background: #161616; color: #f2f2f2; font: 15px/1.45 sans-serif; }
  main { max-width: 760px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 20px; font-weight: 600; margin: 0 0 8px; }
  p { margin: 0 0 16px; color: #bdbdbd; }
  img { width: min(100%, 512px); height: auto; background: #000; }
  dl { display: grid; grid-template-columns: 8em 1fr; gap: 6px 12px; margin: 16px 0 0; }
  dt { color: #9a9a9a; }
  dd { margin: 0; }
</style>
</head>
<body>
<main>
  <h1>LIBERO 探索</h1>
  <p>模型在探索预算内自己 reset，提交控制器后，在没见过的初始状态上评测。对话在 DSH 页面。</p>
  <img id="frame" alt="agentview">
  <dl>
    <dt>状态</dt><dd id="state">连接中</dd>
    <dt>阶段</dt><dd id="phase">-</dd>
    <dt>本轮</dt><dd id="step">-</dd>
    <dt>预算</dt><dd id="budget">-</dd>
    <dt>轮次</dt><dd id="round">-</dd>
    <dt>初始状态</dt><dd id="init">-</dd>
    <dt>泛化评测</dt><dd id="eval">-</dd>
    <dt>指令</dt><dd id="language">-</dd>
    <dt>原因</dt><dd id="reason">-</dd>
    <dt>错误</dt><dd id="error">-</dd>
  </dl>
</main>
<script>
const stateText = {idle:"空闲", running:"执行中", success:"成功", failed:"失败", error:"出错", offline:"未连接"};
const phaseText = {idle:"空闲", explore:"探索", formal:"试跑控制器", evaluating:"泛化评测", closed:"评测结束"};
async function tick() {
  try {
    const response = await fetch("/status", {cache: "no-store"});
    const status = await response.json();
    document.getElementById("state").textContent = stateText[status.state] || status.state || "-";
    document.getElementById("phase").textContent = phaseText[status.phase] || status.phase || "-";
    document.getElementById("step").textContent = (status.step ?? 0) + " / " + (status.max_steps ?? "-");
    document.getElementById("budget").textContent = (status.budget_used ?? 0) + " / " + (status.budget_total ?? "-");
    document.getElementById("round").textContent = status.round ?? "-";
    document.getElementById("init").textContent = status.init_state_id ?? "-";
    const total = Number(status.eval_total || 0);
    document.getElementById("eval").textContent = total
      ? (status.eval_success ?? 0) + " / " + total + (status.eval_done ? " 完成" : " 进行中")
      : "-";
    document.getElementById("language").textContent = status.language || "-";
    document.getElementById("reason").textContent = status.reason || "-";
    document.getElementById("error").textContent = status.eval_error || status.error || "-";
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


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO 探索会话：预算内随机 reset，提交控制器，再在全部初始场景上重放。")
    parser.add_argument("--task", default=str(server.LIBERO_TASK_ID), help="LIBERO-90 任务编号，或 all 表示全部任务")
    parser.add_argument("--budget", type=int, default=2400, help="探索阶段环境步上限，评测不计入")
    parser.add_argument("--horizon", type=int, default=400, help="单轮步上限，LIBERO-90 官方是 400")
    parser.add_argument("--eval-inits", default="all", help="每个任务评测的初始状态数，默认 all 为全部")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if str(args.task).lower() == "all":
        args.task = "all"
    else:
        try:
            args.task = int(args.task)
        except ValueError:
            parser.error("task 是任务编号，或 all")
        if args.task < 0:
            parser.error("task 是任务编号，或 all")
    if args.budget < 1 or args.horizon < 1:
        parser.error("budget、horizon 必须为正")
    if str(args.eval_inits).lower() == "all":
        args.eval_inits = None
    else:
        try:
            args.eval_inits = int(args.eval_inits)
        except ValueError:
            parser.error("eval-inits 是正整数，或 all")
        if args.eval_inits < 1:
            parser.error("eval-inits 是正整数，或 all")
    return args


def start_worker(args):
    log_path = server.RUN_DIR / "worker.log"
    log = open(log_path, "ab", buffering=0)
    env = server.worker_environment(0 if args.task == "all" else args.task, args.horizon)
    if args.task == "all":
        env["LIBERO_TASK_ID"] = "all"
    env["LIBERO_BUDGET"] = str(args.budget)
    env["LIBERO_EVAL_INITS"] = "all" if args.eval_inits is None else str(args.eval_inits)
    library_path = os.environ.get("LD_LIBRARY_PATH")
    if library_path:
        env["LD_LIBRARY_PATH"] = library_path
    return subprocess.Popen(
        [str(server.PYTHON), str(EXPLORE_PLUGIN / "worker.py")],
        cwd=str(server.RUN_DIR),
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
    """This session's DSH profile: the explore robot tools and imajev."""
    profile = server.SANDBOX_HOME / "profiles" / "web"
    manifest_path = profile / "package.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dependencies"] = {
        "@local/libero-explore": f"file:{EXPLORE_PLUGIN}",
        "@local/imajev": f"file:{server.IMAJEV_PLUGIN}",
    }
    manifest.setdefault("dsh", {}).setdefault("profile", {})["bundles"] = [
        "@deepseek-ai/dsh-base",
        "@deepseek-ai/dsh-web-app",
        "@local/libero-explore",
        "@local/imajev",
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    install_bundle(profile, "libero-explore", EXPLORE_PLUGIN)
    install_bundle(profile, "imajev", server.IMAJEV_PLUGIN)


def write_task(all_tasks):
    status = json.loads((server.RUN_DIR / "status.json").read_text(encoding="utf-8"))
    if all_tasks:
        scope = (
            f"This session is every LIBERO-90 task, not one task.\n"
            f"The round already open says: {status.get('language')}.\n"
            f"Round {status.get('round')} is already open. reset samples another task and another scene. "
            f"You are not told which. Read the instruction again.\n"
        )
        rollout = "After submit the same program is rolled out on every initial scene of every task.\n"
    else:
        scope = (
            f"LIBERO-90 task {status.get('task_id')}: {status.get('language')}.\n"
            f"Round {status.get('round')} is already open. reset samples the next scene at random.\n"
        )
        rollout = "After submit the same program is rolled out on every initial scene of this task.\n"
    text = (
        f"Read SKILL.md and follow it.\n"
        f"{scope}"
        f"Exploration budget: {status.get('budget_used', 0)} / {status.get('budget_total')} environment steps.\n"
        f"This round's step cap: {status.get('max_steps')}.\n"
        f"Write code/<name>/controller.py, trial it with run, reset, and submit.\n"
        f"{rollout}"
    )
    (server.AGENT / "TASK.txt").write_text(text, encoding="utf-8")
    return status


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
            limit = report.get("eval_inits")
            scenes = "全部初始场景" if not limit else f"前 {limit} 个初始场景"
            print(f"{scope}的{scenes}评测 {successes}/{total} = {rate:.1%}", flush=True)
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
    if shutil.which("bwrap") is None:
        raise SystemExit("找不到 bwrap，控制器无法在隔离环境里评测。")
    args = parse_args()
    server.use_lane(MONITOR_PORT, WORKER_PORT, DSH_PORT, args.gpu)
    server.require_ports_free()
    server.LIBERO_TASK_ID = args.task
    server.PLUGIN = EXPLORE_PLUGIN
    session_id = __import__("uuid").uuid4().hex[:8]
    session_root = RUN_ROOT / "sessions" / session_id
    skill_text = SKILL_PATH.read_text(encoding="utf-8")
    server.prepare_workspace(session_root=session_root, skill_text=skill_text)
    for path in server.AGENT.glob("*.md"):
        if path.name != "SKILL.md":
            path.unlink()
    explore_code = ROOT / "skills" / "libero-explore" / "code"
    if explore_code.is_dir():
        shutil.copytree(explore_code, server.AGENT / "code", dirs_exist_ok=True)
    template = server.AGENT / "code" / "template"
    template.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPLORE_PLUGIN / "examples" / "controller.py", template / "controller.py")
    install_plugin()
    signal.signal(signal.SIGTERM, server._interrupt)
    signal.signal(signal.SIGHUP, server._interrupt)
    scenes = "全部初始场景" if args.eval_inits is None else f"前 {args.eval_inits} 个初始场景"
    if args.task == "all":
        print(
            f"探索预算 {args.budget} 步，单轮上限 {args.horizon}，LIBERO-90 全部任务。"
            f"reset 随机换任务和场景，提交后在每个任务的{scenes}上重放。",
            flush=True,
        )
    else:
        print(
            f"探索预算 {args.budget} 步，单轮上限 {args.horizon}，任务 {args.task}。"
            f"reset 随机开一局，提交后在该任务的{scenes}上重放。",
            flush=True,
        )
    print("评测步数不计入预算。旧的 server.py 和 libero90 插件没有改。", flush=True)
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
        print("正在打开第一轮。", flush=True)
        server.open_episode()
        status = write_task(args.task == "all")
        print(f"任务：{status.get('language')}", flush=True)
        server.PAGE = PAGE
        monitor = server.start_monitor()
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
