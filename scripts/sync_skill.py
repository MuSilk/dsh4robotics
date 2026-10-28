#!/usr/bin/env python3
"""Copy the newest session skill back into skills/robotwin-explore.

The session workspace is where the agent edits SKILL.md, the markdown it links
(REFERENCE.md, HISTORY.md, tasks/), figures/, and code/lib/. This copies those
files when the session copy is newer. It also copies any code/<path> named in
those documents. Episode notes, experiments, and bytecode stay in the session.

    python3 scripts/sync_skill.py
    python3 scripts/sync_skill.py --dry-run
    python3 scripts/sync_skill.py --session cf3042cf
"""
import argparse
import re
import shutil
from pathlib import Path

ROOT = Path("/data/changzou/deepseek4robotics")
SESSIONS = ROOT / "runs" / "robotwin-explore" / "sessions"
SKILL = ROOT / "skills" / "robotwin-explore"
CODE_REF = re.compile(r"code/[\w./-]+\.(?:py|md)")
DOC_REF = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.md)")


def read_text(path):
    return path.read_text(encoding="utf-8")


def linked_markdown(workspace, texts):
    """Markdown files named by the skill, excluding episode experiments."""
    found = []
    for name in sorted(set(DOC_REF.findall("\n".join(texts)))):
        if name == "SKILL.md" or name.startswith("experiments/") or ".." in name.split("/"):
            continue
        path = workspace / name
        try:
            path.resolve().relative_to(workspace.resolve())
        except ValueError:
            continue
        if path.is_file():
            found.append(path)
    return found


def skill_texts(workspace):
    paths = [workspace / "SKILL.md"]
    tasks = workspace / "tasks"
    if tasks.is_dir():
        paths.extend(sorted(tasks.glob("*.md")))
    texts = [read_text(path) for path in paths if path.is_file()]
    extra = linked_markdown(workspace, texts)
    texts.extend(read_text(path) for path in extra)
    return extra, texts


def newest_workspace():
    found = []
    for path in SESSIONS.glob("*/workspace/SKILL.md"):
        workspace = path.parent
        extra, _texts = skill_texts(workspace)
        score = max([path.stat().st_mtime] + [item.stat().st_mtime for item in extra])
        found.append((score, workspace))
    if not found:
        raise SystemExit(f"在 {SESSIONS} 里没有找到 SKILL.md。")
    found.sort(key=lambda item: item[0])
    return found[-1][1]


def workspace_for(session_id):
    path = SESSIONS / session_id / "workspace"
    if not (path / "SKILL.md").is_file():
        raise SystemExit(f"会话 {session_id} 里没有 SKILL.md。")
    return path


def referenced_code(texts):
    names = set()
    for text in texts:
        names.update(CODE_REF.findall(text))
    return names


def should_copy(source, destination):
    if not destination.exists():
        return True
    return source.stat().st_mtime > destination.stat().st_mtime + 1e-6


def copy_file(source, destination, dry_run):
    if not source.is_file():
        return "missing"
    if not should_copy(source, destination):
        return "same"
    state = "new" if not destination.exists() else "update"
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return state


def library_files(workspace):
    library = workspace / "code" / "lib"
    if not library.is_dir():
        return []
    return [path for path in library.rglob("*.py") if "__pycache__" not in path.parts]


def sync(workspace, dry_run):
    actions = []
    seen = set()

    def take(source, destination):
        if destination in seen:
            return
        seen.add(destination)
        state = copy_file(source, destination, dry_run)
        if state == "missing":
            actions.append(("missing", source.relative_to(workspace)))
        elif state != "same":
            actions.append((state, destination.relative_to(SKILL)))

    extra, texts = skill_texts(workspace)
    take(workspace / "SKILL.md", SKILL / "SKILL.md")
    for path in extra:
        take(path, SKILL / path.relative_to(workspace))
    tasks = workspace / "tasks"
    if tasks.is_dir():
        for path in sorted(tasks.glob("*.md")):
            take(path, SKILL / "tasks" / path.name)
    figures = workspace / "figures"
    if figures.is_dir():
        for path in sorted(figures.iterdir()):
            if path.is_file():
                take(path, SKILL / "figures" / path.name)
    for path in library_files(workspace):
        take(path, SKILL / "code" / "lib" / path.relative_to(workspace / "code" / "lib"))
    for name in sorted(referenced_code(texts)):
        take(workspace / name, SKILL / name)
    return actions


def main():
    parser = argparse.ArgumentParser(description="把最新会话里的技能同步回 skills/robotwin-explore")
    parser.add_argument("--session", help="指定会话编号。默认用 SKILL.md 最新的一次会话。")
    parser.add_argument("--dry-run", action="store_true", help="只打印会复制的文件，不写入。")
    args = parser.parse_args()
    workspace = workspace_for(args.session) if args.session else newest_workspace()
    print(f"来源 {workspace.parent.name}", flush=True)
    actions = sync(workspace, args.dry_run)
    if not actions:
        print("没有比技能目录更新的文件。", flush=True)
        return
    label = {"new": "新增", "update": "更新", "missing": "技能点了名但会话里没有"}
    for state, path in actions:
        print(f"{label[state]} {path}", flush=True)


if __name__ == "__main__":
    main()
