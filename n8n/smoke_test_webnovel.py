#!/usr/bin/env python3
"""webnovel_autowrite 초기화 스모크 테스트."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    script_root = base_dir / "scripts" / "webnovel_autowrite"
    relative_project_root = Path("projects") / "smoke_test_cli"
    project_root = script_root / relative_project_root

    if project_root.exists():
        shutil.rmtree(project_root)

    cmd = [
        sys.executable,
        str(script_root / "main.py"),
        "--project-root",
        str(relative_project_root),
        "--project-name",
        "smoke-test",
        "--init-only",
        "--dry-run",
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=base_dir)
    if completed.returncode != 0:
        print(completed.stderr.strip() or completed.stdout.strip())
        return completed.returncode

    meta = json.loads(completed.stdout.strip())
    if not meta.get("ok"):
        print("init 결과에 ok=true가 없습니다.")
        return 1
    if meta.get("project_root") != "projects/smoke_test_cli":
        print(f"project_root 표시값이 예상과 다릅니다: {meta.get('project_root')}")
        return 1
    files = meta.get("files", {})
    if files.get("state") != "state.json":
        print(f"state 파일 표시값이 예상과 다릅니다: {files.get('state')}")
        return 1

    state_path = project_root / "state.json"
    world_path = project_root / "world_setting.json"
    manuscript_path = project_root / "episodes" / "episode_001" / "manuscript.md"
    review_path = project_root / "episodes" / "episode_001" / "review.json"

    for path in [state_path, world_path, manuscript_path, review_path]:
        if not path.exists():
            print(f"필수 파일이 생성되지 않았습니다: {path}")
            return 1

    state = json.loads(state_path.read_text(encoding="utf-8"))
    required_state_keys = {"project", "runtime", "workflow", "llm", "episodes", "checkpoints"}
    missing = required_state_keys - set(state.keys())
    if missing:
        print(f"state.json 키 누락: {sorted(missing)}")
        return 1

    runtime = state.get("runtime", {})
    if runtime.get("current_episode") != 1:
        print("current_episode 기본값이 1이 아닙니다.")
        return 1
    if runtime.get("last_success_checkpoint") != "bootstrapped":
        print("last_success_checkpoint 기본값이 올바르지 않습니다.")
        return 1

    print("webnovel smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

