#!/usr/bin/env python3
"""analyze_mp3.py의 mock 모드 스모크 테스트."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    sample_mp3 = base_dir / "sample.mp3"
    output_json = base_dir / "out" / "smoke_result.json"

    # 실제 오디오 디코딩을 하지 않으므로 스모크 테스트용 더미 바이트로 충분합니다.
    sample_mp3.write_bytes(b"ID3" + b"\x00" * 128)

    cmd = [
        sys.executable,
        str(base_dir / "scripts" / "analyze_mp3" / "analyze_mp3.py"),
        "--input",
        str(sample_mp3),
        "--provider",
        "mock",
        "--format",
        "json",
        "--output",
        str(output_json),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr.strip())
        return completed.returncode

    meta = json.loads(completed.stdout.strip())
    if not meta.get("ok"):
        print("메타 출력에서 ok=true를 찾지 못했습니다.")
        return 1

    if not output_json.exists():
        print(f"결과 파일이 생성되지 않았습니다: {output_json}")
        return 1

    parsed = json.loads(output_json.read_text(encoding="utf-8"))
    required_keys = {"track_analysis", "mv_concept", "storyboard", "deliverables"}
    missing = required_keys - set(parsed.keys())
    if missing:
        print(f"결과 JSON 키 누락: {sorted(missing)}")
        return 1

    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
