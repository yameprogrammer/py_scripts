#!/usr/bin/env python3
"""호환성 래퍼: 기존 경로 호출을 새 계층 스크립트로 전달합니다."""

from pathlib import Path
import runpy


def main() -> None:
    target = Path(__file__).resolve().parent / "scripts" / "analyze_mp3" / "analyze_mp3.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
