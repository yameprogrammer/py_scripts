from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents import EditorAgent, LLMClient, ProviderConfig, WriterAgent, extract_target_length
from storage import (
    bootstrap_project,
    ensure_episode_files,
    read_json,
    save_episode_outputs,
    update_state_after_stage,
    write_json,
)


def resolve_from_script_root(script_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (script_root / candidate).resolve()


def display_path(path: Path, base_root: Path) -> str:
    try:
        return str(path.relative_to(base_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_bootstrap_preview(script_root: Path, project_root: Path, bootstrap_result: dict[str, str]) -> dict[str, str]:
    return {
        "project_root": display_path(project_root, script_root),
        "project_root_absolute": str(project_root),
        "world_setting": display_path(Path(bootstrap_result["world_setting"]), project_root),
        "state": display_path(Path(bootstrap_result["state"]), project_root),
        "first_episode_manuscript": display_path(Path(bootstrap_result["first_episode_manuscript"]), project_root),
        "first_episode_review": display_path(Path(bootstrap_result["first_episode_review"]), project_root),
    }


def build_project_relative_paths(project_root: Path, paths: dict[str, str]) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for key, value in paths.items():
        if key.endswith("_path"):
            formatted[key] = display_path(Path(value), project_root)
            continue
        formatted[key] = value
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI 웹소설 자동 집필 시스템 초기화 및 실행")
    parser.add_argument("--project-root", default="projects/default", help="프로젝트 데이터 루트 (상대경로는 실행 스크립트 기준)")
    parser.add_argument("--project-name", default="default", help="프로젝트 이름")
    parser.add_argument("--provider", choices=["ollama", "gemini", "openai"], help="실행 시 사용할 LLM provider")
    parser.add_argument("--model", help="실행 시 사용할 모델명 override")
    parser.add_argument("--init-only", action="store_true", help="디렉토리와 기본 JSON만 생성")
    parser.add_argument("--prepare-next-episode", action="store_true", help="현재 화수용 episode 폴더를 준비")
    parser.add_argument("--run-once", action="store_true", help="작가/편집자 1회 실행")
    parser.add_argument("--dry-run", action="store_true", help="실제 모델 호출 없이 프롬프트/설정만 점검")
    return parser.parse_args()


def load_context(project_root: Path) -> tuple[dict, dict]:
    state = read_json(project_root / "state.json")
    world_setting = read_json(project_root / "world_setting.json")
    return state, world_setting


def prepare_episode(project_root: Path, state: dict) -> dict[str, str]:
    return ensure_episode_files(project_root, int(state["runtime"]["current_episode"]))


def get_quality_goals(state: dict[str, Any], world_setting: dict[str, Any]) -> dict[str, Any]:
    quality_gate = state.get("workflow", {}).get("quality_gate", {})
    raw_target_score = quality_gate.get("target_score", 80)
    raw_max_revision_rounds = quality_gate.get("max_revision_rounds", 5)

    try:
        target_score = float(raw_target_score)
    except (TypeError, ValueError):
        target_score = 80.0

    try:
        max_revision_rounds = max(int(raw_max_revision_rounds), 1)
    except (TypeError, ValueError):
        max_revision_rounds = 5

    return {
        "target_score": target_score,
        "max_revision_rounds": max_revision_rounds,
        "target_length": extract_target_length(world_setting),
    }


def build_attempt_summary(attempt_number: int, review: dict[str, Any]) -> dict[str, Any]:
    manuscript_stats = review.get("manuscript_stats", {})
    quality_gate = review.get("quality_gate", {})
    return {
        "attempt": attempt_number,
        "overall": review.get("scores", {}).get("overall"),
        "length_adherence": review.get("scores", {}).get("length_adherence"),
        "char_count_no_space": manuscript_stats.get("char_count_no_space"),
        "passed": quality_gate.get("passed", False),
        "reasons": quality_gate.get("reasons", []),
    }


def run_revision_loop(
    writer: WriterAgent,
    editor: EditorAgent,
    *,
    world_setting: dict[str, Any],
    state: dict[str, Any],
    episode_number: int,
    quality_goals: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    target_length = quality_goals["target_length"]
    target_score = float(quality_goals["target_score"])
    max_revision_rounds = int(quality_goals["max_revision_rounds"])

    manuscript = writer.generate_draft(
        world_setting=world_setting,
        state=state,
        episode_number=episode_number,
        target_length=target_length,
        attempt_number=1,
    )

    final_review: dict[str, Any] = {}
    attempt_history: list[dict[str, Any]] = []
    passed = False

    for attempt_number in range(1, max_revision_rounds + 1):
        review = editor.review_draft(
            world_setting=world_setting,
            episode_number=episode_number,
            manuscript=manuscript,
            target_length=target_length,
            target_score=target_score,
        )
        attempt_history.append(build_attempt_summary(attempt_number, review))
        final_review = review

        if review.get("quality_gate", {}).get("passed", False):
            passed = True
            break

        if attempt_number == max_revision_rounds:
            break

        manuscript = writer.revise_draft(
            world_setting=world_setting,
            state=state,
            episode_number=episode_number,
            manuscript=manuscript,
            review=review,
            target_length=target_length,
            attempt_number=attempt_number + 1,
        )

    final_review = {
        **final_review,
        "attempt_history": attempt_history,
        "final_attempt": len(attempt_history),
        "final_passed": passed,
        "target_score": target_score,
        "target_length": target_length,
    }
    return manuscript, final_review, passed


def run_episode_cycle(
    project_root: Path,
    state: dict,
    world_setting: dict,
    *,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> dict[str, Any]:
    episode_number = int(state["runtime"]["current_episode"])
    config = ProviderConfig.from_state(state, provider_override=provider_override, model_override=model_override)
    client = LLMClient(config)
    writer = WriterAgent(client)
    editor = EditorAgent(client)

    quality_goals = get_quality_goals(state, world_setting)
    draft, review, passed = run_revision_loop(
        writer,
        editor,
        world_setting=world_setting,
        state=state,
        episode_number=episode_number,
        quality_goals=quality_goals,
    )

    output_paths = save_episode_outputs(
        project_root,
        episode_number,
        manuscript=draft,
        review=review,
    )

    updated_state = update_state_after_stage(
        state,
        episode_number,
        "finalize" if passed else "revise",
        note=(
            f"자동 집필 루프 완료: {review['final_attempt']}회 시도 끝에 목표 점수 달성"
            if passed
            else f"자동 집필 루프 종료: {review['final_attempt']}회 시도 후에도 목표 점수 미달"
        ),
        error=None if passed else "; ".join(review.get("quality_gate", {}).get("reasons", [])),
    )
    write_json(project_root / "state.json", updated_state)

    return {
        "provider": config.provider,
        "model": config.model,
        "episode": episode_number,
        "passed": passed,
        "attempts": review.get("final_attempt", 1),
        "final_score": review.get("scores", {}).get("overall"),
        "output_paths": output_paths,
    }


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    project_root = resolve_from_script_root(script_root, args.project_root)

    bootstrap_result = bootstrap_project(project_root, args.project_name)
    state, world_setting = load_context(project_root)
    bootstrap_preview = build_bootstrap_preview(script_root, project_root, bootstrap_result)

    if args.prepare_next_episode:
        episode_paths = prepare_episode(project_root, state)
        updated_state = update_state_after_stage(
            state,
            int(state["runtime"]["current_episode"]),
            "prepared",
            note="현재 화수용 원고/검수 파일 준비 완료",
        )
        write_json(project_root / "state.json", updated_state)
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "prepare-next-episode",
                    "project_root": display_path(project_root, script_root),
                    "paths": build_project_relative_paths(project_root, episode_paths),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.init_only or not args.run_once:
        preview = {
            "ok": True,
            "action": "init",
            "project_root": bootstrap_preview["project_root"],
            "project_root_absolute": bootstrap_preview["project_root_absolute"],
            "files": {
                "world_setting": bootstrap_preview["world_setting"],
                "state": bootstrap_preview["state"],
                "first_episode_manuscript": bootstrap_preview["first_episode_manuscript"],
                "first_episode_review": bootstrap_preview["first_episode_review"],
            },
            "current_episode": state["runtime"]["current_episode"],
            "last_success_checkpoint": state["runtime"]["last_success_checkpoint"],
            "active_provider": state["llm"]["active_provider"],
        }
        if args.dry_run:
            config = ProviderConfig.from_state(
                state,
                provider_override=args.provider,
                model_override=args.model,
            )
            preview["provider_preview"] = {
                "provider": config.provider,
                "model": config.model,
                "temperature": config.temperature,
            }
        print(json.dumps(preview, ensure_ascii=False))
        return 0

    if args.dry_run:
        config = ProviderConfig.from_state(
            state,
            provider_override=args.provider,
            model_override=args.model,
        )
        quality_goals = get_quality_goals(state, world_setting)
        writer = WriterAgent(LLMClient(config))
        prompt_preview = writer.build_prompt(
            world_setting=world_setting,
            state=state,
            episode_number=int(state["runtime"]["current_episode"]),
            target_length=quality_goals["target_length"],
            attempt_number=1,
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "action": "dry-run",
                    "provider": config.provider,
                    "model": config.model,
                    "target_score": quality_goals["target_score"],
                    "target_length": quality_goals["target_length"],
                    "prompt_preview": prompt_preview[:1000],
                },
                ensure_ascii=False,
            )
        )
        return 0

    result = run_episode_cycle(
        project_root,
        state,
        world_setting,
        provider_override=args.provider,
        model_override=args.model,
    )
    result["output_paths"] = build_project_relative_paths(project_root, result["output_paths"])
    response = {
        "ok": bool(result.get("passed")),
        "action": "run-once",
        "project_root": display_path(project_root, script_root),
        "project_root_absolute": str(project_root),
        **result,
    }
    print(json.dumps(response, ensure_ascii=False))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())


