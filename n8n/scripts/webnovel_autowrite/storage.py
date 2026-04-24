from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SERIES_TITLE = "새 웹소설 프로젝트"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_default_world_setting(project_name: str) -> dict[str, Any]:
    return {
        "series": {
            "project_name": project_name,
            "title": DEFAULT_SERIES_TITLE,
            "logline": "주인공이 세계의 균열을 추적하며 자신의 비밀을 마주하는 성장형 판타지 웹소설.",
            "genre": ["판타지", "미스터리", "성장"],
            "target_length": {
                "episodes": 100,
                "words_per_episode": 2500,
            },
            "tone": ["긴장감", "서정성", "몰입감"],
        },
        "world_setting": {
            "era": "근미래와 중세 판타지가 공존하는 혼합 문명",
            "core_rules": [
                "균열이 열릴 때마다 기억이 뒤섞인다.",
                "마력 사용에는 기억 소모라는 대가가 따른다.",
                "도시 국가들은 균열 자원을 두고 경쟁한다.",
            ],
            "factions": [
                {
                    "name": "회색탑",
                    "goal": "균열 에너지를 통제해 세계 질서를 재편한다.",
                    "resources": ["아카이브", "마도 공학", "정찰 조직"],
                },
                {
                    "name": "새벽연맹",
                    "goal": "기억 소모 없는 마력 사용법을 찾는다.",
                    "resources": ["현장 요원", "치유술", "지역 네트워크"],
                },
            ],
            "locations": [
                {
                    "name": "부유도시 루멘",
                    "description": "상층 귀족과 하층 노동 지구가 극명하게 갈린 공중 도시",
                },
                {
                    "name": "침묵의 균열 지대",
                    "description": "들어가면 과거의 후회가 환청으로 되살아나는 위험 지역",
                },
            ],
        },
        "characters": [
            {
                "id": "protagonist",
                "name": "윤서",
                "role": "주인공",
                "goal": "사라진 형의 흔적을 찾아 가족의 진실을 밝힌다.",
                "conflict": "마력을 쓸수록 소중한 기억을 잃는다.",
                "traits": ["집요함", "공감 능력", "즉흥성"],
                "arc_hint": "상실을 두려워하던 인물에서 선택의 책임을 감당하는 인물로 성장",
            },
            {
                "id": "editorial_rival",
                "name": "라온",
                "role": "협력자이자 라이벌",
                "goal": "회색탑보다 먼저 균열의 근원을 해독한다.",
                "conflict": "주인공을 이용해야 하는 임무와 인간적 신뢰 사이에서 흔들린다.",
                "traits": ["분석적", "냉정함", "숨은 다정함"],
                "arc_hint": "통제 중심 태도에서 동료의식을 배우는 방향",
            },
        ],
        "plot": {
            "main_arc": "균열의 진실을 추적하는 과정에서 주인공 형의 실종과 도시 권력 암투가 연결되어 있음을 밝혀낸다.",
            "episode_beats": [
                {
                    "episode": 1,
                    "summary": "윤서가 첫 균열 임무에서 형의 흔적을 발견한다.",
                },
                {
                    "episode": 2,
                    "summary": "라온과 불안한 동맹을 맺고 금지 구역으로 잠입한다.",
                },
            ],
            "long_term_questions": [
                "형은 왜 균열 내부에서 흔적만 남긴 채 사라졌는가?",
                "기억 소모 없는 마력 운용은 가능한가?",
                "회색탑은 무엇을 은폐하고 있는가?",
            ],
        },
    }


def build_default_state(project_name: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "project": {
            "name": project_name,
            "created_at": now,
            "updated_at": now,
        },
        "runtime": {
            "current_episode": 1,
            "last_success_checkpoint": "bootstrapped",
            "last_completed_stage": None,
            "last_error": None,
        },
        "workflow": {
            "episode_pipeline": ["plan", "draft", "review", "revise", "finalize"],
            "quality_gate": {
                "target_score": 80,
                "max_revision_rounds": 5,
            },
            "last_success": {
                "episode": 0,
                "stage": "bootstrapped",
                "timestamp": now,
            },
        },
        "llm": {
            "active_provider": "ollama",
            "providers": {
                "ollama": {
                    "enabled": True,
                    "base_url": "http://127.0.0.1:11434",
                    "model": "qwen2.5:7b-instruct",
                    "temperature": 0.7,
                },
                "gemini": {
                    "enabled": False,
                    "model": "gemini-2.5-flash",
                    "api_key_env": "GEMINI_API_KEY",
                    "temperature": 0.7,
                },
                "openai": {
                    "enabled": False,
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4.1-mini",
                    "api_key_env": "OPENAI_API_KEY",
                    "temperature": 0.7,
                },
            },
        },
        "episodes": {
            "planned": 100,
            "completed": 0,
            "items": {
                "episode_001": {
                    "status": "planned",
                    "title": "1화 초안 준비",
                    "manuscript_path": "episodes/episode_001/manuscript.md",
                    "review_path": "episodes/episode_001/review.json",
                    "updated_at": now,
                }
            },
        },
        "checkpoints": [
            {
                "episode": 0,
                "stage": "bootstrapped",
                "timestamp": now,
                "note": "초기 디렉토리와 기본 설정 파일 생성 완료",
            }
        ],
    }


def build_episode_manuscript(episode_number: int) -> str:
    return "\n".join(
        [
            f"# {episode_number}화 초안",
            "",
            "## 로그라인",
            "- 이번 화의 핵심 사건을 한 줄로 정리하세요.",
            "",
            "## 본문",
            "- AI 작가 에이전트가 생성한 초안을 여기에 저장합니다.",
            "",
            "## 메모",
            "- 복선, 다음 화 연결 포인트, 수정 메모를 적습니다.",
            "",
        ]
    )


def build_episode_review(episode_number: int) -> dict[str, Any]:
    return {
        "episode": episode_number,
        "status": "pending",
        "scores": {
            "plot_coherence": None,
            "character_consistency": None,
            "immersion": None,
            "style": None,
            "length_adherence": None,
            "overall": None,
        },
        "strengths": [],
        "issues": [],
        "revision_requests": [],
        "reviewed_at": None,
        "manuscript_stats": {
            "char_count": 0,
            "char_count_no_space": 0,
            "word_count": 0,
            "line_count": 0,
            "paragraph_count": 0,
            "target_chars": 0,
            "min_chars": 0,
            "meets_length": False,
        },
        "quality_gate": {
            "target_score": 80,
            "passed": False,
            "reasons": [],
        },
        "attempt_history": [],
        "final_attempt": 0,
        "final_passed": False,
    }


def bootstrap_project(project_root: Path, project_name: str = "default") -> dict[str, str]:
    project_root.mkdir(parents=True, exist_ok=True)
    episodes_root = project_root / "episodes"
    first_episode_root = episodes_root / "episode_001"
    first_episode_root.mkdir(parents=True, exist_ok=True)

    world_setting_path = project_root / "world_setting.json"
    state_path = project_root / "state.json"
    manuscript_path = first_episode_root / "manuscript.md"
    review_path = first_episode_root / "review.json"

    if not world_setting_path.exists():
        write_json(world_setting_path, build_default_world_setting(project_name))
    if not state_path.exists():
        write_json(state_path, build_default_state(project_name))
    if not manuscript_path.exists():
        manuscript_path.write_text(build_episode_manuscript(1), encoding="utf-8")
    if not review_path.exists():
        write_json(review_path, build_episode_review(1))

    return {
        "project_root": str(project_root),
        "world_setting": str(world_setting_path),
        "state": str(state_path),
        "first_episode_manuscript": str(manuscript_path),
        "first_episode_review": str(review_path),
    }


def ensure_episode_files(project_root: Path, episode_number: int) -> dict[str, str]:
    episode_key = f"episode_{episode_number:03d}"
    episode_root = project_root / "episodes" / episode_key
    episode_root.mkdir(parents=True, exist_ok=True)

    manuscript_path = episode_root / "manuscript.md"
    review_path = episode_root / "review.json"

    if not manuscript_path.exists():
        manuscript_path.write_text(build_episode_manuscript(episode_number), encoding="utf-8")
    if not review_path.exists():
        write_json(review_path, build_episode_review(episode_number))

    return {
        "episode_key": episode_key,
        "manuscript_path": str(manuscript_path),
        "review_path": str(review_path),
    }


def save_episode_outputs(
    project_root: Path,
    episode_number: int,
    manuscript: str,
    review: dict[str, Any],
) -> dict[str, str]:
    paths = ensure_episode_files(project_root, episode_number)
    manuscript_path = Path(paths["manuscript_path"])
    review_path = Path(paths["review_path"])
    manuscript_path.write_text(manuscript.strip() + "\n", encoding="utf-8")
    write_json(review_path, review)
    return paths


def update_state_after_stage(
    state: dict[str, Any],
    episode_number: int,
    stage: str,
    *,
    note: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(state)
    now = utc_now()

    updated["project"]["updated_at"] = now
    updated["runtime"]["current_episode"] = episode_number
    updated["runtime"]["last_success_checkpoint"] = stage
    updated["runtime"]["last_completed_stage"] = stage
    updated["runtime"]["last_error"] = error
    updated["workflow"]["last_success"] = {
        "episode": episode_number,
        "stage": stage,
        "timestamp": now,
    }
    updated.setdefault("checkpoints", []).append(
        {
            "episode": episode_number,
            "stage": stage,
            "timestamp": now,
            "note": note or "",
        }
    )

    episode_key = f"episode_{episode_number:03d}"
    episodes = updated.setdefault("episodes", {}).setdefault("items", {})
    episode_item = episodes.setdefault(
        episode_key,
        {
            "status": "planned",
            "title": f"{episode_number}화",
            "manuscript_path": f"episodes/{episode_key}/manuscript.md",
            "review_path": f"episodes/{episode_key}/review.json",
            "updated_at": now,
        },
    )
    episode_item["updated_at"] = now
    episode_item["status"] = stage
    updated["episodes"]["completed"] = sum(
        1 for item in episodes.values() if item.get("status") == "finalize"
    )
    return updated

