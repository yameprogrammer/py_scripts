from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any
from urllib import error, request


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


REVIEW_TRAIT_SCORE_KEYS = (
    "plot_coherence",
    "character_consistency",
    "immersion",
    "style",
)

REVIEW_SCORE_KEYS = REVIEW_TRAIT_SCORE_KEYS + (
    "length_adherence",
    "overall",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def calculate_manuscript_stats(manuscript: str) -> dict[str, int]:
    normalized = manuscript.strip()
    no_space = re.sub(r"\s+", "", normalized)
    words = [token for token in re.split(r"\s+", normalized) if token]
    return {
        "char_count": len(normalized),
        "char_count_no_space": len(no_space),
        "word_count": len(words),
        "line_count": len(normalized.splitlines()) if normalized else 0,
        "paragraph_count": len([line for line in normalized.splitlines() if line.strip()]),
    }


def extract_target_length(world_setting: dict[str, Any]) -> dict[str, Any]:
    target_length = world_setting.get("series", {}).get("target_length", {})
    raw_target_chars = target_length.get("chars_per_episode") or target_length.get("words_per_episode") or 2500
    try:
        target_chars = max(int(raw_target_chars), 1)
    except (TypeError, ValueError):
        target_chars = 2500

    min_chars = max(int(round(target_chars * 0.9)), 1)
    soft_max_chars = int(round(target_chars * 1.15))
    return {
        "target_chars": target_chars,
        "min_chars": min_chars,
        "soft_max_chars": soft_max_chars,
        "source_field": "chars_per_episode" if target_length.get("chars_per_episode") else "words_per_episode",
    }


def normalize_score_to_100(value: Any) -> float | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            value = float(stripped)
        except ValueError:
            return None
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    numeric = float(value)
    if numeric <= 5:
        numeric *= 20
    elif numeric <= 10:
        numeric *= 10
    return round(max(0.0, min(100.0, numeric)), 1)


def compute_length_score(manuscript_stats: dict[str, int], target_length: dict[str, Any]) -> tuple[float, bool]:
    target_chars = int(target_length.get("target_chars", 2500))
    min_chars = int(target_length.get("min_chars", target_chars))
    actual_chars = int(manuscript_stats.get("char_count_no_space", 0))

    if actual_chars >= target_chars:
        return 100.0, True

    ratio = actual_chars / max(target_chars, 1)
    score = round(max(0.0, min(100.0, ratio * 100)), 1)
    return score, actual_chars >= min_chars


def normalize_json_text(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\ufeff": "",
        "\u00a0": " ",
    }
    normalized = text.strip()
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def extract_first_json_object(text: str) -> str | None:
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]

    return None


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def load_json_loose(text: str) -> dict[str, Any]:
    candidates: list[str] = []
    for candidate in (text, strip_code_fence(text), extract_first_json_object(text) or ""):
        candidate = normalize_json_text(candidate)
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    errors: list[str] = []
    for candidate in candidates:
        for variant in (candidate, remove_trailing_commas(candidate)):
            try:
                parsed = json.loads(variant)
            except json.JSONDecodeError as exc:
                errors.append(f"{exc.msg} at line {exc.lineno} column {exc.colno}")
                continue
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("JSON 최상위 객체는 dict여야 합니다.")

    detail = " | ".join(errors[-4:]) if errors else "유효한 JSON 객체를 찾지 못했습니다."
    raise ValueError(detail)


def coerce_review_payload(
    payload: dict[str, Any],
    episode_number: int,
    manuscript_stats: dict[str, int],
    target_length: dict[str, Any],
    target_score: float,
) -> dict[str, Any]:
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        scores = {}

    raw_overall = normalize_score_to_100(scores.get("overall"))
    normalized_scores: dict[str, float] = {}
    for key in REVIEW_TRAIT_SCORE_KEYS:
        value = normalize_score_to_100(scores.get(key))
        if value is None:
            value = raw_overall if raw_overall is not None else 0.0
        normalized_scores[key] = value

    length_score, meets_length = compute_length_score(manuscript_stats, target_length)
    normalized_scores["length_adherence"] = length_score
    normalized_scores["overall"] = round(
        sum(normalized_scores[key] for key in REVIEW_TRAIT_SCORE_KEYS + ("length_adherence",))
        / (len(REVIEW_TRAIT_SCORE_KEYS) + 1),
        1,
    )

    def ensure_string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value).strip()]

    issues = ensure_string_list(payload.get("issues"))
    revision_requests = ensure_string_list(payload.get("revision_requests"))
    strengths = ensure_string_list(payload.get("strengths"))

    if not meets_length:
        issues.append(
            "분량 미달: 현재 공백 제외 "
            f"{manuscript_stats['char_count_no_space']}자이며 최소 {target_length['min_chars']}자, 목표 {target_length['target_chars']}자 이상이 필요합니다."
        )
        revision_requests.append(
            "본문 밀도를 유지한 채 분량을 확장하세요. 공백 제외 최소 "
            f"{target_length['min_chars']}자, 권장 {target_length['target_chars']}자 이상으로 작성해야 합니다."
        )

    passed = normalized_scores["overall"] >= target_score and meets_length
    quality_gate_reasons: list[str] = []
    if normalized_scores["overall"] < target_score:
        quality_gate_reasons.append(
            f"종합 점수 {normalized_scores['overall']}점으로 목표 {target_score}점에 미달했습니다."
        )
    if not meets_length:
        quality_gate_reasons.append(
            "분량 기준 미달입니다. "
            f"현재 공백 제외 {manuscript_stats['char_count_no_space']}자 / 최소 {target_length['min_chars']}자."
        )

    return {
        "episode": int(payload.get("episode") or episode_number),
        "status": str(payload.get("status") or "reviewed"),
        "scores": normalized_scores,
        "strengths": strengths,
        "issues": issues,
        "revision_requests": revision_requests,
        "reviewed_at": payload.get("reviewed_at") or utc_now(),
        "manuscript_stats": {
            **manuscript_stats,
            "target_chars": int(target_length["target_chars"]),
            "min_chars": int(target_length["min_chars"]),
            "meets_length": meets_length,
        },
        "quality_gate": {
            "target_score": float(target_score),
            "passed": passed,
            "reasons": quality_gate_reasons,
        },
    }


def parse_editor_response(
    response: str,
    episode_number: int,
    manuscript_stats: dict[str, int],
    target_length: dict[str, Any],
    target_score: float,
) -> dict[str, Any]:
    return coerce_review_payload(
        load_json_loose(response),
        episode_number,
        manuscript_stats,
        target_length,
        target_score,
    )


@dataclass
class ProviderConfig:
    provider: str
    model: str
    temperature: float = 0.7
    base_url: str | None = None
    api_key: str | None = None

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> "ProviderConfig":
        llm_settings = state.get("llm", {})
        active_provider = provider_override or os.getenv("WEBNOVEL_PROVIDER") or llm_settings.get("active_provider", "ollama")
        providers = llm_settings.get("providers", {})
        provider_settings = providers.get(active_provider, {})

        if not provider_settings:
            raise ValueError(f"지원하지 않는 provider 설정입니다: {active_provider}")

        api_key_env = provider_settings.get("api_key_env")
        api_key = os.getenv(api_key_env) if api_key_env else None
        provider_base_url_env = {
            "ollama": "OLLAMA_BASE_URL",
            "openai": "OPENAI_BASE_URL",
        }.get(active_provider)
        provider_model_env = {
            "ollama": "OLLAMA_MODEL",
            "gemini": "GEMINI_MODEL",
            "openai": "OPENAI_MODEL",
        }.get(active_provider)

        base_url = provider_settings.get("base_url")
        if provider_base_url_env:
            base_url = os.getenv(provider_base_url_env, base_url)

        model = (
            model_override
            or os.getenv("WEBNOVEL_MODEL")
            or (os.getenv(provider_model_env) if provider_model_env else None)
            or provider_settings.get("model")
        )
        if not model:
            raise ValueError("모델명이 비어 있습니다.")

        return cls(
            provider=active_provider,
            model=model,
            temperature=float(provider_settings.get("temperature", 0.7)),
            base_url=base_url,
            api_key=api_key,
        )


class LLMClient:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if self.config.provider == "ollama":
            return self._generate_with_ollama(prompt, system_prompt=system_prompt)
        if self.config.provider == "gemini":
            return self._generate_with_gemini(prompt, system_prompt=system_prompt)
        if self.config.provider == "openai":
            return self._generate_with_openai(prompt, system_prompt=system_prompt)
        raise ValueError(f"지원하지 않는 provider입니다: {self.config.provider}")

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"연결 실패: {exc.reason}") from exc
        return json.loads(raw)

    def _generate_with_ollama(self, prompt: str, *, system_prompt: str | None = None) -> str:
        base_url = self.config.base_url or "http://127.0.0.1:11434"
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        response = self._post_json(f"{base_url.rstrip('/')}/api/generate", payload)
        return response.get("response", "").strip()

    def _generate_with_gemini(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not self.config.api_key:
            raise RuntimeError("GEMINI_API_KEY 환경변수가 필요합니다.")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={self.config.api_key}"
        )
        parts = []
        if system_prompt:
            parts.append({"text": system_prompt})
        parts.append({"text": prompt})
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {"temperature": self.config.temperature},
        }
        response = self._post_json(url, payload)
        candidates = response.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini 응답에 candidates가 없습니다: {response}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts).strip()

    def _generate_with_openai(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not self.config.api_key:
            raise RuntimeError("OPENAI_API_KEY 환경변수가 필요합니다.")
        base_url = self.config.base_url or "https://api.openai.com/v1"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        response = self._post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        choices = response.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenAI 응답에 choices가 없습니다: {response}")
        return choices[0].get("message", {}).get("content", "").strip()


WRITER_SYSTEM_PROMPT = """당신은 장편 웹소설 전문 작가 AI입니다.
세계관, 캐릭터 설정, 기존 진행 상태를 절대 훼손하지 말고 유지하세요.
매 화는 사건 진전, 감정 변화, 다음 화 훅을 반드시 포함해야 합니다.
불필요한 메타 설명 없이 결과만 작성하세요.
"""

EDITOR_SYSTEM_PROMPT = """당신은 웹소설 편집자 AI입니다.
초안을 읽고 서사 일관성, 캐릭터 일관성, 몰입감, 문체 완성도를 엄격하게 평가하세요.
반드시 JSON으로만 응답하세요.
"""

EDITOR_JSON_REPAIR_SYSTEM_PROMPT = """당신은 깨진 JSON을 복구하는 도우미입니다.
설명 없이 오직 하나의 유효한 JSON 객체만 반환하세요.
JSON 밖의 텍스트, 코드펜스, 주석을 절대 포함하지 마세요.
"""


class WriterAgent:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def build_prompt(
        self,
        *,
        world_setting: dict[str, Any],
        state: dict[str, Any],
        episode_number: int,
        target_length: dict[str, Any],
        attempt_number: int = 1,
    ) -> str:
        return (
            "다음 정보를 참고해 웹소설 초안을 작성하세요.\n\n"
            f"[현재 화수]\n{episode_number}화\n\n"
            f"[시도 횟수]\n{attempt_number}차 시도\n\n"
            f"[진행 상태]\n{json.dumps(state.get('runtime', {}), ensure_ascii=False, indent=2)}\n\n"
            f"[세계관 및 캐릭터]\n{json.dumps(world_setting, ensure_ascii=False, indent=2)}\n\n"
            f"[분량 규칙]\n- 공백 제외 최소 {target_length['min_chars']}자\n- 권장 공백 제외 {target_length['target_chars']}자 이상\n- 지나치게 요약하지 말고 장면, 감정, 감각 묘사를 충분히 확장\n\n"
            "[출력 형식]\n"
            "- 화 제목 1개\n"
            "- 4~6줄 분량의 핵심 요약\n"
            "- 본문 초안: 장면 전개, 감정 변화, 감각 묘사를 충분히 포함한 장문 서술\n"
            "- 다음 화 훅 2개\n"
        )

    def generate_draft(
        self,
        *,
        world_setting: dict[str, Any],
        state: dict[str, Any],
        episode_number: int,
        target_length: dict[str, Any],
        attempt_number: int = 1,
    ) -> str:
        prompt = self.build_prompt(
            world_setting=world_setting,
            state=state,
            episode_number=episode_number,
            target_length=target_length,
            attempt_number=attempt_number,
        )
        return self.client.generate(prompt, system_prompt=WRITER_SYSTEM_PROMPT)

    def build_revision_prompt(
        self,
        *,
        world_setting: dict[str, Any],
        state: dict[str, Any],
        episode_number: int,
        manuscript: str,
        review: dict[str, Any],
        target_length: dict[str, Any],
        attempt_number: int,
    ) -> str:
        return (
            "다음 원고를 편집자 피드백에 맞춰 전면 수정하세요.\n\n"
            f"[현재 화수]\n{episode_number}화\n\n"
            f"[재작성 차수]\n{attempt_number}차 시도\n\n"
            f"[진행 상태]\n{json.dumps(state.get('runtime', {}), ensure_ascii=False, indent=2)}\n\n"
            f"[세계관 및 캐릭터]\n{json.dumps(world_setting, ensure_ascii=False, indent=2)}\n\n"
            f"[분량 규칙]\n- 공백 제외 최소 {target_length['min_chars']}자\n- 권장 공백 제외 {target_length['target_chars']}자 이상\n- 기존보다 짧아지면 안 됨\n\n"
            f"[직전 원고]\n{manuscript}\n\n"
            f"[검수 결과]\n{json.dumps(review, ensure_ascii=False, indent=2)}\n\n"
            "[수정 지침]\n"
            "- 세계관 설정과 캐릭터 성격은 유지\n"
            "- 지적된 문제를 직접 해결\n"
            "- 감정선, 장면 묘사, 인과관계를 더 촘촘하게 보강\n"
            "- 분량을 채우기 위해 무의미한 반복을 넣지 말고 사건/감정/배경을 확장\n"
            "- 출력 형식은 초안과 동일: 화 제목, 핵심 요약, 본문, 다음 화 훅\n"
        )

    def revise_draft(
        self,
        *,
        world_setting: dict[str, Any],
        state: dict[str, Any],
        episode_number: int,
        manuscript: str,
        review: dict[str, Any],
        target_length: dict[str, Any],
        attempt_number: int,
    ) -> str:
        prompt = self.build_revision_prompt(
            world_setting=world_setting,
            state=state,
            episode_number=episode_number,
            manuscript=manuscript,
            review=review,
            target_length=target_length,
            attempt_number=attempt_number,
        )
        return self.client.generate(prompt, system_prompt=WRITER_SYSTEM_PROMPT)


class EditorAgent:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def build_prompt(
        self,
        *,
        world_setting: dict[str, Any],
        episode_number: int,
        manuscript: str,
        target_length: dict[str, Any],
        target_score: float,
        manuscript_stats: dict[str, int],
    ) -> str:
        rubric = {
            "plot_coherence": "서사 전개와 사건 연결이 자연스러운가",
            "character_consistency": "캐릭터 말투와 동기가 설정과 일치하는가",
            "immersion": "긴장감과 흡입력이 유지되는가",
            "style": "문체, 리듬, 가독성이 안정적인가",
            "length_adherence": "목표 분량을 충분히 만족하는가",
            "overall": "종합 점수(0~100)",
        }
        return (
            "다음 웹소설 초안을 검수하세요.\n\n"
            f"[평가 기준]\n{json.dumps(rubric, ensure_ascii=False, indent=2)}\n\n"
            f"[세계관 참고]\n{json.dumps(world_setting, ensure_ascii=False, indent=2)}\n\n"
            f"[대상 화수]\n{episode_number}화\n\n"
            f"[분량 목표]\n- 목표 공백 제외 {target_length['target_chars']}자 이상\n- 최소 공백 제외 {target_length['min_chars']}자\n- 현재 원고 공백 제외 {manuscript_stats['char_count_no_space']}자\n\n"
            f"[합격 기준]\n- 종합 점수 {target_score}점 이상\n- 분량 기준 충족 필수\n\n"
            f"[원고]\n{manuscript}\n\n"
            "[반드시 아래 JSON 스키마만 반환]\n"
            "{\n"
            '  "episode": 1,\n'
            '  "status": "reviewed",\n'
            '  "scores": {\n'
            '    "plot_coherence": 0,\n'
            '    "character_consistency": 0,\n'
            '    "immersion": 0,\n'
            '    "style": 0,\n'
            '    "length_adherence": 0,\n'
            '    "overall": 0\n'
            "  },\n"
            '  "strengths": ["..."],\n'
            '  "issues": ["..."],\n'
            '  "revision_requests": ["..."],\n'
            '  "quality_gate": {"target_score": 80, "passed": false, "reasons": ["..."]}\n'
            "}\n"
        )

    def build_repair_prompt(self, raw_response: str, episode_number: int) -> str:
        schema = {
            "episode": episode_number,
            "status": "reviewed",
            "scores": {key: 0 for key in REVIEW_SCORE_KEYS},
            "strengths": ["..."],
            "issues": ["..."],
            "revision_requests": ["..."],
            "quality_gate": {"target_score": 80, "passed": False, "reasons": ["..."]},
            "reviewed_at": None,
        }
        return (
            "아래 텍스트를 같은 의미를 유지한 채 유효한 JSON 객체 하나로만 복구하세요.\n\n"
            "[출력 제약]\n"
            "- 반드시 JSON 객체 하나만 반환\n"
            "- 코드블록 금지\n"
            "- 문자열 내부 따옴표는 JSON 규칙에 맞게 escape\n"
            "- 후행 쉼표 금지\n"
            f"- episode는 {episode_number}로 유지\n\n"
            f"[목표 스키마]\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"[복구 대상 원문]\n{raw_response}\n"
        )

    def review_draft(
        self,
        *,
        world_setting: dict[str, Any],
        episode_number: int,
        manuscript: str,
        target_length: dict[str, Any] | None = None,
        target_score: float = 80.0,
    ) -> dict[str, Any]:
        resolved_target_length = target_length or extract_target_length(world_setting)
        manuscript_stats = calculate_manuscript_stats(manuscript)
        prompt = self.build_prompt(
            world_setting=world_setting,
            episode_number=episode_number,
            manuscript=manuscript,
            target_length=resolved_target_length,
            target_score=target_score,
            manuscript_stats=manuscript_stats,
        )
        response = self.client.generate(prompt, system_prompt=EDITOR_SYSTEM_PROMPT)
        try:
            return parse_editor_response(
                response,
                episode_number,
                manuscript_stats,
                resolved_target_length,
                target_score,
            )
        except ValueError as first_error:
            repair_prompt = self.build_repair_prompt(response, episode_number)
            repaired_response = self.client.generate(repair_prompt, system_prompt=EDITOR_JSON_REPAIR_SYSTEM_PROMPT)
            try:
                return parse_editor_response(
                    repaired_response,
                    episode_number,
                    manuscript_stats,
                    resolved_target_length,
                    target_score,
                )
            except ValueError as second_error:
                raise RuntimeError(
                    "편집자 응답을 JSON으로 해석하지 못했습니다. "
                    f"1차 오류: {first_error}; 2차 오류: {second_error}; "
                    f"원본 응답: {response}; 복구 응답: {repaired_response}"
                ) from second_error


