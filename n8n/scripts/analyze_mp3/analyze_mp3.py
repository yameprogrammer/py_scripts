#!/usr/bin/env python3
"""n8n Execute Command에서 호출하는 로컬 Hugging Face 기반 MP3 분석 스크립트."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast

try:
    import librosa
except ImportError:
    librosa = None

try:
    from transformers import pipeline
    from transformers.utils import logging as hf_logging
except ImportError:
    pipeline = None
    hf_logging = None

try:
    import torch
except ImportError:
    torch = None

EXIT_OK = 0
EXIT_BAD_INPUT = 2
EXIT_MODEL_ERROR = 3
EXIT_PARSE_ERROR = 4
EXIT_IO_ERROR = 5

MODEL_ALIAS = {
    # Hugging Face에서 접근 가능한 기본 Gemma 계열 모델(필요 시 --hf-model-id로 직접 지정)
    "gemma4": "google/gemma-2-9b-it",
    "qwen2-audio": "Qwen/Qwen2-Audio-7B-Instruct",
}


@dataclass
class CliConfig:
    input_path: Path
    model: str
    output_format: str
    output_path: Path | None
    provider: str
    hf_model_id: str | None
    hf_token: str | None
    hf_token_env: str
    asr_model_id: str
    asr_chunk_sec: int
    asr_batch_size: int
    asr_return_timestamps: bool
    device: str
    dtype: str
    trust_remote_code: bool
    suppress_transformers_warnings: bool
    whisper_task: str
    whisper_language: str | None
    max_new_tokens: int
    language: str
    temperature: float
    save_raw: Path | None


def parse_args() -> CliConfig:
    parser = argparse.ArgumentParser(
        description="MP3를 로컬 Hugging Face 모델로 분석해 MV 콘티를 생성합니다."
    )
    parser.add_argument("--input", required=True, help="분석할 mp3 파일 경로")
    parser.add_argument(
        "--model",
        default="qwen2-audio",
        choices=["gemma4", "qwen2-audio"],
        help="사용할 모델 별칭",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        default="json",
        choices=["json", "md"],
        help="최종 출력 형식",
    )
    parser.add_argument("--output", help="결과 저장 경로(.json 또는 .md)")
    parser.add_argument(
        "--provider",
        default="hf-local",
        choices=["hf-local", "mock"],
        help="hf-local: 로컬 Hugging Face 실행, mock: 테스트 데이터 반환",
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="로컬 생성 모델 ID. 미지정 시 --model 별칭 기본값 사용",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face 토큰(권장: 직접 입력 대신 --hf-token-env 사용)",
    )
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Hugging Face 토큰을 읽을 환경변수 이름(기본: HF_TOKEN)",
    )
    parser.add_argument(
        "--asr-model-id",
        default="openai/whisper-large-v3-turbo",
        help="로컬 음성인식(전사) 모델 ID",
    )
    parser.add_argument(
        "--asr-chunk-sec",
        type=int,
        default=28,
        help="ASR 청크 길이(초). 긴 오디오 안정성을 위해 30초 미만 권장",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=8,
        help="ASR 배치 크기(RTX4090 권장 시작값: 8)",
    )
    parser.add_argument(
        "--asr-return-timestamps",
        action="store_true",
        help="ASR 결과에 타임스탬프를 포함해 긴 오디오 처리 안정성을 높임",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="추론 디바이스",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1800,
        help="생성 최대 토큰 수",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="생성 온도")
    parser.add_argument("--language", default="ko", help="출력 언어 코드")
    parser.add_argument("--save-raw", help="원본 모델 출력 저장 경로")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="모델 가중치 dtype (RTX4090 권장: float16)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="모델 로더에서 trust_remote_code=True 사용",
    )
    parser.add_argument(
        "--suppress-transformers-warnings",
        action="store_true",
        help="transformers 경고 로그를 최소화합니다.",
    )
    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Whisper 태스크. transcribe(원문 전사) / translate(영문 번역)",
    )
    parser.add_argument(
        "--whisper-language",
        default=None,
        help="Whisper 언어 힌트(예: ko, en). 미지정 시 자동 감지",
    )

    args = parser.parse_args()
    return CliConfig(
        input_path=Path(args.input),
        model=args.model,
        output_format=args.output_format,
        output_path=Path(args.output) if args.output else None,
        provider=args.provider,
        hf_model_id=args.hf_model_id,
        hf_token=args.hf_token,
        hf_token_env=args.hf_token_env,
        asr_model_id=args.asr_model_id,
        asr_chunk_sec=args.asr_chunk_sec,
        asr_batch_size=args.asr_batch_size,
        asr_return_timestamps=args.asr_return_timestamps,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        suppress_transformers_warnings=args.suppress_transformers_warnings,
        whisper_task=args.whisper_task,
        whisper_language=args.whisper_language,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        temperature=args.temperature,
        save_raw=Path(args.save_raw) if args.save_raw else None,
    )


def validate_input(input_path: Path) -> None:
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    if input_path.suffix.lower() != ".mp3":
        raise ValueError(f"입력 파일은 .mp3 이어야 합니다: {input_path}")


def get_llm_model_id(config: CliConfig) -> str:
    if config.hf_model_id:
        return config.hf_model_id
    return MODEL_ALIAS[config.model]


def get_device_for_pipeline(device: str) -> int | str:
    if device == "cpu":
        return -1
    if device == "cuda":
        return 0
    if torch is not None and cast(Any, torch).cuda.is_available():
        return 0
    return -1


def extract_audio_features(input_path: Path) -> Dict[str, Any]:
    if librosa is None:
        return {"duration_sec": None, "tempo_bpm_estimate": None}

    librosa_mod = cast(Any, librosa)
    y, sr = librosa_mod.load(str(input_path), sr=None, mono=True)
    duration = float(len(y) / sr) if sr else None

    tempo_estimate = None
    try:
        tempo, _ = librosa_mod.beat.beat_track(y=y, sr=sr)
        tempo_estimate = float(tempo)
    except Exception:
        tempo_estimate = None

    return {
        "duration_sec": round(duration, 2) if duration else None,
        "tempo_bpm_estimate": round(tempo_estimate, 1) if tempo_estimate else None,
    }


def get_torch_dtype(dtype_name: str) -> Any:
    if torch is None or dtype_name == "auto":
        return None

    torch_mod = cast(Any, torch)
    mapping = {
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
    }
    return mapping.get(dtype_name)


def build_model_kwargs(config: CliConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    dtype = get_torch_dtype(config.dtype)
    if dtype is not None:
        # transformers 최신 버전은 dtype 키를 권장합니다.
        kwargs["dtype"] = dtype
    return kwargs


def resolve_hf_token(config: CliConfig) -> str | None:
    if config.hf_token and config.hf_token.strip():
        return config.hf_token.strip()

    from_env = os.getenv(config.hf_token_env, "").strip()
    if from_env:
        return from_env

    fallback = os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    return fallback or None


def configure_runtime(config: CliConfig) -> None:
    if not config.suppress_transformers_warnings:
        return

    # 필요 시 허브 symlink 경고와 transformers 경고를 줄입니다.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if hf_logging is not None:
        hf_logging.set_verbosity_error()


def build_pipeline_kwargs(config: CliConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model_kwargs": build_model_kwargs(config),
        "trust_remote_code": config.trust_remote_code,
    }
    token = resolve_hf_token(config)
    if token:
        kwargs["token"] = token
    return kwargs


def create_pipeline_with_fallback(
    task: str,
    model: str,
    device_value: int | str,
    config: CliConfig,
    ignore_warning: bool = False,
    use_device_map_auto: bool = False,
) -> Any:
    if pipeline is None:
        raise RuntimeError("transformers가 설치되지 않았습니다. requirements를 설치하세요.")

    pipeline_fn = cast(Any, pipeline)
    kwargs: Dict[str, Any] = {
        "task": task,
        "model": model,
        **build_pipeline_kwargs(config),
    }
    if use_device_map_auto:
        kwargs["device_map"] = "auto"
    else:
        kwargs["device"] = device_value
    if ignore_warning:
        kwargs["ignore_warning"] = True

    try:
        return pipeline_fn(**kwargs)
    except TypeError as exc:
        msg = str(exc)
        # 버전 호환: 구버전은 token 대신 use_auth_token 사용
        if "token" in msg and "unexpected keyword argument" in msg and "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
            return pipeline_fn(**kwargs)
        # 버전 호환: ignore_warning 미지원 버전
        if "ignore_warning" in msg and "unexpected keyword argument" in msg:
            kwargs.pop("ignore_warning", None)
            return pipeline_fn(**kwargs)
        raise


def normalize_hf_load_error(exc: Exception, model_id: str, token_env_name: str) -> RuntimeError:
    msg = str(exc)
    marker = (
        "is not a local folder and is not a valid model identifier",
        "Repository Not Found",
        "401",
        "403",
        "gated",
    )
    if any(m in msg for m in marker):
        return RuntimeError(
            f"모델 로딩 실패: '{model_id}' 접근 권한/식별자를 확인하세요. "
            f"필요 시 `hf auth login` 또는 환경변수 `{token_env_name}`(또는 `--hf-token`)를 설정하세요."
        )
    return RuntimeError(f"모델 로딩 실패: {msg}")


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg"):
        return
    raise RuntimeError(
        "ffmpeg가 설치되어 있지 않습니다. Windows에서는 'winget install Gyan.FFmpeg' 후 새 터미널에서 다시 실행하세요."
    )


def extract_transcript_from_asr_result(result: Any) -> str:
    if isinstance(result, dict):
        chunks = result.get("chunks")
        if isinstance(chunks, list) and chunks:
            parts = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    text = str(chunk.get("text", "")).strip()
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts).strip()
        return str(result.get("text", "")).strip()
    return str(result).strip()


def transcribe_mp3(input_path: Path, asr_model_id: str, device: str, config: CliConfig) -> str:
    if pipeline is None:
        raise RuntimeError("transformers가 설치되지 않았습니다. requirements를 설치하세요.")

    ensure_ffmpeg_available()

    try:
        asr = create_pipeline_with_fallback(
            task="automatic-speech-recognition",
            model=asr_model_id,
            device_value=get_device_for_pipeline(device),
            config=config,
            ignore_warning=config.suppress_transformers_warnings,
        )
    except Exception as exc:
        raise normalize_hf_load_error(exc, asr_model_id, config.hf_token_env) from exc

    # Whisper 계열은 30초 초과 입력에서 timestamp 예측 모드가 필요합니다.
    should_use_timestamps = config.asr_return_timestamps or "whisper" in asr_model_id.lower()
    call_kwargs: Dict[str, Any] = {
        "chunk_length_s": config.asr_chunk_sec,
        "batch_size": config.asr_batch_size,
        "generate_kwargs": {
            "task": config.whisper_task,
        },
    }
    if config.whisper_language:
        call_kwargs["generate_kwargs"]["language"] = config.whisper_language
    if should_use_timestamps:
        call_kwargs["return_timestamps"] = True

    try:
        result = asr(str(input_path), **call_kwargs)
    except Exception as exc:
        msg = str(exc)
        if "more than 3000 mel input features" in msg:
            raise RuntimeError(
                "ASR 장시간 오디오 처리 실패: --asr-chunk-sec 20~28, --asr-return-timestamps 옵션으로 다시 시도하세요."
            ) from exc
        raise

    transcript = extract_transcript_from_asr_result(result)
    if not transcript:
        raise RuntimeError("ASR 전사 결과가 비어 있습니다. 모델/오디오를 확인하세요.")
    return transcript


def build_prompt(language: str, transcript: str, audio_features: Dict[str, Any]) -> str:
    transcript = transcript or "(가사/보컬 전사 결과 없음)"
    return (
        "당신은 음악 분석가이자 뮤직비디오 연출 감독입니다. 아래 정보를 분석해 반드시 JSON만 출력하세요.\n\n"
        f"[전사 텍스트]\n{transcript}\n\n"
        f"[오디오 메타]\n{json.dumps(audio_features, ensure_ascii=False)}\n\n"
        "[출력 스키마]\n"
        "{\n"
        '  "track_analysis": {\n'
        '    "genre": "",\n'
        '    "tempo_bpm_estimate": 0,\n'
        '    "key_estimate": "",\n'
        '    "mood_keywords": [""],\n'
        '    "structure": [\n'
        '      {"section": "intro|verse|pre-chorus|chorus|bridge|outro", "start_sec": 0, "end_sec": 0, "energy": 1-10}\n'
        "    ],\n"
        '    "notable_sonic_elements": [""],\n'
        '    "lyric_theme_guess": ""\n'
        "  },\n"
        '  "mv_concept": {\n'
        '    "one_line_concept": "",\n'
        '    "visual_tone": [""],\n'
        '    "color_palette": [""],\n'
        '    "references": [""],\n'
        '    "locations": [""],\n'
        '    "characters": [""],\n'
        '    "props": [""],\n'
        '    "style_notes": ""\n'
        "  },\n"
        '  "storyboard": [\n'
        "    {\n"
        '      "shot_no": 1,\n'
        '      "time_range": "00:00-00:10",\n'
        '      "section": "intro",\n'
        '      "visual": "",\n'
        '      "camera": "",\n'
        '      "transition": "",\n'
        '      "editing_rhythm": "",\n'
        '      "vfx_or_graphics": "",\n'
        '      "production_notes": ""\n'
        "    }\n"
        "  ],\n"
        '  "deliverables": {\n'
        '    "thumbnail_ideas": [""],\n'
        '    "short_form_cut_points": ["00:15", "00:30"],\n'
        '    "risk_points": [""],\n'
        '    "next_actions": [""],\n'
        '    "confidence": "low|medium|high"\n'
        "  }\n"
        "}\n\n"
        f"출력 언어는 {language}를 우선 사용하세요."
    )


def generate_with_hf_local(config: CliConfig, transcript: str, audio_features: Dict[str, Any]) -> str:
    if pipeline is None:
        raise RuntimeError("transformers가 설치되지 않았습니다. requirements를 설치하세요.")

    llm_model_id = get_llm_model_id(config)
    prompt = build_prompt(config.language, transcript, audio_features)

    try:
        generator = create_pipeline_with_fallback(
            task="text-generation",
            model=llm_model_id,
            device_value=get_device_for_pipeline(config.device),
            config=config,
            use_device_map_auto=(config.device == "auto"),
        )
    except Exception as exc:
        raise normalize_hf_load_error(exc, llm_model_id, config.hf_token_env) from exc

    response = generator(
        prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        do_sample=True,
        return_full_text=False,
    )

    if not response:
        raise RuntimeError("모델 출력이 비어 있습니다.")

    if isinstance(response, list) and isinstance(response[0], dict):
        text = str(response[0].get("generated_text", "")).strip()
    else:
        text = str(response).strip()

    if config.save_raw:
        config.save_raw.parent.mkdir(parents=True, exist_ok=True)
        config.save_raw.write_text(text, encoding="utf-8")

    return text


def build_mock_response(language: str) -> str:
    text = (
        "{\n"
        "  \"track_analysis\": {\n"
        "    \"genre\": \"cinematic pop\",\n"
        "    \"tempo_bpm_estimate\": 108,\n"
        "    \"key_estimate\": \"A minor\",\n"
        "    \"mood_keywords\": [\"melancholy\", \"hopeful\", \"uplifting\"],\n"
        "    \"structure\": [\n"
        "      {\"section\": \"intro\", \"start_sec\": 0, \"end_sec\": 18, \"energy\": 3},\n"
        "      {\"section\": \"verse\", \"start_sec\": 18, \"end_sec\": 46, \"energy\": 5},\n"
        "      {\"section\": \"chorus\", \"start_sec\": 46, \"end_sec\": 76, \"energy\": 8}\n"
        "    ],\n"
        "    \"notable_sonic_elements\": [\"wide pads\", \"tight kick\", \"reverse fx\"],\n"
        "    \"lyric_theme_guess\": \"상실 이후의 재도약\"\n"
        "  },\n"
        "  \"mv_concept\": {\n"
        "    \"one_line_concept\": \"도시의 새벽을 달리며 상처를 빛으로 바꾸는 여정\",\n"
        "    \"visual_tone\": [\"dreamy\", \"urban\", \"cinematic\"],\n"
        "    \"color_palette\": [\"deep blue\", \"neon magenta\", \"warm amber\"],\n"
        "    \"references\": [\"long exposure city lights\", \"slow shutter portrait\"],\n"
        "    \"locations\": [\"roof top\", \"subway platform\", \"empty street\"],\n"
        "    \"characters\": [\"lead artist\", \"shadow self\"],\n"
        "    \"props\": [\"old camcorder\", \"mirror fragments\"],\n"
        "    \"style_notes\": \"후렴으로 갈수록 카메라 이동과 컷 전환을 빠르게\"\n"
        "  },\n"
        "  \"storyboard\": [\n"
        "    {\n"
        "      \"shot_no\": 1,\n"
        "      \"time_range\": \"00:00-00:10\",\n"
        "      \"section\": \"intro\",\n"
        "      \"visual\": \"새벽 옥상에서 도시를 내려다보는 실루엣\",\n"
        "      \"camera\": \"slow dolly in\",\n"
        "      \"transition\": \"fade from black\",\n"
        "      \"editing_rhythm\": \"long take\",\n"
        "      \"vfx_or_graphics\": \"light leak\",\n"
        "      \"production_notes\": \"바람 소품과 헤어 무빙 강조\"\n"
        "    }\n"
        "  ],\n"
        "  \"deliverables\": {\n"
        "    \"thumbnail_ideas\": [\"네온 역광 실루엣\", \"거울 파편 클로즈업\"],\n"
        "    \"short_form_cut_points\": [\"00:17\", \"00:46\"],\n"
        "    \"risk_points\": [\"야간 촬영 노이즈\", \"플랫폼 촬영 허가\"],\n"
        "    \"next_actions\": [\"로케이션 헌팅\", \"콘티 프리비즈\"],\n"
        "    \"confidence\": \"medium\"\n"
        "  }\n"
        "}"
    )
    if language.lower().startswith("en"):
        return text.replace("상실 이후의 재도약", "rebound after loss")
    return text


def parse_json_loose(text: str) -> Dict[str, Any]:
    raw = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*})\s*```", raw)
    if fence_match:
        raw = fence_match.group(1)

    start_idx = raw.find("{")
    end_idx = raw.rfind("}")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError("유효한 JSON 객체를 찾을 수 없습니다.")

    return json.loads(raw[start_idx : end_idx + 1])


def render_markdown(analysis: Dict[str, Any]) -> str:
    track = analysis.get("track_analysis", {})
    concept = analysis.get("mv_concept", {})
    board = analysis.get("storyboard", [])
    deliverables = analysis.get("deliverables", {})

    lines = [
        "# MV 분석 리포트",
        "",
        "## 1) 트랙 분석",
        f"- 장르: {track.get('genre', '-')}",
        f"- BPM 추정: {track.get('tempo_bpm_estimate', '-')}",
        f"- 키 추정: {track.get('key_estimate', '-')}",
        f"- 무드 키워드: {', '.join(track.get('mood_keywords', []))}",
        f"- 가사 테마 추정: {track.get('lyric_theme_guess', '-')}",
        "",
        "## 2) MV 콘셉트",
        f"- 한 줄 콘셉트: {concept.get('one_line_concept', '-')}",
        f"- 비주얼 톤: {', '.join(concept.get('visual_tone', []))}",
        f"- 컬러 팔레트: {', '.join(concept.get('color_palette', []))}",
        f"- 로케이션: {', '.join(concept.get('locations', []))}",
        "",
        "## 3) 콘티(샷 리스트)",
    ]

    if board:
        for shot in board:
            lines.extend(
                [
                    f"### Shot {shot.get('shot_no', '-')}: {shot.get('time_range', '-')}",
                    f"- 섹션: {shot.get('section', '-')}",
                    f"- 비주얼: {shot.get('visual', '-')}",
                    f"- 카메라: {shot.get('camera', '-')}",
                    f"- 전환: {shot.get('transition', '-')}",
                    f"- 편집 리듬: {shot.get('editing_rhythm', '-')}",
                    f"- VFX/그래픽: {shot.get('vfx_or_graphics', '-')}",
                    f"- 제작 메모: {shot.get('production_notes', '-')}",
                    "",
                ]
            )
    else:
        lines.append("- 콘티 정보가 없습니다.")

    lines.extend(
        [
            "## 4) 실행 항목",
            f"- 썸네일 아이디어: {', '.join(deliverables.get('thumbnail_ideas', []))}",
            f"- 쇼츠 컷포인트: {', '.join(deliverables.get('short_form_cut_points', []))}",
            f"- 리스크: {', '.join(deliverables.get('risk_points', []))}",
            f"- 다음 액션: {', '.join(deliverables.get('next_actions', []))}",
            f"- 신뢰도: {deliverables.get('confidence', '-')}",
            "",
        ]
    )
    return "\n".join(lines)


def format_output(analysis: Dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    return render_markdown(analysis)


def write_output(content: str, output_path: Path | None) -> None:
    if output_path is None:
        print(content)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output_path),
                "bytes": len(content.encode("utf-8")),
            },
            ensure_ascii=False,
        )
    )


def run(config: CliConfig) -> int:
    validate_input(config.input_path)
    configure_runtime(config)

    if config.provider == "mock":
        generated_text = build_mock_response(config.language)
    else:
        transcript = transcribe_mp3(config.input_path, config.asr_model_id, config.device, config)
        audio_features = extract_audio_features(config.input_path)
        generated_text = generate_with_hf_local(config, transcript, audio_features)

    analysis = parse_json_loose(generated_text)
    output = format_output(analysis, config.output_format)
    write_output(output, config.output_path)
    return EXIT_OK


def main() -> int:
    try:
        return run(parse_args())
    except json.JSONDecodeError as exc:
        print(f"PARSE_ERROR: JSON 파싱 실패 - {exc}", file=sys.stderr)
        return EXIT_PARSE_ERROR
    except ValueError as exc:
        print(f"INPUT_ERROR: {exc}", file=sys.stderr)
        return EXIT_BAD_INPUT
    except RuntimeError as exc:
        print(f"MODEL_ERROR: {exc}", file=sys.stderr)
        return EXIT_MODEL_ERROR
    except OSError as exc:
        print(f"IO_ERROR: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR


if __name__ == "__main__":
    sys.exit(main())
