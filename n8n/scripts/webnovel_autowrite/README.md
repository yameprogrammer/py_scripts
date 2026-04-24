# AI 웹소설 자동 집필 시스템 초안

`n8n/scripts/webnovel_autowrite/` 아래에 웹소설 자동 집필용 구조를 분리했습니다.
기본 LLM은 **Ollama 로컬 모델**을 사용하고, 필요 시 **Gemini API** 또는 **OpenAI GPT API**로 전환할 수 있게 설계했습니다.

## 프로젝트 구조

```text
webnovel_autowrite/
├─ main.py                     # 전체 흐름 제어
├─ agents.py                   # 작가/편집자 프롬프트 및 LLM 호출
├─ storage.py                  # state/world/episode 저장 유틸
├─ .env.example                # 선택적 환경변수 예시
├─ README.md
└─ projects/
   └─ default/
      ├─ world_setting.json    # 세계관, 캐릭터 시트, 주요 플롯
      ├─ state.json            # 현재 화수, 마지막 성공 지점, provider 설정
      └─ episodes/
         └─ episode_001/
            ├─ manuscript.md   # 각 화 원고
            └─ review.json     # 각 화 검수 점수/피드백
```

## 저장 방식 제안

### `world_setting.json`
- 시리즈 메타데이터
- 세계관 규칙
- 캐릭터 시트
- 장기 플롯/에피소드 비트

### `state.json`
- `runtime.current_episode`: 현재 작업 중인 화수
- `runtime.last_success_checkpoint`: 마지막 성공 지점
- `workflow.last_success`: 마지막 성공 시각과 단계
- `workflow.quality_gate.target_score`: 합격 목표 점수(기본 80)
- `workflow.quality_gate.max_revision_rounds`: 자동 재작성 최대 횟수(기본 5)
- `llm.active_provider`: 현재 기본 provider
- `llm.providers.*`: Ollama / Gemini / OpenAI 개별 설정
- `episodes.items`: 각 화별 상태와 파일 경로 인덱스

### `episodes/episode_xxx/`
- `manuscript.md`: 작가 에이전트 초안과 수정본
- `review.json`: 편집자 점수, 분량 점수, 문제점, 수정 요청, 시도 이력

## 빠른 시작

> `--project-root`에 상대경로를 넣으면 **현재 터미널 위치가 아니라 `main.py`가 있는 `webnovel_autowrite/` 기준**으로 해석됩니다.
> 예: `--project-root projects/my_series` → `n8n/scripts/webnovel_autowrite/projects/my_series`

### 1) 초기 구조 생성 확인

```powershell
python .\n8n\scripts\webnovel_autowrite\main.py --init-only
```

### 2) 현재 화수용 파일 준비

```powershell
python .\n8n\scripts\webnovel_autowrite\main.py --prepare-next-episode
```

### 3) 실제 모델 호출 없이 설정 확인

```powershell
python .\n8n\scripts\webnovel_autowrite\main.py --init-only --dry-run
```

### 4) Ollama로 1회 실행

```powershell
$env:OLLAMA_BASE_URL="http://127.0.0.1:11434"
python .\n8n\scripts\webnovel_autowrite\main.py --run-once --provider ollama --model qwen2.5:7b-instruct
```

이 명령은 이제 단순 1회 생성이 아니라 다음 루프로 동작합니다.

1. 초안 작성
2. 편집자 검수
3. 목표 분량/점수 미달 시 재작성
4. `overall >= 80` 이고 분량 기준을 만족하면 종료
5. 최대 재작성 횟수까지 실패하면 종료 코드 `1` 반환

### 5) Gemini 또는 OpenAI 사용

```powershell
$env:GEMINI_API_KEY="your_key"
python .\n8n\scripts\webnovel_autowrite\main.py --run-once --provider gemini --model gemini-2.5-flash

$env:OPENAI_API_KEY="your_key"
python .\n8n\scripts\webnovel_autowrite\main.py --run-once --provider openai --model gpt-4.1-mini
```

## 참고
- 실제 운영에서는 `world_setting.json`을 먼저 충분히 채운 뒤 `--run-once`를 사용하는 것을 권장합니다.
- `series.target_length.chars_per_episode`를 쓰면 한국어 기준 분량 제어가 더 정확합니다. 없으면 기존 `words_per_episode` 값을 분량 목표로 사용합니다.
- `review.json`에는 최종 점수뿐 아니라 `attempt_history`, `manuscript_stats`, `quality_gate`가 함께 저장됩니다.

