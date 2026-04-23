# n8n MP3 -> MV 콘티 분석 스크립트

`n8n` 아래 여러 스크립트가 섞이지 않도록, 이번 기능은 `n8n/scripts/analyze_mp3/` 계층으로 분리했습니다.

- 실제 실행 스크립트: `n8n/scripts/analyze_mp3/analyze_mp3.py`
- 호환 래퍼(기존 경로 유지): `n8n/analyze_mp3.py`

## 1) 실행 전제

- Python 3.10+ 권장
- 로컬 Hugging Face 추론용 패키지 설치 필요
- Windows에서 ASR 동작을 위해 `ffmpeg` 필요

```powershell
cd "C:\Users\parkp\workspace\py_scripts"
python -m pip install -r .\n8n\requirements.txt
winget install Gyan.FFmpeg
```

## 2) 기본 실행

### A. 테스트(mock, 모델 다운로드 없이)

```powershell
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input .\music.mp3 --provider mock --format json
```

### B. 로컬 Hugging Face 실행

```powershell
$env:HF_TOKEN="hf_xxx"
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input .\music.mp3 --provider hf-local --model qwen2-audio --device cuda --dtype float16 --trust-remote-code --suppress-transformers-warnings --asr-return-timestamps --format md --output .\n8n\out\mv_plan.md
```

기본 동작은 다음과 같습니다.
- ASR 모델(`--asr-model-id`, 기본 `openai/whisper-large-v3-turbo`)로 MP3를 전사
- LLM 모델(`--hf-model-id` 또는 `--model` 별칭 기본값)로 JSON 콘티 생성

모델 별칭 기본 매핑:
- `gemma4` -> `google/gemma-4-9b-it`
- `qwen2-audio` -> `Qwen/Qwen2-Audio-7B-Instruct`

> 참고: `--model gemma4`, `--model qwen2-audio`는 별칭이며 실제 로컬 모델은 `--hf-model-id`로 명시 가능합니다.

## 3) n8n Execute Command 연동 예시

- Command: `python`
- Arguments:
  ```text
  C:\Users\parkp\workspace\py_scripts\n8n\scripts\analyze_mp3\analyze_mp3.py
  --input
  C:\data\music.mp3
  --provider
  hf-local
  --model
  qwen2-audio
  --format
  json
  --output
  C:\data\mv_plan.json
  ```

## 4) n8n 출력 규칙

- `--output` 사용 시 파일 저장 후 stdout에 메타 JSON 1줄 출력
  - 예: `{"ok": true, "output": "C:\\data\\mv_plan.json", "bytes": 12345}`
- `--output` 미사용 시 본문(JSON/Markdown)을 stdout 출력
- 오류 시 stderr 접두어 + 종료코드
  - `INPUT_ERROR`, `MODEL_ERROR`, `PARSE_ERROR`, `IO_ERROR`

## 5) 주요 인자

- `--input` (필수): mp3 파일 경로
- `--model`: `gemma4` | `qwen2-audio` (실제 HF 모델 ID 별칭)
- `--hf-model-id`: 로컬 생성 모델 ID (우선 적용)
- `--hf-token`: HF 토큰 직접 입력(보안상 환경변수 권장)
- `--hf-token-env`: HF 토큰 환경변수 이름(기본 `HF_TOKEN`)
- `--asr-model-id`: 로컬 전사 모델 ID
- `--asr-chunk-sec`: ASR 청크 길이(초), 기본 28
- `--asr-batch-size`: ASR 배치 크기, 기본 8
- `--asr-return-timestamps`: 긴 오디오 안정 처리용 timestamp 모드 강제
- `--whisper-task`: `transcribe` | `translate`
- `--whisper-language`: Whisper 언어 힌트(예: `ko`, `en`)
- `--provider`: `hf-local` | `mock`
- `--device`: `auto` | `cpu` | `cuda`
- `--dtype`: `auto` | `float16` | `bfloat16` | `float32`
- `--trust-remote-code`: remote code 허용이 필요한 모델 로딩 시 사용
- `--suppress-transformers-warnings`: transformers 경고 로그 최소화
- `--format`: `json` | `md`
- `--output`: 출력 파일 경로
- `--save-raw`: 원본 생성 텍스트 저장 경로

## 6) 종료코드

- `0`: 성공
- `2`: 입력 오류
- `3`: 모델/추론 오류
- `4`: 응답 파싱 오류
- `5`: 파일 입출력 오류

## 7) 빠른 검증(스모크 테스트)

```powershell
python .\n8n\smoke_test.py
```

스모크 테스트는 더미 mp3를 만들고 `--provider mock` 모드로 실행해 출력 파일과 핵심 JSON 키를 확인합니다.

## 8) RTX4090 권장 실행 순서

```powershell
cd "C:\Users\parkp\workspace\py_scripts"
python -m pip install -r .\n8n\requirements.txt
huggingface-cli login
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input "C:\data\music.mp3" --provider hf-local --model gemma4 --device cuda --dtype float16 --format json --output "C:\data\mv_plan_gemma4.json"
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input "C:\data\music.mp3" --provider hf-local --model qwen2-audio --device cuda --dtype float16 --trust-remote-code --format json --output "C:\data\mv_plan_qwen2_audio.json"
```

`huggingface-cli login`은 gated 모델 접근 권한이 필요한 경우에만 필요합니다.

## 9) Windows symlink 경고 안내

아래 메시지는 **오류가 아니라 경고**입니다. 캐시를 symlink 대신 복사 방식으로 써서 디스크 사용량이 늘 수 있다는 안내입니다.

- `huggingface_hub cache-system uses symlinks by default ...`

원하면 경고를 숨길 수 있습니다.

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
```

또는 Windows Developer Mode 활성화/관리자 권한 실행으로 symlink를 사용할 수 있습니다.

## 10) 자주 막히는 지점

- `MODEL_ERROR: ffmpeg가 설치되어 있지 않습니다...`
  - `winget install Gyan.FFmpeg` 설치 후 **새 PowerShell**에서 재실행
- `INPUT_ERROR: ... more than 3000 mel input features ...`
  - Whisper 장시간 오디오 제약입니다. 아래처럼 청크/타임스탬프 옵션으로 재실행

```powershell
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input "F:\generated_songs\Amadeus in the Future.mp3" --provider hf-local --model gemma4 --device cuda --dtype float16 --asr-chunk-sec 28 --asr-batch-size 8 --asr-return-timestamps --format json --output "mv_plan_gemma4.json"
```
- `IO_ERROR/MODEL_ERROR: ... not a valid model identifier ...`
  - 모델 ID 오타 또는 권한 이슈입니다. `--hf-model-id`를 정확히 지정하거나 `HF_TOKEN` 설정 후 재시도

```powershell
$env:HF_TOKEN="hf_xxx"
python .\n8n\scripts\analyze_mp3\analyze_mp3.py --input "C:\data\music.mp3" --provider hf-local --hf-model-id "google/gemma-2-9b-it" --device cuda --dtype float16 --format json --output "C:\data\mv_plan.json"
```
- `huggingface_hub cache-system uses symlinks...`
  - 치명 오류 아님(경고). 필요 시 `HF_HUB_DISABLE_SYMLINKS_WARNING=1` 설정
