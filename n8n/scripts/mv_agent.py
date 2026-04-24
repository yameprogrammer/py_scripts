import torch
import requests
import json
import librosa
import numpy as np
import warnings
import os

# -------------------------------------------------------------------
# 0. 경고 문구 및 불필요한 로그 완전 차단 (Clean Console)
# -------------------------------------------------------------------
# 파이썬 기본 경고 무시
warnings.filterwarnings("ignore")

# 허깅페이스 인증 경고 및 심볼릭 링크 경고 환경변수 차단
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import transformers

# 변환기(Transformers) 라이브러리의 로그 출력 수준을 오류(Error)로만 제한
transformers.logging.set_verbosity_error()

from transformers import pipeline

# -------------------------------------------------------------------
# 1. 환경 설정 및 모델 불러오기
# -------------------------------------------------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"
file_name = "고유 시간의 선율"
audio_path = f"F:\\generated_songs\\{file_name}.mp3"  # 분석할 음악 파일 경로

print(f"[{device}] 환경에서 오디오 분석을 시작합니다. (경고 문구 차단됨)")

# 일괄 처리기(Pipeline) 설정
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,
    device=device,
    return_timestamps=True,
    ignore_warning=True  # 실험적 기능(chunk_length_s) 사용에 대한 내부 경고 무시
)

# -------------------------------------------------------------------
# 2. 오디오 분석 수행
# -------------------------------------------------------------------
print("음악을 듣고 텍스트로 변환하는 중... (잠시만 기다려 주세요)")
result = transcriber(
    audio_path,
    generate_kwargs={"language": "ko", "task": "transcribe"}
)

full_text = result["text"]

# 분석된 원본 텍스트 파일 저장
# 실행 스크립트 위치에 out 디렉토리가 결과물이 저장되는 기본 베이스 경로로 지정
base_out_path = os.path.dirname(os.path.abspath(__file__))
base_out_path = os.path.join(base_out_path, "out")

with open(f"{base_out_path}\\{file_name}_transcription_result.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ 오디오 분석 완료! 결과가 '{base_out_path}\\{file_name}_transcription_result.txt'에 저장되었습니다.")
print("-" * 50)
print(full_text[:100] + "...")
print("-" * 50)

# -------------------------------------------------------------------
# [새로 추가된 항목] 2.5 음악적 특성 수치화 (박자, 에너지 흐름)
# -------------------------------------------------------------------
print("\n음악의 박자와 구간별 웅장함을 수학적으로 분석하는 중...")
y, sr = librosa.load(audio_path)

# 1. 분당 박자 수 추출
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
bpm_value = round(tempo[0]) if isinstance(tempo, np.ndarray) else round(tempo)

# 2. 소리 크기 흐름 추출
rms = librosa.feature.rms(y=y)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

energy_log = []
# 10초 단위로 소리 크기 평균을 내어 기록 (언어 모델이 이해하기 쉽도록 정리)
for start_t in range(0, int(times[-1]), 10):
    end_t = start_t + 10
    idx = np.where((times >= start_t) & (times < end_t))
    if len(idx[0]) > 0:
        avg_energy = float(np.mean(rms[idx]))
        energy_log.append(f"[{start_t}초 ~ {end_t}초] 소리 크기: {avg_energy:.4f}")

music_features = f"분당 박자 수: {bpm_value} 박자\n\n[시간대별 소리 크기 변화]\n" + "\n".join(energy_log)

# 분석된 features 텍스트 파일 저장
with open(f"{base_out_path}\\{file_name}_music_features_result.txt", "w", encoding="utf-8") as f:
    f.write(music_features)

print(f"✅ 음악 특성 분석 완료 (박자: {bpm_value})")

total_audio_duration = int(times[-1])
print(f"총 음악 길이: {total_audio_duration}초")

# -------------------------------------------------------------------
# 3. 로컬 Ollama를 이용한 뮤직비디오 콘티 생성
# -------------------------------------------------------------------
print("\nOllama를 통해 비주얼 기획서를 작성합니다...")

ollama_url = "http://localhost:11434/api/generate"
image_model_name = "Flux2" # SD3.5, Flux2, novaOrangeXL_exV20, realcartoonPony_v3
# 파이프라인 코드에서 음악의 총 길이(초)를 계산해 total_audio_duration 변수로 넘겨준다고 가정합니다.
# 예: total_audio_duration = 210 (3분 30초)

system_instruction = f"""
You are a world-class music video director and visual artist known for the aesthetic styles of A24 and Denis Villeneuve.
Your objective is to analyze the provided lyrics ({full_text}) and audio features ({music_features}) to design a high-end, visually explosive music video storyboard. 
The total duration of the track is {total_audio_duration} seconds.

[CORE DIRECTIVES]
1. **Visual Continuity**: Ensure consistent character appearance, wardrobe, and environmental themes across all scenes. 
2. **Audio-Visual Sync**: Sync visual energy with audio RMS. Low energy = static/slow push-in. High energy = dynamic motion/whip pans.
3. **Cinematography & Aesthetics**: Strictly avoid cheap terms like 'Anime' or 'Cartoon'. Use precise cinematography vocabulary ('Cinematic 35mm film', 'Volumetric fog').
4. **Prompting Guidelines**:
   - **positive_prompt (SD 3.5)**: Fluid, highly detailed natural language for a static masterpiece. Include [Subject] + [Environment] + [Lighting] + [Camera shot].
   - **scene_prompt (LTX-Video)**: Focus ONLY on motion. Describe [Subject's movement] + [Camera movement].
5. **COMPLETE COVERAGE (CRITICAL)**: You MUST map out the entire song from 0 to exactly {total_audio_duration} seconds. Do not stop early. Do not skip any sections. The `end_sec` of the very last scene in your JSON array MUST exactly match {total_audio_duration}.

[OUTPUT FORMAT]
You must output ONLY valid JSON.
**TIME CONSTRAINT**: Each scene must be between 3 to 20 seconds. `duration_sec` (`end_sec` - `start_sec`) must be ≤ 20.

{{
  "theme_setup": {{
    "main_character_design": "...",
    "color_palette_and_mood": "..."
  }},
  "timeline": [
    {{
      "start_sec": 0,
      "end_sec": 5,
      "duration_sec": 5,
      "visual_concept": "Creative reasoning",
      "description": "한국어 1문장 씬 요약",
      "positive_prompt": "Highly detailed English prompt...",
      "negative_prompt": "static, boring, low quality...",
      "scene_prompt": "Motion description..."
    }}
    // ... continue until the final scene's end_sec equals {total_audio_duration} ...
  ],
  "is_complete": true
}}
"""

# ollama 반응 안하면 taskkill /f /im ollama.exe /t
payload = {
    "model": "gemma4:e4b", # qwen3.5:9b, gemma4:e4b, 교체하신 모델명 유지
    "prompt": system_instruction,
    "stream": True,
    # "format": "json",  <-- 이 옵션을 과감히 주석 처리하거나 삭제하세요!
    "options": {
        "temperature": 0.1,
        "repeat_penalty": 1.2,
        "num_predict": -1 #  -1 생성량 제한 해제 (8192 Ollama가 자체적으로 멈출 때까지 생성)
    }
}

print("\nOllama가 기획서를 작성 중입니다... (실시간 출력)\n" + "-" * 50)

try:
    response = requests.post(ollama_url, json=payload, stream=True)
    response.raise_for_status()

    conti_data = ""

    for line in response.iter_lines():
        if line:
            # [디버깅] 서버가 실제로 보내는 순수 데이터 구조 확인
            raw_text = line.decode('utf-8')
            print(f"RAW: {raw_text}") # 주석 해제 시 Ollama의 원시 로그를 볼 수 있습니다.

            chunk = json.loads(raw_text)

            # Ollama 내부에서 에러를 던진 경우 캐치
            if "error" in chunk:
                print(f"\n❌ Ollama 내부 에러 감지: {chunk['error']}")
                break

            word = chunk.get("response", "")
            conti_data += word
            print(word, end="", flush=True)

    print("\n" + "-" * 50)

    # 0바이트 파일 방지 로직 추가
    if not conti_data.strip():
        print("⚠️ 경고: Ollama가 아무런 내용도 생성하지 않았습니다. 프롬프트나 모델 상태를 점검하세요.")
    else:
        # 파일 저장 위치를 명확히 지정 (절대 경로 추천)
        file_name = f"{file_name}_conti_plan.json"
        save_path = os.path.join(base_out_path, file_name)  # 원하는 저장 경로로 변경하세요
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(conti_data)
        print(f"💾 기획서 저장 완료! 경로: {save_path}")

except Exception as e:
    print(f"\n❌ 스크립트 실행 중 오류 발생: {e}")