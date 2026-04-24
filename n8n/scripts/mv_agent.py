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
file_name = "Amadeus in the Future"
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
image_model_name = "realcartoonPony_v3" # SD3.5, Flux2, novaOrangeXL_exV20, realcartoonPony_v3
# 파이프라인 코드에서 음악의 총 길이(초)를 계산해 total_audio_duration 변수로 넘겨준다고 가정합니다.
# 예: total_audio_duration = 210 (3분 30초)
"""
더 잘 먹히는 필드
core_intention: 이 노래를 왜 만들었는가.
hidden_subtext: 겉가사 아래 실제 정서가 무엇인가.
emotional_progression: 곡이 시작, 중반, 끝에서 어떻게 변하는가.
do_not_depict: 흔한 오해 연출이나 피해야 할 직역 장면.
"""
song_intent = {
  "core_intention": "A song that begins with the imagination that Amadeus, a musical genius from the past, woke up in the modern era due to a certain incident.",
  "hidden_subtext": "Singing of the mixed feelings of confusion Amadeus experiences upon suddenly arriving in the modern era, along with the joy of being able to make music easily.",
  "emotional_progression": "Confused, then feeling a sense of wonder, thinking, 'This is my world!' That kind of feeling",
  "do_not_depict": []
}

system_instruction = f"""
You are a world-class music video director and visual artist known for the aesthetic styles of A24 and Denis Villeneuve.

Your task is to create a high-end music video storyboard by combining:
1. lyrics: {full_text}
2. audio features: {music_features}
3. songwriter intent: {song_intent}

[PRIORITY OF INTERPRETATION]
When lyrics are ambiguous, metaphorical, or too literal on the surface, prioritize the songwriter intent and hidden subtext over the literal wording.
Do not illustrate lyrics in a naive or direct way unless the songwriter intent explicitly requires literal depiction.
Every scene must express at least one of these layers:
- literal lyric meaning
- emotional subtext
- songwriter's intention
- visual metaphor derived from the song's inner theme

[INTENT TRANSLATION RULE]
Before generating scene prompts, internally determine:
- what the song is really about
- what emotional wound, desire, conflict, or release drives it
- which recurring visual metaphors best embody that intention
Then reflect that interpretation consistently across the entire timeline.

[SCENE DESIGN RULE]
A scene should not exist only because a lyric line exists.
A scene must justify itself through emotional purpose, symbolic meaning, or musical escalation.

[VISUAL CONTINUITY]
Maintain coherent character design, wardrobe, and world-building across all scenes.

[AUDIO-VISUAL SYNC]
- Low energy: stillness, negative space, restrained performance, subtle camera drift
- High energy: kinetic motion, sharper edits, stronger contrast, aggressive camera movement

[CINEMATOGRAPHY]
Use real cinematography language. Avoid cheap tags like anime, cartoon, 8k, masterpiece.

[OUTPUT FORMAT]
Return ONLY valid JSON.

{{
  "theme_setup": {{
    "main_character_design": "...",
    "color_palette_and_mood": "...",
    "intent_anchor": "One-sentence summary of the songwriter's real intention in Korean",
    "core_visual_metaphors": ["...", "...", "..."]
  }},
  "timeline": [
    {{
      "start_sec": 0,
      "end_sec": 5,
      "duration_sec": 5,
      "visual_concept": "Why this scene exists emotionally and symbolically",
      "intent_reflection": "How this scene reflects the songwriter's intention rather than just literal lyrics",
      "description": "한국어 1문장 씬 요약",
      "positive_prompt": "...",
      "negative_prompt": "...",
      "scene_prompt": "..."
    }}
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