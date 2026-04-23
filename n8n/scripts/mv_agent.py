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
with open(f"{file_name}_transcription_result.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"✅ 오디오 분석 완료! 결과가 '{file_name}_transcription_result.txt'에 저장되었습니다.")
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
with open(f"{file_name}_music_features_result.txt", "w", encoding="utf-8") as f:
    f.write(music_features)

print(f"✅ 음악 특성 분석 완료 (박자: {bpm_value})")

# -------------------------------------------------------------------
# 3. 로컬 Ollama를 이용한 뮤직비디오 콘티 생성
# -------------------------------------------------------------------
print("\nOllama를 통해 비주얼 기획서를 작성합니다...")

ollama_url = "http://localhost:11434/api/generate"
system_instruction = f"""
너는 A24와 빌뇌브 감독 스타일의 미학을 가진 세계적인 뮤직비디오 감독이자 시각 예술가야.
제공된 가사({full_text})와 소리 분석 데이터({music_features})를 결합하여, 음악의 감정선이 시각적으로 폭발하는 하이엔드 기획서를 작성해.

[작업 가이드라인]
1. **분석 단계**: 음악의 에너지(RMS/Energy)가 높은 구간은 역동적인 컷을, 낮은 구간은 정적인 미장센을 배치할 것. 가사의 메타포(은유)를 시각적 상징으로 치환해.
2. **시각적 스타일**: 'Anime'나 'Cartoon' 같은 저렴한 표현은 배제한다. 'Cinematic 35mm film', 'Anamorphic lens flare', 'Volumetric lighting' 등 실제 영화 촬영 용어를 사용해.
3. **필드별 작성 요령**:
   - **positive_prompt (SD 3.5용)**: 정지 영상의 마스터피스를 만든다 생각하고 묘사해. (구도, 인물 외형, 의상 질감, 배경의 디테일, 조명의 각도와 색온도, 필름 그레인)
   - **scene_prompt (LTX-Video용)**: 24fps 영상의 '움직임'을 서술해. (카메라의 Push-in/Pull-out, Pan, Tilt, 피사체의 머릿결 휘날림, 눈동자의 떨림, 연기의 확산 등)

[작성 규칙]
1. 모든 프롬프트(positive, negative, scene)는 **영문**으로만 작성한다.
2. 각 장면의 길이는 음악의 비트와 가사 한 줄의 호흡에 맞춰 3~7초 사이로 유연하게 배분해.
3. 중복되는 표현이나 무의미한 '8k, masterpiece' 같은 단어는 지양하고, 구체적인 상황 묘사에 집중해.

반드시 아래 JSON 구조를 엄격히 지켜서 출력해:
{{
  "timeline": [
    {{
      "start_sec": 0,
      "end_sec": 5,
      "description": "한국어 1문장 요약",
      "positive_prompt": "Cinematic shot of [Subject], shot on 35mm lens, [Lighting details], [Texture details], [Color grading]",
      "negative_prompt": "static, boring, low quality, deformed, text, watermark",
      "scene_prompt": "The camera slowly zooms into the character's eyes as the wind blows through their hair. Soft focus background shifts slightly."
    }}
  ]
}}
"""

# system_instruction = f"""
# 너는 뮤직비디오 전문 영상 감독이야. 제공된 가사와 소리 크기 분석 결과를 종합하여 영상 기획서를 작성해.
# 추상적인 이미지나 영상의 나열이 아닌, 가사를 기반한 영화적인 연출의 영상을 제작 하기 위한 구체적이고 명확한 시각적 요소들을 생성해야 한다.
# 일반적인 애니메이션이나 영화처럼 인물, 사물, 배경, 색감, 조명, 카메라 움직임 등 영상의 시각적 요소들을 상세히 묘사하는 키워드들을 생성해야 한다.
#
# [분석 내용]
# {full_text}
# {music_features}
#
# [작성 규칙]
# 1. description: 이 구간의 분위기와 연출 의도를 한국어로 1문장으로 요약할 것.
# 2. positive_prompt: ComfyUI에서 바로 사용할 수 있도록, 영상의 시각적 요소를 묘사하는 영문 키워드들을 쉼표로 구분하여 작성할 것. (예: "cyberpunk city, heavy rain, neon lights, 8k resolution, cinematic lighting")
# 3. negative_prompt: 피해야 할 시각적 요소나 분위기를 묘사하는 영문 키워드들을 쉼표로 구분하여 작성할 것. (예: "low quality, blurry, bad anatomy, deformed, extra limbs")
# 4. scene_prompt: LTX 영상 생성을 위한 프롬프트 필드로, 영상의 동적인 연출을 묘사하는 프롬프트를 작성하세요.
# 4. 절대 같은 단어나 무의미한 기호를 반복하지 마라.
#
# 반드시 아래 JSON 구조를 지켜서 작성해:
# {{
#   "timeline": [
#     {{
#       "start_sec": 0,
#       "end_sec": 10,
#       "description": "한국어 연출 설명",
#       "positive_prompt": "english, keywords, for, comfyui, generation",
#       "negative_prompt": "english, keywords, for, comfyui, avoidance"
#       "scene_prompt": "english, keywords, for, comfyui, scene, generation"  <-- 이 부분은 나중에 LTX-Video 작업에서 활용할 수 있도록 추가한 필드입니다. Ollama가 이 필드도 작성하도록 지시하세요!
#     }}
#   ]
# }}
# """

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
        save_path = os.path.join(r"C:\Users\parkp\workspace\py_scripts", file_name)  # 원하는 저장 경로로 변경하세요
        #save_path = r"C:\Users\parkp\workspace\py_scripts\mv_conti_plan.json"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(conti_data)
        print(f"💾 기획서 저장 완료! 경로: {save_path}")

except Exception as e:
    print(f"\n❌ 스크립트 실행 중 오류 발생: {e}")