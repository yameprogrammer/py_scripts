import json
import urllib.request
import random
import shutil
import os
import time

# --- [환경 설정] 콤피유아이 경로를 PC 환경에 맞게 수정하세요 ---
COMFY_DIR = r"F:\ComfyUI_windows_portable\ComfyUI"
OUTPUT_DIR = os.path.join(COMFY_DIR, "output")
INPUT_DIR = os.path.join(COMFY_DIR, "input")
SERVER_URL = "http://127.0.0.1:8188/prompt"
HISTORY_URL = "http://127.0.0.1:8188/history"

base_out_path = os.path.dirname(os.path.abspath(__file__))
base_out_path = os.path.join(base_out_path, "out")

workflow_path = os.path.dirname(os.path.abspath(__file__))
workflow_path = os.path.join(workflow_path, "workflow")

file_name = "Amadeus in the Future"
# ----------------------------------------------------------------

print("\n" + "=" * 50)
print("🎬 2단계: LTX-Video 2.3 영상 생성 자동화를 시작합니다...")

# 1. LTX-Video 구조도 및 기획서 읽기
video_flow_path = os.path.join(workflow_path, "video_ltx2_3_i2v.json")
with open(video_flow_path, "r", encoding="utf-8-sig") as f:
    video_workflow = json.load(f)

# (이전 단계에서 생성한 mv_conti_plan.json 파일을 읽어왔다고 가정합니다)
conti_path = os.path.join(base_out_path, f"{file_name}_conti_plan.json")
with open(conti_path, "r", encoding="utf-8") as f:
    plan_data = json.load(f)

fps = video_workflow["267:260"]["inputs"]["value"]

theme_setup = plan_data["theme_setup"]
main_character = theme_setup["main_character_design"]
color_palette = theme_setup["color_palette_and_mood"]

# cinematography_style 없으면 빈 문자열로 대체
style = theme_setup["cinematography_style"] if theme_setup.get("cinematography_style") else ""

for scene in plan_data["timeline"]:
    start_sec = scene["start_sec"]
    end_sec = scene["end_sec"]

    prompt_text = f"{style},{color_palette},"
    prompt_text += scene["visual_concept"]

    # 메인캐릭터 정보와 컬러 팔레트 정보를 프롬프트에 반영합니다.
    prompt_text = f",{main_character},"
    prompt_text += scene["positive_prompt"]

    negative_prompt = scene["negative_prompt"]
    scene_prompt = scene["scene_prompt"]

    duration = end_sec - start_sec
    total_frames = int((duration * fps) + 1)

    print(f"\n🎬 Scene: {start_sec}s ~ {end_sec}s ({duration}s)")
    print(f"📊 Calculating frames: {duration}s * {fps}fps + 1 = {total_frames} frames")

    # 2. 파일 이동 로직 (Output -> Input)
    source_image_filename = f"scene_{start_sec:03d}sec_00001_.png"
    if not os.path.exists(os.path.join(OUTPUT_DIR, source_image_filename)):
        print(f"⚠️ Skip: {source_image_filename} not found.")
        continue
    shutil.copy2(os.path.join(OUTPUT_DIR, source_image_filename), os.path.join(INPUT_DIR, source_image_filename))

    # 3. 데이터 주입 (Dynamic Injection)
    video_workflow["269"]["inputs"]["image"] = source_image_filename
    video_workflow["267:266"]["inputs"]["value"] = scene_prompt

    # [핵심] 계산된 프레임 수를 'Length' 노드("267:225")에 주입
    video_workflow["267:225"]["inputs"]["value"] = total_frames

    # Seed 랜덤화
    seed = random.randint(1, 999999999999999)
    video_workflow["267:216"]["inputs"]["noise_seed"] = seed
    video_workflow["267:237"]["inputs"]["noise_seed"] = seed

    # 3. LTX-Video 구조도 데이터 교체
    video_workflow["75"]["inputs"]["filename_prefix"] = f"scene_video_{start_sec:03d}"

    # 4. 서버로 영상 생성 요청
    request_data = json.dumps({"prompt": video_workflow}).encode('utf-8')
    request = urllib.request.Request(SERVER_URL, data=request_data)

    try:
        response = urllib.request.urlopen(request)
        response_data = json.loads(response.read())
        prompt_id = response_data["prompt_id"]

        print(f"▶ {start_sec}초 구간 동영상 렌더링 요청... (작업 ID: {prompt_id})")

        # 5. 영상 생성 완료 대기 (상태 폴링)
        while True:
            history_request = urllib.request.Request(HISTORY_URL)
            history_response = urllib.request.urlopen(history_request)
            history_data = json.loads(history_response.read())

            if prompt_id in history_data:
                print(f"✅ {start_sec}초 구간 동영상(.mp4) 생성 완료!")
                break

            time.sleep(5)

    except Exception as error:
        print(f"❌ {start_sec}초 영상 생성 중 오류 발생: {error}")

print("🎉 모든 영상 조각 생성이 끝났습니다! 이제 마지막 FFmpeg 조립만 남았습니다.")