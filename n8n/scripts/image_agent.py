import json
import os
import urllib.request
import random  # 시드값 랜덤 생성을 위해 추가
import time # 작업 대기를 위해 추가

# 1. 만들어둔 기획서와 콤피유아이 구조도 파일을 읽어옵니다.
base_out_path = os.path.dirname(os.path.abspath(__file__))
base_out_path = os.path.join(base_out_path, "out")

workflow_path = os.path.dirname(os.path.abspath(__file__))
workflow_path = os.path.join(workflow_path, "workflow")

song_name = "고유 시간의 선율"
conti_path = os.path.join(base_out_path, f"{song_name}_conti_plan.json")

with open(conti_path, "r", encoding="utf-8") as f:
    conti = json.load(f)

# 이미지 생성 워크플로우 목록  image_flux2_text_to_image_9b.json, sd3.5_simple_example.json, 90's_anime_workflow.json, realcartoonpony.json
# 선택적으로 진행 할 수 있도록 분기 제작
WORKFLOW_CONFIGS = {
    "sd3.5": {
        "file": "sd3.5_simple_example.json",
        "positive_node": "16",
        "positive_key": "text",
        "negative_node": "40",
        "negative_key": "text",
        "save_node": "9",
        "save_key": "filename_prefix",
        "seed_node": "3",
        "seed_key": "seed",
    },
    "flux2": {
        "file": "image_flux2_text_to_image_9b.json",
        "positive_node": "76",
        "positive_key": "value",
        "negative_node": "75:67",
        "negative_key": "text",
        "save_node": "9",
        "save_key": "filename_prefix",
        "seed_node": "75:73",
        "seed_key": "noise_seed",
    },
    "90s_anime": {
        "file": "90's_anime_workflow.json",
        "positive_node": "1",
        "positive_key": "positive",
        "negative_node": "1",
        "negative_key": "negative",
        "save_node": "14",
        "save_key": "filename_prefix",
        "seed_node": "2",
        "seed_key": "seed",
    },
    "realcartoonpony": {
        "file": "realcartoonpony.json",
        "positive_node": "1",
        "positive_key": "positive",
        "negative_node": "1",
        "negative_key": "negative",
        "save_node": "9",
        "save_key": "filename_prefix",
        "seed_node": "2",
        "seed_key": "seed",
    },
}

# 기본값은 sd3.5이며, 환경변수 IMAGE_WORKFLOW=flux2 로 전환 가능합니다.
selected_workflow = os.getenv("IMAGE_WORKFLOW", "sd3.5").strip().lower()
if selected_workflow not in WORKFLOW_CONFIGS:
    available = ", ".join(WORKFLOW_CONFIGS.keys())
    raise ValueError(f"지원하지 않는 워크플로우 선택값입니다: {selected_workflow}. 사용 가능: {available}")

workflow_config = WORKFLOW_CONFIGS[selected_workflow]
image_workflow_path = os.path.join(workflow_path, workflow_config["file"])
with open(image_workflow_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)


def set_workflow_input(workflow_data, node_id, input_key, value):
    try:
        workflow_data[node_id]["inputs"][input_key] = value
    except KeyError as err:
        raise KeyError(
            f"워크플로우 입력 경로를 찾을 수 없습니다: node={node_id}, key={input_key}"
        ) from err

comfyui_api = "http://127.0.0.1:8188/prompt"

print(f"선택된 워크플로우: {selected_workflow}")
print("그림 생성 자동화 요청을 시작합니다...")

theme_setup = conti["theme_setup"]
main_character = theme_setup["main_character_design"]
color_palette = theme_setup["color_palette_and_mood"]
# cinematography_style 없으면 빈 문자열로 대체
style = theme_setup["cinematography_style"] if theme_setup.get("cinematography_style") else ""

# 2. 기획서의 시간대별로 반복 작업을 수행합니다.
for scene in conti["timeline"]:
    start_time = scene["start_sec"]

    # 메인캐릭터 정보와 컬러 팔레트 정보를 프롬프트에 반영합니다.
    prompt = f"{style},{color_palette},"
    prompt += scene["visual_concept"]

    prompt += f",{main_character},"
    prompt += scene["positive_prompt"]

    negative_prompt = scene["negative_prompt"]

    # 3. 워크플로우별 프롬프트 입력 위치에 현재 시간대 지시문을 반영합니다.
    set_workflow_input(workflow, workflow_config["positive_node"], workflow_config["positive_key"], prompt)
    set_workflow_input(workflow, workflow_config["negative_node"], workflow_config["negative_key"], negative_prompt)

    # 4. 그림 파일의 이름이 시간대별로 저장되도록 설정합니다.
    set_workflow_input(
        workflow,
        workflow_config["save_node"],
        workflow_config["save_key"],
        f"scene_{start_time:03d}sec",
    )

    # 시드(Seed)값 랜덤 지정 - 매번 새로운 구도의 그림을 위해
    set_workflow_input(
        workflow,
        workflow_config["seed_node"],
        workflow_config["seed_key"],
        random.randint(1, 999999999999999),
    )

    # 5. 콤피유아이 서버로 작업을 지시합니다.
    request_data = json.dumps({"prompt": workflow}).encode('utf-8')
    request = urllib.request.Request(comfyui_api, data=request_data)

    try:
        response = urllib.request.urlopen(request)
        response_data = json.loads(response.read())
        job_id = response_data["prompt_id"]

        print(f"▶ {start_time}초 구간 생성을 요청했습니다. (작업 ID: {job_id})")
        print("   그림을 그리는 중입니다... 기다려 주세요.")

        while True:
            # 콤피유아이의 작업 기록(History)을 열람합니다.
            write_success_request = urllib.request.Request("http://127.0.0.1:8188/history")
            response = urllib.request.urlopen(write_success_request)
            response_data = json.loads(response.read())

            # 우리의 작업 ID가 기록에 올라왔다면 그림 생성이 끝난 것입니다.
            if job_id in response_data:
                print(f"✅ {start_time}초 구간 그림 생성 완료!")
                break  # 다음 시간대 작업으로 넘어감

            # 묻지마 폭격을 막기 위해 2초간 멈췄다가 다시 확인합니다.
            time.sleep(2)
    except Exception as err:
        print(f"❌ 요청 중 오류 발생: {err}")
