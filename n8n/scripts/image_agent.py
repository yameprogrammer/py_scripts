import json
import urllib.request
import random  # 시드값 랜덤 생성을 위해 추가
import time # 작업 대기를 위해 추가

# 1. 만들어둔 기획서와 콤피유아이 구조도 파일을 읽어옵니다.
conti_path = r"C:\Users\parkp\workspace\py_scripts\Amadeus in the Future_conti_plan.json"
with open(conti_path, "r", encoding="utf-8") as f:
    conti = json.load(f)

workflow_path = r"C:\Users\parkp\workspace\py_scripts\n8n\scripts\sd3.5_simple_example.json"
with open(workflow_path, "r", encoding="utf-8") as f:
    workflow = json.load(f)

comfyui_api = "http://127.0.0.1:8188/prompt"

print("그림 생성 자동화 요청을 시작합니다...")

# 2. 기획서의 시간대별로 반복 작업을 수행합니다.
for scene in conti["timeline"]:
    start_time = scene["start_sec"]
    prompt = scene["positive_prompt"]
    negative_prompt = scene["negative_prompt"]

    # 3. 구조도 내의 지시문 내용을 현재 시간대의 내용으로 교체합니다.
    # 주의: "6"이라는 숫자는 야매플머님의 긍정 지시문 작업 단위 번호로 꼭 바꿔주세요!
    workflow["16"]["inputs"]["text"] = prompt
    # 주의: "7"이라는 숫자는 야매플머님의 부정 지시문 작업 단위 번호로 꼭 바꿔주세요!
    workflow["40"]["inputs"]["text"] = negative_prompt

    # 4. 그림 파일의 이름이 시간대별로 저장되도록 설정합니다.
    # 그림 저장 작업 단위(Save Image)의 번호가 "9"라고 가정합니다.
    workflow["9"]["inputs"]["filename_prefix"] = f"scene_{start_time:03d}sec"

    # 시드(Seed)값 랜덤 지정 (노드 3번) - 매번 새로운 구도의 그림을 위해
    workflow["3"]["inputs"]["seed"] = random.randint(1, 999999999999999)

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
