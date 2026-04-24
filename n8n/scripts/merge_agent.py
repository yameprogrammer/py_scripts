import json
import subprocess
import os

# --- [Configuration] ---
# Ensure these paths match your local environment
file_name = "고유 시간의 선율"

base_out_path = os.path.dirname(os.path.abspath(__file__))
base_out_path = os.path.join(base_out_path, "out")

workflow_path = os.path.dirname(os.path.abspath(__file__))
workflow_path = os.path.join(workflow_path, "workflow")

COMFY_OUTPUT_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output"
AUDIO_FILE_PATH = f"{file_name}.mp3"
PLAN_JSON_PATH = f"{file_name}_conti_plan.json"
FINAL_VIDEO_NAME = f"{file_name}_Music_Video_Master.mp4"
CONCAT_LIST_PATH = "video_join_list.txt"

# FFmpeg encoding settings for high quality
VIDEO_CODEC = "libx264"
CRF_VALUE = "18"  # Higher quality (18-23 is standard)
PIXEL_FORMAT = "yuv420p"  # Ensures compatibility with most players


# -----------------------

def assemble_music_video():
    print(f"\n" + "=" * 50)
    print("🚀 Starting Final Video Assembly...")

    # 1. Load the planning data
    if not os.path.exists(PLAN_JSON_PATH):
        print(f"❌ Error: {PLAN_JSON_PATH} not found.")
        return

    with open(PLAN_JSON_PATH, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    # 2. Generate the FFmpeg Concat List
    # We use the 'duration' logic to ensure exact sync with the audio timeline
    try:
        with open(CONCAT_LIST_PATH, "w", encoding="utf-8") as f:
            for scene in plan_data["timeline"]:
                start_sec = scene["start_sec"]
                # Match the filename pattern generated in the previous LTX-Video step
                # Expected: scene_video_000_00001.mp4, scene_video_005_00001.mp4...
                video_filename = f"scene_video_{start_sec:03d}_00001.mp4"
                video_full_path = os.path.join(COMFY_OUTPUT_DIR, video_filename)

                if os.path.exists(video_full_path):
                    # FFmpeg concat format requires forward slashes or escaped backslashes
                    normalized_path = video_full_path.replace("\\", "/")
                    f.write(f"file '{normalized_path}'\n")
                else:
                    print(f"⚠️ Warning: Missing clip {video_filename}. Timeline may shift!")

        print(f"📝 Concat list generated at: {CONCAT_LIST_PATH}")

    except Exception as e:
        print(f"❌ Error during concat list generation: {e}")
        return

    # 3. Execute FFmpeg Command
    # -f concat: Joins the files in the list
    # -i AUDIO: Adds the master audio track
    # -shortest: Ends the video when either the audio or video stream ends
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", CONCAT_LIST_PATH,
        "-i", AUDIO_FILE_PATH,
        "-c:v", VIDEO_CODEC,
        "-crf", CRF_VALUE,
        "-pix_fmt", PIXEL_FORMAT,
        "-c:a", "aac",
        "-b:a", "320k",
        "-map", "0:v:0",  # Take video from the first input (concat)
        "-map", "1:a:0",  # Take audio from the second input (mp3)
        "-shortest",
        FINAL_VIDEO_NAME
    ]

    print("🛠️  Encoding Final Master... This might take a minute.")

    try:
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        print(f"✅ Success! Your music video is ready: {FINAL_VIDEO_NAME}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr}")
    finally:
        # Clean up temporary concat file if needed
        # os.remove(CONCAT_LIST_PATH)
        pass


if __name__ == "__main__":
    assemble_music_video()