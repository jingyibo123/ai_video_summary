import sys
import os
from openai import OpenAI

# Redirect stdout to both console and a log file
class Logger(object):
    def __init__(self, filename="scratch/test_asr_output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

api_base = "http://localhost:8089/v1"
api_key = "none"
model = "Qwen/Qwen3-ASR-1.7B"

audio_path = r"C:\Users\yibo\Documents\workspace\ai_video_summary\samples\ai_summary\audio.mp3"

if not os.path.exists(audio_path):
    print(f"Error: audio not found at {audio_path}")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=api_base)

# Test with response_format="json"
print("Testing response_format='json'...")
try:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f, prompt="DMA", response_format="json")
    print("SUCCESS with json format! Response:", resp)
except Exception as e:
    print("FAILED with json format:", e)

# Test with default format (which is JSON)
print("Testing default response format...")
try:
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f, prompt="DMA")
    print("SUCCESS with default format! Response:", resp)
except Exception as e:
    print("FAILED with default format:", e)
