import os
import json
import soundfile as sf
import sounddevice as sd
import numpy as np
import webrtcvad
import whisper
import ollama
import tempfile
import subprocess
import logging

# Remove all existing handlers (to suppress console output)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,             # capture INFO and above
    filename="app_log.txt",         # log file path
    filemode="w",                   # overwrite on each run
    format="%(asctime)s %(levelname)s: %(message)s"
)

logging.info("Logging initialized and redirected to app_log.txt")

from TTS.api import TTS

# ----------------------------
# CONFIG
# ----------------------------
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.7  # seconds
FRAME_DURATION_MS = 30   # Allowed: 10, 20, 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # samples per frame
CONVO_FILE = "conversation.json"
SYSTEM_PROMPT = "You are a helpful assistant."
VAD = webrtcvad.Vad(2)  # 0=aggressive, 3=most sensitive

# ----------------------------
# Conversation Storage
# ----------------------------
def load_conversation():
    if os.path.exists(CONVO_FILE):
        try:
            with open(CONVO_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except json.JSONDecodeError:
            print("⚠️ Conversation file corrupted. Starting new conversation.")
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def save_conversation(conv):
    with open(CONVO_FILE, "w") as f:
        json.dump(conv, f)

# ----------------------------
# Audio Recording with VAD
# ----------------------------
def record_until_silence():
    print("🎙️ Speak now (0.7s silence to stop)...")
    recording = []
    silence_time = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SIZE) as stream:
        while True:
            block, _ = stream.read(FRAME_SIZE)
            if len(block) == 0:
                continue
            try:
                is_speech = VAD.is_speech(block.tobytes(), SAMPLE_RATE)
            except Exception:
                continue  # skip bad frames
            recording.append(block)
            if not is_speech:
                silence_time += FRAME_DURATION_MS / 1000
                if silence_time > SILENCE_THRESHOLD:
                    break
            else:
                silence_time = 0

    return np.concatenate(recording, axis=0)

# ----------------------------
# Transcription
# ----------------------------
whisper_model = whisper.load_model("base", device="cpu")
def transcribe(audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_path = tmp.name
        sf.write(temp_path, audio_data, SAMPLE_RATE)  # <-- fixed here
    try:
        result = whisper_model.transcribe(temp_path, fp16=False)
        return result["text"]
    finally:
        os.remove(temp_path)

# ----------------------------
# LLM Query (Ollama)
# ----------------------------
def query_llm(conv):
    response = ollama.chat(model="smollm", messages=conv)
    return response["message"]["content"]

# ----------------------------
# Text-to-Speech
# ----------------------------
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_path = tmp.name
        tts.tts_to_file(text=text, file_path=tts_path)
        subprocess.run(["afplay", tts_path])
        os.remove(tts_path)

# ----------------------------
# MAIN LOOP
# ----------------------------
if __name__ == "__main__":
    print("✅ Ollama is installed, running, and ready.")
    conversation = load_conversation()

    try:
        while True:
            audio = record_until_silence()
            user_text = transcribe(audio)
            print(f"👤 You: {user_text}")
            conversation.append({"role": "user", "content": user_text})

            ai_text = query_llm(conversation)
            print(f"🤖 AI: {ai_text}")
            conversation.append({"role": "assistant", "content": ai_text})

            save_conversation(conversation)
            speak(ai_text)

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
