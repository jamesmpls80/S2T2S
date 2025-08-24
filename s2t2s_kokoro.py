import os, asyncio, queue, threading, collections
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import requests
from kokoro import KPipeline
import faulthandler

# --- Safety knobs ---
faulthandler.enable()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# --- Config ---
IN_SR = 16000
OUT_SR = 24000
BLOCKSIZE = 1024
CHANNELS = 1
SLICE_SEC = 3
ROLLING_SEC = 5
STOP_WORD = "stop"

# --- Queues/flags ---
prompt_q = queue.Queue()  # for main transcription
stop_q = queue.Queue()    # for STOP detection
play_q = queue.Queue()
stop_play = threading.Event()
playing_event = threading.Event()  # True when TTS is playing

# --- Audio callback: duplicate mic data into both queues ---
def in_cb(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    raw_bytes = bytes(indata)
    prompt_q.put(raw_bytes)
    stop_q.put(raw_bytes)

def drain_queue_to_buffer(q, rolling):
    """Drain queue → append to rolling numpy buffer (float32 mono)."""
    chunks = []
    try:
        while True:
            data = q.get_nowait()
            pcm16 = np.frombuffer(data, dtype=np.int16)
            f32 = pcm16.astype(np.float32) / 32768.0
            chunks.append(f32)
    except queue.Empty:
        pass
    if chunks:
        new = np.concatenate(chunks)
        rolling = np.concatenate([rolling, new])
        # keep last N seconds
        max_len = IN_SR * ROLLING_SEC
        if rolling.size > max_len:
            rolling = rolling[-max_len:]
    return rolling

# --- Playback thread ---
def playback_worker():
    with sd.OutputStream(samplerate=OUT_SR, channels=1, dtype="float32") as stream:
        while True:
            buf = play_q.get()
            if buf is None:
                break
            if stop_play.is_set():
                continue
            stream.write(buf.reshape(-1, 1))
    print("Playback thread exiting.")

threading.Thread(target=playback_worker, daemon=True).start()

# --- Models ---
whisper_model = whisper.load_model("base")
tts = KPipeline(lang_code="a")

async def speak(text):
    stop_play.clear()
    playing_event.set()
    audio_chunks = []
    for _, _, audio in tts(text, voice="af_heart", speed=1.2):
        if stop_play.is_set():
            break
        chunk = np.array(audio, dtype=np.float32)
        audio_chunks.append(chunk)
        play_q.put(chunk)
    playing_event.clear()
    if audio_chunks and not stop_play.is_set():
        sf.write("last_tts.wav", np.concatenate(audio_chunks), OUT_SR)

def ollama(prompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "smollm:latest", "prompt": prompt, "stream": False},
            timeout=30
        )
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""

async def stop_listener():
    """Continuously listen for the STOP keyword, even during TTS."""
    local_buf = collections.deque(maxlen=IN_SR * 2)  # 2 sec buffer
    while True:
        await asyncio.sleep(0.5)
        # Drain stop_q into local buffer
        try:
            while True:
                data = stop_q.get_nowait()
                pcm16 = np.frombuffer(data, dtype=np.int16)
                f32 = pcm16.astype(np.float32) / 32768.0
                local_buf.extend(f32)
        except queue.Empty:
            pass

        if len(local_buf) < IN_SR:  # need at least 1 sec
            continue

        audio = np.array(local_buf, dtype=np.float32)
        result = whisper_model.transcribe(audio, fp16=False)
        text = result.get("text", "").strip().lower()
        local_buf.clear()  # ✅ Clear buffer after each pass

        if STOP_WORD in text:
            print("🛑 STOP detected — halting playback.")
            stop_play.set()
            # Flush playback queue
            while not play_q.empty():
                try:
                    play_q.get_nowait()
                except queue.Empty:
                    break

async def main():
    rolling = np.zeros(0, dtype=np.float32)
    with sd.RawInputStream(
        samplerate=IN_SR,
        blocksize=BLOCKSIZE,
        channels=CHANNELS,
        dtype="int16",
        callback=in_cb
    ):
        print("🎤 Listening…")
        asyncio.create_task(stop_listener())  # run STOP listener in background
        while True:
            await asyncio.sleep(SLICE_SEC)
            if playing_event.is_set():
                continue  # don't collect prompts while TTS is playing
            rolling = drain_queue_to_buffer(prompt_q, rolling)
            if rolling.size < IN_SR:  # need at least ~1s
                continue
            audio = rolling.copy()
            result = whisper_model.transcribe(audio, fp16=False)
            text = result.get("text", "").strip()
            rolling = np.zeros(0, dtype=np.float32)  # ✅ Reset after processing

            if not text:
                continue
            print(f"👤 You: {text}")
            reply = ollama(text) or "I heard you."
            print(f"🤖 AI: {reply}")
            await speak(reply)

if __name__ == "__main__":
    asyncio.run(main())
