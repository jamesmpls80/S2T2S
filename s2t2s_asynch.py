#!/usr/bin/env python3
import os
import time
import json
import queue
import wave
import tempfile
import subprocess
import threading
import logging
import warnings
import requests
import sounddevice as sd
import webrtcvad
import whisper
from TTS.api import TTS
from pydub import AudioSegment
from pydub.playback import play
import multiprocessing as mp
import sys
import traceback

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
BLOCKSIZE = 160
VAD_MODE = 2
MODEL_NAME = "hf.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF:Q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/chat"
SPEAKER_WAV = "ariana.wav"

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

#===============================
# LOGGING
#===============================

logging.basicConfig(
    level=logging.INFO,               # Adjust level if needed (DEBUG, INFO, WARNING, ERROR)
    filename="app_log.txt",           # Log file name
    filemode="w",                    # Overwrite the log file each run ("a" to append)
    format="%(asctime)s %(levelname)s: %(message)s"
)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

file_handler = logging.FileHandler("app_log.txt")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)
logging.getLogger("tts").setLevel(logging.WARNING)

def redirect_stdout_to_file(filename="tts_output.log"):
    sys.stdout = open(filename, 'w')
    sys.stderr = sys.stdout

def restore_stdout():
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Usage example
redirect_stdout_to_file()
# Load your TTS model here (prints will go to file)
# tts_model = ...
restore_stdout()

# Example to test logging
logging.info("Logging initialized and redirected to app_log.txt")

# When you load your models, their logs will now go to app_log.txt
# ==============================
# Helper: start ollama if desired
# ==============================
def maybe_start_ollama():
    try:
        print("🚀 Starting Ollama server (if not already running)...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
    except Exception as e:
        print(f"⚠️ Could not start ollama automatically: {e}")

# ==============================
# TTS process function
# Runs in a separate process to isolate crashes (SIGILL etc.)
# ==============================
def tts_process_main(tts_queue: mp.Queue, ctrl_queue: mp.Queue, speaker_wav: str):
    """
    This runs in a child process. It creates its own TTS model instance (so C/C++ libs load here).
    It blocks reading from the tts_queue. For each text chunk it synthesizes to a temp wav and plays it.
    """
    try:
        print("[TTS proc] Starting TTS process, loading model...")
        # Instantiate TTS in child process to avoid parent being killed by native faults
        xtts_local = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        print("[TTS proc] TTS model loaded.")
    except Exception as e:
        print("[TTS proc] Failed to load TTS model:", e)
        traceback.print_exc()
        # signal parent we failed to initialize
        ctrl_queue.put({"status": "init_failed", "error": str(e)})
        return

    ctrl_queue.put({"status": "ready"})

    try:
        while True:
            try:
                item = tts_queue.get()  # blocking
            except (EOFError, KeyboardInterrupt):
                break
            if item is None:
                # shutdown sentinel
                print("[TTS proc] Shutdown sentinel received.")
                break

            text_chunk = item.get("text", "")
            if not text_chunk:
                continue

            try:
                # Create a temporary wav file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp_path = tmp.name
                # Synthesize to file (wrapping in try so exceptions are caught)
                xtts_local.tts_to_file(text=text_chunk, speaker_wav=speaker_wav, file_path=tmp_path)
                # Play the file
                audio = AudioSegment.from_wav(tmp_path)
                play(audio)
                os.remove(tmp_path)
                ctrl_queue.put({"status": "played", "len": len(text_chunk)})
            except Exception as synth_err:
                print("[TTS proc] Error during synth/play:", synth_err)
                traceback.print_exc()
                # Report back the error to parent so it can decide to restart
                ctrl_queue.put({"status": "error", "error": str(synth_err)})
                # continue to try to process more requests; if native fault occurs,
                # process will likely crash (SIGILL) and parent will restart it.
    except Exception as e:
        print("[TTS proc] Fatal error:", e)
        traceback.print_exc()
    finally:
        print("[TTS proc] Exiting TTS process.")

# ==============================
# Parent process: monitor and restart TTS process
# ==============================
def start_tts_process(tts_mp_queue, ctrl_mp_queue, speaker_wav):
    proc = mp.Process(target=tts_process_main, args=(tts_mp_queue, ctrl_mp_queue, speaker_wav), daemon=True)
    proc.start()
    return proc

# ==============================
# Main script threads (record/transcribe/ollama streaming)
# ==============================
def record_audio_loop(record_queue):
    vad = webrtcvad.Vad(VAD_MODE)
    while True:
        q = queue.Queue()
        frames = []

        def callback(indata, *_):
            q.put(bytes(indata))

        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCKSIZE,
                               dtype='int16', channels=1, callback=callback):
    #--uncomment to debug print("🎙️ Speak now (0.7s silence to stop)...")
            silence = 0
            max_silence = int(0.9 / 0.01)
            while True:
                frame = q.get()
                frames.append(frame)
                if vad.is_speech(frame, SAMPLE_RATE):
                    silence = 0
                else:
                    silence += 1
                    if silence > max_silence:
                        break

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            with wave.open(tmp_wav.name, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SAMPLE_RATE)
                f.writeframes(b''.join(frames))
            record_queue.put(tmp_wav.name)

def transcriber_loop(record_queue, text_queue, whisper_model):
    while True:
        wav_path = record_queue.get()
        try:
            result = whisper_model.transcribe(wav_path, fp16=False)
            text = result['text'].strip()
            if text:
                print(f"📜 Transcript: {text}")
                text_queue.put(text)
        except Exception as e:
            print("❌ Transcription error:", e)
        finally:
            try:
                os.remove(wav_path)
            except Exception:
                pass

def ollama_stream_loop(text_queue, response_queue):
    while True:
        prompt = text_queue.get()
        try:
            conversation = [
                {"role": "system", "content": "You are Ariana, respond concisely."},
                {"role": "user", "content": prompt}
            ]
            with requests.post(
                OLLAMA_URL,
                json={"model": MODEL_NAME, "messages": conversation, "stream": True, "options": {"num_ctx": 4096}
                      },
                stream=True,
                timeout=(10, None)  # connect timeout, read timeout=None (wait indefinitely)
            ) as r:
                r.raise_for_status()
                print("🧠 LLM generating...")
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        if "message" in obj and "content" in obj["message"]:
                            chunk = obj["message"]["content"]
                            if chunk.strip():
                                # print streaming content without newline
                                print(chunk, end="", flush=True)
                                response_queue.put(chunk)
                        if obj.get("done"):
                            print()
                            break
                    except json.JSONDecodeError:
                        # ignore non-json lines
                        continue
        except Exception as e:
            print("❌ LLM error:", e)

# ==============================
# Buffering & TTS-sender thread
# - Collects streaming chunks and sends sentence-like chunks to TTS process queue
# - Uses punctuation or max_chars as flush triggers
# ==============================
def buffering_sender(response_queue, tts_mp_queue):
    buffer = ""
    max_chars = 240  # flush if chunk grows this large
    flush_timeout = 1.0  # seconds to wait for more tokens before flushing if something is buffered

    last_receive_time = None

    while True:
        try:
            chunk = response_queue.get(timeout=flush_timeout)
            now = time.time()
            last_receive_time = now
            buffer += chunk

            # If buffer contains sentence-ending punctuation, flush up to last punctuation
            last_punct_idx = max((buffer.rfind(p) for p in (".", "!", "?")))
            if last_punct_idx != -1 and last_punct_idx >= 0:
                to_send = buffer[:last_punct_idx + 1].strip()
                remainder = buffer[last_punct_idx + 1:].lstrip()
                if to_send:
                    tts_mp_queue.put({"text": to_send})
                buffer = remainder
                continue

            # If buffer too large, flush it (to avoid extremely long waits)
            if len(buffer) >= max_chars:
                tts_mp_queue.put({"text": buffer.strip()})
                buffer = ""
                continue

            # otherwise keep collecting; loop will try to get again and wait up to flush_timeout
        except queue.Empty:
            # timeout: if we have buffered text, flush it so it eventually speaks
            if buffer.strip():
                tts_mp_queue.put({"text": buffer.strip()})
                buffer = ""
            continue
        except Exception as e:
            print("❌ buffering_sender error:", e)
            traceback.print_exc()

# ==============================
# Main runner
# ==============================
def main():
    maybe_start_ollama()

    print("📦 Loading Whisper model (main process)...")
    whisper_model = whisper.load_model("base")
    print("📦 Whisper loaded.")

    # Note: we do NOT load TTS in this process to avoid native crash risk; child will load it.
    print("⏳ Preloading model via Ollama (warm-up)...")
    try:
        conversation = [{"role":"user","content":"Hello"}]
        r = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "messages": conversation, "stream": False}, timeout=120)
        r.raise_for_status()
        print("✅ Ollama model warm-up OK.")
    except Exception as e:
        print("⚠️ Ollama warmup failed or timed out:", e)

    # Queues for threads (thread-safe)
    record_queue = queue.Queue()
    text_queue = queue.Queue()
    response_queue = queue.Queue()

    # Multiprocessing queues for TTS proc
    mp.set_start_method('spawn', force=True) if sys.platform == "darwin" else None
    tts_mp_queue = mp.Queue()
    ctrl_mp_queue = mp.Queue()

    # Start TTS process and monitor object
    tts_proc = start_tts_process(tts_mp_queue, ctrl_mp_queue, SPEAKER_WAV)

    # Wait for TTS readiness (simple blocking wait with timeout)
    ready = False
    try:
        # wait up to 30s for 'ready' message from tts proc
        t0 = time.time()
        while time.time() - t0 < 30:
            try:
                msg = ctrl_mp_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if isinstance(msg, dict) and msg.get("status") == "ready":
                ready = True
                print("✅ TTS process ready.")
                break
            elif isinstance(msg, dict) and msg.get("status") == "init_failed":
                print("❌ TTS failed to initialize:", msg.get("error"))
                break
    except Exception:
        pass

    if not ready:
        print("⚠️ TTS process did not report ready; continuing but TTS may not work.")

    # Start threads
    t_record = threading.Thread(target=record_audio_loop, args=(record_queue,), daemon=True)
    t_trans = threading.Thread(target=transcriber_loop, args=(record_queue, text_queue, whisper_model), daemon=True)
    t_ollama = threading.Thread(target=ollama_stream_loop, args=(text_queue, response_queue), daemon=True)
    t_buffer = threading.Thread(target=buffering_sender, args=(response_queue, tts_mp_queue), daemon=True)

    t_record.start()
    t_trans.start()
    t_ollama.start()
    t_buffer.start()

    print("🔁 Concurrent Voice Assistant running. Ctrl+C to exit.")

    try:
        while True:
            # monitor tts process health and restart if it died
            if not tts_proc.is_alive():
                exitcode = tts_proc.exitcode
                print(f"⚠️ TTS process died with exitcode {exitcode}. Restarting...")
                # drain ctrl queue
                while not ctrl_mp_queue.empty():
                    try:
                        ctrl_mp_queue.get_nowait()
                    except Exception:
                        break
                tts_proc = start_tts_process(tts_mp_queue, ctrl_mp_queue, SPEAKER_WAV)
                # wait for ready signal briefly
                t0 = time.time()
                ready2 = False
                while time.time() - t0 < 10:
                    try:
                        msg = ctrl_mp_queue.get(timeout=1.0)
                        if isinstance(msg, dict) and msg.get("status") == "ready":
                            ready2 = True
                            print("✅ TTS restarted and ready.")
                            break
                    except queue.Empty:
                        pass
                if not ready2:
                    print("⚠️ Restarted TTS did not report ready.")
            """ check ctrl messages for errors/warnings from the tts proc
            try:
                while not ctrl_mp_queue.empty():
                    msg = ctrl_mp_queue.get_nowait()
                    print("[TTS ctrl]", msg)
            except Exception:
                pass """
            time.sleep(0.5) 
    except KeyboardInterrupt:
        print("\n👋 Shutting down... (sending sentinel to TTS proc)")
        try:
            tts_mp_queue.put(None)  # tell TTS process to shutdown nicely
            tts_proc.join(timeout=5)
        except Exception:
            pass
        print("Bye.")

if __name__ == "__main__":
    main()
