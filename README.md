Speech2Text2Speech (S2T2S)
Speech2Text2Speech is a Python-based voice pipeline for macOS (Apple Silicon) that:

Listens to your microphone in real time.
Transcribes your speech to text using OpenAI Whisper.
Processes/Generates AI responses with Ollama using the smollm model.
Speaks the generated output back to you using Coqui TTS.

It’s designed as a quick-start local assistant without reliance on external cloud APIs (other than model downloads).

Platform Support:
This project is currently tested only on macOS with Apple Silicon (M4).
Other platforms may work but are not supported in the current install process.

Requirements:
macOS 13+ (Ventura or later recommended)
Apple Silicon chip (M4)
Homebrew installed
Miniconda or Anaconda installed
Python 3.10.x (critical — newer Python versions break dependencies due to TTS and numpy pinning)

🔧 Installation:
Clone this repository:
git clone https://github.com/yourusername/Speech2Text2Speech.git
cd Speech2Text2Speech
chmod +x install.sh
./install.sh

The installer will:
Create a Conda environment (s2t2s by default, change in script as needed).
Install pinned Python dependencies to avoid numpy / TTS version conflicts.
Install FFmpeg via Homebrew.
Install and attempt to start Ollama.
Pull the smollm model for Ollama.
Pre-download the Coqui TTS model tts_models/en/ljspeech/tacotron2-DDC.

⚠️ Known Issues & Notes
Ollama Service Start Failure
On some systems, brew services start ollama fails with:
Bootstrap failed: 5: Input/output error
The install script will log the error and continue. You can try starting it manually:

sudo launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/homebrew.mxcl.ollama.plist

Verbose Output: By default, Coqui TTS and Whisper produce a lot of setup logs. I am still working to reduce this noise in the console.

CPU Mode Only: Whisper is forced to run on CPU (device="cpu") with FP32 for compatibility.

▶️ Usage
After installation:

conda activate s2t2s
pip install ollama  # In case Ollama Python bindings are needed
python s2t2s.py
Speak after the prompt:
🎙️ Speak now (0.7s silence to stop)...

📝 Logging
The install.sh script writes a detailed log to install.log in the project folder.
This is useful for debugging failed installs or dependency conflicts.

The basic script executes in sequence, s2t2s executes asynchronously so that it is listening while it is talking. be sure to use headphones or the character will start arguing with itself.  Future versions will enable the user to interrrupt the virtual assistant, have a more robust long term memory and use of character profiles.

📄 License
MIT License — see LICENSE for details.

