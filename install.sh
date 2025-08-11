#!/bin/bash
set -e

ENV_NAME="s2t2s"

echo "📦 Creating conda environment..."
conda create -y -n "$ENV_NAME" python=3.10

echo "📦 Activating conda environment..."
# Ensure conda commands are available even if 'conda init' was never run
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "📦 Installing Python dependencies (pinned versions to avoid conflicts)..."
pip install \
    numpy==1.22.0 \
    sounddevice==0.4.6 \
    webrtcvad==2.0.10 \
    openai-whisper==20230124 \
    TTS==0.22.0 \
    pydub==0.25.1 \
    aiohttp==3.9.5 \
    requests==2.32.3


echo "📦 Installing FFmpeg (via brew)..."
brew install ffmpeg

echo "📦 Installing Ollama (via brew)..."
brew install ollama

echo "📦 Starting Ollama service..."
brew services start ollama
sleep 5  # Give Ollama a few seconds to start

echo "📦 Pulling Ollama model: smollm..."
ollama pull smollm

echo "📦 Pre-downloading XTTS model..."
python -c "from TTS.api import TTS; TTS(model_name='tts_models/en/ljspeech/tacotron2-DDC')"

echo "✅ Installation complete!"
echo "Run the program with:"
echo "conda activate $ENV_NAME, then 'pip install ollama', and lastly, 'python s2t2s.py"
