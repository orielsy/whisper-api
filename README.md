# Local Whisper Streaming API

A real-time, low-latency speech-to-text WebSocket API built with **FastAPI**, **OpenAI Whisper**, and **PyAnnote Voice Activity Detection (VAD)**.

---

## Features

- **Real-Time WebSocket Streaming (`/listen`)**: Ingests raw 16 kHz 16-bit mono PCM chunks from browser/desktop audio clients and returns streaming transcripts with segment metadata.
- **Non-blocking Model Inference**: Offloads compute-heavy neural inference to worker threads via `asyncio.to_thread` to maintain high WebSocket throughput and low event-loop latency.
- **Voice Activity Detection**: Optional PyAnnote segmentation pipeline to detect speech boundaries and trim silence.
- **Hardware Acceleration**: Automatic CUDA GPU acceleration fallback to CPU.
- **Batch Endpoint (`/whisper`)**: Multi-part file upload endpoint for asynchronous batch audio transcription.
- **Built-in Client**: Includes a browser microphone capture interface for rapid local testing (`http://localhost:8000`).

---

## Installation & Setup

### 1. Requirements
- Python 3.10+
- PyTorch with CUDA (or CPU)
- ffmpeg installed and on PATH

### 2. Python Environment
```bash
pip install -r requirements.txt
pip install "git+https://github.com/openai/whisper.git"
```

### 3. Running the Server
```bash
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Running via Docker
```bash
docker build -t whisper-api .
docker run -p 8000:8000 whisper-api
```

---

## Configuration

Set via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `WHISPER_MODEL` | Whisper model size (`tiny`, `base`, `medium`, `large-v3`) | `base` |
| `SAMPLING_RATE` | Expected input audio sample rate in Hz | `16000` |
| `WHISPER_LANGUAGE` | Language code for transcription (e.g. `en`, `es`) | `en` |
| `HF_TOKEN` | Hugging Face token for PyAnnote models (optional) | None |

---

## License
MIT
