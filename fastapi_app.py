import asyncio
import json
import logging
import os
from tempfile import NamedTemporaryFile
from typing import List, Optional

import torch
import whisper
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.audio_utils import save_audio_to_file
from src.client import Client
from src.vad.vad_factory import VADFactory

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("whisper_api")

# Hardware & Model Configuration
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "16000"))
DEFAULT_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

logger.info("Initializing Whisper model '%s' on device '%s'...", MODEL_NAME, DEVICE)
model = whisper.load_model(MODEL_NAME, device=DEVICE)
logger.info("Whisper model '%s' loaded successfully.", MODEL_NAME)

app = FastAPI(
    title="Local Whisper Streaming API",
    description="Real-time speech-to-text inference with WebSockets, VAD, and Whisper",
    version="1.0.0",
)

# Allow CORS for local desktop and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="client")


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    sampling_rate: int


class TranscriptionItem(BaseModel):
    filename: str
    transcript: str


class BatchTranscriptionResponse(BaseModel):
    results: List[TranscriptionItem]


class ConnectionManager:
    """Manages active streaming audio WebSocket connections and chunk processing."""

    def __init__(self, sampling_rate: int = DEFAULT_SAMPLING_RATE):
        self.active_connections: list[WebSocket] = []
        self.sampling_rate = sampling_rate
        self.samples_width = 2
        self.vad_pipeline = None

    def get_vad(self):
        if self.vad_pipeline is None:
            try:
                self.vad_pipeline = VADFactory.create_vad_pipeline()
            except Exception as e:
                logger.warning("VAD pipeline unavailable (%s); falling back to direct chunking", e)
        return self.vad_pipeline

    async def connect(self, websocket: WebSocket) -> Client:
        await websocket.accept()
        self.active_connections.append(websocket)
        client = Client(
            client_id=id(websocket),
            sampling_rate=self.sampling_rate,
            samples_width=self.samples_width,
        )
        client.config["language"] = DEFAULT_LANGUAGE
        logger.info("Client %s connected (total active: %d)", client.client_id, len(self.active_connections))
        return client

    def disconnect(self, websocket: WebSocket, client_id: str | int = ""):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("Client %s disconnected (remaining: %d)", client_id, len(self.active_connections))

    async def handle_audio_chunk(self, data: bytes, client: Client, websocket: WebSocket):
        client.append_audio_data(data)

        chunk_length_bytes = (
            int(client.config.get("chunk_length_seconds", 3))
            * client.sampling_rate
            * client.samples_width
        )

        if len(client.buffer) >= chunk_length_bytes:
            client.scratch_buffer.extend(client.buffer)
            client.buffer.clear()
            asyncio.create_task(self.process_buffer(client, websocket))

    async def process_buffer(self, client: Client, websocket: WebSocket):
        if len(client.scratch_buffer) == 0:
            return

        client.increment_file_counter()
        file_path = save_audio_to_file(
            audio_data=bytes(client.scratch_buffer),
            file_name=client.get_file_name(),
            sampling_rate=client.sampling_rate,
            samples_width=client.samples_width,
        )

        try:
            language = client.config.get("language")
            transcribe_kwargs = {}
            if language:
                transcribe_kwargs["language"] = language

            # Run compute-heavy inference in background worker thread to keep event loop free
            transcription = await asyncio.to_thread(
                model.transcribe,
                file_path,
                **transcribe_kwargs,
            )

            text = transcription.get("text", "").strip()
            if text:
                logger.info("[%s] Transcript: %s", client.client_id, text)
                payload = {
                    "text": text,
                    "segments": transcription.get("segments", []),
                    "language": transcription.get("language", language),
                }
                await websocket.send_text(json.dumps(payload))
        except Exception as err:
            logger.error("Error processing transcription: %s", err, exc_info=True)
        finally:
            client.scratch_buffer.clear()
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass


manager = ConnectionManager()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "sampling_rate": DEFAULT_SAMPLING_RATE,
    }


@app.get("/", response_class=HTMLResponse)
async def get_test_client(request: Request):
    if os.path.exists("client/index.html"):
        return templates.TemplateResponse("index.html", {"request": request})
    return HTMLResponse("<h1>Whisper API is running</h1>")


@app.post("/whisper", response_model=BatchTranscriptionResponse)
async def transcribe_files(files: List[UploadFile] = File(...)):
    """Batch transcribe uploaded audio files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    results = []
    for file in files:
        suffix = os.path.splitext(file.filename or "audio.wav")[1]
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            try:
                content = await file.read()
                temp.write(content)
                temp.flush()
                temp_name = temp.name
                temp.close()

                transcription = await asyncio.to_thread(model.transcribe, temp_name)
                results.append(
                    TranscriptionItem(
                        filename=file.filename or "audio",
                        transcript=transcription.get("text", "").strip(),
                    )
                )
            finally:
                if os.path.exists(temp_name):
                    try:
                        os.remove(temp_name)
                    except OSError:
                        pass

    return BatchTranscriptionResponse(results=results)


@app.websocket("/listen")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time streaming audio ingestion endpoint."""
    client = await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            await manager.handle_audio_chunk(data, client, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket, client.client_id)
    except Exception as e:
        logger.warning("WebSocket error on client %s: %s", client.client_id, e)
        manager.disconnect(websocket, client.client_id)
