import os
import asyncio
from typing import List, Dict, Any

from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from .vad_interface import VADInterface
from src.audio_utils import save_audio_to_file


class PyannoteVAD(VADInterface):
    """
    Voice Activity Detection pipeline backed by PyAnnote audio segmentation.
    """

    def __init__(self, **kwargs) -> None:
        model_name = kwargs.get("model_name", "pyannote/segmentation")
        auth_token = kwargs.get(
            "auth_token",
            os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        )

        pyannote_args = kwargs.get(
            "pyannote_args",
            {
                "onset": 0.684,
                "offset": 0.577,
                "min_duration_on": 0.181,
                "min_duration_off": 0.037,
            },
        )
        self.model = Model.from_pretrained(model_name, use_auth_token=auth_token)
        self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.vad_pipeline.instantiate(pyannote_args)

    async def detect_activity(self, client) -> List[Dict[str, float]]:
        """
        Runs voice activity detection on the client's current scratch buffer.
        """
        audio_file_path = save_audio_to_file(
            audio_data=bytes(client.scratch_buffer),
            file_name=client.get_file_name(),
            sampling_rate=client.sampling_rate,
            samples_width=client.samples_width,
        )

        try:
            # Run blocking VAD model inference in worker thread
            vad_results = await asyncio.to_thread(self.vad_pipeline, audio_file_path)
            vad_segments = []
            if vad_results is not None:
                vad_segments = [
                    {"start": segment.start, "end": segment.end, "confidence": 1.0}
                    for segment in vad_results.itersegments()
                ]
            return vad_segments
        finally:
            if os.path.exists(audio_file_path):
                try:
                    os.remove(audio_file_path)
                except OSError:
                    pass
