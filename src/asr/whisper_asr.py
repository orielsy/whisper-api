from transformers import pipeline
from .asr_interface import ASRInterface
from src.audio_utils import save_audio_to_file
import os


class WhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        # model_name = kwargs.get("model_name", "openai/whisper-large-v2")
        model_name = kwargs.get("model_name", "openai/whisper-medium")
        self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)

    async def transcribe(self, client):
        file_path = await save_audio_to_file(
            client.scratch_buffer, client.get_file_name()
        )

        if os.name == "nt":
            os_file_path = file_path.replace("\\", "/")

        """
        cmd = (
            'whisper "{0}" --model medium --task transcribe --language Spanish'.format(
                file_path
            )
        )
        """

        cmd = 'docker run --gpus all -it --rm -v ".:/app" -v whisper_cache:/.cache whisperx:no_model -- --model medium  --language es  --output_dir audio_files/transcriptions --output_format json "{0}"'.format(
            os_file_path
        )
        # os.system(cmd)

        if client.config["language"] is not None:
            to_return = self.asr_pipeline(
                file_path, generate_kwargs={"language": client.config["language"]}
            )["text"]
        else:
            to_return = self.asr_pipeline(file_path)["text"]

        os.remove(file_path)
        return to_return
