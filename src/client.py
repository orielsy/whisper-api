from typing import Dict, Any


class Client:
    """
    Maintains real-time audio buffer and session state for a connected client.

    Attributes:
        client_id (str | int): Identifier for the connected client/session.
        sampling_rate (int): Audio sampling rate in Hz (default: 16000 for Whisper/PyAnnote).
        samples_width (int): Bytes per sample (default: 2 for 16-bit PCM).
        buffer (bytearray): Active streaming audio buffer.
        scratch_buffer (bytearray): Chunk buffer currently being processed.
        config (dict): Session transcription and chunking parameters.
    """

    def __init__(
        self,
        client_id: Any = "default",
        sampling_rate: int = 16000,
        samples_width: int = 2,
    ) -> None:
        self.client_id = client_id
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        self.config: Dict[str, Any] = {
            "language": "en",
            "chunk_length_seconds": 3,
            "chunk_offset_seconds": 1,
        }
        self.file_counter = 0
        self.total_samples = 0

    def update_config(self, config_data: Dict[str, Any]) -> None:
        self.config.update(config_data)

    def append_audio_data(self, audio_data: bytes) -> None:
        self.buffer.extend(audio_data)
        self.total_samples += len(audio_data) // self.samples_width

    def clear_buffer(self) -> None:
        self.buffer.clear()
        self.scratch_buffer.clear()

    def increment_file_counter(self) -> int:
        self.file_counter += 1
        return self.file_counter

    def get_file_name(self) -> str:
        return f"{self.client_id}_{self.file_counter}.wav"
