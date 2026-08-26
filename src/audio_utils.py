import os
import wave
from typing import Optional


def save_audio_to_file(
    audio_data: bytes,
    file_name: str,
    audio_dir: str = "audio_files",
    sampling_rate: int = 16000,
    samples_width: int = 2,
    channels: int = 1,
) -> str:
    """
    Saves raw PCM audio data to a standard WAV file.

    :param audio_data: Raw PCM audio bytes.
    :param file_name: Output filename.
    :param audio_dir: Directory where audio files will be saved.
    :param sampling_rate: Audio sample rate in Hz (default: 16000).
    :param samples_width: Sample width in bytes (2 for 16-bit PCM).
    :param channels: Number of audio channels (1 for mono).
    :return: Full path to the created WAV file.
    """
    os.makedirs(audio_dir, exist_ok=True)
    file_path = os.path.join(audio_dir, file_name)

    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(samples_width)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_data)

    return file_path
