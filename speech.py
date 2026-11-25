"""Optional speech adapters.

The main application remains usable without these dependencies. Install
requirements-voice.txt to enable local speech transcription and TTS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """Transcribe an audio file using Faster-Whisper, loaded lazily."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Speech recognition is optional. Install requirements-voice.txt."
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


async def synthesize_speech(text: str, voice: str = "en-IN-NeerjaNeural") -> str:
    """Create an MP3 response using Edge TTS; returns the generated file path."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError(
            "Text-to-speech is optional. Install requirements-voice.txt."
        ) from exc

    output = Path(tempfile.gettempdir()) / "aahvani_response.mp3"
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(str(output))
    return str(output)
