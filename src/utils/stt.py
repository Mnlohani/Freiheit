import io
import os
import ctranslate2
from faster_whisper import WhisperModel

# Determining device: use CUDA (GPU) if available, otherwise fall back to CPU.
_device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

# Load the Whisper model globally at startup. ensures the model stays in memory
_model = WhisperModel(
    "small",  # Model size variant (e.g., tiny, base, small, medium, large)
    device=_device,
    compute_type="float16" if _device == "cuda" else "int8",
    # Optimize CPU threading: use half of available CPU cores, with a minimum of 1.
    cpu_threads=max(1, (os.cpu_count() or 8) // 2),
    num_workers=1,
)


def transcribe_STT(audio_bytes: bytes) -> tuple[str, str]:
    """Process the raw audio bytes and transcribe them using the pre-loaded model.
    Handles webm and mp4 (iPhone) audio.
    Save input audio bytes in memory to transcribe with Whisper

    Parameters
    -----------
    audio_bytes : bytes
        Raw audio bytes

    Returns
    -------
    transcribed_text : str
        The transcribed text from the audio.
    language_code : str
        Detected 2-letter language code e.g. 'en', 'fi'.
    language_probability : float
        Confidence score of the detected language (0.0 to 1.0).
    """

    segments, info = _model.transcribe(
        io.BytesIO(audio_bytes),
        beam_size=1,
        temperature=0.0,
        vad_filter=True,  # Filter out background noise and silence
        without_timestamps=True,  # Skip generating timestamp markers
        condition_on_previous_text=False,
    )

    # Combine all transcribed text segments into a single clean string.
    transcribed_text = " ".join(s.text.strip() for s in segments).strip()

    # Return the final transcription and the detected language code.
    return transcribed_text, info.language
