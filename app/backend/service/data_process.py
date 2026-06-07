from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import librosa
import torch
import torchaudio

from config.loader import get_preprocess_kwargs

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".flv",
    ".m4v",
    ".wmv",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
    ".wma",
    ".opus",
}

_PREPROCESS_CFG = get_preprocess_kwargs("backend_inference")
_SAMPLE_RATE = int(_PREPROCESS_CFG.get("sample_rate", 16000))
_EXPECTED_SAMPLES = int(_PREPROCESS_CFG.get("expected_samples", 96000))


def _validate_input_path(path: str | Path) -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    return file_path


def _extract_audio_from_video(video_path: Path) -> Path:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg is required to convert video to mp3. Install ffmpeg first."
        )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
        mp3_path = Path(mp3_tmp.name)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        if mp3_path.exists():
            mp3_path.unlink()
        raise RuntimeError(
            f"ffmpeg failed to convert video to mp3: {result.stderr.strip()}"
        )
    return mp3_path


@contextmanager
def prepared_audio_source(path: str | Path) -> Iterator[Path]:
    file_path = _validate_input_path(path)
    suffix = file_path.suffix.lower()

    if suffix in AUDIO_EXTENSIONS:
        yield file_path
        return
    if suffix in VIDEO_EXTENSIONS:
        temp_audio_path = _extract_audio_from_video(file_path)
        try:
            yield temp_audio_path
        finally:
            if temp_audio_path.exists():
                temp_audio_path.unlink()
        return

    raise ValueError(
        f"Unsupported file type '{suffix or '<none>'}'. "
        f"Supported video: {sorted(VIDEO_EXTENSIONS)}; "
        f"supported audio: {sorted(AUDIO_EXTENSIONS)}"
    )


def load_audio_waveform(path: str | Path, sample_rate: int = _SAMPLE_RATE) -> torch.Tensor:
    file_path = _validate_input_path(path)

    try:
        waveform_np, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception:
        with prepared_audio_source(file_path) as audio_path:
            waveform_np, sr = librosa.load(str(audio_path), sr=None, mono=True)

    waveform = torch.as_tensor(waveform_np, dtype=torch.float32)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=sample_rate,
        )
    return waveform.to(torch.float32)


def preprocess_audio(audio_path: str | Path) -> torch.Tensor:
    waveform = load_audio_waveform(audio_path, sample_rate=_SAMPLE_RATE)
    if waveform.numel() < _EXPECTED_SAMPLES:
        pad = _EXPECTED_SAMPLES - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.numel() > _EXPECTED_SAMPLES:
        waveform = waveform[:_EXPECTED_SAMPLES]
    return waveform.unsqueeze(0)


def preprocess(path: str | Path) -> torch.Tensor:
    return preprocess_audio(path)


def build_contiguous_chunks(
    path: str | Path,
    sample_rate: int = _SAMPLE_RATE,
    chunk_samples: int = _EXPECTED_SAMPLES,
) -> tuple[torch.Tensor, list[int], int]:
    waveform = load_audio_waveform(path, sample_rate=sample_rate)
    total_samples = int(waveform.numel())

    if total_samples == 0:
        raise ValueError("Audio waveform is empty after preprocessing")

    chunks: list[torch.Tensor] = []
    starts: list[int] = []

    for start in range(0, total_samples, chunk_samples):
        chunk = waveform[start : start + chunk_samples]
        if chunk.numel() < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.numel()))
        chunks.append(chunk)
        starts.append(start)

    return torch.stack(chunks, dim=0), starts, total_samples
