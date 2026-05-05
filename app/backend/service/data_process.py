import shutil
import subprocess
import tempfile
from pathlib import Path

import torchaudio
import torch
import librosa
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


def preprocess_video(video_path):
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "ffmpeg is required to convert video to mp3. Install ffmpeg first."
        )

    mp3_tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_tmp:
            mp3_tmp_path = Path(mp3_tmp.name)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_file),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to convert video to mp3: {result.stderr.strip()}"
            )

        return preprocess_audio(str(mp3_tmp_path))
    finally:
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()


def preprocess_audio(audio_path):
    waveform, sr = librosa.load(audio_path, sr=None)  # keep original sample rate
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.transpose(0, 1)
    sample_rate = _SAMPLE_RATE
    expected_samples = _EXPECTED_SAMPLES

    # convert về mono nếu nhiều channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)  # [T]

    # resample nếu cần
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    # Model receives raw waveform and performs mel extraction inside forward().
    waveform = waveform.to(torch.float32)
    if waveform.numel() < expected_samples:
        pad = expected_samples - waveform.numel()
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    elif waveform.numel() > expected_samples:
        waveform = waveform[: expected_samples]

    waveform = waveform.unsqueeze(0)
    return waveform



def preprocess(path):
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return preprocess_video(str(file_path))
    if suffix in AUDIO_EXTENSIONS:
        return preprocess_audio(str(file_path))

    raise ValueError(
        f"Unsupported file type '{suffix or '<none>'}'. "
        f"Supported video: {sorted(VIDEO_EXTENSIONS)}; "
        f"supported audio: {sorted(AUDIO_EXTENSIONS)}"
    )
