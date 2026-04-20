import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class SonicDataset(Dataset):
    def __init__(self, df, sample_rate=16000, duration_seconds=6.0):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.expected_samples = int(sample_rate * duration_seconds)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        audio_path = row["path"]
        fake_type = row["fake_type"]
        label = float(row["label"])

        try:
            wave, sr = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = torch.from_numpy(wave.T)  # [C, T]
            sr = int(sr)
        except Exception as exc:  # pragma: no cover - data/codec dependent
            raise RuntimeError(
                f"Failed to load audio '{audio_path}' with soundfile: {exc}"
            ) from exc

        # convert về mono nếu nhiều channel
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)  # [T]

        # resample nếu cần
        if sr != self.sample_rate:
            waveform_np = waveform.detach().cpu().numpy()
            waveform_np = resample_poly(
                waveform_np, up=self.sample_rate, down=sr, axis=-1
            )
            waveform = torch.from_numpy(np.asarray(waveform_np, dtype=np.float32))

        # Model receives raw waveform and performs mel extraction inside forward().
        waveform = waveform.to(torch.float32)
        if waveform.numel() < self.expected_samples:
            pad = self.expected_samples - waveform.numel()
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        elif waveform.numel() > self.expected_samples:
            waveform = waveform[: self.expected_samples]

        return {
            "x": waveform,
            "y": torch.tensor(int(label), dtype=torch.long),
            "filename": filename,
            "fake_type": fake_type,
            "path": audio_path,
        }
