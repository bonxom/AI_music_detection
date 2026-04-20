import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class SonicDataset(Dataset):
    def __init__(self, df, sample_rate=16000, n_mels=128):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        filename = row["filename"]
        audio_path = row["path"]
        fake_type = row["fake_type"]
        label = float(row["label"])

        waveform, sr = librosa.load(audio_path, sr=None, mono=True)

        if sr != self.sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=self.sample_rate
            )

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=self.n_mels,
            fmin=20,
            fmax=8000,
        )

        mel = librosa.power_to_db(mel, ref=np.max)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        return {
            "x": mel,
            "y": torch.tensor(label, dtype=torch.float32),
            "filename": filename,
            "fake_type": fake_type,
            "path": audio_path,
        }
