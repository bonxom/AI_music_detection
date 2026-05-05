import numpy as np
import pandas as pd
import torch
from torch import nn
import torchaudio
from model_audio_input.model import SpecTTTra
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from model_audio_input.dataset import SonicDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
num_epoch = 1

hyper_param_6s = [
    { # alpha
        "input_spec_dim": 128,
        "input_temp_dim": 188,
        "embed_dim": 384,
        "f_clip": 1,
        "t_clip": 3,
        "num_heads": 6,
        "num_layers": 12,
        "pos_drop_rate": 0
    },
    { # beta
        "input_spec_dim": 128,
        "input_temp_dim": 188,
        "embed_dim": 384,
        "f_clip": 3,
        "t_clip": 5,
        "num_heads": 6,
        "num_layers": 12,
        "pos_drop_rate": 0
    },
    { # gamma
        "input_spec_dim": 128,
        "input_temp_dim": 188,
        "embed_dim": 384,
        "f_clip": 5,
        "t_clip": 7,
        "num_heads": 6,
        "num_layers": 12,
        "pos_drop_rate": 0
    }
]

train_loss_hist, train_acc_hist, test_acc_hist = [], [], []

def train(len_clip, version = 1, lr=3e-4, weight_decay=1e-4, use_balanced_sampler=True):
    def get_model_param(len_clip, version):
        if len_clip == 6:
            return hyper_param_6s[version-1]
        return hyper_param_120s[version-1]

    if len_clip not in [6, 120]:
        raise ValueError("Clip len only 6s or 120s")
    if version not in [1, 2, 3]:
        raise ValueError("Not availible model version")

    df = pd.read_csv(f"crop_data/crop{len_clip}.csv")

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    print(f"Train data: {df_train.shape}")
    print(f"Test_data: {df_test.shape}")

    # reset history for each new training run
    train_loss_hist.clear()
    train_acc_hist.clear()
    test_acc_hist.clear()

    train_data = SonicDataset(df_train, duration_seconds=len_clip)
    test_data = SonicDataset(df_test, duration_seconds=len_clip)

    if use_balanced_sampler:
        train_labels = df_train["label"].astype(int).tolist()
        label_weights = {
            0: len(train_labels) / (2.0 * max(1, train_labels.count(0))),
            1: len(train_labels) / (2.0 * max(1, train_labels.count(1))),
        }
        sample_weights = torch.tensor(
            [label_weights[y] for y in train_labels],
            dtype=torch.double,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=2)
    else:
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    hyper_param = get_model_param(len_clip, version).copy()
    model = SpecTTTra(**hyper_param).to(device)

    class_counts = (
        df_train["label"]
        .value_counts()
        .sort_index()
        .reindex([0, 1], fill_value=0)
    )
    class_weights = torch.tensor(
        [len(df_train) / (2.0 * max(1, class_counts[0])),
         len(df_train) / (2.0 * max(1, class_counts[1]))],
        dtype=torch.float32,
        device=device,
    )
    print(f"Class counts(train): {class_counts.to_dict()} | class_weights: {class_weights.tolist()}")

    loss = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        # train
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_sample = 0
        train_pred_count = torch.zeros(2, dtype=torch.long)

        for batch in train_loader:
            X = batch["x"].to(device)
            y = batch["y"].to(device).long()

            optimizer.zero_grad()

            logits = model(X)
            cur_loss = loss(logits, y)
            cur_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += cur_loss.item() * X.size(0)
            total_sample += X.size(0)

            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == y).sum().item()
            train_pred_count += torch.bincount(pred.detach().cpu(), minlength=2)

        epoch_loss = total_loss / total_sample
        epoch_acc = total_correct / total_sample

        # eval
        model.eval()
        test_sample = 0
        test_correct = 0
        test_pred_count = torch.zeros(2, dtype=torch.long)
        test_cm = torch.zeros((2, 2), dtype=torch.long)
        with torch.no_grad():
            for batch in test_loader:
                X = batch["x"].to(device)
                y = batch["y"].to(device).long()

                logits = model(X)
                pred = torch.argmax(logits, dim=1)
                test_sample += X.size(0)
                test_correct += (y == pred).sum().item()
                test_pred_count += torch.bincount(pred.detach().cpu(), minlength=2)
                test_cm += torch.bincount((y.detach().cpu() * 2 + pred.detach().cpu()), minlength=4).reshape(2, 2)

        test_acc = test_correct / test_sample
        recall_fake = test_cm[0, 0].item() / max(1, test_cm[0].sum().item())
        recall_real = test_cm[1, 1].item() / max(1, test_cm[1].sum().item())
        balanced_acc = (recall_fake + recall_real) / 2.0

        train_loss_hist.append(epoch_loss)
        train_acc_hist.append(epoch_acc)
        test_acc_hist.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epoch}] "
            f"Train Loss: {epoch_loss:.4f} | "
            f"Train Acc: {epoch_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Bal Acc: {balanced_acc:.4f} | "
            f"Recall[Fake/Real]: [{recall_fake:.4f}, {recall_real:.4f}] | "
            f"Train Pred[0/1]: {train_pred_count.tolist()} | "
            f"Test Pred[0/1]: {test_pred_count.tolist()} | "
            f"Test CM [[TN,FP],[FN,TP]]: {test_cm.tolist()}")
        


train(6)
