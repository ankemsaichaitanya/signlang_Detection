import os
import csv
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mediapipe as mp
import cv2

# -----------------------------
# CONFIG
# -----------------------------
DATA_ROOT = Path("dataset/ISL_CSLRT_Corpus")
FRAMES_ROOT = DATA_ROOT / "Frames_Sentence_Level"
CSV_PATH = DATA_ROOT / "corpus_csv_files" / "ISL Corpus sign glosses.csv"
SEQS_OUT = DATA_ROOT / "sequences_npz"

KEYPOINT_DIM = 42  # 21 keypoints x, y
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sentence_gloss_map() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row["Sentence"].strip()
            glosses = row["SIGN GLOSSES"].strip().split()
            mapping[sentence] = glosses
    return mapping


mp_hands = mp.solutions.hands


def extract_landmarks_from_frame(img_bgr) -> np.ndarray:
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    ) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return np.zeros(KEYPOINT_DIM, dtype=np.float32)
        hand = res.multi_hand_landmarks[0]
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y])
        return np.asarray(pts, dtype=np.float32)


def prepare_sequences():
    SEQS_OUT.mkdir(parents=True, exist_ok=True)
    sent2gloss = load_sentence_gloss_map()

    vocab: Dict[str, int] = {}
    for glosses in sent2gloss.values():
        for g in glosses:
            if g not in vocab:
                vocab[g] = len(vocab)
    blank_id = len(vocab)

    vocab_path = SEQS_OUT / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for g, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{g}\n")
        f.write(f"{blank_id}\t<blank>\n")

    seq_count = 0
    for sentence_dir in FRAMES_ROOT.iterdir():
        if not sentence_dir.is_dir():
            continue
        sentence = sentence_dir.name.strip()
        if sentence not in sent2gloss:
            continue
        glosses = sent2gloss[sentence]
        gloss_ids = [vocab[g] for g in glosses if g in vocab]
        if not gloss_ids:
            continue

        for signer_dir in sentence_dir.iterdir():
            if not signer_dir.is_dir():
                continue
            frames = sorted(
                [
                    p
                    for p in signer_dir.iterdir()
                    if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
                ],
                key=lambda p: p.name,
            )
            if not frames:
                continue

            seq = []
            for fpath in frames:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                lp = extract_landmarks_from_frame(img)
                seq.append(lp)
            if len(seq) < 4:
                continue

            seq_arr = np.stack(seq, axis=0)
            gloss_ids_arr = np.asarray(gloss_ids, dtype=np.int64)
            out_path = SEQS_OUT / f"{sentence_dir.name}__{signer_dir.name}.npz"
            np.savez_compressed(out_path, landmarks=seq_arr, gloss_ids=gloss_ids_arr)
            seq_count += 1

    print(f"Prepared {seq_count} sequences in {SEQS_OUT}")


class ISLSequenceDataset(Dataset):
    def __init__(self, npz_dir: Path):
        self.files = sorted(npz_dir.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        x = data["landmarks"]
        y = data["gloss_ids"]
        return torch.from_numpy(x).float(), torch.from_numpy(y).long()


def collate_ctc(batch):
    xs, ys = zip(*batch)
    lengths_x = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    lengths_y = torch.tensor([y.size(0) for y in ys], dtype=torch.long)

    max_T = lengths_x.max().item()
    max_L = lengths_y.max().item()

    padded_x = torch.zeros(len(xs), max_T, KEYPOINT_DIM)
    padded_y = torch.full((len(ys), max_L), -1, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        padded_x[i, : x.size(0)] = x
        padded_y[i, : y.size(0)] = y

    return padded_x, padded_y, lengths_x, lengths_y


class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size_plus_blank, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, vocab_size_plus_blank)

    def forward(self, x, lengths):
        x = self.input_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        logits = self.fc(out)
        return logits


def train():
    if not SEQS_OUT.exists() or not any(SEQS_OUT.glob("*.npz")):
        print("No sequences found, running preparation ...")
        prepare_sequences()

    vocab_file = SEQS_OUT / "vocab.txt"
    vocab_size = sum(1 for _ in open(vocab_file, encoding="utf-8")) - 1
    blank_id = vocab_size

    dataset = ISLSequenceDataset(SEQS_OUT)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_ctc
    )

    model = CTCModel(
        KEYPOINT_DIM, 256, vocab_size_plus_blank=vocab_size + 1
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x, y, len_x, len_y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x, len_x)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            targets = []
            target_lengths = []
            for i in range(y.size(0)):
                valid = y[i][y[i] >= 0]
                targets.append(valid)
                target_lengths.append(len(valid))
            targets = torch.cat(targets).to(DEVICE)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long).to(DEVICE)

            loss = ctc_loss(log_probs, targets, len_x, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / max(1, len(loader))
        print(f"Epoch {epoch}/{EPOCHS} - CTC loss: {avg:.4f}")

    out_dir = Path("isl_models")
    out_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_dir / "isl_ctc_model.pt")
    print("Saved model to", out_dir / "isl_ctc_model.pt")
    print("IMPORTANT: Keep", vocab_file, "for decoding.")


if __name__ == "__main__":
    train()

