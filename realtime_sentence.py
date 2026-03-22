import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn

KEYPOINT_DIM = 42
WINDOW_FRAMES = 96
STRIDE = 2
CONF_THRESH = 0.8         # average confidence threshold to show text
MIN_TOKENS_TO_SHOW = 2    # require at least this many tokens
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path("isl_models/isl_ctc_model.pt")
VOCAB_PATH = Path("dataset/ISL_CSLRT_Corpus/sequences_npz/vocab.txt")


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
        return self.fc(out)


def load_vocab():
    idx2gloss = {}
    with open(VOCAB_PATH, encoding="utf-8") as f:
        for line in f:
            idx_str, token = line.strip().split("\t")
            idx = int(idx_str)
            idx2gloss[idx] = token
    blank_id = max(idx2gloss.keys())
    return idx2gloss, blank_id


mp_hands = mp.solutions.hands


def extract_landmarks(img_bgr):
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            return np.zeros(KEYPOINT_DIM, dtype=np.float32)
        hand = res.multi_hand_landmarks[0]
        pts = []
        for lm in hand.landmark:
            pts.extend([lm.x, lm.y])
        return np.asarray(pts, dtype=np.float32)


def ctc_greedy_decode(logits, blank_id, idx2gloss, conf_thresh=CONF_THRESH):
    # logits: 1 x T x V
    probs = logits.softmax(dim=-1)[0]  # T x V
    max_probs, ids = probs.max(dim=-1)
    avg_conf = float(max_probs.mean().item())

    # If overall confidence is low, treat as no-sign segment
    if avg_conf < conf_thresh:
        return [], avg_conf

    tokens = []
    prev = blank_id
    for i in ids.tolist():
        if i != blank_id and i != prev:
            tokens.append(idx2gloss[i])
        prev = i
    return tokens, avg_conf


def main():
    if not MODEL_PATH.exists():
        print("Model not found. Train first with: python train_isl_ctc.py")
        return

    idx2gloss, blank_id = load_vocab()
    vocab_size_plus_blank = blank_id + 1

    model = CTCModel(KEYPOINT_DIM, 256, vocab_size_plus_blank).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    buffer = deque(maxlen=WINDOW_FRAMES)
    frame_counter = 0
    last_sentence = ""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lp = extract_landmarks(frame)
        buffer.append(lp)
        frame_counter += 1

        if len(buffer) >= 10 and frame_counter % STRIDE == 0:
            seq = np.stack(buffer, axis=0)
            x = torch.from_numpy(seq).unsqueeze(0).float().to(DEVICE)
            lengths = torch.tensor([x.size(1)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = model(x, lengths)
            tokens, conf = ctc_greedy_decode(logits, blank_id, idx2gloss)

            # Only update when we have enough stable tokens
            if len(tokens) >= MIN_TOKENS_TO_SHOW:
                last_sentence = " ".join(tokens[-8:])
            else:
                last_sentence = ""

        cv2.putText(
            frame,
            last_sentence,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1,
        )

        cv2.imshow("ISL Continuous Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

