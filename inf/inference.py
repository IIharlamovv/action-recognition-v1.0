

import os, time
import cv2
import numpy as np
import torch
from torch import nn
from pytorchvideo.models.hub import x3d_s
from collections import deque

CKPT_PATH  = "inf/best_x3d_s.pt"
VIDEO_PATH = "inf/session_07.mp4"
OUT_PATH   = "inf/output.mp4"

# --- Параметры (можешь менять) ---
CLIP_LEN = 16          # если хочешь меньше задержку, пробуй 16
MODEL_STRIDE = 1       # 1 = быстрее реакция
PRED_EVERY = 2         # предсказывать каждые N кадров
SMOOTH_WIN = 3         # сглаживание (не делай большим)
OUT_SIZE = 224

def letterbox_bgr(frame_bgr, out_size=224):
    h, w = frame_bgr.shape[:2]
    scale = out_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    top = (out_size - new_h) // 2
    left = (out_size - new_w) // 2
    out[top:top+new_h, left:left+new_w] = resized
    return out

def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    labels = ckpt["labels"]
    model = x3d_s(pretrained=False)
    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, len(labels))
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().to(device)
    return model, labels

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)
if device == "cuda":
    torch.backends.cudnn.benchmark = True

model, labels = load_model(CKPT_PATH, device)

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
use_amp = (device == "cuda")

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Cannot open video"

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-6:
    fps = 30.0

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_PATH, fourcc, fps, (w, h))

need = 1 + (CLIP_LEN - 1) * MODEL_STRIDE
buf = deque(maxlen=need)

recent = deque(maxlen=SMOOTH_WIN)
cur_label, cur_conf = "...", 0.0

infer_times = []

frame_idx = 0
with torch.no_grad():
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # Кладём в буфер кадр для модели (224, RGB)
        fr224_bgr = letterbox_bgr(frame_bgr, OUT_SIZE)
        fr224_rgb = fr224_bgr[..., ::-1]  # BGR->RGB
        buf.append(fr224_rgb)

        if len(buf) == need and (frame_idx % PRED_EVERY == 0):
            clip = [buf[i] for i in range(0, need, MODEL_STRIDE)]
            arr = np.stack(clip, axis=0)  # [T,224,224,3]

            x = torch.from_numpy(arr).to(device).float() / 255.0
            x = x.permute(0,3,1,2)                 # [T,3,224,224]
            x = (x - mean) / std
            x = x.permute(1,0,2,3).unsqueeze(0).contiguous()  # [1,3,T,224,224]

            t0 = time.perf_counter()
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x)[0]
            else:
                logits = model(x)[0]
            dt = time.perf_counter() - t0
            infer_times.append(dt)

            prob = torch.softmax(logits, dim=0)
            cls = int(prob.argmax().item())
            conf = float(prob[cls].item())

            recent.append(cls)
            cls_sm = max(set(recent), key=recent.count)

            cur_label = labels[cls_sm]
            cur_conf = conf

        # Наложение метки на исходный кадр
        cv2.putText(frame_bgr, f"{cur_label} ({cur_conf:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3, cv2.LINE_AA)

        writer.write(frame_bgr)
        frame_idx += 1

cap.release()
writer.release()

if infer_times:
    print(f"Avg infer: {np.mean(infer_times)*1000:.1f} ms | p95: {np.percentile(infer_times,95)*1000:.1f} ms")
print("Saved:", OUT_PATH)