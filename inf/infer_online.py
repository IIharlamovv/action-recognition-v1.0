import time
import cv2
import numpy as np
import torch
from torch import nn
from pytorchvideo.models.hub import x3d_s
from collections import deque
from queue import Queue, Full, Empty
import threading

CKPT_PATH  = "inf/best_x3d_s.pt"
VIDEO_PATH = "inf/session_07.mp4"

# --- Параметры ---
CLIP_LEN = 16
MODEL_STRIDE = 1
PRED_EVERY = 2
SMOOTH_WIN = 3
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

class SharedPred:
    def __init__(self):
        self.lock = threading.Lock()
        self.cur_label = "..."
        self.cur_conf = 0.0

    def set(self, label, conf):
        with self.lock:
            self.cur_label = label
            self.cur_conf = conf

    def get(self):
        with self.lock:
            return self.cur_label, self.cur_conf

def infer_worker(task_q, pred, stop_event, model, labels, device, mean, std, use_amp):
    recent = deque(maxlen=SMOOTH_WIN)
    infer_ms = deque(maxlen=100)

    with torch.no_grad():
        while not stop_event.is_set():
            try:
                clip = task_q.get(timeout=0.1)  
            except Empty:
                continue

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
            dt = (time.perf_counter() - t0) * 1000.0
            infer_ms.append(dt)

            prob = torch.softmax(logits, dim=0)
            cls = int(prob.argmax().item())
            conf = float(prob[cls].item())

            recent.append(cls)
            cls_sm = max(set(recent), key=recent.count)

            pred.set(labels[cls_sm], conf)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    model, labels = load_model(CKPT_PATH, device)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    use_amp = (device == "cuda")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0
    frame_dt = 1.0 / fps

    need = 1 + (CLIP_LEN - 1) * MODEL_STRIDE
    buf = deque(maxlen=need)

    # очередь задач на инференс: держим максимум 1 (самый свежий клип)
    task_q = Queue(maxsize=1)
    pred = SharedPred()
    stop_event = threading.Event()

    worker = threading.Thread(
        target=infer_worker,
        args=(task_q, pred, stop_event, model, labels, device, mean, std, use_amp),
        daemon=True
    )
    worker.start()

    frame_idx = 0
    next_show = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # буфер для модели я
        fr224_bgr = letterbox_bgr(frame_bgr, OUT_SIZE)
        fr224_rgb = fr224_bgr[..., ::-1]  # BGR->RGB
        buf.append(fr224_rgb)

        # отправляем задачу, если есть полный буфер
        if len(buf) == need and (frame_idx % PRED_EVERY == 0):
            clip = [buf[i] for i in range(0, need, MODEL_STRIDE)]
            try:
                task_q.put_nowait(clip)
            except Full:
                # воркер занят — не тормозим показ
                pass

        cur_label, cur_conf = pred.get()
        cv2.putText(frame_bgr, f"{cur_label} ({cur_conf:.2f})",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Online Action Recognition", frame_bgr)

        # показ по таймеру (чтобы видео шло ровно)
        now = time.perf_counter()
        if now < next_show:
            time.sleep(next_show - now)
        next_show += frame_dt

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        frame_idx += 1

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
