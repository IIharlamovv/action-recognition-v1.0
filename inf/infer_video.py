import json
import torch
import torch.nn.functional as F
from torch import nn
from decord import VideoReader, cpu
from pytorchvideo.models.hub import x3d_s

# --- пути  ---
CKPT_PATH = "inf/best_x3d_s.pt" 
VIDEO_PATH = "inf/session_07.mp4"
OUT_JSON = "inf/session_07_predictions.json"

# --- препроцессинг (так как видео у меня в вертикальном формате) ---
def letterbox_batch_to_square(frames_tchw: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    T, C, H, W = frames_tchw.shape
    scale = out_size / max(H, W)
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    resized = F.interpolate(frames_tchw, size=(new_h, new_w), mode="bilinear", align_corners=False)

    pad_h = out_size - new_h
    pad_w = out_size - new_w
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

# --- загружаем модель ---
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location="cpu")
labels = ckpt["labels"]
num_classes = len(labels)

model = x3d_s(pretrained=False)
model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
model.load_state_dict(ckpt["model"], strict=True)
model.eval().to(device)

mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# --- инференс ---
CLIP_LEN = 32 # сколько кадров модель берет за один раз (мое видео грузилось в формате 30 fps)
MODEL_STRIDE = 2          # шаг выборки кадров внутри одного клипа (при stride 2 64 кадра)
STRIDE_INFER = 8          # частота обновления инференса - то есть 1 раз в 8 кадров
SMOOTH_WIN = 5            # некое сглаживание предсказаний, чтобы метка была стабильнее


vr = VideoReader(VIDEO_PATH, ctx=cpu(0))
fps = float(vr.get_avg_fps())
n_frames = len(vr)

need = 1 + (CLIP_LEN - 1) * MODEL_STRIDE  

recent = []
preds = []  

with torch.no_grad():
    for start in range(0, n_frames - need, STRIDE_INFER):
        idxs = [start + i * MODEL_STRIDE for i in range(CLIP_LEN)]
        frames = vr.get_batch(idxs).asnumpy()                         # [T,H,W,3] uint8
        frames = torch.from_numpy(frames).permute(0,3,1,2).float()/255 # [T,3,H,W]

        frames = letterbox_batch_to_square(frames, 224)               
        frames = (frames - mean) / std

        x = frames.permute(1,0,2,3).unsqueeze(0).to(device)           # [1,3,T,224,224]
        logits = model(x)[0]
        prob = torch.softmax(logits, dim=0)
        cls = int(prob.argmax().item())
        conf = float(prob[cls].item())

        recent.append(cls)
        if len(recent) > SMOOTH_WIN:
            recent.pop(0)
        cls_sm = max(set(recent), key=recent.count)

        center_frame = start + need // 2
        t_sec = center_frame / fps
        preds.append((t_sec, labels[cls_sm], conf))

# --- склеиваем сегменты ---
segments = []
if preds:
    cur_label = preds[0][1]
    seg_start = preds[0][0]
    last_t = preds[0][0]
    for t, lab, conf in preds[1:]:
        if lab != cur_label:
            segments.append({"label": cur_label, "start_sec": float(seg_start), "end_sec": float(last_t)})
            cur_label = lab
            seg_start = t
        last_t = t
    segments.append({"label": cur_label, "start_sec": float(seg_start), "end_sec": float(last_t)})

# --- сохраняем ---
out = {
    "video": VIDEO_PATH,
    "fps": fps,
    "clip_len": CLIP_LEN,
    "model_stride": MODEL_STRIDE,
    "stride_infer": STRIDE_INFER,
    "segments": segments,
}
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_JSON)
for s in segments[:15]:
    print(s)
