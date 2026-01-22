# src/data/dataset.py
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

def letterbox_batch_to_square(frames_tchw: torch.Tensor, out_size: int = 224) -> torch.Tensor:
    """
    frames_tchw: [T, 3, H, W] float tensor 0..1
    returns:     [T, 3, out_size, out_size]
    """
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

    padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)
    return padded

def load_split_list(split_txt_path: str):
    with open(split_txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

class ActionClipDataset(Dataset):
    """
    Из untrimmed сегментов делает клипы:
      X: [3, T, 224, 224]
      y: int
    """
    def __init__(
        self,
        annotations_csv: str,
        videos_dir: str,
        split_txt: str,
        clip_len: int = 32,
        out_size: int = 224,
        stride: int = 1,
        seed: int = 42,
        normalize_imagenet: bool = True,
    ):
        if not _HAS_DECORD:
            raise ImportError("decord is not installed. Install it with: pip install decord")

        self.df = pd.read_csv(annotations_csv)
        split_videos = set(load_split_list(split_txt))
        self.df = self.df[self.df["video"].isin(split_videos)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split {split_txt}. Check split file and annotations.")

        self.videos_dir = videos_dir
        self.clip_len = clip_len
        self.out_size = out_size
        self.stride = stride
        self.rnd = random.Random(seed)
        self.normalize_imagenet = normalize_imagenet

        self._vr_cache = {}

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

    def _get_vr(self, video_name: str):
        path = os.path.join(self.videos_dir, video_name)
        if path not in self._vr_cache:
            self._vr_cache[path] = VideoReader(path, ctx=cpu(0))
        return self._vr_cache[path]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video = str(row["video"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        label_id = int(row["label_id"]) if "label_id" in row else int(row["label"])

        vr = self._get_vr(video)
        fps = float(vr.get_avg_fps())
        n_frames = len(vr)

        start_f = max(0, int(np.floor(start_sec * fps)))
        end_f = min(n_frames - 1, int(np.ceil(end_sec * fps)))

        need = 1 + (self.clip_len - 1) * self.stride
        seg_len = end_f - start_f + 1

        if seg_len <= 1:
            start_idx = start_f
        else:
            max_start = start_f + max(0, seg_len - need)
            start_idx = self.rnd.randint(start_f, max_start) if max_start >= start_f else start_f

        frame_idxs = [start_idx + i * self.stride for i in range(self.clip_len)]
        frame_idxs = [min(i, end_f) for i in frame_idxs]
        frame_idxs = [min(i, n_frames - 1) for i in frame_idxs]

        frames = vr.get_batch(frame_idxs).asnumpy()  # [T,H,W,3] uint8
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]

        frames = letterbox_batch_to_square(frames, out_size=self.out_size)  # [T,3,224,224]

        if self.normalize_imagenet:
            frames = (frames - self.mean) / self.std

        x = frames.permute(1, 0, 2, 3).contiguous()  # [3,T,224,224]
        y = torch.tensor(label_id, dtype=torch.long)
        return x, y
