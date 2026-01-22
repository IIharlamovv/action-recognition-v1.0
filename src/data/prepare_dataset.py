
import os
import random
import pandas as pd

REQUIRED_COLS = {"video", "start_sec", "end_sec", "label"}

def read_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    seen = set()
    uniq = []
    for lab in labels:
        if lab in seen:
            raise ValueError(f"Duplicate label in labels.txt: {lab}")
        seen.add(lab)
        uniq.append(lab)
    label2id = {lab: i for i, lab in enumerate(uniq)}
    return uniq, label2id

def read_annotations(ann_path: str, label2id: dict):
    try:
        df = pd.read_csv(ann_path)
        if len(set(df.columns) & REQUIRED_COLS) < 4:
            df = pd.read_csv(ann_path, sep=";")
    except Exception:
        df = pd.read_csv(ann_path, sep=";")

    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(f"annotations.csv must have columns {REQUIRED_COLS}. Found: {df.columns.tolist()}")

    df["video"] = df["video"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)

    bad = df[df["end_sec"] <= df["start_sec"]]
    if len(bad) > 0:
        raise ValueError(f"Found segments with end_sec <= start_sec:\n{bad.head(10)}")

    unknown = df[~df["label"].isin(label2id)]
    if len(unknown) > 0:
        raise ValueError(
            "Found labels in annotations.csv that are not in labels.txt, examples:\n"
            f"{unknown[['video','start_sec','end_sec','label']].head(10)}"
        )

    df["label_id"] = df["label"].map(label2id).astype(int)
    return df

def list_videos(videos_dir: str):
    vids = [f for f in os.listdir(videos_dir) if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))]
    vids.sort()
    return vids

def make_video_split(video_list, seed=42, n_train=5, n_val=1, n_test=0):
    if len(video_list) < (n_train + n_val + n_test):
        raise ValueError(f"Need at least {n_train+n_val+n_test} videos, got {len(video_list)}")
    rnd = random.Random(seed)
    vids = video_list[:]
    rnd.shuffle(vids)
    return {
        "train": vids[:n_train],
        "val": vids[n_train:n_train+n_val],
        "test": vids[n_train+n_val:n_train+n_val+n_test],
    }

def save_split_files(splits: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for k, v in splits.items():
        path = os.path.join(out_dir, f"{k}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for name in v:
                f.write(name + "\n")

def main(
    data_dir="data",
    raw_dir="data/raw",
    ann_path="data/annotations.csv",
    labels_path="data/labels.txt",
    out_ann_path="data/processed/annotations_clean.csv",
    splits_dir="data/splits",
    seed=42,
):
    labels, label2id = read_labels(labels_path)
    df = read_annotations(ann_path, label2id)

    all_videos = list_videos(raw_dir)
    missing = sorted(set(df["video"].unique()) - set(all_videos))
    if missing:
        raise FileNotFoundError(f"Videos referenced in annotations.csv not found in {raw_dir}: {missing}")

    os.makedirs(os.path.dirname(out_ann_path), exist_ok=True)
    df.to_csv(out_ann_path, index=False)
    print(f"Saved cleaned annotations to: {out_ann_path}")

    splits = make_video_split(all_videos, seed=seed, n_train=5, n_val=1, n_test=0)
    save_split_files(splits, splits_dir)
    print("Saved splits:", splits)
    print(f"Split files are in: {splits_dir}")

if __name__ == "__main__":
    main()
