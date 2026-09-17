import hashlib
import pandas as pd
from pathlib import Path


def make_sample_id(dataset: str, split: str, idx: int) -> str:
    """Sinh sample_id ổn định, không đổi dù file được đọc lại hay sắp xếp lại."""
    raw = f"{dataset}_{split}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_vihsd(raw_dir: str) -> pd.DataFrame:
    """Đọc 3 file split chính thức của ViHSD, gộp thành 1 bảng có cột 'split'."""
    raw_dir = Path(raw_dir)
    files = {"train": "train.csv", "dev": "dev.csv", "test": "test.csv"}
    frames = []
    for split, fname in files.items():
        df = pd.read_csv(raw_dir / fname)
        df = df.rename(columns={df.columns[0]: "text", df.columns[-1]: "label"})
        df["split"] = split
        df["dataset"] = "vihsd"
        df["original_index"] = df.index
        df["sample_id"] = [
            make_sample_id("vihsd", split, i) for i in df.index
        ]
        frames.append(df[["sample_id", "dataset", "split", "text", "label", "original_index"]])
    return pd.concat(frames, ignore_index=True)


def load_victsd(raw_dir: str) -> pd.DataFrame:
    """Đọc 3 file split chính thức của UIT-ViCTSD. 'valid' được đổi tên thành 'dev'
    để thống nhất tên gọi giữa hai bộ dữ liệu."""
    raw_dir = Path(raw_dir)
    files = {"train": "ViCTSD_train.csv", "dev": "ViCTSD_valid.csv", "test": "ViCTSD_test.csv"}
    frames = []
    for split, fname in files.items():
        df = pd.read_csv(raw_dir / fname)
        # LƯU Ý: ViCTSD có cả nhãn constructiveness và toxicity.
        # Chỉ lấy cột toxicity — kiểm tra tên cột thật trong notebook EDA rồi sửa lại dòng dưới.
        df = df.rename(columns={"Comment": "text", "Toxicity": "label"})
        df["split"] = split
        df["dataset"] = "victsd"
        df["original_index"] = df.index
        df["sample_id"] = [
            make_sample_id("victsd", split, i) for i in df.index
        ]
        frames.append(df[["sample_id", "dataset", "split", "text", "label", "original_index"]])
    return pd.concat(frames, ignore_index=True)