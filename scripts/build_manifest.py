"""Sinh split_manifest.csv và test_clean.csv cho cả 2 bộ dữ liệu.

Chạy:  python scripts/build_manifest.py
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.loaders import load_vihsd, load_victsd
from src.data.preprocess import minimal_preprocess

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))

vihsd = load_vihsd(ROOT / cfg["paths"]["raw_vihsd"])
victsd = load_victsd(ROOT / cfg["paths"]["raw_victsd"])
full = pd.concat([vihsd, victsd], ignore_index=True)

# Tiền xử lý tối thiểu + loại bản ghi rỗng
before = len(full)
full["text_clean"] = full["text"].apply(minimal_preprocess)
full = full[full["text_clean"].notna()].copy()
print(f"Đã loại {before - len(full)} bản ghi rỗng/lỗi.")

# --- SC1: kiểm tra tính toàn vẹn (§13) ---
assert full["sample_id"].is_unique, "LỖI: sample_id bị trùng!"
for ds in ["vihsd", "victsd"]:
    sub = full[full["dataset"] == ds]
    print(f"\n[{ds}] phân bố nhãn theo split:")
    print(pd.crosstab(sub["split"], sub["label"]))

out = ROOT / cfg["paths"]["processed"]
out.mkdir(parents=True, exist_ok=True)

full[["sample_id", "dataset", "split", "label", "original_index"]].to_csv(
    out / "split_manifest.csv", index=False
)
full[full["split"] == "test"][["sample_id", "dataset", "text_clean", "label"]].to_csv(
    out / "test_clean.csv", index=False
)
# Lưu cả bản đầy đủ để bước train dùng lại, khỏi phải đọc file raw nữa
full.to_csv(out / "all_clean.csv", index=False)
print("\nĐã ghi xong vào", out)