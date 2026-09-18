"""Sinh split_manifest.csv, test_clean.csv, all_clean.csv cho cả 2 bộ dữ liệu.
Đã tích hợp: tiền xử lý tối thiểu, khử trùng lặp train<->eval (§4.1),
kiểm tra chống rò rỉ dữ liệu.

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


def dedup_train_against_eval(full: pd.DataFrame) -> pd.DataFrame:
    """Loại khỏi train các câu đã xuất hiện ở dev hoặc test (§4.1).
    Test và dev giữ nguyên vì đó là split chính thức của dataset gốc."""
    cleaned = []
    for ds in full["dataset"].unique():
        sub = full[full["dataset"] == ds]
        eval_texts = set(sub[sub["split"].isin(["dev", "test"])]["text_clean"])
        train = sub[sub["split"] == "train"]
        mask_leak = train["text_clean"].isin(eval_texts)
        n_removed = mask_leak.sum()
        print(f"[{ds}] Loại {n_removed}/{len(train)} câu khỏi train "
              f"({n_removed / len(train) * 100:.2f}%)")
        cleaned.append(train[~mask_leak])
        cleaned.append(sub[sub["split"] != "train"])
    return pd.concat(cleaned, ignore_index=True)


# ==== 1. Đọc dữ liệu thô ====
vihsd = load_vihsd(ROOT / cfg["paths"]["raw_vihsd"])
victsd = load_victsd(ROOT / cfg["paths"]["raw_victsd"])
full = pd.concat([vihsd, victsd], ignore_index=True)

# ==== 2. Tiền xử lý tối thiểu + loại bản ghi rỗng ====
before = len(full)
full["text_clean"] = full["text"].apply(minimal_preprocess)
full = full[full["text_clean"].notna()].copy()
print(f"Đã loại {before - len(full)} bản ghi rỗng/lỗi.")

# ==== 3. SC1: kiểm tra tính toàn vẹn (§13) ====
assert full["sample_id"].is_unique, "LỖI: sample_id bị trùng!"

# ==== 4. KHỬ TRÙNG LẶP train<->eval (§4.1) — PHẢI LÀM TRƯỚC KHI GHI FILE ====
print("\n=== Khử trùng lặp train<->eval ===")
full = dedup_train_against_eval(full)

# ==== 5. In phân bố nhãn SAU khi đã khử trùng lặp ====
for ds in ["vihsd", "victsd"]:
    sub = full[full["dataset"] == ds]
    print(f"\n[{ds}] phân bố nhãn theo split (sau khử trùng lặp):")
    print(pd.crosstab(sub["split"], sub["label"]))

# ==== 6. Ghi file ====
out = ROOT / cfg["paths"]["processed"]
out.mkdir(parents=True, exist_ok=True)

full[["sample_id", "dataset", "split", "label", "original_index"]].to_csv(
    out / "split_manifest.csv", index=False
)
full[full["split"] == "test"][["sample_id", "dataset", "text_clean", "label"]].to_csv(
    out / "test_clean.csv", index=False
)
full.to_csv(out / "all_clean.csv", index=False)
print("\nĐã ghi xong vào", out)

# ==== 7. Kiểm tra lại: sau khử trùng lặp phải ra 0 ====
print("\n=== Xác nhận sau khử trùng lặp (phải = 0) ===")
for ds in ["vihsd", "victsd"]:
    sub = full[full["dataset"] == ds]
    train_texts = set(sub[sub["split"] == "train"]["text_clean"])
    test_texts = set(sub[sub["split"] == "test"]["text_clean"])
    dev_texts = set(sub[sub["split"] == "dev"]["text_clean"])
    print(f"[{ds}] Trùng train<->test: {len(train_texts & test_texts)} câu")
    print(f"[{ds}] Trùng train<->dev:  {len(train_texts & dev_texts)} câu")