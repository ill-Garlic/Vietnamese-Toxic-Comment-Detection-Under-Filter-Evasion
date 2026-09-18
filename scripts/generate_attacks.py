"""Sinh toàn bộ tập biến đổi kiểm thử — CHẠY ĐÚNG MỘT LẦN.

Sau khi chạy xong, commit test_attacks.csv và KHÔNG chạy lại,
trừ khi QC phát hiện lỗi >5% (khi đó sửa code rồi sinh lại toàn bộ).

Chạy:  python scripts/generate_attacks.py
"""
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.perturbation.a1_accent import apply_a1
from src.perturbation.a2_separator import apply_a2
from src.perturbation.a3_qwerty import apply_a3

# ─── Load config ─────────────────────────────────────────────────────────────
cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))

ATTACK_FNS = {"A1": apply_a1, "A2": apply_a2, "A3": apply_a3}
ATTACK_SEED: int = cfg["seeds"]["attack"]           # 2026
SEVERITIES: dict = cfg["attacks"]["severities"]     # S1/S2/S3 → 0.10/0.20/0.30

# ─── Đọc tập test sạch ───────────────────────────────────────────────────────
input_file = ROOT / cfg["paths"]["processed"] / "test_clean.csv"
if not input_file.exists():
    print(
        "⚠️  Chưa có test_clean.csv!\n"
        "   Chạy: python scripts/build_manifest.py   trước."
    )
    sys.exit(1)

test_clean = pd.read_csv(input_file)
print(f"Số mẫu test: {len(test_clean)}")
print(test_clean.groupby("dataset")["sample_id"].count())

# ─── Sinh biến đổi ───────────────────────────────────────────────────────────
out_dir = ROOT / cfg["paths"]["perturbed"]
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for dataset in test_clean["dataset"].unique():
    subset = test_clean[test_clean["dataset"] == dataset]
    for atk_name, atk_fn in ATTACK_FNS.items():
        for sev_name, ratio in SEVERITIES.items():
            # RNG RIÊNG cho từng (dataset × attack × severity)
            # → tái lập độc lập từng cấu hình, không phụ thuộc thứ tự vòng lặp
            rng = random.Random(f"{ATTACK_SEED}_{dataset}_{atk_name}_{sev_name}")
            for _, r in subset.iterrows():
                perturbed, n_edit = atk_fn(r["text_clean"], ratio, rng)
                rows.append({
                    "sample_id":      r["sample_id"],   # GIỮ NGUYÊN id của bản sạch
                    "dataset":        dataset,
                    "text_clean":     r["text_clean"],
                    "text_perturbed": perturbed,
                    "label":          r["label"],
                    "attack_type":    atk_name,
                    "severity":       sev_name,
                    "attack_seed":    ATTACK_SEED,
                    "edit_count":     n_edit,
                    "attack_applied": int(n_edit > 0),
                })

df = pd.DataFrame(rows)

# ─── Ghi file ────────────────────────────────────────────────────────────────
out_path = out_dir / "test_attacks.csv"
df.to_csv(out_path, index=False)

# ─── Thống kê kiểm tra ───────────────────────────────────────────────────────
print("\n=== Thống kê cấu hình ===")
summary = df.groupby(["dataset", "attack_type", "severity"]).agg(
    n=("sample_id", "count"),
    applied_rate=("attack_applied", "mean"),
    avg_edits=("edit_count", "mean"),
).round(3)
print(summary)

total_applied = df["attack_applied"].mean()
print(f"\nTỷ lệ attack thực sự được áp dụng (tổng): {total_applied:.1%}")
print(f"\nĐã ghi {len(df):,} hàng → {out_path}")
print("✅ Commit file này và KHÔNG chạy lại generate_attacks.py nữa.")
