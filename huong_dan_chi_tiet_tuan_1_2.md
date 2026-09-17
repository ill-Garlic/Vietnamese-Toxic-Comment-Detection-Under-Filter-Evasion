# HƯỚNG DẪN TUẦN 1 & 2 (BẢN 2)
### Viết lại theo đúng cây thư mục thực tế của repo `vihsd-robust-toxic-detection`

---

## 0. Hai thay đổi so với bản hướng dẫn cũ

**Thay đổi 1 — Bỏ toàn bộ phần tự chia dữ liệu.**
Cây thư mục cho thấy cả 2 bộ dữ liệu đều đã có sẵn phân chia chính thức:
- `data/raw/vihsd/` → `train.csv`, `dev.csv`, `test.csv`
- `data/raw/ViCTSD/` → `ViCTSD_train.csv`, `ViCTSD_valid.csv`, `ViCTSD_test.csv`

Theo §4 của Quy trình v1.0, có split chính thức thì **ưu tiên dùng luôn**. Vậy nên `split_seed = 42` và đoạn code `train_test_split` trong bản hướng dẫn cũ **không dùng đến nữa**. Bạn chỉ cần *đọc* split có sẵn và gắn nhãn `split` cho từng dòng.

**Thay đổi 2 — Ánh xạ lại đường dẫn file.**
Bản cũ viết theo cấu trúc gợi ý trong §14 của Quy trình; bản này viết theo đúng cấu trúc bạn đã dựng.

| Bản cũ | Repo thực tế của bạn |
|---|---|
| `configs/protocol_lock.yaml` | `configs/config.yaml` (đã có) |
| `train/preprocess.py` | `src/data/preprocess.py` |
| `attacks/*.py` | `src/perturbation/*.py` |
| `data_manifests/make_split.py` | `scripts/build_manifest.py` |
| `split_manifest.csv`, `test_clean.csv` | `data/processed/` |
| tập biến đổi kiểm thử | `data/perturbed/test_perturbed/` |

> Thư mục `data_manifests/` ở gốc repo đang rỗng và bị trùng vai trò với `data/processed/`. Khuyến nghị: xóa `data_manifests/`, giữ mọi manifest trong `data/processed/` cho nhất quán.

**Nguyên tắc chung về nơi đặt code:** `src/` chứa **hàm/module tái sử dụng** (không tự chạy), `scripts/` chứa **file chạy được** (`python scripts/xxx.py`) gọi vào `src/`. Tách như vậy để sau này notebook, script train, script sinh biến đổi đều dùng chung một logic — tránh tình trạng mỗi nơi viết một kiểu chuẩn hóa văn bản rồi kết quả lệch nhau.

---

# TUẦN 1

## Ngày 1 — Khóa cấu hình + EDA

### 1.1. Cập nhật `configs/config.yaml`

**→ File:** `configs/config.yaml`

```yaml
# ==== SEED ĐÃ KHÓA (Quy trình v1.0 §8) ====
# Không có split_seed vì cả 2 bộ dữ liệu đều dùng split chính thức
seeds:
  model: [42, 43, 44]     # 3 seed cho mô hình có tính ngẫu nhiên
  attack: 2026            # seed sinh tập biến đổi kiểm thử

# ==== ĐƯỜNG DẪN ====
paths:
  raw_vihsd: "data/raw/vihsd"
  raw_victsd: "data/raw/ViCTSD"
  processed: "data/processed"
  perturbed: "data/perturbed/test_perturbed"
  results: "results"
  models: "models"
  logs: "logs"

# ==== ÁNH XẠ NHÃN ĐỘC HẠI / KHÔNG ĐỘC HẠI (§4) ====
# Dùng để tính Recall nội dung độc hại và ASR
harmful_mapping:
  vihsd:
    harmful: ["OFFENSIVE", "HATE"]
    non_harmful: ["CLEAN"]
  victsd:
    harmful: ["TOXIC"]
    non_harmful: ["NONE"]

# ==== ĐỊNH NGHĨA BIẾN ĐỔI (§6) ====
attacks:
  types: ["A1", "A2", "A3"]
  severities:
    S1: 0.10
    S2: 0.20
    S3: 0.30
  min_token_len:
    A1: 3
    A2: 4
    A3: 3
  separators: [" ", ".", "-", "_"]
```

**Vì sao làm bước này trước tiên:** mọi seed và mọi ngưỡng đều nằm ở một chỗ duy nhất. Nếu để rải rác trong từng file `.py`, chỉ cần một lần ai đó sửa `0.10` thành `0.15` ở một file mà quên file khác là toàn bộ kết quả mất tính tái lập (SC6), và rất khó phát hiện.

### 1.2. Hoàn tất EDA

**→ File:** `notebooks/01_eda_vihsd.ipynb` (đang làm dở) và tạo thêm `notebooks/02_eda_victsd.ipynb`

Mỗi notebook chỉ cần trả lời 5 câu hỏi:

```python
import pandas as pd

train = pd.read_csv("../data/raw/vihsd/train.csv")
dev   = pd.read_csv("../data/raw/vihsd/dev.csv")
test  = pd.read_csv("../data/raw/vihsd/test.csv")

for name, df in [("train", train), ("dev", dev), ("test", test)]:
    print(f"--- {name} ---")
    print("Số dòng:", len(df))
    print("Tên cột:", list(df.columns))
    print("Thiếu dữ liệu:\n", df.isnull().sum())
    print("Phân bố nhãn:\n", df.iloc[:, -1].value_counts())
    print("Số dòng văn bản rỗng:", (df.iloc[:, 0].astype(str).str.strip() == "").sum())
    print()

# Kiểm tra trùng văn bản GIỮA các tập (nguy cơ rò rỉ dữ liệu — §4.1)
train_texts = set(train.iloc[:, 0].astype(str))
test_texts  = set(test.iloc[:, 0].astype(str))
print("Số văn bản xuất hiện ở cả train và test:", len(train_texts & test_texts))
```

**Vì sao dòng cuối quan trọng nhất:** nếu một câu xuất hiện ở cả train và test, mô hình đã "nhìn thấy đáp án" trước khi thi. Kết quả Macro-F1 sẽ cao giả tạo, và mọi kết luận về độ bền vững sau đó đều sai lệch. Phát hiện ở ngày 1 thì xử lý dễ; phát hiện ở tuần 4 thì phải chạy lại mọi thứ. Ghi lại con số này vào notebook để sau còn báo cáo được.

---

## Ngày 2 — Manifest và sample_id

### 2.1. Module đọc dữ liệu

**→ File:** `src/data/loaders.py` (tạo mới)

```python
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
```

**Lưu ý thực tế:** tên cột của ViCTSD gần như chắc chắn khác với đoán ở trên. Chạy EDA trước (ngày 1), xem tên cột thật rồi sửa lại dòng `rename` — đừng đoán.

### 2.2. Module tiền xử lý tối thiểu

**→ File:** `src/data/preprocess.py` (tạo mới)

```python
import unicodedata


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về NFC (§5)."""
    return unicodedata.normalize("NFC", str(text))


def minimal_preprocess(text) -> str | None:
    """Tiền xử lý tối thiểu theo §5. Trả về None nếu bản ghi rỗng (để loại bỏ).

    TUYỆT ĐỐI KHÔNG làm các việc sau ở đây:
      - bỏ dấu tiếng Việt      (đó là biến đổi A1)
      - chèn/xóa dấu câu       (đó là biến đổi A2)
      - sửa chính tả           (sẽ xóa mất tín hiệu của A3)
      - co khoảng trắng bất thường
      - chuẩn hóa teencode
    Lý do: đây là nghiên cứu về độ bền vững TRƯỚC các biến đổi đó.
    Nếu tiền xử lý đã tự động "sửa" chúng thì không còn gì để đo.
    """
    if text is None:
        return None
    text = normalize_unicode(text)
    if not text.strip():
        return None
    return text
```

### 2.3. Script sinh manifest

**→ File:** `scripts/build_manifest.py` (tạo mới — đây là file bạn *chạy*)

```python
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
```

**Vì sao `sample_id` phải sinh ở bước này, trước mọi thứ khác:** mọi phiên bản biến đổi (9 điều kiện: A1–A3 × S1–S3) của một câu đều phải mang **đúng cùng** `sample_id` với bản sạch của nó. Đó là điều kiện để tuần 4–5 so sánh theo cặp (paired) — tức là so "câu này khi sạch" với "chính câu này khi bị biến đổi", chứ không phải so hai tập khác nhau về mặt thống kê. Nếu sample_id sinh sau, hoặc sinh lại mỗi lần chạy, thì cặp bị đứt và toàn bộ phần bootstrap ở §11 không thực hiện được.

**Kết quả cuối ngày 2:** `data/processed/` có 3 file: `split_manifest.csv`, `test_clean.csv`, `all_clean.csv`. **Commit và từ đây không sửa nữa.**

---

## Ngày 3–5 — Mô-đun biến đổi A1

### 3.1. Logic dùng chung cho cả 3 loại biến đổi

**→ File:** `src/perturbation/base.py` (tạo mới)

```python
import math
import re


URL_MENTION_PATTERN = re.compile(r"^(https?://|www\.|@|#)")


def is_eligible(token: str, min_len: int) -> bool:
    """Đơn vị từ đủ điều kiện theo §6.2."""
    if URL_MENTION_PATTERN.match(token):
        return False
    if "@" in token and "." in token:      # email
        return False
    alnum = re.sub(r"[^\w]", "", token, flags=re.UNICODE)
    return len(alnum) >= min_len


def perturb_sentence(text: str, ratio: float, min_len: int, token_fn, rng):
    """Khung chung: chọn token đủ điều kiện theo tỷ lệ rồi áp dụng token_fn lên từng token.

    Trả về (văn bản đã biến đổi, số token thực sự bị sửa).
    Dùng chung cho A1, A2, A3 — chỉ khác nhau ở min_len và token_fn.
    """
    tokens = text.split()
    eligible = [i for i, t in enumerate(tokens) if is_eligible(t, min_len)]
    if not eligible:
        return text, 0                      # attack_applied = 0, vẫn phải ghi log (§6.2)

    n = max(1, math.ceil(len(eligible) * ratio))   # làm tròn lên, tối thiểu 1
    n = min(n, len(eligible))
    chosen = rng.sample(eligible, k=n)              # không biến đổi 1 token 2 lần

    edit_count = 0
    for i in chosen:
        new_token = token_fn(tokens[i], rng)
        if new_token != tokens[i]:
            tokens[i] = new_token
            edit_count += 1
    return " ".join(tokens), edit_count
```

**Vì sao tách hàm khung này ra riêng:** ba loại biến đổi A1/A2/A3 chỉ khác nhau ở *cách sửa một token*, còn *cách chọn token nào để sửa* thì giống hệt nhau (cùng quy tắc tỷ lệ, cùng quy tắc làm tròn lên, cùng quy tắc không sửa trùng). Viết một lần ở đây rồi dùng lại 3 lần đảm bảo cả 3 loại tuân thủ đúng cùng một định nghĩa mức độ S1/S2/S3. Nếu copy-paste logic này sang 3 file, chỉ cần một chỗ quên `math.ceil` là mức độ của loại đó sai lệch so với hai loại kia, và QC 30 mẫu rất có thể không phát hiện ra.

### 3.2. A1 — Bỏ dấu tiếng Việt

**→ File:** `src/perturbation/a1_accent.py` (tạo mới)

```python
import unicodedata

from .base import perturb_sentence


def strip_accents_token(token: str, rng=None) -> str:
    """Bỏ dấu tiếng Việt, giữ nguyên các ký tự khác. Xử lý riêng đ/Đ vì
    ký tự này không tách được bằng chuẩn hóa NFD."""
    token = token.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", token)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def apply_a1(text: str, ratio: float, rng, min_len: int = 3):
    return perturb_sentence(text, ratio, min_len, strip_accents_token, rng)
```

**Chi tiết dễ bỏ sót:** `đ` không phải là `d` + dấu, mà là một ký tự riêng trong Unicode. Nếu chỉ dùng NFD + lọc dấu thì `đ` vẫn còn nguyên, trong khi người né bộ lọc thật sẽ gõ thành `d`. Đây đúng là loại lỗi mà QC ở tuần 2 phải bắt được.

### 3.3. Test nhanh

**→ File:** `tests/test_perturbation.py` (tạo mới)

```python
from src.perturbation.a1_accent import strip_accents_token


def test_strip_accents():
    assert strip_accents_token("tiếng") == "tieng"
    assert strip_accents_token("Đồ") == "Do"
    assert strip_accents_token("abc123") == "abc123"
    assert strip_accents_token("😀") == "😀"
```

Chạy: `pytest tests/ -v`

**Vì sao viết test dù đang gấp:** biến đổi là thứ *duy nhất* mà nếu cài sai, bạn sẽ không bao giờ tự phát hiện qua kết quả — mô hình vẫn cho ra số liệu đẹp đẽ, chỉ là số liệu của một phép biến đổi khác với phép bạn mô tả trong bài báo. Vài dòng test rẻ hơn nhiều so với việc sinh lại toàn bộ 18 cấu hình ở tuần 2.

**Mốc cuối Tuần 1:** `data/processed/` khóa xong; `src/perturbation/a1_accent.py` chạy được và qua test.

---

# TUẦN 2

## Ngày 6 — A2 (chèn khoảng trắng/dấu câu)

**→ File:** `src/perturbation/a2_separator.py` (tạo mới)

```python
from .base import perturb_sentence

SEPARATORS = [" ", ".", "-", "_"]


def insert_separator_token(token: str, rng) -> str:
    """Chèn 1 ký tự phân cách vào GIỮA token (không chèn ở đầu/cuối)."""
    if len(token) < 2:
        return token
    pos = rng.randint(1, len(token) - 1)
    sep = rng.choice(SEPARATORS)
    return token[:pos] + sep + token[pos:]


def apply_a2(text: str, ratio: float, rng, min_len: int = 4):
    return perturb_sentence(text, ratio, min_len, insert_separator_token, rng)
```

**Chú ý `min_len=4`:** A2 yêu cầu token dài tối thiểu 4 ký tự, khác A1/A3 (3 ký tự) — đây là quy định ở §6.2, không phải tùy chọn. Lý do hợp lý: chèn dấu phân cách vào giữa một từ 3 ký tự thường phá vỡ từ đến mức không còn đọc được, sẽ rớt QC.

## Ngày 7 — A3 (lỗi gõ QWERTY)

**→ File:** `src/perturbation/a3_qwerty.py` (tạo mới)

```python
from .base import perturb_sentence

QWERTY_NEIGHBORS = {
    "q": "wa",     "w": "qes",    "e": "wrd",    "r": "etf",    "t": "ryg",
    "y": "tuh",    "u": "yij",    "i": "uok",    "o": "ipl",    "p": "o",
    "a": "qsz",    "s": "awdx",   "d": "sefc",   "f": "drgv",   "g": "fthb",
    "h": "gyjn",   "j": "hukm",   "k": "jil",    "l": "ko",
    "z": "asx",    "x": "zsdc",   "c": "xdfv",   "v": "cfgb",   "b": "vghn",
    "n": "bhjm",   "m": "njk",
}


def qwerty_typo_token(token: str, rng) -> str:
    """Thay 1 ký tự chữ bằng ký tự lân cận trên bàn phím QWERTY.
    Ký tự tiếng Việt có dấu không có ánh xạ -> bỏ qua (§6)."""
    candidates = [i for i, c in enumerate(token) if c.lower() in QWERTY_NEIGHBORS]
    if not candidates:
        return token                     # không có ánh xạ hợp lệ -> giữ nguyên
    i = rng.choice(candidates)
    c = token[i]
    repl = rng.choice(QWERTY_NEIGHBORS[c.lower()])
    repl = repl.upper() if c.isupper() else repl
    return token[:i] + repl + token[i + 1:]


def apply_a3(text: str, ratio: float, rng, min_len: int = 3):
    return perturb_sentence(text, ratio, min_len, qwerty_typo_token, rng)
```

**Điểm cần ghi lại trong báo cáo:** với văn bản tiếng Việt có dấu, nhiều ký tự (`ế`, `ơ`, `ạ`...) không nằm trong bảng ánh xạ QWERTY, nên A3 sẽ tác động lên ít ký tự hơn so với văn bản tiếng Anh. Đây không phải lỗi, nhưng phải nêu rõ trong phần Phương pháp, vì nó giải thích vì sao A3 có thể tỏ ra "nhẹ" hơn A1/A2 trong kết quả.

## Ngày 8 — Sinh tập biến đổi kiểm thử chính thức

**→ File:** `scripts/generate_attacks.py` (tạo mới)

```python
"""Sinh toàn bộ tập biến đổi kiểm thử — CHẠY ĐÚNG MỘT LẦN.

Chạy:  python scripts/generate_attacks.py
"""
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.perturbation.a1_accent import apply_a1
from src.perturbation.a2_separator import apply_a2
from src.perturbation.a3_qwerty import apply_a3

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))

ATTACK_FNS = {"A1": apply_a1, "A2": apply_a2, "A3": apply_a3}
ATTACK_SEED = cfg["seeds"]["attack"]          # 2026
SEVERITIES = cfg["attacks"]["severities"]     # S1/S2/S3 -> 0.10/0.20/0.30

test_clean = pd.read_csv(ROOT / cfg["paths"]["processed"] / "test_clean.csv")
out_dir = ROOT / cfg["paths"]["perturbed"]
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for dataset in ["vihsd", "victsd"]:
    subset = test_clean[test_clean["dataset"] == dataset]
    for atk_name, atk_fn in ATTACK_FNS.items():
        for sev_name, ratio in SEVERITIES.items():
            # RNG riêng cho từng cấu hình -> tái lập độc lập từng cấu hình,
            # không phụ thuộc thứ tự vòng lặp
            rng = random.Random(f"{ATTACK_SEED}_{dataset}_{atk_name}_{sev_name}")
            for _, r in subset.iterrows():
                perturbed, n_edit = atk_fn(r["text_clean"], ratio, rng)
                rows.append({
                    "sample_id": r["sample_id"],      # GIỮ NGUYÊN id của bản sạch
                    "dataset": dataset,
                    "text_clean": r["text_clean"],
                    "text_perturbed": perturbed,
                    "label": r["label"],
                    "attack_type": atk_name,
                    "severity": sev_name,
                    "attack_seed": ATTACK_SEED,
                    "edit_count": n_edit,
                    "attack_applied": int(n_edit > 0),
                })

df = pd.DataFrame(rows)
df.to_csv(out_dir / "test_attacks.csv", index=False)

# Thống kê để đối chiếu và đưa vào báo cáo
print(df.groupby(["dataset", "attack_type", "severity"]).agg(
    n=("sample_id", "count"),
    applied_rate=("attack_applied", "mean"),
    avg_edits=("edit_count", "mean"),
))
print("\nĐã ghi:", out_dir / "test_attacks.csv")
```

**Vì sao mỗi cấu hình một RNG riêng, thay vì một RNG chung cho cả vòng lặp:** nếu dùng chung một `random.Random(2026)` xuyên suốt, kết quả của cấu hình thứ 5 phụ thuộc vào việc 4 cấu hình trước đã "rút" bao nhiêu số ngẫu nhiên. Chỉ cần sau này bạn muốn sinh lại riêng cấu hình `A2_S3` để kiểm tra, kết quả sẽ khác — mất tính tái lập. Seed theo chuỗi tên cấu hình thì mỗi cấu hình độc lập hoàn toàn, sinh lại lúc nào cũng ra đúng kết quả cũ.

**Từ sau khi chạy file này: KHÔNG chạy lại**, trừ khi QC phát hiện lỗi >5% (khi đó sửa code rồi sinh lại toàn bộ). Commit file `test_attacks.csv` luôn.

## Ngày 9 — QC thủ công

**→ File:** `scripts/sample_for_qc.py` (tạo mới)

```python
"""Lấy 30 mẫu ngẫu nhiên cho mỗi cấu hình để kiểm tra chất lượng thủ công (§6.3)."""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load(open(ROOT / "configs/config.yaml", encoding="utf-8"))

df = pd.read_csv(ROOT / cfg["paths"]["perturbed"] / "test_attacks.csv")

qc = (df.groupby(["dataset", "attack_type", "severity"], group_keys=False)
        .apply(lambda g: g.sample(n=min(30, len(g)), random_state=2026)))

qc = qc[["sample_id", "dataset", "attack_type", "severity",
         "text_clean", "text_perturbed", "label", "edit_count"]]
qc["loi_doi_nhan"] = ""        # người kiểm điền: 1 = có lỗi, 0 = không
qc["loi_khong_doc_duoc"] = ""
qc["loi_sai_loai_bien_doi"] = ""
qc["ghi_chu"] = ""

out = ROOT / "reports" / "qc_samples.csv"
qc.to_csv(out, index=False, encoding="utf-8-sig")   # utf-8-sig để mở Excel không lỗi font
print(f"Đã xuất {len(qc)} mẫu QC vào {out}")
```

Mở `reports/qc_samples.csv` bằng Excel, hai người chia nhau đọc 540 mẫu, điền 3 cột lỗi. Sau đó tính tỷ lệ lỗi theo từng cấu hình — **nếu bất kỳ cấu hình nào vượt 5%**, sửa code rồi sinh lại toàn bộ cấu hình đó và QC lại.

## Ngày 10 — Bắt đầu M1 (TF-IDF + SVM)

**→ File:** `src/features/tfidf.py` và `src/models/svm_model.py`, script chạy ở `scripts/train_m1.py`

Điểm quan trọng nhất của ngày này, cũng là lỗi kinh điển dễ mắc:

```python
# ĐÚNG
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train["text_clean"])   # fit CHỈ trên train
X_dev   = vectorizer.transform(dev["text_clean"])          # transform, không fit lại
X_test  = vectorizer.transform(test["text_clean"])         # transform, không fit lại

# SAI — rò rỉ dữ liệu
X_all = vectorizer.fit_transform(all_data["text_clean"])   # từ vựng đã "nhìn thấy" test
```

**Vì sao:** `fit` sẽ học từ vựng và trọng số IDF từ dữ liệu. Nếu fit trên toàn bộ, mô hình gián tiếp biết được những từ nào xuất hiện trong tập test và chúng hiếm/phổ biến ra sao. Kết quả cao hơn thực tế, và §4.1 cấm rõ điều này. Đây là lỗi không báo lỗi, không crash, chỉ âm thầm làm sai toàn bộ kết luận.

Chọn `C` trong `{0.1, 1, 10}` bằng tập dev, không đụng test.

---

## Bảng tổng hợp file cần tạo trong 2 tuần

| Đường dẫn | Loại | Ngày |
|---|---|---|
| `configs/config.yaml` | cấu hình (cập nhật) | 1 |
| `notebooks/01_eda_vihsd.ipynb` | notebook (hoàn tất) | 1 |
| `notebooks/02_eda_victsd.ipynb` | notebook (mới) | 1 |
| `src/data/loaders.py` | module | 2 |
| `src/data/preprocess.py` | module | 2 |
| `scripts/build_manifest.py` | script chạy | 2 |
| `src/perturbation/base.py` | module | 3–5 |
| `src/perturbation/a1_accent.py` | module | 3–5 |
| `tests/test_perturbation.py` | test | 3–5 |
| `src/perturbation/a2_separator.py` | module | 6 |
| `src/perturbation/a3_qwerty.py` | module | 7 |
| `scripts/generate_attacks.py` | script chạy 1 lần | 8 |
| `scripts/sample_for_qc.py` | script chạy | 9 |
| `src/features/tfidf.py`, `src/models/svm_model.py`, `scripts/train_m1.py` | module + script | 10 |

**File dữ liệu sinh ra:**
- `data/processed/split_manifest.csv`, `test_clean.csv`, `all_clean.csv` (ngày 2)
- `data/perturbed/test_perturbed/test_attacks.csv` (ngày 8)
- `reports/qc_samples.csv` (ngày 9)

---

## Ba điều dễ sai nhất, tóm lại

1. **`fit` bộ vector hóa trên toàn bộ dữ liệu** thay vì chỉ trên train — rò rỉ dữ liệu, không có thông báo lỗi nào.
2. **Sinh lại `test_attacks.csv` nhiều lần** trong quá trình làm — mỗi mô hình sẽ bị đánh giá trên một tập khác nhau, phá vỡ so sánh theo cặp.
3. **Để tiền xử lý "dọn dẹp" văn bản quá tay** (bỏ dấu, co khoảng trắng) — xóa mất chính tín hiệu mà đề tài cần đo.