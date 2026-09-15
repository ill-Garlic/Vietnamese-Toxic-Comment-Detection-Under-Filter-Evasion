# HƯỚNG DẪN CHI TIẾT TUẦN 1 & TUẦN 2
### Kế hoạch nén tiến độ HCMUE 1,5 tháng — chi tiết hóa đến từng ngày, kèm giải thích "vì sao"

Quy ước: mỗi ngày làm việc tính ~4–6 giờ. Nếu nhóm học song song với lịch học chính khóa, có thể co giãn số giờ nhưng **không được đảo thứ tự các đầu việc trong cùng một ngày/tuần**, vì lý do đã giải thích ở tài liệu trước (chống rò rỉ dữ liệu, tránh lan lỗi).

---

# TUẦN 1 — Dữ liệu, hạ tầng, bắt đầu mô-đun biến đổi

## Ngày 1 (thứ 2): Thu thập và kiểm tra dữ liệu thô

**Việc cần làm — Trần Kim Xuyến:**
1. Tải ViHSD và UIT-ViCTSD từ nguồn công bố chính thức của tác giả (không dùng bản sao chép không rõ nguồn gốc, vì cần biết chính xác không gian nhãn gốc).
2. Đọc dữ liệu vào DataFrame, kiểm tra:
   - Số dòng, số cột, tên cột.
   - Số bản ghi có nhãn rỗng/thiếu.
   - Số bản ghi có văn bản rỗng hoặc lỗi encoding.
   - Phân bố nhãn (CLEAN/OFFENSIVE/HATE cho ViHSD; TOXIC/NONE cho UIT-ViCTSD).
3. Kiểm tra trùng lặp/gần trùng (theo văn bản đã chuẩn hóa sơ bộ — chỉ để phát hiện, chưa xử lý vội).

```python
import pandas as pd

df = pd.read_csv("vihsd_raw.csv")
print(df.shape)
print(df.isnull().sum())
print(df["label"].value_counts())

# Phát hiện trùng lặp văn bản
dup_mask = df.duplicated(subset=["text"], keep=False)
print(f"Số bản ghi trùng văn bản: {dup_mask.sum()}")
```

**Việc cần làm — Nguyễn Minh Hiển:**
1. Dựng khung thư mục repo (đúng §14 của Quy trình v1.0):

```bash
mkdir -p project/{configs,data_manifests,attacks,models,train,evaluate,metrics,notebooks,figures,tables,logs,demo,paper_hcmue}
mkdir -p project/results/{clean,attacks,augmented}
cd project && git init
```

2. Tạo file `configs/protocol_lock.yaml` ghi lại các giá trị khởi tạo đã khóa ngay từ đầu, để không ai vô tình đổi sau này:

```yaml
split_seed: 42
model_seeds: [42, 43, 44]
attack_seed: 2026
split_ratio: [0.70, 0.10, 0.20]
```

**Giải thích vì sao làm ngày 1 thế này:**
- Kiểm tra dữ liệu thô *trước khi* làm bất cứ điều gì khác, vì nếu dữ liệu có lỗi nghiêm trọng (ví dụ file tải về bị lệch cột, nhãn map sai), phát hiện càng sớm càng đỡ phải làm lại các bước sau (đặc biệt là split và sample_id, vốn phải "đóng băng" ngay Tuần 1).
- Việc khóa các giá trị khởi tạo (seed) vào một file cấu hình ngay từ đầu, thay vì hardcode rải rác trong code, là để tránh tình huống về sau ai đó vô tình dùng seed khác nhau giữa các script → phá vỡ tính tái lập (yêu cầu SC6 và §8).

---

## Ngày 2 (thứ 3): Phân chia dữ liệu và gán sample_id

**Việc cần làm — Xuyến:**
1. Kiểm tra xem ViHSD và UIT-ViCTSD có phân chia train/dev/test chính thức đi kèm không (đọc lại paper gốc/repo gốc). Nếu có, dùng luôn — đây là lựa chọn ưu tiên theo §4.
2. Nếu không có phân chia chính thức phù hợp, thực hiện phân tầng 70/10/20 theo nhãn, seed=42:

```python
from sklearn.model_selection import train_test_split

train_val, test = train_test_split(
    df, test_size=0.20, stratify=df["label"], random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.125,  # 0.125 * 0.80 = 0.10 tổng thể
    stratify=train_val["label"], random_state=42
)
```

3. Gán `sample_id` ổn định (dùng chỉ số gốc hoặc hash nội dung, không dùng số thứ tự sẽ đổi nếu shuffle lại):

```python
import hashlib

def make_sample_id(dataset_name, original_index):
    raw = f"{dataset_name}_{original_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

df["sample_id"] = df.index.map(lambda i: make_sample_id("vihsd", i))
```

4. Xuất `split_manifest.csv` với đúng cột bắt buộc ở §4.2: `sample_id, dataset, split, label, original_index/hash`.
5. Xuất `test_clean.csv`: `sample_id, text_clean, label` — chỉ chứa các mẫu thuộc tập test.

**Lặp lại toàn bộ bước trên cho UIT-ViCTSD** — xử lý độc lập, không trộn hai bộ dữ liệu.

**Giải thích vì sao làm ngày 2 thế này:**
- `sample_id` phải được gán **trước khi** sinh bất kỳ phiên bản biến đổi nào, vì mọi phiên bản biến đổi của một mẫu (A1/A2/A3 × S1/S2/S3) phải giữ **cùng** sample_id với bản sạch gốc (yêu cầu ở §4.1) — đây là điều kiện để có thể so sánh theo cặp (paired comparison) ở bước phân tích thống kê sau này (§11).
- Dùng hash thay vì số thứ tự (index) đơn thuần để sample_id không bị đổi nếu file được đọc lại, sort lại, hay xử lý trên máy khác — đảm bảo tính tái lập (SC6).
- Việc xử lý 2 bộ dữ liệu hoàn toàn tách biệt (không dùng chung code trộn lẫn) là để tránh vô tình làm rò rỉ thông tin từ bộ này sang bộ kia — đúng nguyên tắc "hai bài toán độc lập" ở §4.

**Mốc kiểm tra cuối ngày 2 (SC1):** không có sample_id trùng giữa train/dev/test; phân bố nhãn ở mỗi tập hợp lý (gần với phân bố toàn bộ dữ liệu, sai lệch không quá vài phần trăm do stratify).

---

## Ngày 3 (thứ 4): Tiền xử lý tối thiểu + bắt đầu môi trường GPU

**Việc cần làm — Hiển:**
1. Viết hàm chuẩn hóa Unicode NFC:

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)
```

2. Viết hàm tiền xử lý tối thiểu theo đúng giới hạn ở §5 — **chỉ làm đúng những gì được phép, không làm thêm**:

```python
def minimal_preprocess(text: str) -> str:
    text = normalize_unicode(text)
    if not text.strip():
        return None  # đánh dấu để loại bỏ bản ghi rỗng
    return text
    # KHÔNG: xóa dấu câu, KHÔNG: bỏ dấu tiếng Việt,
    # KHÔNG: co khoảng trắng bất thường, KHÔNG: sửa chính tả,
    # KHÔNG: chuẩn hóa teencode — tất cả các thao tác này để dành
    # cho các mô-đun BIẾN ĐỔI A1–A3, không phải tiền xử lý chung.
```

3. Áp dụng lên `test_clean.csv` và tập train/dev, loại bản ghi rỗng, ghi log số lượng bị loại.

**Việc cần làm — Xuyến (song song):**
1. Thiết lập Colab Pro cho cả 2 tài khoản.
2. Test load thử PhoBERT-base, ViSoBERT, ViHateT5 từ HuggingFace (chỉ load, chưa fine-tune) để phát hiện sớm nếu có vấn đề về quyền truy cập, phiên bản thư viện, hoặc bộ tách từ riêng.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

for model_name in ["vinai/phobert-base", "uitnlp/visobert", "tarudesu/ViHateT5-base"]:
    tok = AutoTokenizer.from_pretrained(model_name)
    print(model_name, "OK, vocab size:", tok.vocab_size)
```

**Giải thích vì sao làm ngày 3 thế này:**
- Tiền xử lý phải **tối thiểu** vì đây là nghiên cứu về độ bền vững trước biến đổi né lọc — nếu tiền xử lý chung đã tự động bỏ dấu hay co khoảng trắng, thì đến khi mô-đun A1/A2 "cố ý" làm việc đó, sẽ không đo được đúng hiệu ứng, vì tín hiệu biến đổi đã bị pha loãng bởi bước tiền xử lý phía trước. Đây là lý do §5 nhấn rất rõ danh sách "KHÔNG được làm".
- Test load mô hình sớm (ngày 3, không phải đợi đến ngày cần fine-tune ở Tuần 3) để phát hiện sớm rủi ro về hạ tầng (tên model trên HuggingFace sai, thiếu quyền truy cập, xung đột phiên bản thư viện) khi vẫn còn nhiều thời gian dự phòng để xử lý — nếu đợi đến Tuần 3 mới phát hiện lỗi này, sẽ trực tiếp ăn vào thời gian fine-tune vốn đã eo hẹp.

---

## Ngày 4–5 (thứ 5–6): Bắt đầu mô-đun biến đổi A1

**Việc cần làm — Hiển (việc chính của 2 ngày này):**
1. Xác định "đơn vị từ đủ điều kiện" theo §6.2: ít nhất 3 ký tự chữ/số với A1, không phải URL/email/mention/hashtag.
2. Viết hàm bỏ dấu tiếng Việt (A1):

```python
import re
import unicodedata

def is_eligible_token_a1(token: str) -> bool:
    if re.match(r"(https?://|www\.|@|#)", token):
        return False
    alnum = re.sub(r"[^\w]", "", token, flags=re.UNICODE)
    return len(alnum) >= 3

def strip_vietnamese_accents(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

def apply_a1(text: str, ratio: float, rng) -> str:
    tokens = text.split()
    eligible_idx = [i for i, t in enumerate(tokens) if is_eligible_token_a1(t)]
    if not eligible_idx:
        return text  # attack_applied = 0, vẫn phải ghi log riêng
    n_to_perturb = max(1, -(-len(eligible_idx) * ratio // 1))  # làm tròn lên, tối thiểu 1
    chosen = rng.sample(eligible_idx, k=min(int(n_to_perturb), len(eligible_idx)))
    for i in chosen:
        tokens[i] = strip_vietnamese_accents(tokens[i])
    return " ".join(tokens)
```

3. Test nhanh trên vài chục câu mẫu thủ công (chưa phải QC chính thức — chỉ để tự kiểm tra code chạy đúng cú pháp).
4. Bắt đầu viết khung hàm chung cho A2, A3 (chưa cần hoàn thiện logic, chỉ dựng interface thống nhất) để sang Tuần 2 lắp tiếp nhanh hơn.

**Giải thích vì sao làm ngày 4–5 thế này:**
- Bắt đầu A1 (đơn giản nhất trong 3 loại biến đổi — chỉ cần bỏ dấu, không cần bảng ánh xạ bàn phím như A3, không cần tập ký tự phân cách như A2) trước, để có một phiên bản chạy được sớm, dễ debug logic chung (cách chọn từ đủ điều kiện, cách tính tỷ lệ theo mức độ S1/S2/S3, cách ghi log) — logic này sẽ dùng lại gần như nguyên vẹn cho A2, A3 ở Tuần 2.
- `rng.sample` dùng bộ sinh số ngẫu nhiên **riêng** (không dùng chung random state với việc chia dữ liệu) và sẽ được seed bằng `attack_seed=2026` khi sinh tập kiểm thử chính thức ở Tuần 2 — nhưng ở giai đoạn viết code này (ngày 4–5) chưa cần cố định seed, vì đây chỉ là bước phát triển/debug, chưa phải bước sinh tập kiểm thử chính thức (việc đó chỉ làm **một lần, cố định** ở Tuần 2 theo đúng §6.2).

**Mốc cuối Tuần 1:** `split_manifest.csv` và `test_clean.csv` khóa cho cả 2 bộ dữ liệu; môi trường GPU đã test chạy được; A1 có bản chạy được (chưa QC chính thức).

---

# TUẦN 2 — Hoàn thiện biến đổi, QC, bắt đầu train mô hình cổ điển

## Ngày 6 (thứ 2): Hoàn thiện A2 (chèn khoảng trắng/dấu câu)

**Việc cần làm — Hiển:**

```python
SEPARATORS = [" ", ".", "-", "_"]

def is_eligible_token_a2(token: str) -> bool:
    if re.match(r"(https?://|www\.|@|#)", token):
        return False
    alnum = re.sub(r"[^\w]", "", token, flags=re.UNICODE)
    return len(alnum) >= 4  # A2 yêu cầu tối thiểu 4 ký tự theo §6.2

def apply_a2(token: str, rng) -> str:
    if len(token) < 2:
        return token
    insert_pos = rng.randint(1, len(token) - 1)
    sep = rng.choice(SEPARATORS)
    return token[:insert_pos] + sep + token[insert_pos:]
```

- Áp dụng cùng logic chọn tỷ lệ theo mức độ S1/S2/S3 đã viết ở A1 (tái sử dụng hàm chung, chỉ đổi `is_eligible_token` và hàm biến đổi từng token).

## Ngày 7 (thứ 3): Hoàn thiện A3 (lỗi gõ QWERTY)

**Việc cần làm — Hiển:**
1. Xây bảng ánh xạ ký tự lân cận trên bàn phím QWERTY tiếng Việt (bao gồm cả chữ có dấu, vì văn bản gốc trước khi bị A3 tác động vẫn có thể còn dấu — A3 độc lập với A1).
2. Nếu một ký tự không có ánh xạ hợp lệ (ví dụ ký tự đặc biệt), **không áp dụng** biến đổi lên token đó — đúng quy định ở §6 (mã A3: "không áp dụng nếu không có ánh xạ hợp lệ").

```python
QWERTY_NEIGHBORS = {
    "a": ["q", "w", "s", "z"],
    "s": ["a", "w", "e", "d", "x", "z"],
    # ... xây đầy đủ cho toàn bộ bảng chữ cái, xử lý riêng ký tự có dấu
}

def apply_a3(token: str, rng) -> str:
    letters_idx = [i for i, c in enumerate(token) if c.lower() in QWERTY_NEIGHBORS]
    if not letters_idx:
        return token  # không có ánh xạ hợp lệ -> không áp dụng
    i = rng.choice(letters_idx)
    c = token[i]
    neighbor = rng.choice(QWERTY_NEIGHBORS[c.lower()])
    replacement = neighbor.upper() if c.isupper() else neighbor
    return token[:i] + replacement + token[i+1:]
```

**Giải thích vì sao tách A2 và A3 ra 2 ngày riêng thay vì làm cùng lúc A1–A3 trong 1 ngày:**
- Mỗi loại biến đổi có một điều kiện "đủ điều kiện" khác nhau (A1/A3: ≥3 ký tự; A2: ≥4 ký tự) và logic áp dụng khác nhau — dồn cả 3 vào 1 ngày dễ nhầm lẫn điều kiện giữa các loại (ví dụ lỡ dùng ngưỡng 3 ký tự cho A2). Tách riêng từng ngày giúp test kỹ từng hàm trước khi ghép chung, giảm rủi ro lỗi hệ thống mà QC ở ngày 8–9 mới phát hiện ra — lúc đó sửa sẽ tốn thời gian hơn.

## Ngày 8 (thứ 4): Sinh tập kiểm thử biến đổi chính thức + bắt đầu QC

**Việc cần làm — Hiển:**
1. Cố định `attack_seed = 2026`, sinh **một lần duy nhất** toàn bộ tập biến đổi kiểm thử cho cả 2 bộ dữ liệu × 3 dạng × 3 mức = 18 cấu hình:

```python
import random

rng = random.Random(2026)  # attack_seed cố định

for dataset in ["vihsd", "victsd"]:
    for attack_type in ["A1", "A2", "A3"]:
        for severity, ratio in [("S1", 0.10), ("S2", 0.20), ("S3", 0.30)]:
            # sinh và lưu vào test_attacks.csv với đầy đủ cột:
            # sample_id, text_clean, text_perturbed, attack_type,
            # severity, attack_seed, edit_count
            ...
```

2. **Từ thời điểm này, tuyệt đối không sinh lại tập biến đổi kiểm thử** trừ khi QC phát hiện lỗi >5% (khi đó sửa code rồi sinh lại toàn bộ, không sinh lại riêng lẻ từng phần).
3. Bắt đầu lấy mẫu ngẫu nhiên 30 mẫu/cấu hình để chuẩn bị QC (18 cấu hình × 30 = 540 mẫu).

**Giải thích:**
- "Sinh một lần, cố định" (§6.2, §12.1 bước 4) là điều kiện bắt buộc để **mọi mô hình đều được đánh giá trên đúng cùng một tập biến đổi** — nếu mỗi lần đánh giá một mô hình lại sinh lại tập biến đổi (dù cùng seed nhưng có thể lệch do thứ tự gọi hàm ngẫu nhiên khác nhau trong code), so sánh giữa các mô hình sẽ không còn là so sánh theo cặp (paired) hợp lệ nữa.

## Ngày 9 (thứ 5): Hoàn tất QC thủ công

**Việc cần làm — cả nhóm (chia đôi 540 mẫu, mỗi người đọc ~270 mẫu):**

Với mỗi mẫu, kiểm tra 3 tiêu chí ở §6.3:
1. Nhãn có bị đổi một cách hiển nhiên không (đọc câu, tự hỏi: nếu là người đọc bình thường, nhãn gốc có còn hợp lý không).
2. Văn bản sau biến đổi có còn đọc/diễn giải được không (không bị vỡ nát đến mức vô nghĩa).
3. Biến đổi có đúng loại đã khai báo và đúng tỷ lệ trong sai số làm tròn không.

Ghi kết quả vào bảng theo cấu hình:

| dataset | attack_type | severity | n_checked | n_error | error_rate |
|---|---|---|---|---|---|
| vihsd | A1 | S1 | 30 | ? | ?% |
| ... | | | | | |

**Nếu error_rate > 5% ở bất kỳ dòng nào** → quay lại sửa hàm sinh biến đổi tương ứng, sinh lại **toàn bộ** cấu hình đó (không chỉ 30 mẫu đã QC), rồi QC lại từ đầu cho cấu hình đó.

**Giải thích vì sao QC làm thủ công, không tự động hóa hoàn toàn:**
- Hai tiêu chí đầu (đổi nhãn hiển nhiên, còn đọc được) là phán đoán ngữ nghĩa mà một script khó đánh giá đáng tin cậy — cần con người đọc trực tiếp. Đây là lý do §6.3 nói rõ đây là "kiểm tra chất lượng triển khai" (implementation QC), không phải nghiên cứu có người tham gia quy mô lớn — chỉ cần 30 mẫu/cấu hình là đủ để phát hiện lỗi hệ thống (bug), không nhằm đo lường ý kiến đa dạng của nhiều người đọc.

## Ngày 10 (thứ 6): Bắt đầu train M1, M2 trên dữ liệu sạch

**Việc cần làm — Xuyến:**
1. M1 — TF-IDF + SVM tuyến tính, n-gram 1–2, chọn C trên tập dev từ {0.1, 1, 10}:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train["text_clean"])  # fit CHỈ trên train
X_dev = vectorizer.transform(dev["text_clean"])           # transform, không fit lại

param_grid = {"C": [0.1, 1, 10]}
grid = GridSearchCV(LinearSVC(class_weight="balanced"), param_grid, scoring="f1_macro", cv=3)
grid.fit(X_train, train["label"])
best_C = grid.best_params_["C"]
```

2. M2 — FastText embedding + TextCNN, filter size 3/4/5, dropout 0.5, early stopping theo Macro-F1 trên dev. Bắt đầu code khung train, chạy thử nghiệm nhỏ (chưa cần hoàn tất trong ngày 10, có thể tràn sang đầu Tuần 3 nếu cần, vì Tuần 3 vẫn còn dư việc train M3–M5 song song).

**Giải thích:**
- `vectorizer.fit_transform` chỉ gọi trên `train`, còn `dev`/`test` chỉ dùng `.transform()` — đây là cách hiện thực hóa cụ thể của quy tắc "mọi tiền xử lý có học tham số từ dữ liệu ... chỉ được ước lượng trên tập huấn luyện" (§4.1). Đây là lỗi rất dễ mắc phải (fit trên toàn bộ dữ liệu trước khi split) và một khi mắc phải sẽ làm rò rỉ thông tin từ dev/test vào mô hình mà không ai nhận ra ngay.
- Chọn siêu tham số C chỉ dựa trên tập dev (không đụng vào test) — đúng nguyên tắc ở §7.1.

**Mốc cuối Tuần 2 (SC3 đạt):** `test_attacks.csv` khóa với error_rate ≤5% ở tất cả 18 cấu hình; M1 đã có kết quả trên dev (đang chọn C); M2 đã có khung code chạy được, sẵn sàng hoàn tất đầu Tuần 3.

---

## Bảng tổng hợp nhanh 2 tuần

| Ngày | Đầu việc chính | Vì sao đặt ở đây (không sớm/muộn hơn) |
|---|---|---|
| 1 | Kiểm tra dữ liệu thô + dựng repo | Phát hiện lỗi dữ liệu sớm nhất có thể, trước khi mọi thứ khác phụ thuộc vào nó |
| 2 | Split + sample_id | Phải xong trước khi sinh biến đổi (biến đổi cần dùng lại sample_id) |
| 3 | Tiền xử lý tối thiểu + test GPU | Test hạ tầng sớm để còn thời gian xử lý nếu có sự cố, trước khi Tuần 3 cần dùng thật |
| 4–5 | Code A1 + khung A2/A3 | Bắt đầu phần dễ nhất để hoàn thiện logic chung, tái dùng cho A2/A3 |
| 6–7 | Hoàn thiện A2, A3 | Tách riêng từng loại để tránh nhầm điều kiện đủ-điều-kiện giữa các loại |
| 8 | Sinh tập biến đổi kiểm thử chính thức (1 lần, cố định) | Phải xong trước khi có thể QC và trước khi Tuần 4 cần dùng để đánh giá |
| 9 | QC thủ công 540 mẫu | Bắt buộc trước khi tin dùng tập biến đổi cho toàn bộ Tuần 4–5 |
| 10 | Train M1, bắt đầu M2 | Chạy song song với QC/hoàn thiện biến đổi vì không phụ thuộc lẫn nhau |

Nếu có ngày nào bị trễ (ví dụ QC phát hiện lỗi phải sinh lại), ưu tiên bù giờ vào cuối Tuần 2/đầu Tuần 3 trước khi đụng đến lịch fine-tune Transformer — vì đó là phần ít có dư địa co giãn nhất trong toàn bộ 6 tuần (đã giải thích ở tài liệu "vì sao" trước, mục 4).
