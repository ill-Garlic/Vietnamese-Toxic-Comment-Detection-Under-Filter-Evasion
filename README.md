# Phát hiện Bình luận Độc hại Tiếng Việt trong Điều kiện Né tránh Bộ lọc

Nghiên cứu về độ bền vững (robustness) của các mô hình phân loại bình luận độc hại tiếng Việt (ViHSD) trước các kỹ thuật né tránh bộ lọc phổ biến: bỏ dấu, teencode, và chèn nhiễu ký tự.

## Mục tiêu

1. Xây dựng baseline (SVM, PhoBERT) trên dữ liệu ViHSD gốc.
2. Sinh dữ liệu perturbation (bỏ dấu / teencode / nhiễu ký tự) từ tập train để làm data augmentation, và từ tập test để làm bộ đánh giá robustness.
3. So sánh mức độ suy giảm hiệu năng (performance degradation) giữa mô hình train trên dữ liệu gốc và mô hình train trên dữ liệu đã augment.

## Cấu trúc dự án

```
data/           # raw / interim / processed / perturbed
notebooks/      # EDA, phân tích perturbation, phân tích lỗi
src/
  data/         # load, preprocess, split
  perturbation/ # các hàm sinh nhiễu + orchestrator
  features/     # TF-IDF, tokenizer PhoBERT
  models/       # SVM, PhoBERT, train/predict
  evaluation/   # metrics, robustness_eval, visualize
models/         # checkpoint đã train
reports/        # confusion matrix, bảng kết quả tổng hợp
scripts/        # entrypoint chạy pipeline qua CLI
tests/          # unit test
```

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách chạy pipeline

```bash
# 1. Tiền xử lý + chia train/dev/test
bash scripts/run_preprocessing.sh

# 2. Sinh dữ liệu perturbation (augmentation cho train, eval set cho test)
bash scripts/run_perturbation.sh

# 3. Train baseline trên dữ liệu gốc
bash scripts/run_train_baseline.sh

# 4. Train lại trên dữ liệu augmented
bash scripts/run_train_augmented.sh

# 5. Đánh giá + xuất confusion matrix, bảng so sánh degradation
bash scripts/run_evaluation.sh
```

## Dữ liệu

Dự án sử dụng bộ dữ liệu **ViHSD** (Vietnamese Hate Speech Detection). Do vấn đề bản quyền / dung lượng, dữ liệu gốc **không được commit** lên repo — vui lòng tải về và đặt vào `data/raw/`.

## Kết quả

Bảng tổng hợp F1-score và % performance degradation giữa baseline và augmented model được lưu tại `reports/results.csv`, biểu đồ confusion matrix tại `reports/figures/`.

## License

## Quy ước nhãn

- **ViHSD**: 0 = CLEAN, 1 = OFFENSIVE, 2 = HATE
- **UIT-ViCTSD**: 0 = NONE, 1 = TOXIC
- Nhóm "độc hại" dùng cho các chỉ số Recall/ASR: ViHSD = {1, 2}; ViCTSD = {1}
  (xem `configs/config.yaml` → `harmful_mapping`)