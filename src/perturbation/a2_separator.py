"""A2 — Chèn khoảng trắng / dấu câu vào GIỮA token (§6).

Dùng chung khung perturb_sentence từ base.py.
min_len = 4 vì A2 yêu cầu token ít nhất 4 ký tự (§6.2).
"""
from .base import perturb_sentence

SEPARATORS = [" ", ".", "-", "_"]


def insert_separator_token(token: str, rng) -> str:
    """Chèn 1 ký tự phân cách vào GIỮA token (không chèn ở đầu/cuối).

    Ví dụ: "chết" → "ch.et", "thằng" → "tha_ng"
    """
    if len(token) < 2:
        return token
    pos = rng.randint(1, len(token) - 1)
    sep = rng.choice(SEPARATORS)
    return token[:pos] + sep + token[pos:]


def apply_a2(text: str, ratio: float, rng, min_len: int = 4):
    """Áp dụng biến đổi A2 lên toàn câu.

    Args:
        text:    câu đầu vào (đã qua minimal_preprocess)
        ratio:   tỷ lệ token bị biến đổi (S1=0.10, S2=0.20, S3=0.30)
        rng:     random.Random đã được seed — KHÔNG dùng random module trực tiếp
        min_len: token tối thiểu 4 ký tự (§6.2, khác A1/A3 là 3)

    Returns:
        (text_perturbed, edit_count): câu đã biến đổi và số token thực sự bị sửa
    """
    return perturb_sentence(text, ratio, min_len, insert_separator_token, rng)
