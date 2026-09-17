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