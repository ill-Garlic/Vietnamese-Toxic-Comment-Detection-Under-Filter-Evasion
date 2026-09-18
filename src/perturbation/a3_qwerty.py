"""A3 — Lỗi gõ QWERTY: thay 1 ký tự Latin bằng ký tự lân cận trên bàn phím (§6).

Lưu ý quan trọng (cần nêu trong báo cáo):
    Ký tự tiếng Việt có dấu (ế, ơ, ạ, …) KHÔNG nằm trong bảng QWERTY_NEIGHBORS
    nên A3 chỉ tác động lên ký tự Latin thuần. Điều này làm A3 "nhẹ" hơn A1/A2
    trên văn bản tiếng Việt → cần giải thích rõ trong phần Phương pháp.
"""
from .base import perturb_sentence

# Bàn phím QWERTY chuẩn US — chỉ ánh xạ ký tự thường, case sẽ được giữ lại
QWERTY_NEIGHBORS: dict = {
    "q": "wa",    "w": "qes",   "e": "wrd",   "r": "etf",   "t": "ryg",
    "y": "tuh",   "u": "yij",   "i": "uok",   "o": "ipl",   "p": "o",
    "a": "qsz",   "s": "awdx",  "d": "sefc",  "f": "drgv",  "g": "fthb",
    "h": "gyjn",  "j": "hukm",  "k": "jil",   "l": "ko",
    "z": "asx",   "x": "zsdc",  "c": "xdfv",  "v": "cfgb",  "b": "vghn",
    "n": "bhjm",  "m": "njk",
}


def qwerty_typo_token(token: str, rng) -> str:
    """Thay đúng 1 ký tự Latin bằng ký tự lân cận trên bàn phím QWERTY.

    - Nếu không có ký tự Latin nào trong token → giữ nguyên (attack_applied = 0).
    - Giữ nguyên hoa/thường của ký tự bị thay.
    - Ví dụ: "hate" → "hате" (t → y, lân cận trên QWERTY)
    """
    candidates = [i for i, c in enumerate(token) if c.lower() in QWERTY_NEIGHBORS]
    if not candidates:
        return token  # không có ánh xạ hợp lệ → giữ nguyên

    i = rng.choice(candidates)
    c = token[i]
    replacement = rng.choice(QWERTY_NEIGHBORS[c.lower()])
    # Giữ nguyên case
    replacement = replacement.upper() if c.isupper() else replacement
    return token[:i] + replacement + token[i + 1:]


def apply_a3(text: str, ratio: float, rng, min_len: int = 3):
    """Áp dụng biến đổi A3 lên toàn câu.

    Args:
        text:    câu đầu vào (đã qua minimal_preprocess)
        ratio:   tỷ lệ token bị biến đổi (S1=0.10, S2=0.20, S3=0.30)
        rng:     random.Random đã được seed — KHÔNG dùng random module trực tiếp
        min_len: token tối thiểu 3 ký tự (§6.2)

    Returns:
        (text_perturbed, edit_count): câu đã biến đổi và số token thực sự bị sửa
    """
    return perturb_sentence(text, ratio, min_len, qwerty_typo_token, rng)
