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