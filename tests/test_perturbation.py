"""Unit tests cho toàn bộ mô-đun perturbation (A1, A2, A3).

Chạy: python -m pytest tests/test_perturbation.py -v
"""
import random

from src.perturbation.a1_accent import strip_accents_token, apply_a1
from src.perturbation.a2_separator import insert_separator_token, apply_a2
from src.perturbation.a3_qwerty import qwerty_typo_token, apply_a3

# ══════════════════════════ A1 — Bỏ dấu ══════════════════════════════════════

def test_a1_basic():
    assert strip_accents_token("tiếng") == "tieng"

def test_a1_uppercase():
    assert strip_accents_token("Đồ") == "Do"

def test_a1_no_accent():
    assert strip_accents_token("abc123") == "abc123"

def test_a1_emoji_unchanged():
    assert strip_accents_token("😀") == "😀"

def test_a1_d_special():
    """đ/Đ phải được xử lý riêng vì không tách được bằng NFD."""
    assert strip_accents_token("đường") == "duong"
    assert strip_accents_token("Đại") == "Dai"

def test_a1_sentence():
    rng = random.Random(42)
    result, edits = apply_a1("thằng này nói chuyện xấu quá", ratio=0.5, rng=rng)
    assert isinstance(result, str)
    assert edits >= 0

def test_a1_empty():
    rng = random.Random(42)
    result, edits = apply_a1("", ratio=0.5, rng=rng)
    assert result == "" and edits == 0

# ══════════════════════════ A2 — Chèn dấu câu ════════════════════════════════

def test_a2_insert_adds_one_char():
    """Sau khi chèn, token phải dài hơn đúng 1 ký tự."""
    rng = random.Random(1)
    result = insert_separator_token("chuyện", rng)
    assert len(result) == len("chuyện") + 1

def test_a2_short_token_unchanged():
    """Token 1 ký tự → giữ nguyên."""
    assert insert_separator_token("a", random.Random(1)) == "a"

def test_a2_separator_not_at_edges():
    """Ký tự chèn không được ở vị trí đầu hoặc cuối."""
    rng = random.Random(7)
    token = "toxicc"
    result = insert_separator_token(token, rng)
    assert result[0] == token[0]
    assert result[-1] == token[-1]

def test_a2_sentence():
    rng = random.Random(42)
    result, edits = apply_a2("thằng này nói chuyện xấu quá đây", ratio=0.5, rng=rng)
    assert isinstance(result, str) and isinstance(edits, int)

def test_a2_min_len4():
    """Token < 4 ký tự không bị chọn khi min_len=4."""
    rng = random.Random(42)
    # "tôi đi" — cả 2 token ngắn hơn 4 ký tự alnum
    result, edits = apply_a2("tôi đi", ratio=1.0, rng=rng, min_len=4)
    assert edits == 0

# ══════════════════════════ A3 — QWERTY typo ══════════════════════════════════

def test_a3_replaces_exactly_one_char():
    """Đúng 1 ký tự Latin bị thay, độ dài không đổi."""
    rng = random.Random(5)
    token = "hate"
    result = qwerty_typo_token(token, rng)
    assert len(result) == len(token)
    diffs = sum(a != b for a, b in zip(token, result))
    assert diffs == 1

def test_a3_pure_diacritic_unchanged():
    """Token không có ký tự Latin trong QWERTY map → giữ nguyên."""
    # "ằổ" chỉ chứa ký tự Vietnamese diacritic, không có Latin
    result = qwerty_typo_token("ằổ", random.Random(5))
    assert result == "ằổ"

def test_a3_preserves_case():
    """Hoa → thay bằng hoa; thường → thường."""
    rng = random.Random(0)
    token = "Hate"
    result = qwerty_typo_token(token, rng)
    for orig, repl in zip(token, result):
        if orig != repl:
            assert orig.isupper() == repl.isupper()

def test_a3_sentence():
    rng = random.Random(42)
    result, edits = apply_a3("hate speech is bad here now", ratio=0.5, rng=rng)
    assert isinstance(result, str) and edits >= 0

# ══════════════════════════ Tính tái lập (seed) ══════════════════════════════

def test_same_seed_same_result():
    text = "bình luận độc hại này phải bị xử lý ngay"
    r1, e1 = apply_a1(text, 0.3, random.Random(2026))
    r2, e2 = apply_a1(text, 0.3, random.Random(2026))
    assert r1 == r2 and e1 == e2

def test_different_seed_different_result():
    text = "bình luận độc hại này phải bị xử lý ngay thôi đấy"
    r1, _ = apply_a1(text, 0.5, random.Random(1))
    r2, _ = apply_a1(text, 0.5, random.Random(999))
    assert r1 != r2