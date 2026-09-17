from src.perturbation.a1_accent import strip_accents_token


def test_strip_accents():
    assert strip_accents_token("tiếng") == "tieng"
    assert strip_accents_token("Đồ") == "Do"
    assert strip_accents_token("abc123") == "abc123"
    assert strip_accents_token("😀") == "😀"