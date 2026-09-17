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