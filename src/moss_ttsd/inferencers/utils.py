import re

def keep_zh_en_space(text: str) -> str:
    """
    保留汉字、英文字母和空格；移除其他字符（含标点、数字、emoji 等）。
    会把连续空格压成一个空格，并去掉首尾空格。
    """
    if text is None:
        return ""
    s = re.sub(r"[^A-Za-z\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", s).strip()
