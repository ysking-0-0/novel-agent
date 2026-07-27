"""
novel_pipeline.nodes.text_chunker
文本分片节点（纯代码）—— 主循环步骤 1。
对应设计文档 5.2。

职责：
  从 offset 位置读取文本 → 自动对齐章节边界 → 生成 current_chunk
  读到文末标记 loop_finished=True
"""
import re
from typing import Dict
from state import NovelState


# 常见章节标题正则（适配中文小说）
CHAPTER_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千零0-9]+章\b", re.M),
    re.compile(r"^第[一二三四五六七八九十百千零0-9]+回\b", re.M),
    re.compile(r"^Chapter\s+\d+", re.M | re.I),
    re.compile(r"^卷[一二三四五六七八九十0-9]+\b", re.M),
    re.compile(r"^正文\s+", re.M),
]


def _find_next_chapter_boundary(text: str) -> int:
    """在 text 中找到下一个章节标题位置（跳过开头，避免截到当前章）。"""
    for pat in CHAPTER_PATTERNS:
        matches = list(pat.finditer(text))
        if len(matches) > 1:
            return matches[1].start()
    # 找不到明确章节标题，按段落边界对齐
    # 在后 1/3 区域找最后一个换行
    if len(text) < 100:
        return len(text)
    cut = int(len(text) * 0.8)
    nl = text.rfind("\n", 0, cut)
    return nl + 1 if nl > 0 else len(text)


def text_chunker_node(state: Dict) -> Dict:
    """读取下一段文本，对齐章节边界，更新 offset（按字节偏移，二进制读取避免字符截断）。"""
    file_path: str = state.get("file_path", "")
    offset: int = state.get("offset", 0)
    chunk_size: int = state.get("chunk_size", 8000)

    if not file_path:
        return {"current_chunk": "", "loop_finished": True}

    try:
        with open(file_path, "rb") as f:
            f.seek(offset)
            raw_bytes = f.read(chunk_size + 2000)  # 多读一截用于对齐
    except FileNotFoundError:
        return {"current_chunk": "", "loop_finished": True}

    if not raw_bytes:
        return {"current_chunk": "", "loop_finished": True}

    raw = raw_bytes.decode("utf-8", errors="ignore")

    # 对齐章节边界
    boundary = _find_next_chapter_boundary(raw)
    chunk = raw[:boundary] if boundary < len(raw) else raw
    new_offset = offset + len(chunk.encode("utf-8"))

    # 判断是否真正读到文件末尾
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
    finished = new_offset >= file_size

    return {
        "current_chunk": chunk,
        "offset": new_offset,
        "loop_finished": bool(finished),
    }
