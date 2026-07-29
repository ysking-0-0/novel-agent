"""
novel_pipeline.utils
通用工具：从推理模型（MiniMax-M3 等）的混合输出中鲁棒提取 JSON。

MiniMax-M3 是推理模型，content 中常带思维链文本，纯 json.loads 会失败。
本模块提供 extract_json，能从「推理文本 + JSON」混合内容中提取最后一段 JSON。
"""
import json
import re
from typing import Any, Optional


def _find_last_json_block(text: str) -> Optional[str]:
    """从文本中提取最后一个 ```...``` 代码块内容（优先 json 标记）。"""
    # 找所有完整 ``` 围栏块，取最后一个非空的
    blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    for b in reversed(blocks):
        b = b.strip()
        if b:
            return b
    # 处理只有开头 ```json 但无配对结束 fence 的情况（输出被截断）
    # 从最后一个 ```json 之后取剩余内容作为候选
    m = None
    for m in re.finditer(r"```(?:json)?\s*", text):
        pass
    if m:
        tail = text[m.end():].strip()
        if tail and ('{' in tail or '[' in tail):
            return tail
    return None


def extract_json(content: Any, default: Any = None) -> Any:
    """从 LLM 输出中提取 JSON。

    支持：
      - content 已是 dict/list
      - content 是纯 JSON 字符串
      - content 是带思维链的混合文本（提取最后一段 JSON 数组或对象）
      - content 是 ```代码块包裹的 JSON

    Args:
        content: LLM 返回的 content（str / dict / list）
        default: 提取失败时返回的默认值

    Returns:
        解析后的 Python 对象（dict / list），或 default
    """
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return default

    text = content.strip()

    # 1. 先尝试整体解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 优先：提取最后一个 ``` 代码块内容（推理模型常把 JSON 放在最后的代码块）
    block = _find_last_json_block(text)
    if block:
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", block)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # 3. 无代码块时，从混合文本中提取最后一个 JSON 对象/数组
    #    用括号配平而非贪婪正则，避免思维链中的 { } 干扰
    candidate = _extract_balanced(text, "{", "}")
    if candidate is None:
        candidate = _extract_balanced(text, "[", "]")
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        # 4. 终极容错：MiniMax 有时把 JSON 结构分隔的字面 \n / \t 写成
        #    反斜杠+n/t（双字符），导致 json.loads 失败。在字符串引号之外
        #    把这些字面转义还原为真实字符再尝试。仅做最佳努力，不保证成功。
        repaired = _fix_literal_escapes_outside_strings(candidate)
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                try:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", repaired)
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

    return default


def _fix_literal_escapes_outside_strings(text: str) -> str:
    """把 JSON 字符串引号之外的字面 \\n / \\t / \\r 还原为真实字符。

    仅处理 \" 与字符串内部不触碰。状态机扫描: in_string 标记是否在
    双引号字符串内；遇 \" 反斜杠后跳过下一字符以正确处理内部转义。
    """
    out = []
    i = 0
    n = len(text)
    in_string = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == "\"":
                in_string = False
            i += 1
            continue
        # 引号外
        if ch == "\"":
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n and text[i + 1] in "ntr":
            out.append({"n": "\n", "t": "\t", "r": "\r"}[text[i + 1]])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[str]:
    """从 text 中提取最后一个括号配平的片段（贪婪向后）。

    从后往前找每个 close_ch，再向前配平到对应的 open_ch，
    返回这段子串。避免贪婪正则把思维链里的括号也算进去。
    """
    # 从最后一个 close_ch 开始向前配平
    depth = 0
    end = -1
    for i in range(len(text) - 1, -1, -1):
        if text[i] == close_ch:
            if depth == 0:
                end = i
            depth += 1
        elif text[i] == open_ch:
            depth -= 1
            if depth == 0 and end != -1:
                # 找到配平片段
                return text[i:end + 1]
    return None


def extract_json_list(content: Any) -> list:
    """专门提取 JSON 数组，失败返回 []。"""
    data = extract_json(content, default=[])
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("scenes", "data", "items", "list", "results"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    return []


def extract_json_dict(content: Any) -> Optional[dict]:
    """专门提取 JSON 对象，失败返回 None。"""
    data = extract_json(content, default=None)
    if isinstance(data, dict):
        return data
    return None
