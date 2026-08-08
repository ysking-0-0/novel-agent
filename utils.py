"""
novel_pipeline.utils
通用工具：从推理模型（MiniMax-M3 等）的混合输出中鲁棒提取 JSON。

MiniMax-M3 是推理模型，content 中常带思维链文本，纯 json.loads 会失败。
本模块提供 extract_json，能从「推理文本 + JSON」混合内容中提取最后一段 JSON。
"""
import json
import re
from typing import Any, Optional


def _strip_think_block(text: str) -> str:
    """剥离模型输出的 <think>...</think> 思维链块（MiniMax-M2.7 等推理模型常见）。

    只剥离成对出现的 <think> ... </think>；标签不闭合时不剥离（可能输出被截断，
    此时保留原文本，后续截断恢复逻辑处理）。返回剥离后的文本。
    """
    if "<think>" not in text and "</think>" not in text:
        return text
    # 成对剥离：循环处理直到没有成对标签
    out = text
    while True:
        m = re.search(r"<think>([\s\S]*?)</think>", out)
        if not m:
            break
        out = out[: m.start()] + out[m.end():]
    return out.strip()


def _recover_truncated_json(text: str) -> Optional[str]:
    """尝试从被截断的 JSON 文本恢复：补齐缺失的右括号/引号。

    MiniMax 推理模型输出超长思维链时，真实 JSON 常被 max_tokens 截断——
    缺少右括号、字符串未闭合、数组/对象不完整。这里用括号配平扫描，
    找到第一个未闭合的 { 或 [，向后补齐对应的 ] / }，同时补上末尾未闭合的字符串引号。
    仅做最佳努力，失败返回 None。
    """
    # 先看整体是否已配平
    if _is_balanced_json(text):
        return text
    # 从最后一个 { 或 [ 开始向后补齐（真实 JSON 通常在末尾；
    # 思维链/正文里的伪 JSON 片段靠"取最长恢复结果"跳过）
    candidates = []
    for i, ch in enumerate(text):
        if ch in "{[":
            candidates.append(i)
    if not candidates:
        return None
    best = None
    for start in reversed(candidates):
        body = text[start:]
        # 栈式扫描，补齐闭合
        stack = []
        in_str = False
        esc = False
        i = 0
        n = len(body)
        repaired = []
        ok = True
        while i < n:
            ch = body[i]
            repaired.append(ch)
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                i += 1
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
                else:
                    # 多余的闭合括号：该起点无效，放弃
                    ok = False
                    break
            i += 1
        if not ok:
            continue
        # 字符串未闭合 → 补引号
        if in_str:
            repaired.append('"')
        # 补齐未闭合的括号（反序）
        for ch in reversed(stack):
            repaired.append("}" if ch == "{" else "]")
        candidate = "".join(repaired)
        if _is_balanced_json(candidate):
            if best is None or len(candidate) > len(best):
                best = candidate
    return best


def _is_balanced_json(text: str) -> bool:
    """粗略检查 JSON 文本括号配平且字符串闭合（不做完整语法校验）。"""
    stack = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return False
            if (ch == "}" and stack[-1] != "{") or (ch == "]" and stack[-1] != "["):
                return False
            stack.pop()
    return not in_str and not stack


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

    # 0. 剥离 <think>...</think> 思维链块（推理模型常输出，干扰 JSON 提取）
    stripped = _strip_think_block(text)
    if stripped != text:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

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

    # 4. 截断恢复优先：JSON 被 max_tokens 截断时，整体补齐右括号/引号再尝试。
    #    必须放在 candidate 之前——_extract_balanced 在截断的数组/对象未闭合时，
    #    可能只提取到内部片段（如 tts_meta 里的 {..}），整体恢复能得到更完整的 JSON。
    for base in (stripped, text):
        if not base:
            continue
        recovered = _recover_truncated_json(base)
        if recovered is None:
            continue
        try:
            return json.loads(recovered)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", recovered)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 5. 再尝试 candidate 本身（可能内部片段，也可能是完整对象）
    if candidate is not None:
        # 5.0 候选可能是"截断 JSON 的内部片段"（外层被截断，内层数组/对象配平）。
        #     尝试从候选起始位置向前扩展出真正的外层，再整体做截断恢复。
        outer = _expand_to_outer_json(text, candidate)
        if outer is not None:
            recovered = _recover_truncated_json(outer)
            if recovered is not None:
                try:
                    return json.loads(recovered)
                except json.JSONDecodeError:
                    cleaned = re.sub(r",\s*([}\]])", r"\1", recovered)
                    try:
                        return json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        # 终极容错：MiniMax 有时把 JSON 结构分隔的字面 \n / \t 写成
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


def _expand_to_outer_json(text: str, candidate: str) -> Optional[str]:
    """从括号配平候选的内部片段，向前扩展出真正的外层 JSON 起始。

    场景：截断的 JSON 中内层数组/对象（如 tts_meta 里的 {..}）恰好配平，
    _extract_balanced 会提取到它而非外层。此函数从 candidate 在 text 中的
    起始位置向前找最近的外层 { 或 [（该位置之前的括号深度应比 candidate 起始处少 1），
    返回 text[outer_start:]（保留截断的尾部，由 _recover_truncated_json 补齐）。
    找不到更外层时返回 None。
    """
    # 找到 candidate 在 text 中的起始位置（可能有多处，取最后一个）
    start = text.rfind(candidate)
    if start <= 0:
        return None
    prefix = text[:start]
    # 计算 prefix 中的括号深度：遇到 { [ 深度+1，遇到 } ] 深度-1（忽略字符串内）
    depth = 0
    in_str = False
    esc = False
    # 从后往前找：深度比 candidate 起始处少 1 且字符为 { 或 [ 的位置
    target = 0
    # 先算 candidate 起始处的深度
    d_start = 0
    for ch in prefix:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            d_start += 1
        elif ch in "}]":
            d_start -= 1
    # 从后往前找使深度降到 d_start-1 的 { 或 [
    depth = d_start
    in_str = False
    esc = False
    for i in range(len(prefix) - 1, -1, -1):
        ch = prefix[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "}]":
            depth += 1
        elif ch in "{[":
            depth -= 1
            if depth == d_start - 1:
                return text[i:]  # 从该外层起始保留到尾
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
