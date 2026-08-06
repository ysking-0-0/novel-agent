"""
novel_pipeline.main_supervisor
监督循环：包住 main.run()，出错时用 LLM 分析并自动修复。
能力边界：仅自动重试（断点续跑）+ 调整运行参数（chunk-size / max-retries），不改代码。
"""
import json
import re
from typing import Callable, Dict, Optional

from llm_factory import get_llm
from config import get_config

# 每轮最多自动修复轮数，防死循环
MAX_AUTO_FIX_ROUNDS = 3


class PipelineError(Exception):
    """携带运行上下文的流水线异常。

    trace: 完整 traceback 文本（供 LLM 分析）
    context: 运行状态快照 {done, offset, file_path, target, loop_finished}
    """

    def __init__(self, message: str, trace: str = "", context: Dict = None):
        super().__init__(message)
        self.trace = trace
        self.context = context or {}


_FIX_PROMPT = """你是长篇小说视频生产流水线的错误诊断员。流水线运行报错，请分析并给出自动修复决策。

【错误类型】{etype}
【错误信息】{message}
【Traceback】
{trace}

【运行上下文】
{context}

可选自动修复手段（只能选这些，不能改代码）：
- 自动重试：从断点 resume 重跑当前环节（网络抖动 / API 限流 / 超时 / 瞬时文件问题通常有效）
- 调整参数：
  - chunk_size：文本分片块大小（默认 8000）。若错误与解析超长文本/上下文溢出/截断相关，可调小（如 4000）
  - max_retries：单集最大重试次数（默认 2）。若错误是"重试超限仍不合格"，可调大（如 4）

判断原则：
- 瞬时/外部错误（网络、限流、超时、临时文件缺失）→ retry，不调参
- 可通过调参解决（文本块过大、重试次数不足）→ retry + 调参
- 代码缺陷 / 数据损坏 / API 永久拒绝（如敏感内容 1026/1027）→ human（无法自动修复）

只输出一个 JSON 对象，不要任何其他文字：
{{"action": "retry"|"human", "params": {{"chunk_size": null, "max_retries": null}}, "reason": "一句话中文说明"}}
"""


def _extract_json(text: str) -> Optional[Dict]:
    text = (text or "").strip()
    # MiniMax 可能在最前面输出 <think>...</think> 思考块，先剥离
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # 兼容 LLM 用 ```json 代码块包裹
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 逐 { 尝试 raw_decode（能正确处理嵌套 JSON 与前后说明文字）
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, m.start())
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def analyze_error(err: PipelineError) -> Dict:
    """用生产 LLM 分析错误，返回决策 {action, params, reason}。"""
    try:
        llm = get_llm(role="production")
        resp = llm.invoke(_FIX_PROMPT.format(
            etype=type(err).__name__,
            message=str(err),
            trace=(err.trace or "")[-4000:],
            context=json.dumps(err.context, ensure_ascii=False, default=str),
        ))
        text = getattr(resp, "content", None) or str(resp)
        decision = _extract_json(text)
        if not decision or decision.get("action") not in ("retry", "human"):
            raise ValueError("LLM 决策无法解析: %s" % (text[:200],))
        decision.setdefault("params", {})
        decision.setdefault("reason", "")
        return decision
    except Exception as e:
        print(f"[监督] LLM 分析失败({e})，保守回退为人工介入")
        return {"action": "human", "params": {}, "reason": "LLM 分析不可用"}


def _apply_params(params: Dict) -> Dict:
    """应用 LLM 建议的参数调整，返回需同步进断点 state 的字段。"""
    sync = {}
    if not isinstance(params, dict):
        return sync
    cfg = get_config()
    cs = params.get("chunk_size")
    if isinstance(cs, (int, float)) and cs > 0:
        cfg.run.chunk_size = int(cs)
        sync["chunk_size"] = int(cs)
        print(f"[监督] 调整参数：chunk_size → {int(cs)}")
    mr = params.get("max_retries")
    if isinstance(mr, (int, float)) and mr > 0:
        cfg.run.max_retries = int(mr)
        print(f"[监督] 调整参数：max_retries → {int(mr)}")
    return sync


def run_with_supervisor(run_fn: Callable, **kwargs) -> None:
    """监督循环：执行 run_fn，出错时 LLM 分析并自动重试/调参。

    run_fn 抛 PipelineError 时触发；重试统一走 resume（断点续跑）。
    """
    attempt = 0
    last_sync = {}
    while True:
        try:
            run_fn(**kwargs)
            return
        except PipelineError as e:
            attempt += 1
            if attempt > MAX_AUTO_FIX_ROUNDS:
                print(f"[监督] 已达最大自动修复轮数({MAX_AUTO_FIX_ROUNDS})，请人工介入。")
                print(f"[监督] 最近错误: {type(e).__name__}: {e}")
                return
            decision = analyze_error(e)
            if decision.get("action") != "retry":
                print(f"[监督] 无法自动修复，请人工介入。原因: {decision.get('reason', '')}")
                print(f"[监督] 最近错误: {type(e).__name__}: {e}")
                return
            print(f"[监督] 第{attempt}轮自动修复：{decision.get('reason', '')} → resume 重跑")
            last_sync = _apply_params(decision.get("params") or {})
            kwargs["resume"] = True
            # 调参同步进断点 state，下次 resume 时 text_chunker 用新 chunk_size
            kwargs["_param_sync"] = last_sync
        except KeyboardInterrupt:
            print("\n[监督] 手动中断")
            return
        except SystemExit:
            raise
