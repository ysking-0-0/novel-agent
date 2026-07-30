"""
novel_pipeline.agents.plot_parser
剧情解析 Agent —— 剧情事实基准节点，所有下游内容的事实源头。
对应设计文档 4.1.1。

核心职责：
1. 读取当前章节文本，调用记忆 Agent 召回关联历史上下文
2. 拆分最小剧情单元 Scene，提取每条场景完整因果链：
   前置诱因 → 人物动机 → 核心行为 → 直接结果 → 远期影响
3. 识别新增伏笔、人物状态变化、关键道具/世界观设定
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from prompts import load_prompt
from agents.memory_manager import get_memory_agent


class PlotParserAgent:
    """剧情解析 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")
        self.system_prompt = load_prompt("plot_parser")

    def parse(self, current_chunk: str, offset: int) -> List[Dict]:
        """解析当前文本块，返回结构化场景列表 new_scenes。"""
        memory = get_memory_agent()
        # 召回关联历史上下文
        context = memory.recall_context(current_chunk[:500])

        context_brief = self._format_context(context)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_user_msg(current_chunk, offset, context_brief)),
        ]
        resp = self.llm.invoke(messages)
        return self._parse_response(resp.content)

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口：从 state 取字段，输出 new_scenes。"""
        chunk = state.get("current_chunk", "")
        offset = state.get("offset", 0)
        scenes = self.parse(chunk, offset)
        # 给每个场景打上 raw_order，便于后续追溯
        for i, s in enumerate(scenes):
            s.setdefault("raw_order", offset + i)
        return {"new_scenes": scenes}

    def _format_context(self, context: Dict) -> str:
        lines = []
        for c in context.get("characters", [])[:8]:
            # 传完整人物档案（固定外貌字段），让 LLM 沿用
            parts = [f"已知人物-{c.get('char_id')}: {c.get('name','')}"]
            for k in ("age", "identity", "appearance", "attire", "personality"):
                v = c.get(k, "")
                if v:
                    parts.append(f"{k}={v}")
            parts.append(f"state_change={json.dumps(c.get('state_change', ''), ensure_ascii=False)}")
            lines.append(" | ".join(parts))
        for f in context.get("unresolved_foreshadows", [])[:10]:
            lines.append(f"未解伏笔-{f.get('foreshadow_id')}: {f.get('description','')}")
        for s in context.get("recent_events", [])[:8]:
            lines.append(f"前情-{s.get('event_id','')}: {s.get('summary','')}")
        return "\n".join(lines) if lines else "（暂无历史记忆）"

    def _build_user_msg(self, chunk: str, offset: int, context_brief: str) -> str:
        return f"""【历史上下文（仅供参照，勿照抄）】
{context_brief}

【当前小说文本块 offset={offset}】
{chunk}

请拆分为最小剧情单元 Scene 列表，输出 JSON 数组。"""

    def _parse_response(self, content) -> List[Dict]:
        from utils import extract_json_list
        data = extract_json_list(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "scenes" in data:
            return data["scenes"]
        return []
