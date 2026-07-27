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
from agents.memory_manager import get_memory_agent


SYSTEM_PROMPT = """你是长篇小说剧情解析专家。你的任务是把一段小说原文拆解为最小剧情单元（Scene）列表，每个 Scene 必须包含完整因果链。

对每个场景，提取以下字段：
- scene_id: 场景唯一ID，格式 sc_<3位序号>（本轮内递增）
- title: 场景简短标题（不超过20字）
- summary: 场景内容摘要（80-150字，事实客观，不加工）
- cause: 前置诱因（什么导致了这个场景发生）
- motivation: 人物动机（涉事人物为何这样做）
- core_action: 核心行为（场景中发生的关键事件/动作）
- immediate_result: 直接结果（场景结束时发生了什么）
- long_term_impact: 远期影响（对后续剧情的潜在影响）
- characters: 涉及人物列表，每项 {"char_id": 人物名, "name": 姓名, "state_change": 本场景中该人物的状态变化（可选）}
- foreshadows: 本场景中埋设或回收的伏笔列表，每项 {"foreshadow_id": "fs_<序号>", "type": "plant|resolve", "description": 伏笔描述, "status": "planted|resolved"}
- key_items: 关键道具/世界观设定，字符串列表

严格要求：
1. 只从给定原文提取事实，不得臆造、不得脑补、不得剧透后续
2. 因果链必须完整，缺项填 "未明确"
3. 保留原文细节，不要过度抽象丢失关键信息
4. 人物 char_id 优先使用历史档案中的ID；新人物用其姓名

只输出一个 JSON 数组，不要任何解释文字。"""


class PlotParserAgent:
    """剧情解析 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")

    def parse(self, current_chunk: str, offset: int) -> List[Dict]:
        """解析当前文本块，返回结构化场景列表 new_scenes。"""
        memory = get_memory_agent()
        # 召回关联历史上下文
        context = memory.recall_context(current_chunk[:500])

        context_brief = self._format_context(context)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
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
            lines.append(f"已知人物-{c.get('char_id')}: {c.get('name','')} {json.dumps(c.get('state_change',''), ensure_ascii=False)}")
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
