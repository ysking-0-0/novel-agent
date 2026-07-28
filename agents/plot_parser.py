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
- characters: 涉及人物列表，每项必须包含：
  {
    "char_id": 人物名（优先用历史档案ID；新人物用姓名）,
    "name": 姓名,
    "state_change": 本场景中该人物的状态变化（可选）,
    "appearance": 人物固定外貌描述（发色/发型/眼/眉/身材/五官特征，新人物首次出现时必须填写，已有人物沿用档案值填同值）,
    "age": 年龄或年龄段（如"15岁少年"/"中年"/"幼童"，从原文推断，新人物必填）,
    "identity": 身份地位（如"剑门外门弟子"/"灯中灵体"/"长老"，从原文推断）,
    "attire": 穿着服饰（符合境界身份，如"粗麻兽皮短打"/"素色道袍"，从原文推断，无明确则按身份推测）,
    "personality": 性格特征（如"坚毅果敢"/"慵懒神秘"，新人物必填）,
    "is_new": true/false（新人物首次出现为true，已存在为false）,
    "appearance_override": true/false（当且仅当本场景发生变形/易容/化妆/法术变身/显著外貌改变时设为true，其余情况必须为false或省略）
  }
- foreshadows: 本场景中埋设或回收的伏笔列表，每项 {"foreshadow_id": "fs_<序号>", "type": "plant|resolve", "description": 伏笔描述, "status": "planted|resolved"}
- key_items: 关键道具/世界观设定，字符串列表

【人物档案规则（最高优先级）】
1. 新人物首次出现时（is_new=true），必须完整填写 appearance/age/identity/attire/personality，从原文细节推断，无法确定则按身份合理推测
2. 已存在人物（is_new=false），appearance/age/identity/attire/personality 必须填入与档案相同的值（保持固定不变），仅 state_change 可变
3. appearance 一旦确定，后续绝不改变（人物不会突然变老/变年轻/换发型/换体型）——除非本场景明确发生变形/易容/化妆/法术变身（此时 appearance_override=true 并填写改变后的新外貌）
4. 年龄根据原文时间线推断；若原文有"九岁""六年"等线索据此推算
5. attire 必须符合人物身份地位与修为境界，规则如下：
   - 人族普通地位低、无修为的弟子（如外门弟子）→ 破旧粗布衣衫、粗麻兽皮短打，朴素简陋
   - 内门弟子 → 素色道袍，整洁但朴素
   - 长老/高阶 → 华贵锦袍佩玉
   - 神族/灵体 → 古朴祭祀礼服或特殊灵体形态
   - attire 一旦确定，身份不变则穿着不变（不会突然换装）；只有身份明确变化（如外门升内门）才更新穿着
6. appearance_override=true 的判定标准（必须严格）：
   - 原文明确描写"变形/易容/换脸/化身/变装/化妆/幻化/法术改变外貌"
   - 原文明确描写该人物外貌发生显著变化（如"化作老翁""变成蛇身""披上人皮"）
   - 普通的换衣服、受伤留疤、头发长长等不属于 appearance_override
   - 若不确定，设为false（保持原外貌）
7. 当 appearance_override=true 时，appearance/age 等字段填写改变后的新外貌新身份，系统会据此更新档案
8. 主要角色（戏份多）和主要配角（有名字、多次出现）都必须建立完整档案；一次性龙套（如"路人甲"）可省略 appearance

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
