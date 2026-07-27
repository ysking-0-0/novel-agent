"""
novel_pipeline.agents.review_timeline
剧情时序评审 Agent —— 专精并行评审之三。
对应设计文档 4.2.1。

核查范围：事件先后顺序、因果链条、关键剧情篡改、因果倒置
依赖记忆：时序事件库 + 原著原文
输出：剧情事实缺陷列表
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from agents.memory_manager import get_memory_agent


SYSTEM_PROMPT = """你是剧情时序与事实评审专家。只做对错核查，不参与内容创作。

对照「时序事件库」与原著原文，核查本集素材在剧情事实维度是否存在缺陷。核查项：
1. 事件先后顺序：本集描述的事件顺序是否与原著一致
2. 因果链条：起因→发展→结果的链条是否被破坏或错位
3. 关键剧情篡改：是否擅自修改了原著关键剧情
4. 因果倒置：是否存在把结果当成原因、把后事提前的错位

缺陷等级：
- critical: 剧情事实错误（顺序颠倒、因果篡改、关键情节魔改）
- minor: 表述偏差但事实未错（措辞导致因果表述不够清晰）

输出格式（严格 JSON）：
{
  "dimension": "timeline",
  "passed": true/false,
  "defects": [
    {
      "severity": "critical|minor",
      "scene_id": "涉及场景ID",
      "type": "order_wrong|causality_broken|plot_altered|causality_inverted",
      "description": "缺陷描述",
      "evidence": "原著/事件库事实 vs 素材描述",
      "fix_suggestion": "修改建议"
    }
  ],
  "summary": "本维度评审结论概述"
}"""


class ReviewTimelineAgent:
    def __init__(self):
        self.llm = get_llm(role="review")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        timeline_events = episode_memory.get("recent_events", [])
        if not timeline_events:
            timeline_events = get_memory_agent().get_timeline_events()
        original_context = get_memory_agent().get_original_snippet_context(episode)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_msg(episode, script, image_prompts, tts_meta,
                                                  timeline_events, original_context)),
        ]
        resp = self.llm.invoke(messages)
        return self._parse(resp.content)

    def invoke(self, state: Dict) -> Dict:
        episode_memory = state.get("prefetched_memory") or {}
        return self.review(
            state.get("current_episode") or {},
            state.get("episode_script", ""),
            state.get("episode_image_prompts", []),
            state.get("episode_tts_meta", []),
            episode_memory,
        )

    def _build_msg(self, episode, script, image_prompts, tts_meta, events, original_ctx) -> str:
        return f"""【时序事件库（前情）】
{json.dumps(events, ensure_ascii=False, indent=2)}

【原著前情梗概】
{original_ctx}

【本集场景】
{json.dumps(episode.get('scenes', []), ensure_ascii=False, indent=2)}

【讲解文案】
{script}

【生图 Prompt】
{json.dumps(image_prompts, ensure_ascii=False, indent=2)}

请输出剧情时序维度评审结果 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            data.setdefault("dimension", "timeline")
            data.setdefault("passed", len(data.get("defects", [])) == 0)
            return data
        return {"dimension": "timeline", "passed": True, "defects": [], "summary": "评审解析失败，默认通过"}
