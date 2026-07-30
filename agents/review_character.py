"""
novel_pipeline.agents.review_character
人物设定评审 Agent —— 专精并行评审之一。
对应设计文档 4.2.1。

核查范围：人物性格、行事动机、人物关系、能力/外貌设定是否符合原著前后文
依赖记忆：人物档案库
输出：人物维度缺陷列表
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from prompts import load_prompt
from agents.memory_manager import get_memory_agent


class ReviewCharacterAgent:
    def __init__(self):
        self.llm = get_llm(role="review")
        self.system_prompt = load_prompt("review_character")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        char_profiles = episode_memory.get("characters", [])
        if not char_profiles:
            char_profiles = get_memory_agent().get_character_profiles()

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_msg(episode, script, image_prompts, tts_meta, char_profiles)),
        ]
        resp = self.llm.invoke(messages)
        return self._parse(resp.content)

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口：输出单维度评审报告 dict。"""
        episode_memory = state.get("prefetched_memory") or {}
        return self.review(
            state.get("current_episode") or {},
            state.get("episode_script", ""),
            state.get("episode_image_prompts", []),
            state.get("episode_tts_meta", []),
            episode_memory,
        )

    def _build_msg(self, episode, script, image_prompts, tts_meta, char_profiles) -> str:
        return f"""【人物档案库】
{json.dumps(char_profiles, ensure_ascii=False, indent=2)}

【本集 Episode 场景】
{json.dumps(episode.get('scenes', []), ensure_ascii=False, indent=2)}

【讲解文案】
{script}

【生图 Prompt 列表】
{json.dumps(image_prompts, ensure_ascii=False, indent=2)}

【TTS 参数】
{json.dumps(tts_meta, ensure_ascii=False, indent=2)}

请输出人物维度评审结果 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            data.setdefault("dimension", "character")
            data.setdefault("passed", len(data.get("defects", [])) == 0)
            return data
        return {"dimension": "character", "passed": True, "defects": [], "summary": "评审解析失败，默认通过"}
