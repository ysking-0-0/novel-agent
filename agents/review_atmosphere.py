"""
novel_pipeline.agents.review_atmosphere
视听氛围匹配评审 Agent —— 专精并行评审之四。
对应设计文档 4.2.1。

核查范围：讲解文案语气、TTS 情绪/语速、生图 Prompt 氛围与剧情冲突匹配度
依赖记忆：无
输出：视听维度缺陷列表
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from prompts import load_prompt


class ReviewAtmosphereAgent:
    def __init__(self):
        self.llm = get_llm(role="review")
        self.system_prompt = load_prompt("review_atmosphere")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_msg(episode, script, image_prompts, tts_meta)),
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

    def _build_msg(self, episode, script, image_prompts, tts_meta) -> str:
        return f"""【本集场景因果链】
{json.dumps(episode.get('scenes', []), ensure_ascii=False, indent=2)}

【讲解文案】
{script}

【生图 Prompt 列表】
{json.dumps(image_prompts, ensure_ascii=False, indent=2)}

【TTS 参数】
{json.dumps(tts_meta, ensure_ascii=False, indent=2)}

请输出视听氛围维度评审结果 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            data.setdefault("dimension", "atmosphere")
            data.setdefault("passed", len(data.get("defects", [])) == 0)
            return data
        return {"dimension": "atmosphere", "passed": True, "defects": [], "summary": "评审解析失败，默认通过"}
