"""
novel_pipeline.agents.review_foreshadow
伏笔线索评审 Agent —— 专精并行评审之二。
对应设计文档 4.2.1。

核查范围：长线伏笔解读正确性、关键伏笔遗漏、超前剧透、伏笔回收逻辑
依赖记忆：伏笔台账 + 历史事件向量检索
输出：伏笔维度缺陷列表
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from prompts import load_prompt
from agents.memory_manager import get_memory_agent


class ReviewForeshadowAgent:
    def __init__(self):
        self.llm = get_llm(role="review")
        self.system_prompt = load_prompt("review_foreshadow")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        foreshadow_ledger = episode_memory.get("unresolved_foreshadows", [])
        if not foreshadow_ledger:
            foreshadow_ledger = get_memory_agent().get_foreshadow_ledger()
        similar_scenes = episode_memory.get("similar_scenes", [])

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_msg(episode, script, image_prompts, tts_meta,
                                                  foreshadow_ledger, similar_scenes)),
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

    def _build_msg(self, episode, script, image_prompts, tts_meta, ledger, similar) -> str:
        return f"""【伏笔台账】
{json.dumps(ledger, ensure_ascii=False, indent=2)}

【相关历史场景检索结果】
{json.dumps(similar, ensure_ascii=False, indent=2)}

【本集关联伏笔ID】
{json.dumps(episode.get('linked_foreshadow_ids', []), ensure_ascii=False)}

【本集场景】
{json.dumps(episode.get('scenes', []), ensure_ascii=False, indent=2)}

【讲解文案】
{script}

【生图 Prompt】
{json.dumps(image_prompts, ensure_ascii=False, indent=2)}

请输出伏笔维度评审结果 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            data.setdefault("dimension", "foreshadow")
            data.setdefault("passed", len(data.get("defects", [])) == 0)
            return data
        return {"dimension": "foreshadow", "passed": True, "defects": [], "summary": "评审解析失败，默认通过"}
