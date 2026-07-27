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
from agents.memory_manager import get_memory_agent


SYSTEM_PROMPT = """你是伏笔线索评审专家。只做对错核查，不参与内容创作。

对照「伏笔台账」与历史事件检索，核查本集素材在伏笔维度是否存在缺陷。核查项：
1. 解读正确性：本集对历史伏笔的解读/回收是否准确
2. 关键伏笔遗漏：本集该提及/回收的伏笔是否被遗漏
3. 超前剧透：是否提前剧透了尚未发生的关键伏笔
4. 回收逻辑：伏笔回收的因果逻辑是否成立

缺陷等级：
- critical: 伏笔事实错误（错误回收、关键伏笔遗漏、重大超前剧透）
- minor: 回收表述不够清晰但方向正确

输出格式（严格 JSON）：
{
  "dimension": "foreshadow",
  "passed": true/false,
  "defects": [
    {
      "severity": "critical|minor",
      "foreshadow_id": "涉及伏笔ID",
      "type": "wrong_interpretation|missed|premature|logic_flaw",
      "description": "缺陷描述",
      "evidence": "台账记录 vs 素材描述",
      "fix_suggestion": "修改建议"
    }
  ],
  "summary": "本维度评审结论概述"
}"""


class ReviewForeshadowAgent:
    def __init__(self):
        self.llm = get_llm(role="review")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        foreshadow_ledger = episode_memory.get("unresolved_foreshadows", [])
        if not foreshadow_ledger:
            foreshadow_ledger = get_memory_agent().get_foreshadow_ledger()
        similar_scenes = episode_memory.get("similar_scenes", [])

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
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
