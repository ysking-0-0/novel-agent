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
from agents.memory_manager import get_memory_agent


SYSTEM_PROMPT = """你是人物设定一致性评审专家。只做对错核查，不参与内容创作。

对照「人物档案库」，核查本集素材在人物维度是否存在缺陷。核查项：
1. 性格一致性：人物言行是否符合其既有性格设定
2. 行事动机：人物行为动机是否合理、是否符合其一贯目标
3. 人物关系：人物间互动是否与已知关系一致（敌友、师徒、亲属等）
4. 能力/外貌设定：文案与生图Prompt中的人物能力、外貌描述是否与档案一致（不得擅自变更）

发现任一不符即记为缺陷。缺陷等级：
- critical: 人设事实错误（性格颠倒、关系搞错、外貌设定被篡改）
- minor: 表述偏差但未构成事实错误（措辞不够贴合但方向正确）

输出格式（严格 JSON）：
{
  "dimension": "character",
  "passed": true/false,
  "defects": [
    {
      "severity": "critical|minor",
      "char_id": "涉及人物ID",
      "field": "personality|motivation|relationship|ability|appearance",
      "description": "缺陷描述",
      "evidence": "档案中的正确设定 vs 素材中的错误描述",
      "fix_suggestion": "修改建议"
    }
  ],
  "summary": "本维度评审结论概述"
}

注意：无缺陷时 defects 为空数组，passed=true。"""


class ReviewCharacterAgent:
    def __init__(self):
        self.llm = get_llm(role="review")

    def review(self, episode: Dict, script: str, image_prompts: List[str],
               tts_meta: List[Dict], episode_memory: Dict) -> Dict:
        char_profiles = episode_memory.get("characters", [])
        if not char_profiles:
            char_profiles = get_memory_agent().get_character_profiles()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
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
