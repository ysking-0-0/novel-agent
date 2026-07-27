"""
novel_pipeline.agents.review_arbiter
评审汇总仲裁 Agent —— 多评审结果统一收口者。
对应设计文档 4.2.2。

核心职责：
1. 合并 4 份评审报告，去重同类问题
2. 缺陷分级：严重错误（整集重生成）/ 轻微瑕疵（局部微调）
3. 解决评审意见冲突，以原著原文为唯一标准做最终判定
4. 整理统一修改指令下发回生产节点
"""
import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from agents.memory_manager import get_memory_agent

from llm_factory import get_llm


SYSTEM_PROMPT = """你是多维度评审汇总仲裁专家。你将收到 4 份不同维度的评审报告，需要统一收口。

核心职责：
1. 去重：不同维度可能报告了同一问题，合并为一条
2. 冲突解决：若不同维度意见冲突，以「原著事实」为唯一标准裁定
3. 缺陷分级：
   - critical（严重错误）：剧情/伏笔/人设事实错误 → 整集重生成
   - minor（轻微瑕疵）：措辞/情绪微调 → 局部修改
4. 决策：
   - pass: 全部通过
   - minor_revise: 存在轻微瑕疵，返回生产节点局部微调
   - regenerate: 存在严重错误，整集重生成

输出格式（严格 JSON）：
{{
  "verdict": "pass|minor_revise|regenerate",
  "critical_defects": [
    {{"type": "缺陷类型", "description": "描述", "dimension": "来源维度", "fix_suggestion": "修改建议"}}
  ],
  "minor_defects": [
    {{"type": "缺陷类型", "description": "描述", "dimension": "来源维度", "fix_suggestion": "修改建议"}}
  ],
  "unified_revise_instruction": {{
    "action": "pass|minor_revise|regenerate",
    "instructions": ["统一的修改指令列表，供生产节点执行"],
    "focus_scenes": ["需重点修改的场景ID列表"]
  }},
  "summary": "仲裁结论概述"
}}

严格要求：
1. 严重错误必须触发 regenerate，不得降级
2. 轻微瑕疵触发 minor_revise，避免整集重生成节约算力
3. 修改指令必须具体可执行，不要泛泛而谈"""


class ReviewArbiterAgent:
    def __init__(self):
        self.llm = get_llm(role="review")

    def arbitrate(self, reviews: List[Dict], episode: Dict, original_snippet: str) -> Dict:
        """合并 4 份评审报告，输出仲裁结果。"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_msg(reviews, episode, original_snippet)),
        ]
        resp = self.llm.invoke(messages)
        return self._parse(resp.content, reviews)

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口。"""
        ep = state.get("current_episode") or {}
        original = get_memory_agent().get_original_snippet_context(ep)
        reviews = state.get("review_reports", [])
        result = self.arbitrate(reviews, ep, original)
        # 在仲裁结果里带上 retry 信息供路由使用
        result["_retry_count"] = state.get("retry_count", 0)
        return {"review_result": result}

    def _build_msg(self, reviews: List[Dict], episode: Dict, original_snippet: str) -> str:
        return f"""【4 份专精评审报告】
{json.dumps(reviews, ensure_ascii=False, indent=2)}

【本集 Episode 概况】
摘要: {episode.get('summary','')}
悬念: {episode.get('cliffhanger','')}
场景数: {len(episode.get('scenes', []))}

【对应原著原文片段】
{original_snippet}

请合并仲裁，输出 JSON。"""

    def _parse(self, content, reviews: List[Dict]) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            data.setdefault("verdict", "pass")
            data.setdefault("critical_defects", [])
            data.setdefault("minor_defects", [])
            data.setdefault("unified_revise_instruction", {"action": "pass", "instructions": [], "focus_scenes": []})
            return data
        # 兜底：从 4 份报告快速判定
        all_defects = [d for r in reviews for d in r.get("defects", [])]
        critical = [d for d in all_defects if d.get("severity") == "critical"]
        minor = [d for d in all_defects if d.get("severity") == "minor"]
        if critical:
            verdict = "regenerate"
        elif minor:
            verdict = "minor_revise"
        else:
            verdict = "pass"
        return {
            "verdict": verdict,
            "critical_defects": critical,
            "minor_defects": minor,
            "unified_revise_instruction": {
                "action": verdict,
                "instructions": [d.get("fix_suggestion", "") for d in critical + minor if d.get("fix_suggestion")],
                "focus_scenes": [],
            },
            "summary": "仲裁 LLM 解析失败，按规则兜底判定",
        }
