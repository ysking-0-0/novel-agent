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
from prompts import load_prompt
from agents.memory_manager import get_memory_agent

from llm_factory import get_llm


class ReviewArbiterAgent:
    def __init__(self):
        self.llm = get_llm(role="review")
        self.system_prompt = load_prompt("review_arbiter")

    def arbitrate(self, reviews: List[Dict], episode: Dict, original_snippet: str) -> Dict:
        """合并 4 份评审报告，输出仲裁结果。"""
        messages = [
            SystemMessage(content=self.system_prompt),
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
