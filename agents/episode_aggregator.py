"""
novel_pipeline.agents.episode_aggregator
剧集聚合 Agent —— 单集叙事节奏把控者。
对应设计文档 4.1.2。

核心职责：
1. 合并跨循环遗留的 pending_scenes 与本轮新场景
2. 按「叙事闭环、体量适中、悬念收尾」规则，将连续场景打包为完整 Episode
3. 不足成集的场景更新回 pending_scenes，等待下一轮文本补充
4. 标记本集关联的历史伏笔 ID，供下游生成与评审使用
"""
import json
from typing import Dict, List, Tuple, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from prompts import load_prompt
from agents.memory_manager import get_memory_agent


class EpisodeAggregatorAgent:
    """剧集聚合 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")
        self.system_prompt = load_prompt("episode_aggregator")

    def aggregate(self, pending: List[Dict], new_scenes: List[Dict],
                  loop_finished: bool) -> Tuple[Optional[Dict], List[Dict]]:
        """
        返回 (episode, updated_pending)。
        - episode: 若成集，为完整 Episode dict；否则 None
        - updated_pending: 剩余未成集的场景，留待下一轮
        """
        all_scenes = pending + new_scenes
        # 全本结束强制打包：剩余场景都必须成最终集
        if loop_finished and all_scenes:
            return self._force_pack(all_scenes), []

        if not all_scenes:
            return None, []

        memory = get_memory_agent()
        foreshadow_ledger = memory.get_foreshadow_ledger()

        decision = self._decide(all_scenes, foreshadow_ledger)
        if not decision.get("can_form_episode"):
            return None, all_scenes

        n = int(decision.get("episode_scenes_count", 0)) or len(all_scenes)
        n = max(1, min(n, len(all_scenes)))
        used = all_scenes[:n]
        rest = all_scenes[n:]

        episode = {
            "episode_id": None,  # 由归档节点赋值
            "scenes": used,
            "summary": decision.get("episode_summary", ""),
            "cliffhanger": decision.get("cliffhanger", ""),
            "linked_foreshadow_ids": decision.get("linked_foreshadow_ids", []),
            "source_offset_range": self._offset_range(used),
        }
        return episode, rest

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口。"""
        force = state.get("_force_pack", False)
        loop_finished = bool(state.get("loop_finished")) or force
        episode, pending = self.aggregate(
            state.get("pending_scenes", []),
            state.get("new_scenes", []),
            loop_finished,
        )
        update = {"pending_scenes": pending, "new_scenes": [], "current_episode": episode}
        return update

    def _decide(self, scenes: List[Dict], foreshadow_ledger: List[Dict]) -> Dict:
        scenes_brief = self._scenes_to_brief(scenes)
        fs_brief = "\n".join(
            f"- {f.get('foreshadow_id')}: {f.get('description','')}"
            for f in foreshadow_ledger[:20]
        ) or "（暂无）"

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"【场景列表（共{len(scenes)}条）】\n{scenes_brief}\n\n【已知伏笔台账】\n{fs_brief}\n\n请判定是否可成集。"),
        ]
        resp = self.llm.invoke(messages)
        return self._parse(resp.content)

    def _scenes_to_brief(self, scenes: List[Dict]) -> str:
        lines = []
        for i, s in enumerate(scenes, 1):
            lines.append(
                f"[{i}] {s.get('title','')} | 摘要:{s.get('summary','')} | "
                f"因果:{s.get('cause','')}→{s.get('core_action','')}→{s.get('immediate_result','')} | "
                f"伏笔:{json.dumps(s.get('foreshadows',[]), ensure_ascii=False)}"
            )
        return "\n".join(lines)

    def _offset_range(self, used: List[Dict]) -> List[int]:
        offsets = [s.get("raw_order", 0) for s in used if s.get("raw_order") is not None]
        return [min(offsets), max(offsets)] if offsets else []

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            return data
        return {"can_form_episode": False, "reason": "LLM 返回解析失败"}

    def _force_pack(self, scenes: List[Dict]) -> Dict:
        """全本结束，强制打包剩余场景为最终集。"""
        return {
            "episode_id": None,
            "scenes": scenes,
            "summary": "全本收尾集",
            "cliffhanger": "全书完结",
            "linked_foreshadow_ids": [],
            "source_offset_range": self._offset_range(scenes),
            "is_final_episode": True,
        }
