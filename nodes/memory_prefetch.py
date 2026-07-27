"""
novel_pipeline.nodes.memory_prefetch
统一记忆预检索节点（纯代码调用记忆Agent）—— 主循环步骤 6。
对应设计文档 5.2。

职责：
  一次性拉取本集关联记忆存入 State，供所有评审共享。
  避免每个评审 Agent 重复检索。
"""
from typing import Dict
import json
from agents import get_memory_agent


def memory_prefetch_node(state: Dict) -> Dict:
    """从 Episode 一次性检索人物/事件/伏笔记忆，写入 state，供所有评审共享。"""
    episode = state.get("current_episode") or {}
    mem = get_memory_agent()
    # 直接复用记忆 Agent 的单集预检索接口（已聚合人物/伏笔/事件/向量召回）
    recalled = mem.recall_for_episode(episode)

    # 拼接供评审对照的全文本
    parts = ["【人物档案】"]
    for c in recalled.get("characters", []):
        parts.append(f"- {c.get('char_id','')}: {c.get('name','')} | {json.dumps(c.get('state_change',''), ensure_ascii=False)}")
    parts.append("\n【未解伏笔台账】")
    for f in recalled.get("unresolved_foreshadows", []):
        parts.append(f"- {f.get('foreshadow_id','')}: {f.get('description','')} 状态={f.get('status','')}")
    parts.append("\n【相关历史场景（向量召回）】")
    for s in recalled.get("similar_scenes", []):
        parts.append(f"- {s.get('meta',{}).get('scene_id','')}: {s.get('meta',{}).get('summary','')}")
    parts.append("\n【近期时序事件】")
    for e in recalled.get("recent_events", []):
        parts.append(f"- [{e.get('event_id','')}] {e.get('summary','')}")
    recalled["full_text"] = "\n".join(parts)

    return {"prefetched_memory": recalled}
