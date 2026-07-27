"""
novel_pipeline.nodes.routing
条件路由节点（纯代码）—— 对应设计文档中的条件边。
对应设计文档 5.2 步骤 3、5、8、11。

LangGraph 用 add_conditional_edges 挂接这些路由函数。
"""
from typing import Dict
from config import get_config


# ---------- 步骤 3：剧集聚合后路由 ----------
def route_after_aggregation(state: Dict) -> str:
    """形成完整 Episode → 向下；不足成集 → 回到文本分片读取下一段。"""
    episode = state.get("current_episode")
    if episode is None:
        return "text_chunker"   # 不足成集，继续读文本
    return "material_generator"


# ---------- 步骤 5：格式校验后路由 ----------
def route_after_format_check(state: Dict) -> str:
    """格式正确 → 进入记忆预检索；错误 → 回到素材生成局部修正。"""
    if state.get("format_valid"):
        return "memory_prefetch"
    return "material_generator"   # 局部修正重生成


# ---------- 步骤 8：仲裁后路由 ----------
def route_after_arbiter(state: Dict) -> str:
    """仲裁结果路由：
    - pass → 持久化归档
    - regenerate + 未超限 → 整集重生成
    - regenerate + 超限 → 标记人工复核 → 持久化归档
    - minor_revise → 局部微调（回素材生成）
    """
    verdict = (state.get("review_result") or {}).get("verdict", "pass")
    cfg = get_config()
    max_retry = cfg.run.max_retries
    retry = state.get("retry_count", 0)

    if verdict == "pass":
        return "persistence"
    if verdict == "minor_revise":
        return "material_generator"
    if verdict == "regenerate":
        if retry >= max_retry:
            print(f"[仲裁] 重试超限({retry}>={max_retry})，标记人工复核后继续归档")
            # 把人工复核标记写入 review_result
            rr = state.get("review_result") or {}
            rr["manual_review"] = True
            # 注意：无法在此直接修改 state，通过返回值影响
            return "persistence"
        # retry_count 由 retry_counter 节点递增
        return "retry_counter"
    # 未知 verdict 兜底
    return "persistence"


# ---------- 步骤 11：终止条件判断 ----------
def route_after_persistence(state: Dict) -> str:
    """达到目标集数或读完小说 → END；否则回到文本分片。"""
    cfg = get_config()
    target = state.get("target_episode_count")
    completed = state.get("completed_episode_count", 0)
    loop_finished = state.get("loop_finished", False)

    if target is not None and completed >= target:
        print(f"[终止] 达到目标集数 {target}，优雅停止")
        return "__end__"
    if loop_finished:
        # pending_scenes 已强制打包为最终集走完流程
        pending = state.get("pending_scenes") or []
        if len(pending) == 0:
            print("[终止] 小说读至文末，所有场景已归档，任务结束")
            return "__end__"
    return "text_chunker"


# ---------- 循环出口：text_chunker 读完后判断 ----------
def route_after_chunking(state: Dict) -> str:
    """文末 + 无 pending → 直接 END；否则进入剧情解析。"""
    if state.get("loop_finished"):
        pending = state.get("pending_scenes") or []
        if len(pending) == 0:
            print("[终止] 文末且无遗留场景，结束")
            return "__end__"
        # 有遗留场景 → 强制打包为最终集
        print("[收尾] 文末，强制打包遗留场景为最终集")
        return "episode_aggregator_force"
    return "plot_parser"
