"""
novel_pipeline.graph.builder
LangGraph StateGraph 构建：节点注册 + 边连接 + 条件路由 + SqliteSaver 断点续跑。
对应设计文档 2.1 / 5.2。

流水线（11 步主循环）：
  START → text_chunker
       → [route_after_chunking] → plot_parser / episode_aggregator_force / END
  plot_parser → episode_aggregator
       → [route_after_aggregation] → material_generator / text_chunker(回读)
  material_generator → format_validator
       → [route_after_format_check] → memory_prefetch / material_generator(局部修正)
  memory_prefetch → (并行 4 专精评审) → review_arbiter
       → [route_after_arbiter] → persistence / retry_counter → material_generator / persistence
  persistence → [route_after_persistence] → text_chunker / END
"""
import functools
import os
from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from state import NovelState
from agents import (
    PlotParserAgent, EpisodeAggregatorAgent, MaterialGeneratorAgent,
    ReviewCharacterAgent, ReviewForeshadowAgent, ReviewTimelineAgent,
    ReviewAtmosphereAgent, ReviewArbiterAgent, get_memory_agent,
)
from nodes import (
    text_chunker_node, format_validator_node, memory_prefetch_node,
    persistence_node, retry_counter_node, media_synthesizer_node,
    route_after_aggregation, route_after_format_check,
    route_after_arbiter, route_after_persistence, route_after_chunking,
    route_after_media_quality, route_after_retry,
)
from config import get_config


# ---------- Agent 节点包装 ----------
def _wrap(agent_method):
    """把 Agent.invoke 包装成 LangGraph 节点函数。"""
    @functools.wraps(agent_method)
    def _node(state: Dict) -> Dict:
        return agent_method(state)
    return _node


def _parallel_reviews_node(state: Dict) -> Dict:
    """串行执行 4 个专精评审，收集为 review_reports 列表。

    设计文档要求并行；为兼容单线程 SqliteSaver 与简化实现，这里采用顺序调用，
    每个 Agent 内部独立调用 LLM。后续可用 asyncio.gather 优化为真正并行。
    """
    reports = []
    for cls in (ReviewCharacterAgent, ReviewForeshadowAgent,
                ReviewTimelineAgent, ReviewAtmosphereAgent):
        agent = cls()
        try:
            rep = agent.invoke(state)
            reports.append(rep)
        except Exception as e:
            reports.append({
                "dimension": cls.__name__,
                "defects": [],
                "error": f"{type(e).__name__}: {e}",
            })
    return {"review_reports": reports}


def _force_pack_node(state: Dict) -> Dict:
    """文末强制把 pending_scences 打包为最终集。"""
    aggregator = EpisodeAggregatorAgent()
    # 临时把 loop_finished 透传给聚合器，强制打包
    forced_state = dict(state)
    forced_state["_force_pack"] = True
    return aggregator.invoke(forced_state)


def _termination_check_node(state: Dict) -> Dict:
    """终止判断占位节点：无实际逻辑，仅作为 media_synthesizer 质检合格后的路由跳板。"""
    return {}


def build_graph(db_path: str = None):
    """构建并编译 LangGraph，返回 compiled graph。

    Args:
        db_path: sqlite 断点库路径。None 则用配置中的默认路径。
    """
    if db_path is None:
        db_path = get_config().storage.sqlite_path

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    g = StateGraph(NovelState)

    # ---------- 注册节点 ----------
    g.add_node("text_chunker", text_chunker_node)
    g.add_node("plot_parser", _wrap(PlotParserAgent().invoke))
    g.add_node("episode_aggregator", _wrap(EpisodeAggregatorAgent().invoke))
    g.add_node("episode_aggregator_force", _force_pack_node)
    g.add_node("material_generator", _wrap(MaterialGeneratorAgent().invoke))
    g.add_node("format_validator", format_validator_node)
    g.add_node("memory_prefetch", memory_prefetch_node)
    g.add_node("parallel_reviews", _parallel_reviews_node)
    g.add_node("review_arbiter", _wrap(ReviewArbiterAgent().invoke))
    g.add_node("retry_counter", retry_counter_node)
    g.add_node("persistence", persistence_node)
    g.add_node("media_synthesizer", media_synthesizer_node)
    g.add_node("termination_check", _termination_check_node)

    # ---------- 入口 ----------
    g.add_edge(START, "text_chunker")

    # text_chunker → 条件路由
    g.add_conditional_edges(
        "text_chunker",
        route_after_chunking,
        {
            "plot_parser": "plot_parser",
            "episode_aggregator_force": "episode_aggregator_force",
            "__end__": END,
        },
    )

    # plot_parser → episode_aggregator
    g.add_edge("plot_parser", "episode_aggregator")

    # episode_aggregator → 条件路由
    g.add_conditional_edges(
        "episode_aggregator",
        route_after_aggregation,
        {
            "material_generator": "material_generator",
            "text_chunker": "text_chunker",
        },
    )

    # episode_aggregator_force → material_generator（强制打包后直接生成素材）
    g.add_edge("episode_aggregator_force", "material_generator")

    # material_generator → format_validator
    g.add_edge("material_generator", "format_validator")

    # format_validator → 条件路由
    g.add_conditional_edges(
        "format_validator",
        route_after_format_check,
        {
            "memory_prefetch": "memory_prefetch",
            "material_generator": "material_generator",
            "persistence": "persistence",
        },
    )

    # memory_prefetch → 并行评审
    g.add_edge("memory_prefetch", "parallel_reviews")

    # parallel_reviews → review_arbiter
    g.add_edge("parallel_reviews", "review_arbiter")

    # review_arbiter → 条件路由
    g.add_conditional_edges(
        "review_arbiter",
        route_after_arbiter,
        {
            "persistence": "persistence",
            "retry_counter": "retry_counter",
            "material_generator": "material_generator",
        },
    )

    # retry_counter → 条件路由（评审 regenerate → material_generator；质检失败 → media_synthesizer 重合成）
    g.add_conditional_edges(
        "retry_counter",
        route_after_retry,
        {
            "media_synthesizer": "media_synthesizer",
            "material_generator": "material_generator",
        },
    )

    # persistence → media_synthesizer → 质检路由（合格→终止判断；不合格→整集重生成）
    g.add_edge("persistence", "media_synthesizer")
    g.add_conditional_edges(
        "media_synthesizer",
        route_after_media_quality,
        {
            "termination_check": "termination_check",
            "retry_counter": "retry_counter",
        },
    )
    # 终止判断（原 route_after_persistence）：达到目标/读完 → END，否则回 text_chunker
    g.add_conditional_edges(
        "termination_check",
        route_after_persistence,
        {
            "text_chunker": "text_chunker",
            "__end__": END,
        },
    )

    compiled = g.compile(checkpointer=checkpointer)
    return compiled, conn
