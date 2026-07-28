"""
novel_pipeline.state
LangGraph 全节点共享的全局状态 NovelState。
设计原则：轻量化、无冗余、可持久化。
对应设计文档第 3 节「核心状态设计（NovelState）」。
"""
from typing import TypedDict, Optional, List, Dict, Any


class NovelState(TypedDict, total=False):
    # ───────── 分片控制 ─────────
    # 本地小说 TXT 文件路径，不存储全文本
    file_path: str
    # 文件读取字节偏移游标，断点续跑核心标识
    offset: int
    # 单次读取基础字符上限，可配置
    chunk_size: int
    # 当前轮读取的完整章节文本块，仅本轮有效
    current_chunk: str

    # ───────── 剧情缓存 ─────────
    # 未凑成完整一集的零散场景缓存，跨循环留存
    pending_scenes: List[Dict]
    # 当前轮解析出的新剧情场景列表
    new_scenes: List[Dict]

    # ───────── 当前集产出 ─────────
    # 当前待生产 / 评审的完整剧集数据
    current_episode: Optional[Dict]
    # 整集讲解文案
    episode_script: Optional[str]
    # 整集时序化生图 Prompt 列表
    episode_image_prompts: Optional[List[str]]
    # 整集结构化 TTS 参数列表
    episode_tts_meta: Optional[List[Dict]]
    # 评审汇总报告（缺陷列表、等级、修改建议）
    review_result: Optional[Dict]
    # 当前集重生成次数，防止死循环
    retry_count: int

    # ───────── 生产进度 ─────────
    # 目标生成集数，None 表示跑完全本
    target_episode_count: Optional[int]
    # 已完成归档的有效集数
    completed_episode_count: int
    # 是否读到小说文件末尾
    loop_finished: bool

    # ───────── 全局记忆 ─────────
    # 轻量化全局剧情摘要（人物 / 事件索引），运行期缓存
    global_plot_memory: Dict

    # ───────── 流转控制（图调度用） ─────────
    # 当前应执行的分支：aggregate / generate / regenerate / revise / archive / next_loop / finish
    route: str
    # 本集关联的历史记忆（预检索结果，供所有评审共享）
    episode_memory: Dict
    # 上一轮评审后的统一修改指令（下发回生产节点）
    revise_instruction: Optional[Dict]
    # 人工复核标记（重试超限仍不通过）
    needs_human_review: bool

    # ───────── 多媒体合成（media_synthesizer 产出） ─────────
    # 本集合成视频的本地路径（合成失败为 None）
    video_path: Optional[str]


def initial_state(file_path: str,
                  chunk_size: int = 8000,
                  target_episode_count: Optional[int] = None) -> NovelState:
    """全新任务的初始状态。"""
    return NovelState(
        file_path=file_path,
        offset=0,
        chunk_size=chunk_size,
        current_chunk="",
        pending_scenes=[],
        new_scenes=[],
        current_episode=None,
        episode_script=None,
        episode_image_prompts=None,
        episode_tts_meta=None,
        review_result=None,
        retry_count=0,
        target_episode_count=target_episode_count,
        completed_episode_count=0,
        loop_finished=False,
        global_plot_memory={},
        route="",
        episode_memory={},
        revise_instruction=None,
        needs_human_review=False,
    )


# ───────── 状态序列化辅助（断点快照 / 调试查看） ─────────
def state_to_dict(state: NovelState) -> Dict[str, Any]:
    """将 TypedDict 状态转为普通 dict，供 SqliteSaver 持久化。"""
    return dict(state)


def state_from_dict(data: Dict[str, Any]) -> NovelState:
    """从 dict 还原 NovelState，补齐缺失字段（兼容旧快照）。"""
    base = initial_state(
        file_path=data.get("file_path", ""),
        chunk_size=data.get("chunk_size", 8000),
        target_episode_count=data.get("target_episode_count"),
    )
    base.update({k: v for k, v in data.items() if k in base or True})
    return base  # type: ignore
