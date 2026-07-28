"""
novel_pipeline.main
主入口：启动 / 断点续跑 / 配置加载 / 优雅终止。
对应设计文档 5.1 / 5.3。

用法:
  python main.py --novel /path/to/novel.txt --target 10
  python main.py --config config.yaml        # 使用配置文件覆盖默认值
  python main.py --resume                    # 从断点继续
"""
import argparse
import os
import sys
import uuid
from typing import Dict, Any

from config import get_config, set_config
from state import NovelState
from graph import build_graph
from agents import get_memory_agent


def _initial_state(novel_path: str, target: int = None) -> Dict[str, Any]:
    """构造全新初始状态。"""
    return {
        # 分片控制
        "file_path": novel_path,
        "offset": 0,
        "chunk_size": get_config().run.chunk_size,
        "current_chunk": "",
        # 剧情缓存
        "pending_scenes": [],
        "new_scenes": [],
        # 当前集产出
        "current_episode": None,
        "episode_script": None,
        "episode_image_prompts": [],
        "episode_tts_meta": [],
        "review_result": None,
        "retry_count": 0,
        # 生产进度
        "target_episode_count": target,
        "completed_episode_count": 0,
        "loop_finished": False,
        # 全局记忆
        "global_plot_memory": {},
        # 图运行期字段
        "format_valid": False,
        "format_errors": [],
        "prefetched_memory": None,
        "review_reports": [],
        "current_node": "",
        "episode_counter": 0,
    }


def _resume_state(graph, thread_id: str) -> Dict[str, Any] | None:
    """尝试从 SqliteSaver 读取断点状态。"""
    try:
        snapshots = list(graph.get_state_history(
            {"configurable": {"thread_id": thread_id}}
        ))
        if not snapshots:
            return None
        # 取最新一个有数据的 snapshot
        last = snapshots[0]
        if last and last.values and len(last.values) > 1:
            return last.values
        return None
    except Exception:
        return None


def run(novel_path: str, target: int = None, resume: bool = False, config_file: str = None):
    """主运行入口。"""
    if config_file and os.path.exists(config_file):
        set_config(config_file)

    cfg = get_config()

    # 续跑时若未提供 novel，从断点恢复；全新必须提供
    if not resume and not novel_path:
        print("[错误] 全新运行必须提供 --novel")
        sys.exit(1)
    if not novel_path:
        novel_path = ""  # 续跑时由 checkpointer 恢复 file_path

    # 全新运行校验小说文件；续跑跳过（由断点恢复）
    if not resume and not os.path.exists(novel_path):
        print(f"[错误] 小说文件不存在: {novel_path}")
        sys.exit(1)

    # 构建图
    graph, conn = build_graph(cfg.storage.sqlite_path)

    # 生成 thread_id（断点续跑用）
    thread_id = "novel_main_thread"
    config = {"configurable": {"thread_id": thread_id}}

    # 加载或初始化状态
    if resume:
        state = _resume_state(graph, thread_id)
        if state:
            print(f"[续跑] 加载断点状态成功，offset={state.get('offset',0)}，已完成 {state.get('completed_episode_count',0)} 集")
        else:
            print("[续跑] 未找到断点，全新启动")
            state = _initial_state(novel_path, target)
    else:
        state = _initial_state(novel_path, target)
        print(f"[启动] 全新任务，目标集数: {target if target else '全本'}")

    # 初始化记忆 Agent（首次运行建立空记忆库）
    mem = get_memory_agent()
    print(f"[记忆] 全局记忆库路径: {cfg.storage.memory_dir}")

    # 运行图
    print("=" * 60)
    print("进入主生产循环...")
    try:
        # 断点续跑：stream 传 None，由 checkpointer 恢复；全新：传 state
        input_state = None if resume and state else state
        for output in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, update in output.items():
                # 汇报进度
                if not isinstance(update, dict):
                    continue
                ep = update.get("current_episode")
                completed = update.get("completed_episode_count")
                if completed is not None:
                    print(f"[进度] 已完成 {completed} 集")
                if ep is not None and node_name == "episode_aggregator":
                    eid = ep.get("episode_id", "?")
                    print(f"[生产] 新集已聚合: {eid}")
    except KeyboardInterrupt:
        print("\n[手动停止] LangGraph 已自动保存最近节点状态，下次 --resume 可续跑")
    except Exception as e:
        print(f"[运行错误] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        print("[完成] 运行结束")


def main():
    parser = argparse.ArgumentParser(description="长篇小说多媒体剧集生产多 Agent 系统")
    parser.add_argument("--novel", default=None, help="小说 TXT 文件路径（续跑可不提供）")
    parser.add_argument("--target", type=int, default=None, help="目标生成集数，不设为全本")
    parser.add_argument("--resume", action="store_true", help="从断点继续")
    parser.add_argument("--config", default=None, help="配置文件路径 (yaml/json)")
    parser.add_argument("--chunk-size", type=int, default=None, help="单次读取字符上限")
    parser.add_argument("--max-retries", type=int, default=None, help="单集最大重试次数")
    args = parser.parse_args()

    # 先加载配置文件（设置 api_key 等），再读全局配置
    if args.config and os.path.exists(args.config):
        set_config(args.config)
    cfg = get_config()
    if args.chunk_size:
        cfg.run.chunk_size = args.chunk_size
    if args.max_retries:
        cfg.run.max_retries = args.max_retries

    # 全新运行必须提供 novel；续跑可不提供（从断点恢复 file_path）
    if not args.resume and not args.novel:
        parser.error("全新运行必须提供 --novel 参数")

    run(
        novel_path=args.novel or "",
        target=args.target,
        resume=args.resume,
        config_file=args.config,
    )


if __name__ == "__main__":
    main()
