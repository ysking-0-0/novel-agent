"""
novel_pipeline.main
主入口：启动 / 断点续跑 / 配置加载 / 优雅终止。
对应设计文档 5.1 / 5.3。

用法:
  python main.py --novel /path/to/novel.txt --target 10
  python main.py --config config.yaml        # 使用配置文件覆盖默认值
  python main.py --resume                    # 从断点继续
"""
# ── stdout/stderr 强制 UTF-8（Windows 控制台默认 GBK/CP936，
#    Gradio app.py 用 encoding="utf-8" 读子进程 stdout；若 main.py 输出
#    用 locale 编码，中文字符会被 errors="replace" 替换成乱码方块，
#    且部分字符在 GBK 下无法编码会抛 UnicodeEncodeError ──必须在最前面
#    reconfigure，让后续所有 print 输出 UTF-8 字节，与 app.py 解码器对齐。
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import os
import sys
import uuid
from typing import Dict, Any

from config import get_config, set_config, apply_book
from state import NovelState
from graph import build_graph
from agents import get_memory_agent


def _initial_state(novel_path: str, target: int = None, novel_queue: list = None) -> Dict[str, Any]:
    """构造全新初始状态。novel_queue: 后续待读的 txt 路径列表。"""
    return {
        # 分片控制
        "file_path": novel_path,
        "file_queue": novel_queue or [],
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


def _find_latest_thread_id(graph, thread_base: str = "novel_main_thread") -> str:
    """扫描所有 <thread_base>* 的 thread，返回已完成集数最多的那个 thread_id。
    每次续跑 END 后会创建 <thread_base>_resume_N，需要找最新的。
    兼容旧断点：若新前缀扫不到，回退扫无前缀 'novel_main_thread%'。
    """
    import sqlite3
    try:
        db_path = get_config().storage.sqlite_path
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?",
            (thread_base + "%",)
        ).fetchall()
        # 兼容旧断点：新前缀扫不到时回退扫无 book 前缀的旧 thread
        if not rows and thread_base != "novel_main_thread":
            rows = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'novel_main_thread%'"
            ).fetchall()
        conn.close()
        if not rows:
            return thread_base
        best_tid = thread_base
        best_done = -1
        for (tid,) in rows:
            try:
                snaps = list(graph.get_state_history(
                    {"configurable": {"thread_id": tid}}
                ))
                if snaps:
                    done = snaps[0].values.get("completed_episode_count", 0) if snaps[0].values else 0
                    if isinstance(done, int) and done > best_done:
                        best_done = done
                        best_tid = tid
            except Exception:
                pass
        return best_tid
    except Exception:
        return thread_base


def _next_resume_thread_id(graph, thread_base: str = "novel_main_thread") -> str:
    """生成一个不与已有 thread 撞名的新 thread_id。

    扫描 checkpoints 表里所有 <thread_base>_resume_N，取最大序号 +1。
    避免用 completed_episode_count 拼（可能和已有 thread 重名 → 复用已 END
    的 thread 导致 LangGraph 空转中断）。
    """
    import sqlite3
    import re
    try:
        db_path = get_config().storage.sqlite_path
        conn = sqlite3.connect(db_path)
        resume_prefix = thread_base + "_resume_"
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?",
            (resume_prefix + "%",)
        ).fetchall()
        conn.close()
        max_n = 0
        for (tid,) in rows:
            m = re.match(re.escape(resume_prefix) + r"(\d+)$", tid)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"{resume_prefix}{max_n + 1}"
    except Exception:
        # 退化：用时间戳保证唯一
        import time
        return f"{thread_base}_resume_{int(time.time())}"


def _resume_state(graph, thread_id: str, target_override: int = None) -> Dict[str, Any] | None:
    """尝试从 SqliteSaver 读取断点状态。

    target_override: 若提供，覆盖断点内的 target_episode_count。
    用于"上轮已到目标集数但想继续生产更多集"的场景——
    旧断点已 END(next=空)，直接 invoke 不会动，需要构造可继续的初始状态。
    """
    try:
        snapshots = list(graph.get_state_history(
            {"configurable": {"thread_id": thread_id}}
        ))
        if not snapshots:
            return None
        last = snapshots[0]
        if not (last and last.values and len(last.values) > 1):
            return None
        state = dict(last.values)
        # 注入 target 覆盖（让"已完成2集"后继续跑到新目标3集）
        if target_override is not None:
            state["target_episode_count"] = target_override
            # 关键：重置路由入口，让图从 text_chunker 重新启动主循环
            # loop_finished 保持原值；pending_scenes 保留以聚合为下一集
        return state
    except Exception:
        return None


def run(novel_path: str, target: int = None, resume: bool = False, config_file: str = None,
        novel_queue: list = None, book: str = None):
    """主运行入口。book: 书名，指定后 storage 路径重定向到 novels/<book>/ 下。"""
    if config_file and os.path.exists(config_file):
        set_config(config_file)

    # 多书隔离：在 get_config 之前 apply_book，确保后续所有路径用书的独立目录
    if book:
        apply_book(book)
        print(f"[书] 当前书: {book} → {get_config().storage.output_dir}")
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

    # 生成 thread_id（断点续跑用）。多书隔离：加 book 前缀，防不同书串断点
    book_tag = (book or "default").replace(" ", "_")
    thread_base = f"novel_main_thread__{book_tag}"
    thread_id = thread_base
    config = {"configurable": {"thread_id": thread_id}}

    # 加载或初始化状态
    if resume:
        # 续跑时找已完成集数最多的 thread（每次 END 后会创建 thread_base_resume_N）
        latest_tid = _find_latest_thread_id(graph, thread_base)
        if latest_tid != thread_id:
            print(f"[续跑] 检测到最新断点 thread: {latest_tid}")
            thread_id = latest_tid
            config = {"configurable": {"thread_id": thread_id}}
        # 续跑时 target 语义为"增量"：已完成4集 + --target 1 = 跑到第5集
        # 命令行 --target 优先，未指定则用 config.json 的值
        if target is not None:
            target_increment = target
        else:
            target_increment = cfg.run.target_episode_count
        state = _resume_state(graph, thread_id, target_override=None)  # 先读原始状态，不加 target
        if state:
            done = state.get("completed_episode_count", 0)
            # 增量转绝对值：已完成数 + 增量
            target_override = done + target_increment
            state["target_episode_count"] = target_override
            # 检测断点是否已 END（next 为空）：此时传 None 不会继续，需要全新输入
            snap = list(graph.get_state_history(config))
            is_finished = bool(snap) and snap[0].next == ()
            if is_finished:
                print(f"[续跑] 旧断点已结束（已完成{done}集），追加 {target_increment} 集→新目标 {target_override}，从 offset={state.get('offset',0)} 续跑")
                # 用全新 thread_id 续跑，避免和已 END 的状态冲突。
                # 关键：不能用 "resume_%d" % done 拼——done 可能和已有 thread 撞名
                # （如已存在 resume_5 时再次续跑 done=5 会复用已 END 的 resume_5，
                #  把新 input 塞进已结束的 thread → LangGraph 空转/异常中断，只跑1集就停）。
                # 改为扫描所有 novel_main_thread_resume_N 取最大序号 +1，保证唯一。
                thread_id = _next_resume_thread_id(graph, thread_base)
                config = {"configurable": {"thread_id": thread_id}}
                # state 作为全新输入传入（不走 None 恢复）
                input_state = state
            else:
                print(f"[续跑] 加载断点状态（已完成{done}集），追加 {target_increment} 集→新目标 {target_override}，offset={state.get('offset',0)}")
                # 未 END：用 update_state 注入新 target，再 stream(None) 续跑
                graph.update_state(config, {"target_episode_count": target_override})
                input_state = None
        else:
            print("[续跑] 未找到断点，全新启动")
            state = _initial_state(novel_path, target, novel_queue)
            input_state = state
    else:
        state = _initial_state(novel_path, target, novel_queue)
        input_state = state
        q_info = f"，后续 {len(novel_queue)} 本" if novel_queue else ""
        print(f"[启动] 全新任务，目标集数: {target if target else '全本'}{q_info}")

    # 初始化记忆 Agent（首次运行建立空记忆库）
    mem = get_memory_agent()
    print(f"[记忆] 全局记忆库路径: {cfg.storage.memory_dir}")

    # 运行图
    print("=" * 60)
    print("进入主生产循环...")
    # 节点中文名映射，用于进度展示
    NODE_LABELS = {
        "text_chunker": "文本分片", "plot_parser": "剧情解析",
        "episode_aggregator": "剧集聚合", "episode_aggregator_force": "末尾强制打包",
        "material_generator": "素材生成", "format_validator": "格式校验",
        "memory_prefetch": "记忆预检索", "parallel_reviews": "并行评审",
        "review_arbiter": "评审仲裁", "retry_counter": "重试计数",
        "persistence": "归档", "media_synthesizer": "多媒体合成",
    }
    try:
        for output in graph.stream(input_state, config=config, stream_mode="updates"):
            for node_name, update in output.items():
                if not isinstance(update, dict):
                    continue
                label = NODE_LABELS.get(node_name, node_name)
                # 汇报当前节点执行
                ep = update.get("current_episode")
                completed = update.get("completed_episode_count")
                fe = update.get("format_errors")
                rr = update.get("review_result")
                # 当前集上下文
                done = input_state.get("completed_episode_count", state.get("completed_episode_count", 0)) if isinstance(input_state, dict) else state.get("completed_episode_count", 0)
                target_n = state.get("target_episode_count")
                target_disp = f"/{target_n}" if target_n else ""
                # 评审结果
                if rr and isinstance(rr, dict):
                    verdict = rr.get("verdict", "?")
                    print(f"  [{label}] 评审结论: {verdict}")
                if fe:
                    errs = fe if isinstance(fe, list) else [fe]
                    print(f"  [{label}] 格式错误({len(errs)}): {str(errs[0])[:80]}")
                if completed is not None:
                    print(f"[进度] 已完成 {completed}{target_disp} 集")
                if ep is not None and node_name in ("episode_aggregator", "episode_aggregator_force"):
                    eid = ep.get("episode_id", "?") if isinstance(ep, dict) else "?"
                    sc = len(ep.get("scenes", [])) if isinstance(ep, dict) else 0
                    print(f"[生产] 新集聚合: {eid} ({sc} 场景)")
                elif node_name in ("text_chunker", "plot_parser", "material_generator",
                                   "format_validator", "memory_prefetch", "parallel_reviews",
                                   "review_arbiter", "retry_counter", "persistence", "media_synthesizer"):
                    # 阶段进度
                    extra = ""
                    if node_name == "text_chunker":
                        off = update.get("offset", 0)
                        extra = f" offset={off}"
                    elif node_name == "material_generator":
                        ni = len(update.get("episode_image_prompts") or [])
                        nt = len(update.get("episode_tts_meta") or [])
                        extra = f" 生图prompt={ni} TTS段={nt}"
                    elif node_name == "persistence":
                        vid = update.get("video_path")
                        extra = " 视频=已归档" if vid else ""
                    print(f"  [{label}] 执行{extra}")
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
    parser.add_argument("--novel-queue", nargs="*", default=None,
                        help="后续待读的 TXT 路径列表（超长篇拆成多本时用，按顺序自动衔接）")
    parser.add_argument("--target", type=int, default=None, help="目标生成集数，不设为全本")
    parser.add_argument("--resume", action="store_true", help="从断点继续")
    parser.add_argument("--config", default=None, help="配置文件路径 (yaml/json)")
    parser.add_argument("--chunk-size", type=int, default=None, help="单次读取字符上限")
    parser.add_argument("--max-retries", type=int, default=None, help="单集最大重试次数")
    parser.add_argument("--book", default=None, help="书名（多书隔离：storage 路径重定向到 novels/<book>/）")
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
        novel_queue=args.novel_queue,
        book=args.book,
    )


if __name__ == "__main__":
    main()
