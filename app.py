"""
novel_pipeline.app
Gradio 控制台——长篇小说多媒体剧集生产系统的运行时交互界面。

三区设计：
① 任务配置区：上传小说、目标集数、风格/横竖屏/音量参数
② 实时进度区：当前节点、进度条、滚动日志流
③ 成品浏览区：各集视频播放、script/prompt 文件查看
④ 提示词管理：在线编辑各 agent 的 system prompt（改动即时落盘 prompts/*.md）

启动：python app.py --config config.json
"""
# ── stdout/stderr 强制 UTF-8（Windows 控制台默认 GBK/CP936，
#    避免 app.py 自身 print 中文乱码 / UnicodeEncodeError ──
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import sys
import json
import time
import shutil
import threading
import subprocess
from typing import Optional

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import set_config, get_config, apply_book, StorageConfig
from prompts import load_prompt, save_prompt, list_prompts, ART_STYLES
from nodes.media_synthesizer import resynthesize_video, regenerate_episode_media


# ────────────── 书管理 ──────────────
_NOVELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "novels")


def _book_dir(book_name: str) -> str:
    return os.path.join(_NOVELS_DIR, book_name or "")


def list_books() -> list:
    """列出 novels/ 下所有书名。"""
    if not os.path.isdir(_NOVELS_DIR):
        return []
    return sorted([d for d in os.listdir(_NOVELS_DIR)
                   if os.path.isdir(os.path.join(_NOVELS_DIR, d))])


def book_dropdown_choices():
    return [(b, b) for b in list_books()]


def create_book(book_name: str):
    """新建书目录结构。返回下拉更新 + 状态。"""
    if not book_name or not book_name.strip():
        return gr.update(), "⚠️ 书名不能为空"
    name = book_name.strip()
    bdir = _book_dir(name)
    try:
        os.makedirs(os.path.join(bdir, "data"), exist_ok=True)
        os.makedirs(os.path.join(bdir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(bdir, "memory"), exist_ok=True)
        os.makedirs(os.path.join(bdir, "output"), exist_ok=True)
    except Exception as e:
        return gr.update(), f"❌ 创建失败: {e}"
    choices = book_dropdown_choices()
    return gr.Dropdown(choices=choices, value=name), f"✅ 书 '{name}' 已创建"


def _switch_book(book_name: str):
    """切换当前书：apply_book 重定向 storage 路径，返回书状态文本。
    不持久化到 config.json（运行期内存覆盖，避免写盘污染其他书）。
    """
    if not book_name:
        return "⚠️ 未选书", "", gr.update(), gr.update()
    apply_book(book_name)
    return f"📖 当前书：{book_name}", "", gr.update(), gr.update()


def _book_dir(book_name: str) -> str:
    """返回指定书的根目录绝对路径。"""
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "novels", book_name)


def _scan_book_files(book_name: str) -> list:
    """扫描指定书 data/ 下所有 txt，返回 [{name, size_mb, path}]。"""
    data_dir = os.path.join(_book_dir(book_name), "data")
    if not os.path.isdir(data_dir):
        return []
    files = []
    for fn in sorted(os.listdir(data_dir)):
        fp = os.path.join(data_dir, fn)
        if os.path.isfile(fp) and fn.lower().endswith(".txt"):
            files.append({"name": fn, "size_mb": round(os.path.getsize(fp) / 1024 / 1024, 1), "path": fp})
    return files


def _read_book_progress(book_name: str) -> dict:
    """从 sqlite 断点读取当前书的进度：当前文件/offset/队列/已完成集数。
    用 LangGraph SqliteSaver.get_state_history 读取（兼容各种序列化格式）。
    """
    db_path = os.path.join(_book_dir(book_name), "checkpoints", "checkpoint.sqlite")
    if not os.path.exists(db_path):
        return {}
    try:
        from graph import build_graph
        graph, conn = build_graph(db_path)
        # 扫描所有 novel_main_thread* 的 thread（兼容旧无前缀 + 新带 book 前缀）
        import sqlite3 as _sql
        c2 = _sql.connect(db_path)
        rows = c2.execute(
            "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE 'novel_main_thread%'"
        ).fetchall()
        c2.close()
        if not rows:
            conn.close()
            return {}
        best = {"done": -1}
        for (tid,) in rows:
            try:
                snaps = list(graph.get_state_history(
                    {"configurable": {"thread_id": tid}}
                ))
                if snaps:
                    s = snaps[0]
                    vals = s.values or {}
                    done = vals.get("completed_episode_count", 0)
                    if isinstance(done, int) and done > best["done"]:
                        best = {"thread": tid, "done": done,
                                "file_path": vals.get("file_path", ""),
                                "offset": vals.get("offset", 0),
                                "file_queue": vals.get("file_queue", []),
                                "completed": done,
                                "loop_finished": vals.get("loop_finished", False)}
            except Exception:
                pass
        conn.close()
        return best if best["done"] >= 0 else {}
    except Exception:
        return {}


def refresh_book_files(book_name: str):
    """刷新文件清单 + 进度显示。返回 Markdown 文本。
    标记每本 txt 的状态：🟢当前在读 / ⏳队列待读 / ✅已读完 / ⚪未识别。
    防止续跑到末尾才发现没衔接。"""
    if not book_name:
        return "⚠️ 请先选择书"
    files = _scan_book_files(book_name)
    progress = _read_book_progress(book_name)
    lines = [f"### 📂 {book_name} 文件清单\n"]
    if not files:
        lines.append("（data/ 目录无 txt 文件，请上传）")
    else:
        # 从断点提取当前文件和队列
        cur_file = progress.get("file_path", "") if progress else ""
        cur_file_name = os.path.basename(cur_file) if cur_file else ""
        queue = progress.get("file_queue", []) if progress else []
        queue_names = [os.path.basename(p) for p in queue] if queue else []
        done = progress.get("done", 0) if progress else 0
        finished = progress.get("loop_finished", False) if progress else False
        # 判断每本状态
        lines.append("| 序号 | 文件名 | 大小 | 状态 |")
        lines.append("|------|--------|------|------|")
        for i, f in enumerate(files, 1):
            status = "⚪ 未识别"
            fname = f["name"]
            # 匹配当前在读：断点 file_path 的 basename 等于本文件名
            if cur_file_name and cur_file_name == fname:
                if finished:
                    status = "✅ 已读完"
                else:
                    off = progress.get("offset", 0)
                    status = f"🟢 当前在读 (offset={off})"
            elif fname in queue_names:
                status = "⏳ 队列待读"
            elif done > 0 and not queue_names:
                # 有进度但队列空——当前在读文件可能名字不匹配（旧迁移残留），
                # 只剩一本且已有进度，标记当前在读
                status = f"🟢 当前在读 (offset={progress.get('offset',0)})"
            elif finished and done > 0:
                status = "✅ 已读完"
            lines.append(f"| {i} | {fname} | {f['size_mb']} MB | {status} |")
    lines.append("")
    if progress:
        lines.append(f"**断点**：已完成 {progress.get('done', 0)} 集"
                     + ("，小说已读完" if progress.get("loop_finished") else ""))
    else:
        lines.append("**断点**：无（未启动过生产）")
    return "\n".join(lines)


# ────────────── 文件日志（调试用，/tmp/app_debug.log） ──────────────
def _log_file(fmt, *args):
    import datetime
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] " + (fmt % args if args else fmt)
    with open("/tmp/app_debug.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ────────────── 全局运行状态（进程内共享） ──────────────
class _RunState:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.log_lines: list = []
        self.is_running: bool = False
        self.stop_requested: bool = False
        self.lock = threading.RLock()  # 可重入锁，避免同线程内嵌套获取死锁

    def reset(self):
        with self.lock:
            self.log_lines = []
            self.stop_requested = False


_RUN = _RunState()


def _build_run_cmd(novel_path, target, art_style, orientation, tts_speed,
                   bgm_volume, tts_volume, chunk_size, max_retries, resume=False,
                   novel_queue=None, book=None) -> list:
    """构造 main.py 运行命令。resume=True 加 --resume。novel_queue: 后续 txt 路径列表。"""
    cmd = [sys.executable, "main.py", "--config", "config.json"]
    if book:
        cmd += ["--book", book]
    if resume:
        cmd += ["--resume"]
    elif novel_path:
        cmd += ["--novel", novel_path]
    if novel_queue:
        cmd += ["--novel-queue"] + novel_queue
    if target and int(target) > 0:
        cmd += ["--target", str(int(target))]
    if chunk_size:
        cmd += ["--chunk-size", str(int(chunk_size))]
    if max_retries:
        cmd += ["--max-retries", str(int(max_retries))]
    return cmd


def _apply_runtime_config(art_style, orientation, tts_speed, bgm_volume, tts_volume,
                          enable_subtitles=False):
    """把 UI 参数写入 config.json（运行时覆盖），让 main.py 加载时生效。"""
    cfg_path = "config.json"
    if not os.path.exists(cfg_path):
        return "config.json 不存在"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    media = cfg.setdefault("media", {})
    media["art_style"] = art_style
    if orientation == "横屏 1920x1080 (16:9)":
        media["image_aspect_ratio"] = "16:9"
        media["video_resolution"] = "1920x1080"
    else:
        media["image_aspect_ratio"] = "9:16"
        media["video_resolution"] = "1080x1920"
    media["bgm_volume"] = float(bgm_volume)
    media["tts_volume"] = float(tts_volume)
    media["enable_subtitles"] = bool(enable_subtitles)
    # tts_speed 写入 run 段（material_generator 从 tts_meta.speed 读，这里改默认值无直接通道，
    # 实际 speed 由 material_generator prompt 控制；此处仅记录供参考）
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return None


def start_production(novel_file, target, art_style, orientation,
                     tts_speed, bgm_volume, tts_volume, chunk_size, max_retries,
                     enable_subtitles=False, resume=False, book=None):
    """启动生产进程。resume=True 时从断点续跑（忽略新上传文件）。
    book: 当前书名，指定后文件存到 novels/<book>/data/，命令传 --book。
    """
    import traceback as _tb
    _log_file("start_production 被调用，参数: novel_file=%r type=%s target=%s book=%s",
               novel_file, type(novel_file).__name__, target, book)
    try:
        with _RUN.lock:
            if _RUN.is_running:
                # ⚠️ 警告写 start_feedback（status_box 被 Timer 每2s覆盖，用户看不到）
                return refresh_status_simple(), "⚠️ 已有任务在运行中，请先停止后再点续跑/开始"
            _RUN.reset()
            _RUN.is_running = True

        novel_path = ""
        novel_queue = []   # 后续 txt 路径列表
        # 多书隔离：按书名存到 novels/<book>/data/，无书名回退到根目录 data/
        data_dir = os.path.join("novels", book, "data") if book else "data"
        os.makedirs(data_dir, exist_ok=True)
        if resume:
            _RUN.log_lines.append(f"[续跑] 从上次断点恢复 (书={book or '默认'})，不重新上传文件")
        elif novel_file is not None:
            # novel_file 可能是单文件(str)或多文件(list)——file_count="multiple" 返回 list
            files = novel_file if isinstance(novel_file, (list, tuple)) else [novel_file]
            _log_file("novel_file 数量=%d 类型=%s", len(files), type(novel_file).__name__)
            saved_paths = []
            for fi, f in enumerate(files):
                src = f
                if hasattr(src, "path"):
                    src = src.path
                elif hasattr(src, "name"):
                    src = src.name
                elif isinstance(src, dict):
                    src = src.get("path") or src.get("name") or ""
                if isinstance(src, str) and os.path.exists(src):
                    dest = os.path.join(data_dir, "novel_%d.txt" % fi) if fi else os.path.join(data_dir, "novel.txt")
                    shutil.copy(src, dest)
                    sz_mb = os.path.getsize(dest) / 1024 / 1024
                    orig_name = os.path.basename(src)
                    saved_paths.append(dest)
                    _RUN.log_lines.append(f"[上传] 第{fi+1}本 [{orig_name}] → {dest} ({sz_mb:.1f} MB)")
                else:
                    _RUN.log_lines.append(f"[上传警告] 第{fi+1}本无法定位路径 type={type(f).__name__}")
            if not saved_paths:
                with _RUN.lock:
                    _RUN.is_running = False
                return "❌ 无法定位任何上传文件", []
            novel_path = saved_paths[0]
            novel_queue = saved_paths[1:]   # 后续本进队列
            # 读取顺序总览：1→2→3...
            order_str = " → ".join(os.path.basename(p) for p in saved_paths)
            _RUN.log_lines.append(
                f"[多本] 共 {len(saved_paths)} 本，读取顺序: {order_str}"
            ) if novel_queue else _RUN.log_lines.append(
                f"[单本] 读取: {os.path.basename(novel_path)}"
            )
        else:
            # 未上传新文件：尝试复用该书已存的 novel.txt
            cached = os.path.join(data_dir, "novel.txt")
            if os.path.exists(cached):
                novel_path = cached
                sz_mb = os.path.getsize(novel_path) / 1024 / 1024
                _RUN.log_lines.append(f"[复用] 使用已有文件 {novel_path} ({sz_mb:.1f} MB)")
            else:
                with _RUN.lock:
                    _RUN.is_running = False
                return "❌ 请先上传小说 TXT 文件（首次运行必须上传）", []

        err = _apply_runtime_config(art_style, orientation, tts_speed,
                                     bgm_volume, tts_volume, enable_subtitles)
        if err:
            with _RUN.lock:
                _RUN.is_running = False
            return f"❌ 配置写入失败: {err}", []

        cmd = _build_run_cmd(novel_path, target, art_style, orientation,
                             tts_speed, bgm_volume, tts_volume, chunk_size, max_retries,
                             resume=resume, novel_queue=novel_queue, book=book)
        _RUN.log_lines.append(f"[启动] 命令: {' '.join(cmd)}")
        _RUN.log_lines.append(f"[参数] 风格={art_style} 方向={orientation} "
                              f"BGM={bgm_volume} TTS音量={tts_volume} 字幕={enable_subtitles} 续跑={resume}")
        _log_file("启动命令: %s", ' '.join(cmd))

        def _run_process():
            try:
                import os as _os
                env = dict(_os.environ)
                env["PYTHONUNBUFFERED"] = "1"  # 关键：让 main.py 的 print 立即 flush
                _RUN.process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
                    text=True, bufsize=1, encoding="utf-8", errors="replace",
                    env=env,
                )
                for line in iter(_RUN.process.stdout.readline, ''):
                    if _RUN.stop_requested:
                        break
                    _RUN.log_lines.append(line.rstrip())
                _RUN.process.wait()
                _RUN.log_lines.append(f"[结束] 退出码={_RUN.process.returncode}")
            except Exception as e:
                _RUN.log_lines.append(f"[错误] {e}")
                _log_file("进程错误: %s", e)
            finally:
                with _RUN.lock:
                    _RUN.is_running = False
                    _RUN.process = None

        threading.Thread(target=_run_process, daemon=True).start()
        return "✅ 生产已启动", "\n".join(_RUN.log_lines[-50:])
    except Exception as e:
        _log_file("start_production 异常: %s\n%s", e, _tb.format_exc())
        with _RUN.lock:
            _RUN.is_running = False
        return f"❌ 启动异常: {e}", _tb.format_exc()[-500:]


def resume_production(art_style, orientation, tts_speed, bgm_volume, tts_volume,
                      chunk_size, max_retries, target, enable_subtitles=False, book=None):
    """从断点续跑：复用上次的 offset/pending_scenes/已完成集数。"""
    # target 仍可覆盖（支持"已完成4集，再追加2集"）
    return start_production(
        None, target, art_style, orientation,
        tts_speed, bgm_volume, tts_volume, chunk_size, max_retries,
        enable_subtitles=enable_subtitles, resume=True, book=book,
    )


def stop_production():
    """停止生成进程。"""
    with _RUN.lock:
        if not _RUN.is_running or _RUN.process is None:
            return "⚠️ 无运行中的任务"
        _RUN.stop_requested = True
    # 优雅终止：发送 SIGINT（模拟 Ctrl+C，LangGraph 会保存断点）
    try:
        _RUN.process.send_signal(subprocess.SIGINT)
        _RUN.log_lines.append("[停止] 已发送中断信号，等待保存断点...")
    except Exception as e:
        _RUN.log_lines.append(f"[停止] 发送信号失败: {e}")
    return "⏹️ 正在停止（保存断点中）..."


def refresh_log():
    """刷新日志显示。"""
    return "\n".join(_RUN.log_lines[-200:])


def refresh_status():
    """刷新运行状态。"""
    if _RUN.is_running:
        return "🟢 运行中", gr.update(variant="danger", value="⏹️ 停止生成")
    else:
        return "⚪ 空闲", gr.update(variant="primary", value="▶️ 开始生成")


def refresh_status_simple():
    """周期刷新状态文本（Timer 用，只返回字符串）。"""
    return "🟢 运行中" if _RUN.is_running else "⚪ 空闲"


def list_episodes():
    """列出已完成的集。"""
    out_dir = get_config().storage.output_dir
    eps = []
    if os.path.isdir(out_dir):
        for d in sorted(os.listdir(out_dir)):
            ep_dir = os.path.join(out_dir, d)
            if os.path.isdir(ep_dir) and d.startswith("ep_"):
                mp4 = os.path.join(ep_dir, f"{d}.mp4")
                script = os.path.join(ep_dir, "script.txt")
                info = os.path.join(ep_dir, "episode_info.json")
                meta = {}
                if os.path.exists(info):
                    try:
                        with open(info, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                eps.append({
                    "id": d,
                    "has_video": os.path.exists(mp4),
                    "video_path": mp4,
                    "summary": meta.get("summary", "")[:60],
                    "scene_count": meta.get("scene_count", "?"),
                })
    return eps


def episode_dropdown_choices():
    eps = list_episodes()
    return [(f"第{e['id'].replace('ep_','')}集 · {e['scene_count']}场景", e['id']) for e in eps]


def load_episode_video(ep_id):
    """加载选中集的视频。"""
    if not ep_id:
        return None
    out_dir = get_config().storage.output_dir
    mp4 = os.path.join(out_dir, ep_id, f"{ep_id}.mp4")
    return mp4 if os.path.exists(mp4) else None


def load_episode_script(ep_id):
    """加载选中集的讲解文案。"""
    if not ep_id:
        return ""
    out_dir = get_config().storage.output_dir
    p = os.path.join(out_dir, ep_id, "script.txt")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return "（无文案文件）"


def load_episode_prompts(ep_id):
    """加载选中集的生图 prompt（返回 Python list/dict 供 gr.JSON 渲染）。"""
    if not ep_id:
        return None
    out_dir = get_config().storage.output_dir
    p = os.path.join(out_dir, ep_id, "image_prompts.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_episode_all(ep_id):
    """一次性加载视频+文案+prompts，避免多个 change 事件竞争。"""
    return load_episode_video(ep_id), load_episode_script(ep_id), load_episode_prompts(ep_id)


def refresh_episode_list():
    """刷新集列表并自动选中第一集，触发内容加载。"""
    choices = episode_dropdown_choices()
    first_val = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=first_val), *load_episode_all(first_val)


# ────────────── 单集重生 / 补字幕 ──────────────
def _run_single_ep_task(ep_id, fn, label):
    """后台线程跑 resynthesize_video / regenerate_episode_media，日志回流到 _RUN.log_lines。

    复用主生产的日志通道，Gradio Timer 会自动刷新显示。
    用 stdout tee 让 fn 内部所有 print（含 media_synthesizer_node 深层）都进日志窗。
    """
    if not ep_id:
        return "⚠️ 请先选择剧集"
    with _RUN.lock:
        if _RUN.is_running:
            return "⚠️ 有任务在运行，请先停止"
        _RUN.reset()
        _RUN.is_running = True

    _RUN.log_lines.append(f"[{label}] 目标: {ep_id}")

    class _Tee:
        """同时写原 stdout 和 log_lines 的 stdout 替身。"""
        def __init__(self, real):
            self.real = real
            self.encoding = getattr(real, "encoding", "utf-8")
            self.errors = getattr(real, "errors", "replace")
        def write(self, s):
            if s.strip():
                _RUN.log_lines.append(s.rstrip())
            try: self.real.write(s)
            except Exception: pass
        def flush(self):
            try: self.real.flush()
            except Exception: pass
        def reconfigure(self, **kw): pass

    def _worker():
        import contextlib as _ctx
        old = sys.stdout
        tee = _Tee(old)
        sys.stdout = tee
        try:
            fn(ep_id)
        except Exception as e:
            import traceback as _tb
            _RUN.log_lines.append(f"[{label}] 异常: {e}")
            _RUN.log_lines.append(_tb.format_exc()[-400:])
        finally:
            sys.stdout = old
            _RUN.log_lines.append(f"[{label}] 结束")
            with _RUN.lock:
                _RUN.is_running = False

    threading.Thread(target=_worker, daemon=True).start()
    return f"✅ {label} 已启动: {ep_id}"


def resynth_video(ep_id):
    """仅重合成视频（补字幕 / 换 BGM），秒级完成，不调任何 API。"""
    return _run_single_ep_task(ep_id, resynthesize_video, "补字幕/重合成")


def rerun_episode(ep_id):
    """重跑本集全部媒体（生图+TTS+视频），保留剧本/prompt 不变。"""
    return _run_single_ep_task(ep_id, regenerate_episode_media, "重跑本集")


# ────────────── 提示词管理 ──────────────
def load_prompt_content(name):
    """加载指定 agent 的 prompt 到编辑框。"""
    if not name:
        return ""
    return load_prompt(name, art_style=None) if name == "material_generator" else load_prompt(name)


def save_prompt_content(name, content):
    """保存编辑后的 prompt。"""
    if not name:
        return "⚠️ 未选择 agent"
    save_prompt(name, content)
    return f"✅ {name} 的 prompt 已保存（下一集生效）"


def reset_prompt(name):
    """恢复 prompt（从 git checkout）。"""
    if not name:
        return "⚠️ 未选择 agent", ""
    try:
        subprocess.run(["git", "checkout", f"prompts/{name}.md"],
                       capture_output=True, cwd=os.path.dirname(os.path.abspath(__file__)) or ".")
    except Exception:
        pass
    return f"✅ {name} 已从 git 恢复", load_prompt_content(name)


# ────────────── 书管理 ──────────────
def list_book_choices():
    """列出 novels/ 下所有书名，返回 Gradio Dropdown 选项。"""
    from config import StorageConfig
    books = StorageConfig.list_books()
    return [(b, b) for b in books] if books else []


def create_book(book_name):
    """新建一本书的目录结构。"""
    if not book_name or not book_name.strip():
        return gr.update(), "⚠️ 请输入书名", "⚠️ 请输入书名"
    book_name = book_name.strip()
    bdir = _book_dir(book_name)
    os.makedirs(os.path.join(bdir, "data"), exist_ok=True)
    os.makedirs(os.path.join(bdir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(bdir, "memory"), exist_ok=True)
    os.makedirs(os.path.join(bdir, "output"), exist_ok=True)
    # 更新下拉并选中新书
    return gr.update(choices=list_book_choices(), value=book_name), \
           f"✅ 新建书 [{book_name}] 完成，可上传 txt", refresh_book_files(book_name)


def on_book_change_fileonly(book_name):
    """切换书时刷新文件清单 + apply_book 路径。"""
    if not book_name:
        return "⚠️ 未选书"
    from config import apply_book
    apply_book(book_name)
    return refresh_book_files(book_name)


def on_book_change_episode(book_name):
    """切换书时刷新成品浏览集列表。自行 apply_book 确保路径已切换。"""
    if not book_name:
        return gr.update(), None, "", None
    from config import apply_book
    apply_book(book_name)  # 幂等，重复调用安全
    choices = episode_dropdown_choices()
    first_val = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first_val), *load_episode_all(first_val)


def on_book_change(book_name, ep_dd, ep_video):
    """切换书时：刷新文件清单 + 集列表 + config 路径（需集列表组件传入）。"""
    if not book_name:
        return "⚠️ 未选书", gr.update(), None
    from config import apply_book
    apply_book(book_name)
    file_md = refresh_book_files(book_name)
    ep_choices = episode_dropdown_choices()
    first_val = ep_choices[0][1] if ep_choices else None
    return file_md, gr.update(choices=ep_choices, value=first_val), load_episode_video(first_val)


# ────────────── 角色档案 ──────────────
def list_character_choices():
    """列出当前书的所有角色名。需先 apply_book 切到对应书。"""
    try:
        from agents.memory_manager import get_memory_agent
        mem = get_memory_agent()
        chars = mem.get_character_profiles()
        return [(f"{c.get('char_id','')} · {c.get('name','')}", c.get('char_id', '')) for c in chars]
    except Exception:
        return []


def load_character_profile(char_id):
    """加载角色档案到编辑框。返回 user_description + 完整档案 JSON。"""
    if not char_id:
        return "", {}, "⚠️ 未选择角色"
    try:
        from agents.memory_manager import get_memory_agent
        mem = get_memory_agent()
        c = mem.get_character(char_id) or {}
        user_desc = c.get("user_description", "")
        # 展示完整档案（只读）
        display = {k: v for k, v in c.items() if k != "user_description"}
        return user_desc, display, f"✅ 已加载 [{char_id}]"
    except Exception as e:
        return "", {}, f"❌ 加载失败: {e}"


def save_character_description(char_id, user_description):
    """保存用户编辑的角色描述到 characters.json。"""
    if not char_id:
        return "⚠️ 未选择角色"
    try:
        from agents.memory_manager import get_memory_agent
        mem = get_memory_agent()
        mem.save_user_description(char_id, user_description or "")
        return f"✅ [{char_id}] 的用户描述已保存（下集生图生效）"
    except Exception as e:
        return f"❌ 保存失败: {e}"


# ────────────── 构建 Gradio 界面 ──────────────
def build_ui():
    # ①②两列等高靠 gr.Row(equal_height=True)；干预 flex-grow：
    #   - col_progress 里的按钮/启动反馈框 flex-grow:0（不被拉伸）
    #   - 含 log_box 的 form 独自 flex-grow:1 吃多余空间
    #   - 日志 textarea 固定高度 + 内部滚动 = 滑动窗口
    css = """
        /* ② 标题块只占自然高度，不被撑高，让「运行状态」对齐「上传小说」*/
        #col-progress > .form { flex-grow: 0 !important; }
        #col-progress > .block { flex-grow: 0 !important; flex-shrink: 0 !important; }
        /* ② 含 log-box 的 form 吃掉多余空间 */
        #col-progress > .form:has(#log-box) { flex-grow: 1 !important; }
        /* ② 刷新按钮、启动反馈框不被拉伸 */
        #col-progress > button { flex-grow: 0 !important; flex-shrink: 0 !important; }
        /* ② 日志 textarea 固定高度 + 内部滚动 */
        #log-box textarea { resize: none; height: 520px !important; max-height: 520px !important; overflow-y: auto; }
        /* 下方文案区 */
        #ep-script textarea { resize: none; }
        /* 下方 ep_prompts 与 ep_script 等高：给最小高度 */
        #ep-prompts { min-height: 460px; }
        #ep-prompts .json { min-height: 400px; }
    """
    align_js = "() => {}"
    with gr.Blocks(title="小说剧集生产控制台") as app:
        gr.Markdown("# 📖 长篇小说多媒体剧集生产控制台")

        with gr.Tab("生产控制"):
            # 当前书选择 + 新建书
            with gr.Row():
                book_dd = gr.Dropdown(
                    choices=list_book_choices(), label="当前书",
                    info="多书隔离：每本书独立断点/记忆/成品。新建书先在右侧输入书名",
                    interactive=True, scale=3,
                )
                new_book_input = gr.Textbox(label="新书店名", placeholder="如：人道至尊1-500",
                                            scale=2)
                create_book_btn = gr.Button("📦 新建书", variant="secondary", scale=1)
                refresh_book_btn = gr.Button("🔄 刷新清单", scale=1)
            book_files_md = gr.Markdown("⚠️ 请先选择书", elem_id="book-files-md")
            # 切换书时刷新文件清单 + config 路径（集列表由 ep_choices 的 change 单独绑定）
            book_dd.change(fn=on_book_change_fileonly, inputs=book_dd, outputs=book_files_md)
            create_book_btn.click(fn=create_book, inputs=new_book_input,
                                  outputs=[book_dd, book_files_md, book_files_md])
            refresh_book_btn.click(fn=refresh_book_files, inputs=book_dd, outputs=book_files_md)
            # 周期刷新文件清单（运行中进度变化）
            book_timer = gr.Timer(value=5)
            book_timer.tick(fn=refresh_book_files, inputs=book_dd, outputs=book_files_md)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_id="col-task"):
                    gr.Markdown("### ① 任务配置")
                    novel_input = gr.File(label="小说文本（可多选，列表顺序=读取顺序，支持拖拽调整）",
                                          file_types=[".txt"], type="filepath",
                                          file_count="multiple", allow_reordering=True)
                    target_input = gr.Number(label="目标集数", value=1, precision=0,
                                              info="全新运行=总集数；续跑=追加几集（如已完成4集+填1→跑到第5集）")
                    art_style_dd = gr.Dropdown(
                        choices=["anime", "realistic"],
                        value="anime", label="生图风格",
                        info="anime=动漫 / realistic=写实"
                    )
                    orient_dd = gr.Dropdown(
                        choices=["横屏 1920x1080 (16:9)", "竖屏 1080x1920 (9:16)"],
                        value="横屏 1920x1080 (16:9)", label="视频方向"
                    )
                    tts_speed_dd = gr.Slider(0.8, 1.3, value=1.08, step=0.02,
                                             label="TTS 语速")
                    bgm_vol_sl = gr.Slider(0.0, 0.5, value=0.25, step=0.05,
                                           label="BGM 音量")
                    tts_vol_sl = gr.Slider(0.8, 2.0, value=1.25, step=0.05,
                                          label="TTS 音量增益")
                    subtitle_cb = gr.Checkbox(label="字幕（烧录到视频）", value=False)
                    chunk_input = gr.Number(label="分片字符数", value=8000, precision=0)
                    retries_input = gr.Number(label="单集最大重试", value=2, precision=0)
                    start_btn = gr.Button("▶️ 开始生成", variant="primary")
                    resume_btn = gr.Button("♻️ 从断点续跑", variant="secondary")
                    stop_btn = gr.Button("⏹️ 停止生成", variant="stop")

                with gr.Column(scale=1, elem_id="col-progress"):
                    gr.Markdown("### ② 实时进度")
                    status_box = gr.Textbox(label="运行状态", value="⚪ 空闲",
                                            interactive=False)
                    log_box = gr.Textbox(
                        label="日志流", lines=25, max_lines=50,
                        interactive=False, autoscroll=True, elem_id="log-box",
                    )
                    refresh_btn = gr.Button("🔄 刷新日志")
                    refresh_btn.click(fn=refresh_log, outputs=log_box)
                    timer = gr.Timer(value=2)
                    timer.tick(fn=refresh_log, outputs=log_box)
                    timer.tick(fn=refresh_status_simple, outputs=status_box)
                    start_feedback = gr.Textbox(label="启动反馈", interactive=False)
                    start_btn.click(
                        fn=start_production,
                        inputs=[novel_input, target_input, art_style_dd, orient_dd,
                                tts_speed_dd, bgm_vol_sl, tts_vol_sl,
                                chunk_input, retries_input, subtitle_cb, book_dd],
                        outputs=[status_box, start_feedback],
                    )
                    resume_btn.click(
                        fn=resume_production,
                        inputs=[art_style_dd, orient_dd, tts_speed_dd,
                                bgm_vol_sl, tts_vol_sl, chunk_input, retries_input,
                                target_input, subtitle_cb, book_dd],
                        outputs=[status_box, start_feedback],
                    )
                    stop_btn.click(fn=stop_production, outputs=status_box)

            with gr.Row():
                gr.Markdown("### ③ 成品浏览")
            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    ep_choices = gr.Dropdown(
                        choices=episode_dropdown_choices(),
                        label="选择剧集", interactive=True,
                    )
                    refresh_ep_btn = gr.Button("🔄 刷新集列表")
                    resynth_btn = gr.Button("📝 仅重合成视频（补字幕/换BGM）",
                                            variant="secondary", elem_id="btn-resynth")
                    rerun_btn = gr.Button("♻️ 重跑本集（生图+TTS+视频）",
                                          variant="stop", elem_id="btn-rerun")
                    ep_task_status = gr.Textbox(label="单集任务状态", interactive=False,
                                                value="", elem_id="ep-task-status")
                with gr.Column(scale=3):
                    ep_video = gr.Video(label="视频预览", height=520)
            with gr.Row(equal_height=True, elem_id="row-output"):
                with gr.Column(scale=1):
                    ep_script = gr.Textbox(label="讲解文案", lines=20, interactive=False, elem_id="ep-script")
                with gr.Column(scale=1):
                    ep_prompts = gr.JSON(label="生图 Prompt 列表", elem_id="ep-prompts")

                refresh_ep_btn.click(
                    fn=refresh_episode_list,
                    outputs=[ep_choices, ep_video, ep_script, ep_prompts],
                )
                # 切换书时刷新成品浏览集列表（补绑：ep_choices/ep_video 此时已创建）
                book_dd.change(fn=on_book_change_episode,
                               inputs=book_dd,
                               outputs=[ep_choices, ep_video, ep_script, ep_prompts])
                ep_choices.change(
                    fn=load_episode_all,
                    inputs=ep_choices,
                    outputs=[ep_video, ep_script, ep_prompts],
                )
                resynth_btn.click(
                    fn=resynth_video, inputs=ep_choices, outputs=ep_task_status,
                )
                rerun_btn.click(
                    fn=rerun_episode, inputs=ep_choices, outputs=ep_task_status,
                )
                # 周期刷新集列表下拉：续跑/重跑新生成的集自动出现，不必手动点刷新。
                # 只更新 choices，不覆盖当前 value（用户选中不变）。
                timer2 = gr.Timer(value=5)
                timer2.tick(fn=lambda: gr.Dropdown(choices=episode_dropdown_choices()),
                            outputs=ep_choices)

        with gr.Tab("提示词管理"):
            gr.Markdown("### 📝 在线编辑各 Agent 的 System Prompt\n"
                        "改动保存后即时写入 `prompts/*.md`，**下一集生产时生效**。\n"
                        "`material_generator` 中的 `{{ART_STYLE}}` 占位符会在运行时按风格下拉自动替换。")
            with gr.Row():
                prompt_selector = gr.Dropdown(
                    choices=list_prompts(), label="选择 Agent", interactive=True,
                )
                load_prompt_btn = gr.Button("📂 加载")
                save_prompt_btn = gr.Button("💾 保存", variant="primary")
                reset_prompt_btn = gr.Button("♻️ 从 git 恢复", variant="stop")
            prompt_editor = gr.Textbox(
                label="Prompt 内容（可直接编辑）", lines=30, max_lines=80,
                interactive=True,
            )
            prompt_status = gr.Textbox(label="操作状态", interactive=False)

            load_prompt_btn.click(fn=load_prompt_content, inputs=prompt_selector, outputs=prompt_editor)
            save_prompt_btn.click(fn=save_prompt_content, inputs=[prompt_selector, prompt_editor], outputs=prompt_status)
            reset_prompt_btn.click(fn=reset_prompt, inputs=prompt_selector, outputs=[prompt_status, prompt_editor])
            # 进入标签页自动加载第一个
            prompt_selector.change(fn=load_prompt_content, inputs=prompt_selector, outputs=prompt_editor)

        with gr.Tab("角色档案") as char_tab:
            gr.Markdown("### 🎭 全局角色描述管理\n"
                        "AI 解析小说时自动维护角色档案（外貌/年龄/身份等）。\n"
                        "**用户描述**（下方编辑框）优先级最高——填写后，生图时**严格以此为准**，AI 不会覆盖你的修改。\n"
                        "留空则 AI 继续用自动维护的外貌描述。")
            with gr.Row():
                char_selector = gr.Dropdown(
                    choices=list_character_choices(), label="选择角色", interactive=True,
                )
                char_refresh_btn = gr.Button("🔄 刷新角色列表")
            char_user_desc = gr.Textbox(
                label="用户描述（生图优先依据，AI 永不覆盖）",
                lines=8, max_lines=20, interactive=True,
                placeholder="填写你对角色外貌/服饰/风格的精确描述，留空则用 AI 自动维护的档案",
            )
            with gr.Row():
                char_load_btn = gr.Button("📂 加载档案")
                char_save_btn = gr.Button("💾 保存用户描述", variant="primary")
            char_profile_json = gr.JSON(label="AI 自动维护的档案（只读）")
            char_status = gr.Textbox(label="操作状态", interactive=False)

            char_load_btn.click(fn=load_character_profile, inputs=char_selector,
                                outputs=[char_user_desc, char_profile_json, char_status])
            char_save_btn.click(fn=save_character_description, inputs=[char_selector, char_user_desc],
                                outputs=char_status)
            char_selector.change(fn=load_character_profile, inputs=char_selector,
                                  outputs=[char_user_desc, char_profile_json, char_status])
            char_refresh_btn.click(fn=lambda: gr.update(choices=list_character_choices()),
                                   outputs=char_selector)
            # 进入角色档案 Tab 时自动刷新角色下拉（apply_book 已在切书时重置单例）
            char_tab.select(fn=lambda: gr.update(choices=list_character_choices()),
                            outputs=char_selector)
            # 切换书时也刷新角色下拉
            book_dd.change(fn=lambda b: gr.update(choices=list_character_choices()),
                           inputs=book_dd, outputs=char_selector)

    return app, css, align_js


def main():
    parser = __import__("argparse").ArgumentParser(description="小说剧集生产 Gradio 控制台")
    parser.add_argument("--config", default="config.json", help="配置文件")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    if os.path.exists(args.config):
        set_config(args.config)

    # 启动时若有书，默认选中第一本并 apply_book，让成品预览/角色档案立即可见
    from config import StorageConfig, apply_book
    books = StorageConfig.list_books()
    if books:
        apply_book(books[0])
        print(f"[启动] 默认选中书: {books[0]}")

    app, css, js = build_ui()
    app.launch(server_port=args.port, share=args.share, inbrowser=False,
               max_file_size="500mb", css=css, js=js)


if __name__ == "__main__":
    main()
