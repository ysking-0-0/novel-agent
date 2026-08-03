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
from config import set_config, get_config
from prompts import load_prompt, save_prompt, list_prompts, ART_STYLES
from nodes.media_synthesizer import resynthesize_video, regenerate_episode_media


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
                   bgm_volume, tts_volume, chunk_size, max_retries, resume=False) -> list:
    """构造 main.py 运行命令。resume=True 加 --resume。"""
    cmd = [sys.executable, "main.py", "--config", "config.json"]
    if resume:
        cmd += ["--resume"]
    elif novel_path:
        cmd += ["--novel", novel_path]
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
                     enable_subtitles=False, resume=False):
    """启动生产进程。resume=True 时从断点续跑（忽略新上传文件）。"""
    import traceback as _tb
    _log_file("start_production 被调用，参数: novel_file=%r type=%s target=%s",
               novel_file, type(novel_file).__name__, target)
    try:
        with _RUN.lock:
            if _RUN.is_running:
                return "⚠️ 已有任务在运行中，请先停止", "\n".join(_RUN.log_lines[-50:])
            _RUN.reset()
            _RUN.is_running = True

        novel_path = ""
        if resume:
            _RUN.log_lines.append("[续跑] 从上次断点恢复，不重新上传文件")
        elif novel_file is not None:
            novel_path = os.path.join("data", "novel.txt")
            os.makedirs("data", exist_ok=True)
            src = novel_file
            _log_file("novel_file 原始类型=%s repr=%s", type(src).__name__, repr(src)[:200])
            if hasattr(src, "path"):
                src = src.path
            elif hasattr(src, "name"):
                src = src.name
            elif isinstance(src, dict):
                src = src.get("path") or src.get("name") or ""
            _log_file("解析后 src=%s exists=%s", src, os.path.exists(src) if isinstance(src, str) else "N/A")
            if isinstance(src, str) and os.path.exists(src):
                shutil.copy(src, novel_path)
                sz_mb = os.path.getsize(novel_path) / 1024 / 1024
                _RUN.log_lines.append(f"[上传] 已保存到 {novel_path} ({sz_mb:.1f} MB)")
                _log_file("上传成功 %s %.1fMB", novel_path, sz_mb)
            else:
                _RUN.log_lines.append(f"[上传警告] 无法定位路径，type={type(novel_file).__name__}")
                with _RUN.lock:
                    _RUN.is_running = False
                return f"❌ 无法定位上传文件（type={type(novel_file).__name__}, val={novel_file!r:.80}）", []
        else:
            # 未上传新文件：尝试复用 data/novel.txt
            cached = os.path.join("data", "novel.txt")
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
                             resume=resume)
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
                      chunk_size, max_retries, target, enable_subtitles=False):
    """从断点续跑：复用上次的 offset/pending_scenes/已完成集数。"""
    # target 仍可覆盖（支持"已完成4集，再追加2集"）
    return start_production(
        None, target, art_style, orientation,
        tts_speed, bgm_volume, tts_volume, chunk_size, max_retries,
        enable_subtitles=enable_subtitles, resume=True,
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
    """
    if not ep_id:
        return "⚠️ 请先选择剧集"
    with _RUN.lock:
        if _RUN.is_running:
            return "⚠️ 有任务在运行，请先停止"
        _RUN.reset()
        _RUN.is_running = True

    _RUN.log_lines.append(f"[{label}] 目标: {ep_id}")

    def _worker():
        try:
            fn(ep_id)
        except Exception as e:
            import traceback as _tb
            _RUN.log_lines.append(f"[{label}] 异常: {e}")
            _RUN.log_lines.append(_tb.format_exc()[-400:])
        finally:
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
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_id="col-task"):
                    gr.Markdown("### ① 任务配置")
                    novel_input = gr.File(label="上传小说 TXT（支持大文件，≤500MB）",
                                          file_types=[".txt"], type="filepath")
                    target_input = gr.Number(label="目标集数", value=4, precision=0,
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
                                chunk_input, retries_input, subtitle_cb],
                        outputs=[status_box, start_feedback],
                    )
                    resume_btn.click(
                        fn=resume_production,
                        inputs=[art_style_dd, orient_dd, tts_speed_dd,
                                bgm_vol_sl, tts_vol_sl, chunk_input, retries_input,
                                target_input, subtitle_cb],
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

    return app, css, align_js


def main():
    parser = __import__("argparse").ArgumentParser(description="小说剧集生产 Gradio 控制台")
    parser.add_argument("--config", default="config.json", help="配置文件")
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    args = parser.parse_args()

    if os.path.exists(args.config):
        set_config(args.config)

    app, css, js = build_ui()
    app.launch(server_port=args.port, share=args.share, inbrowser=False,
               max_file_size="500mb", css=css, js=js)


if __name__ == "__main__":
    main()
