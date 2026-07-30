# 更新日志

所有重要版本变化均记录于此。格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。

---

## [Gradio 控制台版] — 2025-07

在 `b077928`（动漫画质+1080p 版）基础上新增 **Gradio Web 控制台**、提示词外部化、黑屏自动修复等功能。此版本即 `main` 分支当前 HEAD。

> Gradio 前的最后纯命令行版本保存在 `pre-gradio` 分支（指向 `b077928`）。

### 新增

#### 1. Gradio Web 控制台（`app.py`）

一、**任务配置区**
- 浏览器直接上传小说 TXT（≤500MB），无需命令行 `--novel`
- 可视化设置目标集数、生图风格（anime / realistic）、视频方向（横屏 1920×1080 / 竖屏 1080×1920）
- TTS 语速、BGM 音量、TTS 音量增益滑块
- 三个按钮：▶️ 开始生产 / ♻️ 从断点续跑 / ⏹️ 停止生产（停止只发信号，不杀进程，下集边界自然退出）

二、**实时进度区**
- 运行状态灯（⚪ 空闲 / 🟢 运行中 / 🟡 停止中）
- 日志流（每 2 秒自动刷新，子进程 stdout 实时回流）

三、**成品浏览区**
- 集列表下拉（格式：`第001集 · 5场景`），🔄 刷新按钮
- 视频预览（高度 520px，浏览器内直接播放）
- 讲解文案 / 生图 Prompt 列表（JSON 渲染）并排展示
- 切集自动加载该集视频+文案+prompt，刷新自动选第一集

四、**提示词管理区**
- 下拉选择 8 个 Agent 之一，加载其 `prompts/*.md` 内容到编辑框
- 💾 保存即时写盘，下一集生产时生效
- ♻️ 从 git 恢复：`git checkout -- prompts/<name>.md` 一键还原

#### 2. 提示词外部化（`prompts/`）
- 8 个 Agent 的 SYSTEM_PROMPT 抽离到独立 `.md` 文件：
  `plot_parser / episode_aggregator / material_generator / review_character / review_foreshadow / review_timeline / review_atmosphere / review_arbiter`
- `prompts/__init__.py` 加载器：`load_prompt(name, art_style)` / `save_prompt(name, content)` / `list_prompts()`
- `material_generator.md` 含 `{{ART_STYLE}}` 占位符，运行时按风格下拉（anime→动漫前缀 / realistic→写实前缀）自动替换

#### 3. 黑屏自动修复（`nodes/media_synthesizer.py`）
- 新增 `_retry_generate_images()`：对失败图/缺图单独重试一轮
- 占位图颜色从深蓝 `0x1a1a2e` 改为亮灰 `0x404050`，便于区分真实黑屏
- 黑屏检测阈值从 0.5% 提到 5%；超阈值时自动定位缺图/占位图→重试生图→重新合成视频

### 变更

#### 配置路径绝对化（`config.py`）
- `output_dir` / `sqlite_path` / `memory_dir` 从相对路径改为基于 `__file__` 的绝对路径
- `set_config()` 末尾 `os.makedirs(..., exist_ok=True)` 自动创建目录
- `config.json` / `config.example.json` 路径字段清空（`""`），让 dataclass 默认值生效
- storage 覆盖逻辑改为 `if v:` 过滤空值，避免空串覆盖默认绝对路径

#### 视频默认参数
- 分辨率 1280×720 → **1920×1080**（横屏 1080p）
- 生图风格默认 **anime**，前缀 `anime style, ancient Chinese mythology art style,`
- TTS 语速 1.0 → **1.08**
- BGM 音量 0.15 → **0.25**
- 新增 `tts_volume=1.25`：amix 前对 TTS 音轨 `volume=1.25` 增益

### 修复
- **Gradio 6 兼容**：`max_file_size` 从 `500`（字节）改为 `"500mb"`；`gr.File` 改 `type="filepath"` 并处理 `NamedString`；组件 `every` 参数废弃 → 改用 `gr.Timer` + `timer.tick()`；多个 `.change` 竞争 → 合并为单 `load_episode_all` 函数
- **死锁**：`_RUN.lock` 从 `Lock()` 改 `RLock()`，解决同线程嵌套获取死锁
- **日志不回流**：子进程加 `env["PYTHONUNBUFFERED"]="1"`
- **SAR 残留**：`setsar=1` 移到最后一次 scale 后
- **审核跳过**：`_call_image_api` / `_call_tts_api` 对 status_code 1026/1027 直接 return None，不重试

### 使用方法

#### 方式一：Gradio Web 控制台（推荐）

```bash
pip install -r requirements.txt          # 含 gradio>=4.0
python app.py --config config.json --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，在页面上完成上传小说→设参→开始生产→看日志→预览成品→改 prompt 全流程。

可选参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--config` | `config.json` | 配置文件 |
| `--port` | `7860` | 服务端口 |
| `--share` | 关 | 生成 gradio.live 公网临时链接 |

#### 方式二：命令行（pre-gradio 分支原有方式，仍保留）

```bash
python main.py --novel ./data/novel.txt --target 10 --config config.json
python main.py --resume --config config.json          # 断点续跑
```

#### 提示词在线编辑
控制台「提示词管理」标签页 → 选 Agent → 加载 → 改 → 保存。改动写入 `prompts/*.md`，**下一集生产时生效**（正在跑的当前集不受影响）。`material_generator` 中的 `{{ART_STYLE}}` 占位符运行时按风格下拉自动替换，无需手写。

---

## [动漫画质+1080p 版] — `b077928` — 2025-07-30

即 `pre-gradio` 分支指向的提交，Gradio 前最后一个版本。

- 生图默认动漫风格：SYSTEM_PROMPT 前缀 `anime style, ancient Chinese mythology art style, `
- 视频默认分辨率 1280×720 → 1920×1080（横屏 1080p）
- TTS 语速 1.0 → 1.08
- BGM 音量 0.15 → 0.25
- 新增 tts_volume=1.25
- SAR 修复：setsar=1 移到最后一次 scale 后
- 审核跳过：status_code 1026/1027 直接跳过不重试

更早历史见 `git log`。
