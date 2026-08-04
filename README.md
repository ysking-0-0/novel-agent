# 长篇小说多媒体剧集生产多 Agent 系统

基于 **LangGraph 状态机** 的长篇小说（百万字级）批量拆解生产系统：从纯文本小说自动拆解为可直接用于短视频制作的多媒体素材包，**端到端产出讲解视频**（口播文案 → 语音合成 → AI 配图 → FFmpeg 拼接成片）。

> 设计文档：`任务文档.md`（阶段二完整版，9 个 LLM Agent + 7 个纯代码节点）
> 更新日志：`CHANGELOG.md`（版本变化记录与使用说明）

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│ 调度层：LangGraph StateGraph 状态机                  │
│ （硬编码流转规则 + 条件边，无LLM调度，100%可预测）    │
├─────────────────────────────────────────────────────┤
│ 生产Agent层（3个，绑定高速通用模型）                 │
│ 剧情解析Agent → 剧集聚合Agent → 多媒体素材生成Agent  │
├─────────────────────────────────────────────────────┤
│ 评审Agent层（5个，绑定强推理长文本模型）             │
│ 人物/伏笔/剧情/氛围 四专精并行 → 评审汇总仲裁Agent   │
├─────────────────────────────────────────────────────┤
│ 支撑服务层（1个，按需调用）                           │
│ 全局记忆管理Agent（人物/事件/伏笔台账 + 向量检索）    │
├─────────────────────────────────────────────────────┤
│ 多媒体合成层（1个纯代码节点）                        │
│ 定妆照生成 → AI生图 → TTS合成 → FFmpeg视频拼接       │
├─────────────────────────────────────────────────────┤
│ 存储层                                               │
│ Sqlite断点快照 + 本地成品文件 + 向量记忆库           │
└─────────────────────────────────────────────────────┘
```

### 流水线（12 步主循环）

```
START → text_chunker
     → [route_after_chunking] → plot_parser / episode_aggregator_force / END
plot_parser → episode_aggregator
     → [route_after_aggregation] → material_generator / text_chunker(回读)
material_generator → format_validator
     → [route_after_format_check] → memory_prefetch / material_generator(局部修正) / persistence(超限放行)
memory_prefetch → parallel_reviews(4专精并行) → review_arbiter
     → [route_after_arbiter] → persistence / retry_counter → material_generator
persistence → media_synthesizer → [route_after_persistence] → text_chunker / END
```

---

## 工作原理与节点详解

### 一、文本分片（text_chunker）

纯代码节点，适配 500 万字 + 长篇。

- **字节 offset 游标**：用 `file.seek(offset)` 二进制读取，不加载全文，内存占用恒定（单次 chunk_size=8000 字符 + 2000 余量）
- **章节边界自动对齐**：内置中文章节标题正则（第X章/回、Chapter N、卷X），读取后向后找下一个章节标题作为切点，避免把章节腰斩；找不到章节标题时退化为段落边界（取 80% 位置的换行符）
- **offset 推进**：`new_offset = old_offset + chunk.encode('utf-8')`，按实际字节推进，断点续跑的核心标识
- **文末检测**：读取字节数 < 预期 → `loop_finished=True`，触发最终集成打包

### 二、生产层（3 个 LLM Agent）

#### 2.1 剧情解析 Agent（plot_parser）

**事实基准节点**——所有下游内容的事实源头。

读入 current_chunk，调用记忆 Agent 召回关联历史上下文，把原文拆解为最小剧情单元 Scene 列表。每个 Scene 必须包含**完整因果链**：

| 字段 | 含义 |
|---|---|
| `cause` | 前置诱因（什么导致这个场景发生） |
| `motivation` | 人物动机（为何这样做） |
| `core_action` | 核心行为（关键事件/动作） |
| `immediate_result` | 直接结果（场景结束时发生了什么） |
| `long_term_impact` | 远期影响（对后续剧情的潜在影响） |
| `characters` | 涉及人物 + 每人的 state_change |
| `foreshadows` | 埋设/回收的伏笔（plant/resolve + status） |
| `key_items` | 关键道具/世界观设定 |

严格约束：只从原文提取事实，不得臆造、不剧透后续、不丢失细节。

#### 2.2 剧集聚合 Agent（episode_aggregator）

把连续 Scene 打包为完整 Episode。判定成集的三标准（同时满足）：
1. **叙事闭环**：有起因-发展-结果，能独立观看
2. **体量适中**：内容足够支撑 2-4 分钟短视频讲解
3. **悬念收尾**：结尾留适度悬念，吸引下一集

- 不足成集的场景留存 `pending_scenes`，下一轮新文本补充后继续聚合（解决按章节硬切导致的剧情腰斩）
- 文末 `loop_finished=True` 时强制把所有遗留 pending 打包为最终集
- 标记本集关联的历史伏笔 ID（`linked_foreshadow_ids`），供下游生成与评审使用

#### 2.3 多媒体素材生成 Agent（material_generator）

基于 Episode 因果链，一次性并行产出三类素材：

| 媒体 | 描述 |
|---|---|
| 讲解文案 | 600-1200 字口播文本，用『』分段，每段对应一个画面节拍 |
| 生图 Prompt | List[dict]，每个含 prompt（以 `ancient Chinese mythology art style, ` 开头）、narration_segment（对应第几段语音）、mood（情绪关键词）。图片粒度细：每个场景/动作一张图，图片数通常多于语音段数 |
| TTS 参数 | 每段 text/voice/emotion/speed/pause_after，与 script 段段对齐 |

约束：不新增剧情、不篡改事实，文案是对原文的口语化转述。调用记忆 Agent 取关联人物档案，确保生图 Prompt 严格沿用人物设定。

### 三、格式校验（format_validator）

纯代码节点，校验三类产出的 JSON 结构、字段完整性、ID 规范性。

**死循环防护（本次修复重点）**：校验失败回 material_generator 局部修正，但复用 `retry_count` 计数；超过 `max_retries` 仍不通过时，打 `manual_review` 标记后**直送 persistence 归档**（绕过 review），不再无限重试。三态路由：

- 通过 → `memory_prefetch`
- 失败未超限 → `material_generator`（retry_count+1）
- 失败超限 → `persistence`（带 manual_review）

### 四、记忆预检索（memory_prefetch）

纯代码节点，一次性从记忆 Agent 拉取本集关联记忆，写入 state 供所有评审共享，避免 4 个评审 Agent 重复检索。拉取内容：
- 本集涉及人物档案
- 未解伏笔台账
- 向量召回的相关历史场景（解决长线伏笔回收）
- 近期时序事件

### 五、评审层（4 个专精并行 + 1 个仲裁）

#### 5.1 四专精评审

4 个评审 Agent 各管一维，只做对错核查、不参与创作，保证客观性：

| Agent | 维度 | 核查项 | 缺陷等级 |
|---|---|---|---|
| `review_character` | 人物 | 性格一致、动机合理、关系正确、外貌设定 | critical / minor |
| `review_foreshadow` | 伏笔 | 埋设声明、回收兑现、跨集连续性 | critical / minor |
| `review_timeline` | 剧情 | 事件时序、因果逻辑、与前集衔接 | critical / minor |
| `review_atmosphere` | 氛围 | 情绪匹配、光影与剧情情绪对应、TTS 情绪/语速贴 | critical / minor |

> 当前为顺序执行（兼容单线程 SqliteSaver），每个 Agent 内独立调 LLM；后续可用 asyncio.gather 优化为真正并行。

#### 5.2 评审汇总仲裁 Agent（review_arbiter）

合并 4 份报告，去重同类问题，解决冲突（以原著事实为唯一标准），输出三态裁定：

| verdict | 含义 | 路由 |
|---|---|---|
| `pass` | 全部通过 | persistence 归档 |
| `minor_revise` | 轻微瑕疵 | 回 material_generator 局部微调 |
| `regenerate` + 未超限 | 严重错误 | retry_counter → 整集重生成 |
| `regenerate` + 超限 | 重试耗尽 | persistence（标记 manual_review） |

### 六、归档（persistence）

纯代码节点。按完成序号赋 `ep_xxx`，每集独立目录写入：
- `episode_info.json`（基础信息、因果概述、关联伏笔、review_result）
- `script.txt` / `image_prompts.json` / `tts_meta.json` / `original_snippet.txt`
- 递增 `completed_episode_count`，触发记忆增量更新（本集场景/伏笔/人物入库）
- 清理临时字段，`retry_count` 归零

### 七、多媒体合成（media_synthesizer）★新增

纯代码节点，persistence 后自动触发，把上一阶段产出的 Prompt/参数落成真实媒体文件并拼成讲解视频。

**人物一致性方案**：为每个主要角色预生成一张「定妆照」（portrait prompt + 中性背景），后续该角色出现的所有画面生成时都带 `reference_image`（定妆照 base64）作为参考，确保外貌在整集内保持一致。定妆照缓存在 `memory/character_portraits/` 跨集复用。

**三阶段流程**：

| 阶段 | 操作 | 接口 |
|---|---|---|
| 1. 生图 | 遍历 image_prompts（List[dict]，含 narration_segment），并发（默认 3）调生图接口，带定妆照参考。Prompt 必须以 `ancient Chinese mythology art style, ` 开头 | MiniMax `/v1/image_generation` (model=image-01) |
| 2. TTS | 遍历 tts_meta，串行（防限流）调语音合成，解码 hex 音频落盘 | MiniMax `/v1/t2a_v2` (model=speech-02-hd) |
| 3. 视频 | 按 narration_segment 把图片分组到对应语音段，同段内多图均分音频时长展示（`-ss` 切片）；每段 = 图片+对应音频切片 → FFmpeg 合成片段 mp4；concat 拼接 → 完整视频 | imageio-ffmpeg (静态 ffmpeg 二进制) |

**关键工程坑与修复**（实测发现）：

| 问题 | 根因 | 修复 |
|---|---|---|
| TTS 返回乱码 | MiniMax `data.audio` 是 **hex 字符串**非 base64 | `_decode_audio` 自适应 hex/base64 |
| TTS 部分 4xx | `tense` 等值不在 MiniMax emotion 白名单 | `_resolve_emotion` 映射到支持的 7 种 (neutral/happy/sad/angry/disgusted/surprised/calm) |
| 片段合成失败 | imageio-ffmpeg 不带 ffprobe | `_audio_duration` 改用 `ffmpeg -i` 解析 stderr |
| concat 拼接失败 | concat demuxer 按相对路径重复拼接 | concat.txt 用绝对路径 |
| 图片 URL 过期 | OSS 签名 URL 24 小时失效 | 生成后即时下载落盘 |
| 字幕显示方块 | libass (ffmpeg `subtitles` 滤镜) 在本环境 HarfBuzz shaping 缺陷，无法渲染 CJK | 改用 **PIL 渲染每句字幕为透明 PNG + ffmpeg overlay 叠加**，绕开 libass |
| 字幕字体找不到 (Windows原生) | `_detect_cjk_font` 仅写 WSL 路径 `/mnt/c/Windows/Fonts/`，Windows 原生 Python 找不到 → fallback `load_default()` 不支持中文 | 候选列表追加 `C:\Windows\Fonts\` 原生路径（读 `%WINDIR%`），三环境全覆盖 |

**音色映射**：`tts_meta.voice` 字段值（角色名或 `narrator`）→ MiniMax voice_id 通过 `voice_mapping` 配置映射。当前内置音色：
- 旁白 → `Chinese_gravelly_storyteller_nv1`（沉稳说书人）
- 女性角色（如薪火）→ `Chinese (Mandarin)_Sweet_Lady`（甜美女声）
- 男性角色（如钟岳）→ `Chinese_worker_male`
- 默认音色 → `Chinese_gravelly_storyteller_nv1`
- 可在 config.json 的 `media.voice_mapping` 按角色名添加/覆盖

### 八、全局记忆管理 Agent（memory_manager）

全系统唯一记忆读写入口，单例，线程安全。维护三大结构化知识库 + 向量索引：

| 知识库 | 键 | 值 |
|---|---|---|
| 人物档案库 | char_id | 性格/外貌/能力/关系 + **user_description（用户手填，AI 不可覆盖）** |
| 时序事件库 | event_id | 因果链/时间顺序 |
| 伏笔台账 | foreshadow_id | 埋设/回收状态 |

- 向量库（FAISS）按需召回相关历史场景，解决长线伏笔回收，避免上下文溢出
- 所有 Agent 不得直接改记忆库，统一走 MemoryManagerAgent
- 每集归档后增量更新（update_from_episode + save）
- **角色档案双区设计**：`appearance`/`age`/`identity`/`attire`/`personality` 由 plot_parser 每集从小说抽取并 upsert（首次写入后锁定，保证人物一致性）；`user_description` 字段由用户通过「角色档案」Tab 手填，AI 的 upsert 在 `_USER_FIELDS` 保护下**永不覆盖**。material_generator 生图时优先用 `user_description`（加"【用户指定·优先】"前缀传给 LLM），为空才回退 AI 维护的 appearance+attire。
- **多书切换重置**：`apply_book()` 切书后调 `reset_memory_store()` 清空进程内单例，下次 `get_memory_agent()` 重新加载新书目录的记忆库

### 九、断点续跑

SqliteSaver 自动保存每个节点完成后的完整 NovelState 快照。崩溃 / 手动停止 / 断电后，`--resume` 从最新 checkpoint 无缝恢复。`offset` 字节游标 + sqlite 快照双重保障。

**多书隔离**：thread_id 拼成 `novel_main_thread__<书名>` 前缀，不同书的断点互不串扰。续跑时自动找当前书下完成集数最多的 thread；旧版本（无书名前缀的 `novel_main_thread`）会自动回退兼容。每次一集 END 后续跑会新建 `novel_main_thread__<书名>_resume_N` 形式的增量 thread。

### 十、条件路由（5 个路由函数）

纯代码，无 LLM 参与，100% 可预测：

| 路由函数 | 触发点 | 分支 |
|---|---|---|
| `route_after_chunking` | text_chunker 后 | 文末+无pending→END / 文末+有pending→force_pack / 正常→plot_parser |
| `route_after_aggregation` | episode_aggregator 后 | 成集→material_generator / 不足→text_chunker(回读) |
| `route_after_format_check` | format_validator 后 | 通过→memory_prefetch / 失败未超限→material_generator / 失败超限→persistence |
| `route_after_arbiter` | review_arbiter 后 | pass→persistence / minor_revise→material_generator / regenerate未超限→retry_counter / regenerate超限→persistence |
| `route_after_persistence` | media_synthesizer 后 | 达目标集数→END / 文末且无pending→END / 否则→text_chunker |

## 快速开始

本系统提供两种使用方式：**Gradio Web 控制台**（推荐，鼠标点选全流程）与**命令行**（脚本化批量跑）。

### 0. 你需要准备什么

运行本系统**只需要两样东西**：

| 材料 | 说明 | 获取方式 |
|---|---|---|
| **MiniMax API Key** | 调用大模型/生图/TTS 三个接口的统一凭证 | 在 [MiniMax 开放平台](https://platform.minimaxi.com/) 注册账号 → 创建 API Key，形如 `sk-cp-xxxxxxxx` |
| **一部小说文本文件** | 纯文本 `.txt`，UTF-8 或 GBK 编码均可（系统自适应解码）。中文小说最佳，带「第X章」类章节标题时分片更精准 | 任意来源（已购电子书、公开版权小说等），存成本地 txt 即可 |

> 不需要：FFmpeg（系统自带静态二进制 via imageio-ffmpeg）、向量数据库（FAISS 内嵌）、GPU（纯 CPU 可跑，所有推理在 MiniMax 云端）。

### 1. 下载代码并安装依赖

```bash
git clone https://github.com/ysking-0-0/novel-agent.git
cd novel-agent          # 项目根目录（即本仓库）
pip install -r requirements.txt
```

依赖清单（requirements.txt）：`langchain-openai`、`langgraph`、`langgraph-checkpoint-sqlite`、`requests`、`Pillow`、`imageio-ffmpeg`、`faiss-cpu`、**`gradio>=4.0`**（Web 控制台）等。Python 3.11+。

### 2. 配置 API Key（二选一）

**方式一：环境变量（推荐，最简单）**
```bash
export MINIMAX_API_KEY="sk-cp-你的真实key"
```

**方式二：配置文件（同时想改模型/音色/分辨率时用）**
```bash
cp config.example.json config.json
```
然后编辑 `config.json`，至少把 `model.api_key` 改成你的真实 key。config.json 已在 .gitignore 中，不会被提交。

> 还可在 config.json 里改：模型名（默认 MiniMax-M2.7-highspeed）、音色映射、视频分辨率（默认 1920x1080）、目标集数等。详见下方「配置参数」表。

### 3-A. 方式一：Gradio Web 控制台（推荐）

无需预先放小说文件——直接启动服务，在浏览器里完成上传→设参→生产→预览→改 prompt 全流程。

```bash
python app.py --config config.json --port 7860
```

浏览器打开 `http://127.0.0.1:7860`，顶部三个 Tab，生产控制 Tab 下多个区块：

| 区块 | 功能 |
|---|---|
| ① 任务配置 | **当前书下拉 + 新建书** → 上传小说 TXT（≤500MB）设目标集数/风格/方向/音量等 |
| 文件清单 | 显示当前书全部 txt 的状态（🟢当前在读 / ⏳队列待读 / ⚪未识别），每 5 秒刷新 |
| ② 实时进度 | 运行状态灯 + 日志流（每 2 秒自动刷新） |
| ③ 成品浏览 | 选集下拉 → 视频预览（520px 高）+ 讲解文案 + 生图 Prompt |
| ④ 提示词管理 | 选 Agent → 加载/编辑/保存其 `prompts/*.md`，下一集即时生效 |
| ⑤ 角色档案（Tab）| 选角色 → 查看AI维护档案 → 编辑「用户描述」（生图优先依据，AI永不覆盖） |

#### Gradio 界面参数详解

**○ 多书管理（生产控制 Tab 顶部）**

每本书在 `novels/<书名>/` 下独立拥有 `checkpoints/ memory/ output/ data/` 四目录，互不干扰——可同时啃多本大长篇，切换书即切换整条进度链（断点/角色档案/成品互不影响）。

| 控件 | 功能 |
|---|---|
| 当前书 | 下拉切换已存在的书；切换后自动刷新文件清单/成品列表/角色档案到该书的目录 |
| 新书店名 | 输入书名（如「人道至尊1-500」）→ 点「📦 新建书」创建目录结构并自动选中 |
| 文件清单（Markdown 表） | 列出该书 data/ 下全部 txt，标注状态：🟢当前在读(offset) / ⏳队列待读 / ✅已读完 / ⚪未识别。每 5 秒自动刷新，**一眼看出续跑到末尾是否会中断（队列是否为空）** |

> 启动时若 `novels/` 下已有书，默认自动选中第一本，省去手动切。

**① 任务配置区**

| 参数 | 类型 | 默认 | 含义与功效 |
|---|---|---|---|
| 上传小说 TXT | 文件 | — | 支持 ≤500MB 纯文本，UTF-8/GBK 自适应。上传后存到 `novels/<当前书>/data/`；同书重跑可不上传（自动复用该书已存文件） |
| 目标集数 | 数字 | 4 | **全新运行**=总集数（从头跑几集）；**续跑**=增量（再跑几集，如已完成4集+填1→跑到第5集） |
| 生图风格 | 下拉 | anime | `anime`=动漫风格（prompt 自动加 `anime style, ancient Chinese mythology art style,` 前缀）；`realistic`=写实风格 |
| 视频方向 | 下拉 | 横屏 | `横屏 1920×1080 (16:9)`=横屏 1080p；`竖屏 1080×1920 (9:16)`=竖屏（适合手机/抖音） |
| TTS 语速 | 滑块 | 1.08 | 语音合成语速倍率，0.8~1.3。值越大语速越快（1.0=正常，1.08=紧凑节奏） |
| BGM 音量 | 滑块 | 0.25 | 背景音乐相对人声的音量比，0.0~0.5。0.25=BGM 声压约为人声的 25%（不抢声） |
| TTS 音量增益 | 滑块 | 1.25 | 对人声音轨的增益倍率，0.8~2.0。1.0=原始音量，1.25=适度放大（确保人声清晰） |
| 字幕（烧录到视频） | 复选框 | 关 | 勾选后从口播文案生成 SRT 字幕并烧录到视频画面（白字黑描边、底部居中）。需系统有 CJK 字体（WSL 自动探测 Windows 微软雅黑） |
| 分片字符数 | 数字 | 8000 | 每次读取小说文本的字符上限，影响剧情解析粒度。太大→LLM 上下文溢出；太小→场景太碎 |
| 单集最大重试 | 数字 | 2 | 单集在格式校验/评审失败时的最大重试次数，超限后标记 `manual_review` 直送归档（防死循环） |

**操作按钮**

| 按钮 | 功能 |
|---|---|
| ▶️ 开始生成 | 全新运行：从小说开头开始，目标集数=总集数。未上传时自动复用当前书 `data/novel.txt` |
| ♻️ 从断点续跑 | 从当前书的断点继续：自动找该书完成集数最多的 thread，目标集数=增量（再跑几集） |
| ⏹️ 停止生成 | 发送停止信号，等当前集生产完成后优雅退出（不杀进程，不丢数据），下次可续跑 |

**② 实时进度区**

运行状态灯 + 日志流。日志每 2 秒自动刷新，显示节点执行进度：
- `[文本分片] offset=7951` — 正在读小说下一段
- `[剧情解析]` — LLM 拆解场景
- `[生产] 新集聚合: ep_005 (6 场景)` — 新集打包完成
- `[素材生成] 生图prompt=12 TTS段=5` — AI 生图+语音参数就绪
- `[评审] 评审结论: pass` — 四专精评审通过
- `[归档]` — 落盘到 `output/ep_xxx/`
- `[合成] ep_005 完成: ep_005.mp4 (28.5 MB)` — 视频产出
- `[进度] 已完成 5/6 集` — 进度汇报

**③ 成品浏览区**

| 操作 | 功能 |
|---|---|
| 🔄 刷新集列表 | 扫描 `output/` 目录，列出已生产的剧集（显示"第N集 · X场景"），自动加载第一集 |
| 选择剧集 | 下拉选集后自动加载该集的视频/文案/生图 Prompt |
| 视频预览 | 520px 高播放器，直接在线预览成品 mp4 |
| 讲解文案 | 该集的口播文本（`script.txt` 内容），可复制 |
| 生图 Prompt 列表 | 该集所有 AI 配图的 prompt（JSON 格式），含 `narration_segment`（对应第几段语音）和 `start_ratio`（出现时机） |

**④ 提示词管理区**

| 操作 | 功能 |
|---|---|
| 选择 Agent | 下拉选 8 个 Agent 之一（plot_parser / episode_aggregator / material_generator / 4个评审 / arbiter） |
| 📂 加载 | 把该 Agent 的 `prompts/*.md` 加载到编辑框 |
| 💾 保存 | 把编辑框内容写回 `prompts/*.md`，**下一集生产时即时生效** |
| ♻️ 从 git 恢复 | `git checkout` 恢复该 prompt 文件到上次提交的版本 |
| 自动加载 | 切换 Agent 时自动加载其 prompt 到编辑框 |

> `material_generator.md` 中的 `{{ART_STYLE}}` 占位符运行时按风格下拉（anime / realistic）自动替换，无需手写。

#### 典型操作场景

| 场景 | 操作 |
|---|---|
| **首次跑 4 集** | 选/建书 → 上传 TXT → 目标集数填 4 → 点 ▶️ 开始生成 |
| **继续跑 1 集** | 选当前书 → 目标集数填 1 → 点 ♻️ 从断点续跑（自动从上次第4集后继续） |
| **同一本书重跑 2 集** | 选当前书（不上传复用）→ 目标集数填 2 → 点 ▶️ 开始生成（从头开始） |
| **换一本书跑** | 「当前书」下拉切换 或 「新书店名」+📦 新建书 → 上传新 TXT → 开始生成 |
| **多本大长篇交替跑** | 两个书各自独立断点，切换书即切换进度链，互不影响 |
| **编辑角色外貌** | 角色档案 Tab → 选角色 → 填「用户描述」→ 保存（下集生图以此为准） |
| **换动漫→写实风格** | 生图风格选 realistic → 重新开始生成或续跑 |
| **加字幕** | 勾选「字幕」→ 开始生成或续跑（字幕烧录到视频画面） |
| **改 Agent 提示词** | 提示词管理 → 选 Agent → 编辑 → 💾 保存 → 下一集生效 |

### 3-B. 方式二：命令行

命令行通过 `--book <书名>` 指定书，storage 路径自动重定向到 `novels/<书名>/` 下（多书隔离）。未传 `--book` 时回退到根目录的旧默认路径（兼容历史用法）。

先把小说 txt 放到对应书的 data 目录：

```bash
mkdir -p "novels/我的书/data"
cp /你的小说路径/某小说.txt "novels/我的书/data/novel.txt"
```

> 也支持放别处用 `--novel /你的路径.txt` 指定。文件越完整越好（几万字起步，百万字也行）；太短（<几千字）可能凑不够一集。

**最简：生成 10 集视频（指定书）**
```bash
python main.py --book "我的书" --novel "novels/我的书/data/novel.txt" --target 10 --config config.json
```

**跑完全本**（自动一直跑到小说末尾，生成 N 集）
```bash
python main.py --book "我的书" --novel "novels/我的书/data/novel.txt" --config config.json
```

**断点续跑**（`--target` 为增量即"再跑几集"；`--book` 指定从哪本书的断点续）
```bash
python main.py --book "我的书" --resume --target 1 --config config.json   # 已完成4集+再跑1集→跑到第5集
python main.py --book "我的书" --resume --config config.json                # 不指定target=用config.json默认值
```

**超长篇拆多本 txt 自动衔接**（`--novel-queue` 后续本按顺序衔接，读完后自动切下一本）
```bash
python main.py --book "人道至尊" --novel "novels/人道至尊/data/novel.txt" \
  --novel-queue "novels/人道至尊/data/novel_1.txt" "novels/人道至尊/data/novel_2.txt" \
  --target 20 --config config.json
```

**用环境变量方式（不写 config.json）**
```bash
export MINIMAX_API_KEY="sk-cp-你的真实key"
python main.py --book "我的书" --novel "novels/我的书/data/novel.txt" --target 5
```

> 运行过程中控制台会实时打印进度：`[生产]`/`[评审]`/`[归档]`/`[合成]`/`[进度] 已完成 N 集`。每集约需 3-8 分钟（取决于 LLM 响应和生图/TTS 并发）。

### 5. 产出在哪里 & 各是什么

成品按单集独立目录归档在 `novels/<书名>/output/` 下（多书隔离）：

```
novels/人道至尊/
├ data/
│  └ novel.txt              # 上传的小说源文本
├ ep_001/                   # 成品按单集独立目录归档
│  ├ episode_info.json      # 单集基础信息（剧情概述/因果链/关联伏笔/评审结果）
│  ├ script.txt             # 整集口播讲解文案（可直接朗读）
│  ├ image_prompts.json     # 时序化图片 Prompt 列表（List[dict]）
│  ├ tts_meta.json          # 整集 TTS 参数列表（每段：文本/音色/情绪/语速/停顿）
│  ├ original_snippet.txt   # 对应的原著原文片段（备查/对照）
│  ├ images/                # AI 生成的每帧图片（001.png ...）
│  ├ audio/                 # 合成的每段语音（001.mp3 ...）
│  └ ep_001.mp4             # ★ 最终讲解视频（图片+语音拼接）
├ ep_002/
├ memory/                   # 该书独立的记忆库
│  ├ characters.json        # 全局人物档案库（含 user_description 字段）
│  ├ events.json            # 时序事件库
│  ├ foreshadows.json       # 伏笔台账
│  └ character_portraits/   # 角色定妆照（跨集复用，保证人物一致）
└ checkpoints/
   └ checkpoint.sqlite      # 断点快照（--resume 靠它续跑，thread_id 含书名前缀）
```

**直接拿 `novels/<书名>/output/ep_XXX/ep_XXX.mp4` 就是成品视频**，可直接播放或在剪辑软件二次加工。

## 目录结构

```
novel_pipeline/
├── requirements.txt          # 依赖（含 gradio>=4.0）
├── config.py                 # 全局配置（model/run/storage/media 四组 + 多书 apply_book）
├── config.example.json       # 示例配置文件
├── llm_factory.py             # LLM 实例工厂（按角色分层实例化）
├── state.py                  # NovelState 核心状态定义
├── app.py                    # ★ Gradio Web 控制台入口（多书管理 + 角色档案 Tab）
├── novels/                   # ★ 多本小说隔离根目录（每书独立 checkpoints/memory/output/data）
│   └── <书名>/
├── prompts/                  # ★ 8 个 Agent 的外部化 System Prompt
│   ├── __init__.py           # 加载器（load_prompt / save_prompt / list_prompts）
│   ├── plot_parser.md
│   ├── episode_aggregator.md
│   ├── material_generator.md # 含 {{ART_STYLE}} 占位符
│   ├── review_character.md
│   ├── review_foreshadow.md
│   ├── review_timeline.md
│   ├── review_atmosphere.md
│   └── review_arbiter.md
├── agents/                   # 9 个 LLM 智能 Agent
│   ├── memory_manager.py     # 支撑层：全局记忆管理（人物/事件/伏笔 + FAISS + user_description 保护）
│   ├── plot_parser.py        # 生产层：剧情解析
│   ├── episode_aggregator.py # 生产层：剧集聚合
│   ├── material_generator.py # 生产层：多媒体素材生成（生图优先用 user_description）
│   ├── review_character.py   # 评审层：人物设定评审
│   ├── review_foreshadow.py  # 评审层：伏笔线索评审
│   ├── review_timeline.py    # 评审层：剧情时序评审
│   ├── review_atmosphere.py  # 评审层：视听氛围评审
│   └── review_arbiter.py     # 评审层：评审汇总仲裁
├── nodes/                    # 7 个纯代码逻辑节点
│   ├── text_chunker.py       # 文本分片（offset 游标 + 章节边界对齐 + 多本队列衔接）
│   ├── format_validator.py   # 格式校验（含超限放行保护）
│   ├── memory_prefetch.py    # 统一记忆预检索
│   ├── persistence.py        # 持久化归档
│   ├── media_synthesizer.py  # 多媒体合成（生图+TTS+FFmpeg视频拼接）
│   ├── routing.py            # 条件路由（5 个路由函数）
│   └── retry_counter.py      # 重试计数
├── graph/
│   └── builder.py            # LangGraph StateGraph 构建 + SqliteSaver
└── main.py                   # 命令行主入口（--book 多书隔离参数）
```

## 配置参数

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| model.api_key | `MINIMAX_API_KEY` | - | MiniMax API Key |
| model.base_url | `MINIMAX_BASE_URL` | `https://api.minimaxi.com/v1` | OpenAI 兼容端点 |
| model.production_model | - | MiniMax-M2.7-highspeed | 生产/支撑层模型 |
| model.review_model | - | MiniMax-M2.7-highspeed | 评审层模型 |
| model.max_tokens | `MODEL_MAX_TOKENS` | 16384 | 单次响应 token 上限（material_generator 输出含完整 JSON，需较大值） |
| run.chunk_size | `CHUNK_SIZE` | 8000 | 单次读取字符上限 |
| run.target_episode_count | `TARGET_EPISODE_COUNT` | 0(全本) | 目标集数 |
| run.max_retries | `MAX_RETRIES` | 2 | 单集最大重试（format + review 共享） |
| run.min/max_scenes_per_episode | - | 3 / 8 | 单集场景数阈值 |
| storage.output_dir | `OUTPUT_DIR` | ./output | 成品目录（多书隔离时由 `apply_book` 重定向到 `novels/<书名>/output`） |
| storage.sqlite_path | `SQLITE_PATH` | ./checkpoints/checkpoint.sqlite | 断点库（多书隔离时重定向到 `novels/<书名>/checkpoints/`） |
| storage.memory_dir | `MEMORY_DIR` | ./memory | 记忆库目录（多书隔离时重定向到 `novels/<书名>/memory`） |
| storage.enable_vector_retrieval | `ENABLE_VECTOR_RETRIEVAL` | true | 是否启用 FAISS |
| media.enable_synthesis | `ENABLE_SYNTHESIS` | true | 是否启用合成阶段 |
| media.image_model | `IMAGE_MODEL` | image-01 | 生图模型 |
| media.tts_model | `TTS_MODEL` | speech-02-hd | TTS 模型 |
| media.image_concurrency | `IMAGE_CONCURRENCY` | 3 | 生图并发数 |
| media.tts_concurrency | `TTS_CONCURRENCY` | 1 | TTS 并发数（限流敏感） |
| media.video_resolution | `VIDEO_RESOLUTION` | 1920x1080 | 视频分辨率 |
| media.video_fps | `VIDEO_FPS` | 30 | 视频帧率 |
| media.image_aspect_ratio | `IMAGE_ASPECT_RATIO` | 16:9 | 生图宽高比 |
| media.voice_mapping | - | 见 config.example.json | voice 字段 → MiniMax voice_id（默认：narrator→Chinese_gravelly_storyteller_nv1，女性→Chinese (Mandarin)_Sweet_Lady，钟岳→Chinese_worker_male）|
| media.default_voice_id | `DEFAULT_VOICE_ID` | Chinese_gravelly_storyteller_nv1 | 兜底音色 |

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 调度框架 | LangGraph (StateGraph + SqliteSaver) | 状态机编排 + 断点续跑 |
| LLM | MiniMax-M2.7-highspeed（OpenAI 兼容协议） | 剧情/素材/评审 |
| LLM 编排 | LangChain (langchain-openai) | ChatOpenAI 接入 |
| 生图 | MiniMax `/v1/image_generation` (image-01) | AI 配图，中国古代神话风格，带 reference_image 保持人物一致 |
| 语音合成 | MiniMax `/v1/t2a_v2` (speech-02-hd) | TTS，三音色映射（旁白/女性/男性），hex 音频解码 |
| 视频合成 | imageio-ffmpeg（静态 ffmpeg 二进制） | 图片+音频→mp4 拼接 |
| 向量检索 | FAISS (faiss-cpu) | 长线伏笔召回 |
| 状态持久化 | SQLite (LangGraph Checkpointer) | 断点快照 |
| HTTP | requests | 生图/TTS 接口调用 |
| Web 控制台 | Gradio (gradio>=4.0) | 浏览器界面上传/设参/预览/改 prompt |
| 语言 | Python 3.11+ | - |

## 实现状态

- ✅ 9 个 LLM Agent 全部实现（生产 3 + 评审 5 + 支撑 1）
- ✅ 7 个纯代码节点全部实现（含新增 media_synthesizer）
- ✅ LangGraph 状态机编排、5 个条件边路由、SqliteSaver 断点续跑
- ✅ 全局记忆库（人物/事件/伏笔 JSON + FAISS 向量索引）
- ✅ offset 字节游标分片 + 章节边界自动对齐
- ✅ 适配推理模型的思维链输出，鲁棒 JSON 提取（extract_json_dict）
- ✅ format 阶段死循环防护（超 max_retries 放行归档）
- ✅ 多媒体合成闭环：定妆照参考 → AI 生图（中国古代神话风格）→ TTS（三音色映射）→ FFmpeg 视频拼接
- ✅ 细粒度图片：每个场景/动作对应一张图（narration_segment 标注归属语音段），图片数 > 语音段数，同段多图均分音频时长展示
- ✅ 端到端实跑验证：ep_001 产出 4 图 + 1 段语音 + 37.8 秒讲解视频 (1.9MB)
- ✅ Gradio Web 控制台：上传/设参/实时日志/成品预览/提示词在线编辑
- ✅ 8 个 Agent 提示词外部化到 `prompts/*.md`，支持风格切换（`{{ART_STYLE}}` 占位符）
- ✅ 黑屏自动修复：检测到黑屏帧自动重试缺图/占位图并重新合成视频
- ✅ **多本小说隔离**：`novels/<书名>/` 下独立目录（checkpoints/memory/output/data），命令行 `--book` 参数，thread_id 加书名前缀防串断点，兼容旧版本无前缀断点
- ✅ **文本识别清单**：Gradio 文件清单 Markdown 实时显示每本 txt 的 🟢当前在读/⏳队列待读/✅已读完/⚪未识别，防续跑到末尾没衔接
- ✅ **角色档案双区 + 用户可编辑**：user_description 字段 AI 永不覆盖，生图优先用；新增「角色档案」Tab 在线编辑

## 注意事项

1. **API Key 安全**：不要把真实 key 提交到版本库，用环境变量或 config.json（已加入 .gitignore）
2. **向量库 Embedding**：memory_manager 默认调用 OpenAI 兼容 embeddings 接口；若未开放会降级为确定性伪向量（保证流程可跑但无真实语义）。生产环境应替换为真实 Embedding 模型
3. **并行评审**：当前为顺序执行（兼容单线程 SqliteSaver），后续可用 asyncio.gather 优化为真正并行
4. **章节标题正则**：text_chunker 内置中文章节标题模式（第X章/回、Chapter N、卷X），非标准格式小说可能对齐失败，退化为段落边界
5. **TTS 限流**：MiniMax TTS 对并发请求敏感，默认 tts_concurrency=1（串行）；emotion 仅支持 neutral/happy/sad/angry/disgusted/surprised/calm，其他值自动映射
6. **生图 URL 时效**：MiniMax 返回的 OSS 签名 URL 24 小时失效，media_synthesizer 生成后即时下载落盘

## 环境兼容性（WSL / Windows 原生）

本项目在 **WSL (Ubuntu)** 下开发与验证，以下问题在 **Windows 原生运行**（直接用 Windows 版 Python 跑 `app.py`/`main.py`）时可能出现，已做兼容处理：

| 问题 | 现象 | 原因 | 处理 |
|---|---|---|---|
| 字幕显示方块/乱码 | 开启「字幕」后视频底部字幕全是 □□□ 或不显示 | libass (ffmpeg `subtitles` 滤镜) 在 WSL/Windows 下 HarfBuzz shaping 缺陷，无法渲染 CJK；早期字体探测只查 WSL 路径 `/mnt/c/Windows/Fonts/`，Windows 原生找不到字体 → PIL fallback 到 `load_default()`（不支持中文） | ① 字幕改用 **PIL 渲染 PNG + ffmpeg overlay**，绕开 libass；② `_detect_cjk_font()` 同时探测 `C:\Windows\Fonts\`（读 `%WINDIR%`）、WSL 挂载路径、Linux noto/wqy 三类路径，自动命中当前环境的中文字体 |
| Gradio 日志乱码 | 控制台「日志流」里中文变成 ?? 或方框、子进程报 `UnicodeEncodeError` | Windows 控制台默认编码 GBK/CP936，`main.py` 的 `print` 默认用 locale 编码输出，而 `app.py` 用 `encoding="utf-8"` 解码子进程 stdout，两端不匹配 → `errors="replace"` 把无法解码的字节替换成乱码 | `main.py` / `app.py` 启动时 `sys.stdout.reconfigure(encoding="utf-8")`，强制所有 `print` 输出 UTF-8 字节，与 Gradio 解码器对齐 |

> **推荐运行环境**：WSL (Ubuntu)。Windows 原生虽已做兼容，但 ffmpeg 静态二进制、字体路径、路径分隔符等仍有边角差异，如遇问题优先回到 WSL。
