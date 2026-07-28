# 长篇小说多媒体剧集生产多 Agent 系统

基于 **LangGraph 状态机** 的长篇小说（百万字级）批量拆解生产系统：从纯文本小说自动拆解为可直接用于短视频制作的多媒体素材包，**端到端产出讲解视频**（口播文案 → 语音合成 → AI 配图 → FFmpeg 拼接成片）。

> 设计文档：`任务文档.md`（阶段二完整版，9 个 LLM Agent + 7 个纯代码节点）

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

| 产出 | 字段 | 要求 |
|---|---|---|
| 讲解文案 | `script` | 600-1200 字口播文本，用『』分段，每段对应一个画面节拍 |
| 生图 Prompt | `image_prompts` | 每段英文/中文 Prompt，含人物外貌（100% 沿用档案）、动作、场景、光影、构图 |
| TTS 参数 | `tts_meta` | 每段 text/voice/emotion/speed/pause_after，与 script 段段对齐 |

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

**人物一致性方案**：为每个主要角色预生成一张「定妆照」（portrait prompt + 中性背景），后续该角色出现的所有画面生成时都带 `reference_image`（定妆照 base64）作为参考，确保外貌在整集内保持一致。定妆照缓存在 `output/_character_refs/` 跨集复用。

**三阶段流程**：

| 阶段 | 操作 | 接口 |
|---|---|---|
| 1. 生图 | 遍历 image_prompts，并发（默认 3）调生图接口，带定妆照参考 | MiniMax `/v1/image_generation` (model=image-01) |
| 2. TTS | 遍历 tts_meta，串行（防限流）调语音合成，解码 hex 音频落盘 | MiniMax `/v1/t2a_v2` (model=speech-02-hd) |
| 3. 视频 | 每段 = 图片 + 音频 → FFmpeg 合成片段 mp4；concat 拼接 → 完整视频 | imageio-ffmpeg (静态 ffmpeg 二进制) |

**关键工程坑与修复**（实测发现）：

| 问题 | 根因 | 修复 |
|---|---|---|
| TTS 返回乱码 | MiniMax `data.audio` 是 **hex 字符串**非 base64 | `_decode_audio` 自适应 hex/base64 |
| TTS 部分 4xx | `tense` 等值不在 MiniMax emotion 白名单 | `_resolve_emotion` 映射到支持的 7 种 (neutral/happy/sad/angry/disgusted/surprised/calm) |
| 片段合成失败 | imageio-ffmpeg 不带 ffprobe | `_audio_duration` 改用 `ffmpeg -i` 解析 stderr |
| concat 拼接失败 | concat demuxer 按相对路径重复拼接 | concat.txt 用绝对路径 |
| 图片 URL 过期 | OSS 签名 URL 24 小时失效 | 生成后即时下载落盘 |

**音色映射**：`tts_meta.voice` 字段值（如 `narrator_male`）→ MiniMax voice_id（如 `male-qn-jingying`）通过 `voice_mapping` 配置映射，可在 config.json 覆盖。

### 八、全局记忆管理 Agent（memory_manager）

全系统唯一记忆读写入口，单例，线程安全。维护三大结构化知识库 + 向量索引：

| 知识库 | 键 | 值 |
|---|---|---|
| 人物档案库 | char_id | 性格/外貌/能力/关系 |
| 时序事件库 | event_id | 因果链/时间顺序 |
| 伏笔台账 | foreshadow_id | 埋设/回收状态 |

- 向量库（FAISS）按需召回相关历史场景，解决长线伏笔回收，避免上下文溢出
- 所有 Agent 不得直接改记忆库，统一走 MemoryManagerAgent
- 每集归档后增量更新（update_from_episode + save）

### 九、断点续跑

SqliteSaver 自动保存每个节点完成后的完整 NovelState 快照（thread_id 固定 `novel_main_thread`）。崩溃 / 手动停止 / 断电后，`--resume` 从最新 checkpoint 无缝恢复。`offset` 字节游标 + sqlite 快照双重保障。

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

### 1. 安装依赖

```bash
cd novel_pipeline
pip install -r requirements.txt
```

### 2. 配置 API Key

方式一：环境变量
```bash
export MINIMAX_API_KEY="sk-cp-你的真实key"
```

方式二：配置文件（复制示例后修改）
```bash
cp config.example.json config.json
# 编辑 config.json 中的 api_key
```

### 3. 准备小说文本

将小说 TXT 文件放于任意路径，例如 `./data/novel.txt`。

### 4. 运行

全新运行（生成 10 集）：
```bash
python main.py --novel ./data/novel.txt --target 10
```

跑完全本：
```bash
python main.py --novel ./data/novel.txt
```

断点续跑（从上次中断处继续）：
```bash
python main.py --resume
```

自定义分片大小与重试上限：
```bash
python main.py --novel ./data/novel.txt --chunk-size 6000 --max-retries 3
```

使用配置文件覆盖：
```bash
python main.py --novel ./data/novel.txt --config config.json
```

### 5. 产出

成品素材按单集独立目录归档：
```
output/
├ ep_001/
│  ├ episode_info.json      # 单集基础信息、因果概述、关联伏笔
│  ├ script.txt             # 整集讲解文案
│  ├ image_prompts.json     # 时序化图片Prompt列表
│  ├ tts_meta.json          # 整集TTS参数列表
│  ├ original_snippet.txt   # 对应原著原文片段（备查）
│  ├ images/                # 生成的每帧图片（001.png ...）
│  ├ audio/                 # 合成的每段语音（001.mp3 ...）
│  └ ep_001.mp4             # 拼接后的讲解视频
├ ep_002/
└ ...
```

## 目录结构

```
novel_pipeline/
├── requirements.txt          # 依赖
├── config.py                 # 全局配置（model/run/storage/media 四组）
├── config.example.json       # 示例配置文件
├── llm_factory.py             # LLM 实例工厂（按角色分层实例化）
├── state.py                  # NovelState 核心状态定义
├── agents/                   # 9 个 LLM 智能 Agent
│   ├── memory_manager.py     # 支撑层：全局记忆管理（人物/事件/伏笔 + FAISS）
│   ├── plot_parser.py        # 生产层：剧情解析
│   ├── episode_aggregator.py # 生产层：剧集聚合
│   ├── material_generator.py # 生产层：多媒体素材生成
│   ├── review_character.py   # 评审层：人物设定评审
│   ├── review_foreshadow.py  # 评审层：伏笔线索评审
│   ├── review_timeline.py    # 评审层：剧情时序评审
│   ├── review_atmosphere.py  # 评审层：视听氛围评审
│   └── review_arbiter.py     # 评审层：评审汇总仲裁
├── nodes/                    # 7 个纯代码逻辑节点
│   ├── text_chunker.py       # 文本分片（offset 游标 + 章节边界对齐）
│   ├── format_validator.py   # 格式校验（含超限放行保护）
│   ├── memory_prefetch.py    # 统一记忆预检索
│   ├── persistence.py        # 持久化归档
│   ├── media_synthesizer.py  # 多媒体合成（生图+TTS+FFmpeg视频拼接）
│   ├── routing.py            # 条件路由（5 个路由函数）
│   └── retry_counter.py      # 重试计数
├── graph/
│   └── builder.py            # LangGraph StateGraph 构建 + SqliteSaver
└── main.py                   # 主入口
```

## 配置参数

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| model.api_key | `MINIMAX_API_KEY` | - | MiniMax API Key |
| model.base_url | `MINIMAX_BASE_URL` | `https://api.minimaxi.com/v1` | OpenAI 兼容端点 |
| model.production_model | - | MiniMax-M3 | 生产/支撑层模型 |
| model.review_model | - | MiniMax-M3 | 评审层模型 |
| run.chunk_size | `CHUNK_SIZE` | 8000 | 单次读取字符上限 |
| run.target_episode_count | `TARGET_EPISODE_COUNT` | 0(全本) | 目标集数 |
| run.max_retries | `MAX_RETRIES` | 2 | 单集最大重试（format + review 共享） |
| run.min/max_scenes_per_episode | - | 3 / 8 | 单集场景数阈值 |
| storage.output_dir | `OUTPUT_DIR` | ./output | 成品目录 |
| storage.sqlite_path | `SQLITE_PATH` | ./checkpoints/checkpoint.sqlite | 断点库 |
| storage.memory_dir | `MEMORY_DIR` | ./memory | 记忆库目录 |
| storage.enable_vector_retrieval | `ENABLE_VECTOR_RETRIEVAL` | true | 是否启用 FAISS |
| media.enable_synthesis | `ENABLE_SYNTHESIS` | true | 是否启用合成阶段 |
| media.image_model | `IMAGE_MODEL` | image-01 | 生图模型 |
| media.tts_model | `TTS_MODEL` | speech-02-hd | TTS 模型 |
| media.image_concurrency | `IMAGE_CONCURRENCY` | 3 | 生图并发数 |
| media.tts_concurrency | `TTS_CONCURRENCY` | 1 | TTS 并发数（限流敏感） |
| media.video_resolution | `VIDEO_RESOLUTION` | 1280x720 | 视频分辨率 |
| media.video_fps | `VIDEO_FPS` | 30 | 视频帧率 |
| media.voice_mapping | - | 见 config.example.json | voice 字段 → MiniMax voice_id |
| media.default_voice_id | `DEFAULT_VOICE_ID` | male-qn-jingying | 兜底音色 |

## 技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| 调度框架 | LangGraph (StateGraph + SqliteSaver) | 状态机编排 + 断点续跑 |
| LLM | MiniMax-M2.7-highspeed / M3（OpenAI 兼容协议） | 剧情/素材/评审 |
| LLM 编排 | LangChain (langchain-openai) | ChatOpenAI 接入 |
| 生图 | MiniMax `/v1/image_generation` (image-01) | AI 配图，带 reference_image 保持人物一致 |
| 语音合成 | MiniMax `/v1/t2a_v2` (speech-02-hd) | TTS，hex 音频解码 |
| 视频合成 | imageio-ffmpeg（静态 ffmpeg 二进制） | 图片+音频→mp4 拼接 |
| 向量检索 | FAISS (faiss-cpu) | 长线伏笔召回 |
| 状态持久化 | SQLite (LangGraph Checkpointer) | 断点快照 |
| HTTP | requests | 生图/TTS 接口调用 |
| 语言 | Python 3.11+ | - |

## 实现状态

- ✅ 9 个 LLM Agent 全部实现（生产 3 + 评审 5 + 支撑 1）
- ✅ 7 个纯代码节点全部实现（含新增 media_synthesizer）
- ✅ LangGraph 状态机编排、5 个条件边路由、SqliteSaver 断点续跑
- ✅ 全局记忆库（人物/事件/伏笔 JSON + FAISS 向量索引）
- ✅ offset 字节游标分片 + 章节边界自动对齐
- ✅ 适配推理模型的思维链输出，鲁棒 JSON 提取（extract_json_dict）
- ✅ format 阶段死循环防护（超 max_retries 放行归档）
- ✅ 多媒体合成闭环：定妆照参考 → AI 生图 → TTS → FFmpeg 视频拼接
- ✅ 端到端实跑验证：ep_001 产出 7 图 + 7 段语音 + 2分38秒讲解视频 (5.6MB)

## 注意事项

1. **API Key 安全**：不要把真实 key 提交到版本库，用环境变量或 config.json（已加入 .gitignore）
2. **向量库 Embedding**：memory_manager 默认调用 OpenAI 兼容 embeddings 接口；若未开放会降级为确定性伪向量（保证流程可跑但无真实语义）。生产环境应替换为真实 Embedding 模型
3. **并行评审**：当前为顺序执行（兼容单线程 SqliteSaver），后续可用 asyncio.gather 优化为真正并行
4. **章节标题正则**：text_chunker 内置中文章节标题模式（第X章/回、Chapter N、卷X），非标准格式小说可能对齐失败，退化为段落边界
5. **TTS 限流**：MiniMax TTS 对并发请求敏感，默认 tts_concurrency=1（串行）；emotion 仅支持 neutral/happy/sad/angry/disgusted/surprised/calm，其他值自动映射
6. **生图 URL 时效**：MiniMax 返回的 OSS 签名 URL 24 小时失效，media_synthesizer 生成后即时下载落盘
