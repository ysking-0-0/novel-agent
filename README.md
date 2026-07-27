# 长篇小说多媒体剧集生产多 Agent 系统

基于 **LangGraph 状态机** 的长篇小说（百万字级）批量拆解生产系统，将小说自动拆解为可直接用于短视频制作的多媒体素材包。

> 设计文档：`任务文档.md`（阶段二完整版，9 个 LLM Agent + 6 个纯代码节点）

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
│ 存储层                                               │
│ Sqlite断点快照 + 本地成品文件 + 向量记忆库           │
└─────────────────────────────────────────────────────┘
```

### 流水线（11 步主循环）

```
START → text_chunker
     → [route_after_chunking] → plot_parser / episode_aggregator_force / END
plot_parser → episode_aggregator
     → [route_after_aggregation] → material_generator / text_chunker(回读)
material_generator → format_validator
     → [route_after_format_check] → memory_prefetch / material_generator(局部修正)
memory_prefetch → parallel_reviews(4专精并行) → review_arbiter
     → [route_after_arbiter] → persistence / retry_counter → material_generator
persistence → [route_after_persistence] → text_chunker / END
```

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
│  └ original_snippet.txt   # 对应原著原文片段（备查）
├ ep_002/
└ ...
```

## 目录结构

```
novel_pipeline/
├── requirements.txt          # 依赖
├── config.py                 # 全局配置（model/run/storage 三组）
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
├── nodes/                    # 6 个纯代码逻辑节点
│   ├── text_chunker.py       # 文本分片（offset 游标 + 章节边界对齐）
│   ├── format_validator.py   # 格式校验
│   ├── memory_prefetch.py    # 统一记忆预检索
│   ├── persistence.py        # 持久化归档
│   ├── routing.py            # 条件路由（5 个路由函数）
│   └── retry_counter.py      # 重试计数
├── graph/
│   └── builder.py            # LangGraph StateGraph 构建 + SqliteSaver
└── main.py                   # 主入口
```

## 核心设计要点

### 1. LangGraph 隐式状态机调度
不设独立 LLM 全局调度 Agent，全部流转规则硬编码，100% 可预测，原生支持断点续跑。条件边负责：聚合是否成集、格式是否合规、仲裁判定路由、终止条件判断。

### 2. offset 游标分片（适配 500 万字 +）
- 文件按字节 offset 读取，不加载全文，内存占用稳定
- 自动对齐章节边界，不强制截断剧情
- 二进制读取 + 解码，避免多字节字符截断乱码

### 3. pending_scenes 跨循环缓存
不足成集的场景留存 `pending_scenes`，下一轮新文本补充后继续聚合，解决按章节硬切导致的剧情腰斩。

### 4. 全局记忆管理 Agent 统一收口
所有 Agent 不得直接修改记忆库，统一通过 MemoryManagerAgent。维护三大结构化知识库：
- **人物档案库**：char_id → 性格/外貌/能力/关系
- **时序事件库**：event_id → 因果链/时间顺序
- **伏笔台账**：foreshadow_id → 埋设/回收状态

向量库（FAISS）按需召回相关历史场景，解决长线伏笔回收，避免上下文溢出。

### 5. 生产与评审分离
评审 Agent 仅做对错核查，不参与内容创作，保证客观性。4 个专精评审并行执行，聚焦单一维度，核查深度更高。

### 6. 缺陷分级重试
仲裁 Agent 输出 `pass / regenerate / minor_revise` 三态：
- `pass` → 归档
- `regenerate` + 未超限 → 整集重生成（retry_count+1）
- `regenerate` + 超限 → 标记人工复核后归档（不阻塞）
- `minor_revise` → 局部微调

### 7. 断点续跑
SqliteSaver 自动保存每个节点完成后的完整 NovelState 快照。崩溃 / 手动停止 / 断电后，`--resume` 无缝恢复。

## 配置参数

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| model.api_key | `MINIMAX_API_KEY` | - | MiniMax API Key |
| model.base_url | `MINIMAX_BASE_URL` | `https://api.minimaxi.com/v1` | OpenAI 兼容端点 |
| run.chunk_size | `CHUNK_SIZE` | 8000 | 单次读取字符上限 |
| run.target_episode_count | `TARGET_EPISODE_COUNT` | 0(全本) | 目标集数 |
| run.max_retries | `MAX_RETRIES` | 2 | 单集最大重试 |
| storage.output_dir | `OUTPUT_DIR` | ./output | 成品目录 |
| storage.sqlite_path | `SQLITE_PATH` | ./checkpoints/checkpoint.sqlite | 断点库 |
| storage.memory_dir | `MEMORY_DIR` | ./memory | 记忆库目录 |
| storage.enable_vector_retrieval | `ENABLE_VECTOR_RETRIEVAL` | true | 是否启用 FAISS |

## 技术栈

| 组件 | 选型 |
|------|------|
| 调度框架 | LangGraph (StateGraph + SqliteSaver) |
| LLM 接口 | MiniMax-M3（OpenAI 兼容协议，ChatOpenAI 接入） |
| 向量检索 | FAISS (faiss-cpu) |
| 状态持久化 | SQLite (LangGraph Checkpointer) |
| LLM 编排 | LangChain (langchain-openai) |
| 语言 | Python 3.11+ |

## 实现状态

已完成设计文档「阶段二完整版」：
- ✅ 9 个 LLM Agent 全部实现（生产 3 + 评审 5 + 支撑 1）
- ✅ 6 个纯代码节点全部实现
- ✅ LangGraph 状态机编排、5 个条件边路由、SqliteSaver 断点续跑
- ✅ 全局记忆库（人物/事件/伏笔 JSON + FAISS 向量索引）
- ✅ offset 字节游标分片 + 章节边界自动对齐
- ✅ 适配推理模型（MiniMax-M3）的思维链输出，鲁棒 JSON 提取
- ✅ 端到端 dry-run 验证通过，已用真实小说（5MB / 170 万字）实跑验证

## 注意事项

1. **API Key 安全**：不要把真实 key 提交到版本库，用环境变量或 config.json（加入 .gitignore）
2. **向量库 Embedding**：memory_manager 默认调用 OpenAI 兼容 embeddings 接口；若 MiniMax 未开放该接口，会降级为确定性伪向量（保证流程可跑，但无真实语义）。生产环境应替换为真实 Embedding 模型
3. **并行评审**：当前为顺序执行（兼容单线程 SqliteSaver），每个 Agent 内独立调用 LLM。后续可用 asyncio.gather 优化为真正并行
4. **章节标题正则**：text_chunker 内置中文章节标题模式（第X章/回、Chapter N、卷X），非标准格式小说可能对齐失败，退化为段落边界
