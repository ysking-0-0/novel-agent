"""
novel_pipeline.config
全局配置：模型参数、分片大小、目标集数、路径、重试上限等。
通过环境变量覆盖，便于不同部署环境。

配置分四组：
  model   —— LLM 绑定（生产/评审/支撑三层模型）
  run     —— 运行参数（分片大小、重试上限、目标集数）
  storage —— 存储路径（sqlite 断点、成品目录、记忆库、向量库开关）
  media   —— 多媒体合成（生图/TTS/视频拼接参数）
"""
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """模型绑定配置。生产层高速通用、评审层强推理、支撑层高速通用。"""
    production_model: str = "MiniMax-M3"
    review_model: str = "MiniMax-M3"
    support_model: str = "MiniMax-M3"
    # OpenAI 兼容接入参数（MiniMax 官方兼容端点）
    api_key: str = os.getenv("MINIMAX_API_KEY", "")
    base_url: str = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "16384"))


@dataclass
class RunConfig:
    """生产流水线运行参数。"""
    novel_file_path: str = os.getenv("NOVEL_FILE_PATH", "./data/novel.txt")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "8000"))
    # 目标生成集数，None 表示跑完全本
    target_episode_count: Optional[int] = (
        (int(os.getenv("TARGET_EPISODE_COUNT", "0")) or None)
    )
    # 单集最大重试次数（防止死循环）
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    # 场景阈值：少于该数量不足以成集，等待下一轮补充
    min_scenes_per_episode: int = int(os.getenv("MIN_SCENES_PER_EPISODE", "3"))
    max_scenes_per_episode: int = int(os.getenv("MAX_SCENES_PER_EPISODE", "8"))


@dataclass
class StorageConfig:
    """存储路径与介质配置。

    多书隔离：每本书在 novels/<book_name>/ 下拥有独立的
    checkpoints/ memory/ output/ data/ 四子目录，互不干扰。
    config.json 保留全局模型参数；storage 路径运行时由
    apply_book(name) 重定向。无 book 时回退到项目根目录（兼容旧版）。
    """
    # 项目根目录（config.py 所在目录），所有路径基于此，不依赖运行时 cwd
    _PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    # 当前书名（None=兼容旧版单书模式，路径用根目录下的默认值）
    book_name: Optional[str] = None
    # 成品输出目录（绝对路径，不存在自动创建）
    output_dir: str = os.getenv("OUTPUT_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
    # 断点快照 Sqlite 路径（绝对路径）
    sqlite_path: str = os.getenv("SQLITE_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "checkpoint.sqlite"))
    # 全局记忆库目录（绝对路径）
    memory_dir: str = os.getenv("MEMORY_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory"))
    # 是否启用向量检索（FAISS）
    enable_vector_retrieval: bool = (
        os.getenv("ENABLE_VECTOR_RETRIEVAL", "true").lower() == "true"
    )
    # 向量库维度（与 embedding 模型一致）
    vector_dim: int = int(os.getenv("VECTOR_DIM", "1536"))

    @classmethod
    def book_dir(cls, book_name: str) -> str:
        """返回指定书名对应的根目录绝对路径。"""
        root = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(root, "novels", book_name)

    @staticmethod
    def list_books() -> list:
        """列出 novels/ 下所有已存在的书名（子目录名）。"""
        root = os.path.dirname(os.path.abspath(__file__))
        novels_root = os.path.join(root, "novels")
        if not os.path.isdir(novels_root):
            return []
        return sorted([d for d in os.listdir(novels_root)
                       if os.path.isdir(os.path.join(novels_root, d))])


def apply_book(book_name: Optional[str]):
    """把当前全局配置的 storage 三路径重定向到 novels/<book_name>/ 下。

    book_name=None 时保持默认（兼容旧版单书模式，路径在项目根目录）。
    会在 set_config 之后、build_graph 之前调用。
    """
    cfg = get_config()
    if not book_name:
        cfg.storage.book_name = None
        return cfg
    bdir = StorageConfig.book_dir(book_name)
    os.makedirs(bdir, exist_ok=True)
    cfg.storage.book_name = book_name
    cfg.storage.output_dir = os.path.join(bdir, "output")
    cfg.storage.sqlite_path = os.path.join(bdir, "checkpoints", "checkpoint.sqlite")
    cfg.storage.memory_dir = os.path.join(bdir, "memory")
    for d in (cfg.storage.output_dir, cfg.storage.memory_dir,
              os.path.dirname(cfg.storage.sqlite_path)):
        os.makedirs(d, exist_ok=True)
    # 重置 memory_agent 单例，让下次 get_memory_agent 重新加载新书的记忆库
    try:
        from agents.memory_manager import reset_memory_store
        reset_memory_store()
    except Exception:
        pass
    return cfg


@dataclass
class MediaConfig:
    """多媒体合成配置：生图/TTS/视频拼接。"""
    # 是否启用合成阶段（persistence 后自动触发生图+TTS+视频）
    enable_synthesis: bool = (
        os.getenv("ENABLE_SYNTHESIS", "true").lower() == "true"
    )
    # 生图并发数（官方 RPM=10、无并发上限，media_synthesizer 有全局 RPM 限速器兜底，3 并发安全）
    image_concurrency: int = int(os.getenv("IMAGE_CONCURRENCY", "3"))
    # TTS 并发数（speech-02-hd 单次约 1.4s，4 并发安全）
    tts_concurrency: int = int(os.getenv("TTS_CONCURRENCY", "4"))
    # 生图模型
    image_model: str = os.getenv("IMAGE_MODEL", "image-01")
    # TTS 模型
    tts_model: str = os.getenv("TTS_MODEL", "speech-02-hd")
    # 图片宽高比
    image_aspect_ratio: str = os.getenv("IMAGE_ASPECT_RATIO", "16:9")
    # 视频分辨率（与图片比例匹配）
    video_resolution: str = os.getenv("VIDEO_RESOLUTION", "1920x1080")
    # 视频帧率
    video_fps: int = int(os.getenv("VIDEO_FPS", "30"))
    # 生图风格：anime=动漫 / realistic=写实（注入 material_generator prompt 的 {{ART_STYLE}}）
    art_style: str = os.getenv("ART_STYLE", "anime")
    # 单幅图片展示目标时长（秒），material_generator 据此估算图片数量
    # 一两句话同一场景描述为一幅，约10秒换一幅
    image_duration_target: float = float(os.getenv("IMAGE_DURATION_TARGET", "10.0"))
    # 背景音乐文件路径（None=不加BGM）。BGM 会被循环/截断对齐视频时长，音量降低做背景
    bgm_path: str = os.getenv("BGM_PATH", "./assets/bgm.mp3")
    # BGM 音量（0.0-1.0，相对TTS为1.0）。0.25 表示BGM声压约为TTS的25%
    bgm_volume: float = float(os.getenv("BGM_VOLUME", "0.25"))
    # TTS 音量增益（1.0=原始，>1放大）。合成时对 TTS 音轨统一施加该增益
    tts_volume: float = float(os.getenv("TTS_VOLUME", "1.25"))
    # 结尾延长秒数（0=关闭）：最后一张图定格 N 秒、讲解声停止、BGM 慢慢淡出（保留底音）
    ending_extend_seconds: float = float(os.getenv("ENDING_EXTEND_SECONDS", "3.0"))
    # 是否在视频上烧录字幕（从 tts_meta 文本生成 SRT，drawtext 渲染）
    enable_subtitles: bool = os.getenv("ENABLE_SUBTITLES", "false").lower() == "true"
    # 字幕字体大小（像素，相对于 1080p 高度）
    subtitle_font_size: int = int(os.getenv("SUBTITLE_FONT_SIZE", "42"))
    # 字幕字体文件路径（空=自动探测系统 CJK 字体；也可手动指定 ttf/ttc 路径）
    subtitle_font: str = os.getenv("SUBTITLE_FONT", "")
    # 角色音色映射：tts_meta.voice 值（角色名/旁白标识）→ MiniMax voice_id
    # 可用 voice_id（MiniMax）：
    #   Chinese_gravelly_storyteller_nv1（中文沉稳旁白/说书人，沙哑磁性）
    #   Chinese (Mandarin)_Sweet_Lady（中文甜美女声）
    #   male-qn-badao（男霸道，占位待替换）
    # 旁白用 narrator，女性角色用 Sweet_Lady
    voice_mapping: dict = field(default_factory=lambda: {
        "narrator": "Chinese_gravelly_storyteller_nv1",
        "narrator_male": "Chinese_gravelly_storyteller_nv1",
        "narrator_female": "Chinese (Mandarin)_Sweet_Lady",
        # 角色→音色（按需在此添加，角色名须与剧情解析一致）
        "钟岳": "Chinese_worker_male",
        "薪火": "Chinese (Mandarin)_Sweet_Lady",
    })
    # 默认音色（voice_mapping 中找不到时用）
    default_voice_id: str = os.getenv("DEFAULT_VOICE_ID", "Chinese_gravelly_storyteller_nv1")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    media: MediaConfig = field(default_factory=MediaConfig)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- 全局单例 ----------
_global_config: Optional[Config] = None


def load_config() -> Config:
    """加载全局配置，合并环境变量。"""
    cfg = Config()
    if not cfg.model.api_key:
        raise RuntimeError(
            "未检测到 MINIMAX_API_KEY，请设置环境变量或直接修改 config.py。"
        )
    return cfg


def get_config() -> Config:
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config_file: Optional[str] = None) -> Config:
    """从 JSON 配置文件加载并覆盖默认配置（运行期覆盖）。

    Args:
        config_file: 配置文件路径（.json）。None 则重新从环境变量加载。
    """
    global _global_config
    # 直接构造默认配置，不走 load_config() 的 key 校验
    # （key 由配置文件提供时，环境变量可能为空，不应在此抛错）
    cfg = Config()
    if config_file and os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 逐层覆盖
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "run" in data:
            for k, v in data["run"].items():
                if hasattr(cfg.run, k):
                    setattr(cfg.run, k, v)
        if "storage" in data:
            for k, v in data["storage"].items():
                # 空值跳过（保留 dataclass 默认的绝对路径），仅非空才覆盖
                if v and hasattr(cfg.storage, k):
                    setattr(cfg.storage, k, v)
        if "media" in data:
            for k, v in data["media"].items():
                if hasattr(cfg.media, k):
                    setattr(cfg.media, k, v)
    # 覆盖后仍无 key 才报错
    if not cfg.model.api_key:
        raise RuntimeError(
            "未检测到 MINIMAX_API_KEY，请设置环境变量或在配置文件 model.api_key 中填写。"
        )
    # 自动创建存储目录（不存在则建），保证下载即用
    for d in (cfg.storage.output_dir, cfg.storage.memory_dir,
              os.path.dirname(cfg.storage.sqlite_path)):
        if d:
            os.makedirs(d, exist_ok=True)
    _global_config = cfg
    return cfg
