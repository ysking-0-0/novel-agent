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
    """存储路径与介质配置。"""
    # 成品输出目录
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    # 断点快照 Sqlite 路径
    sqlite_path: str = os.getenv("SQLITE_PATH", "./checkpoints/checkpoint.sqlite")
    # 全局记忆库目录（人物/事件/伏笔 JSON + FAISS 索引）
    memory_dir: str = os.getenv("MEMORY_DIR", "./memory")
    # 是否启用向量检索（FAISS）
    enable_vector_retrieval: bool = (
        os.getenv("ENABLE_VECTOR_RETRIEVAL", "true").lower() == "true"
    )
    # 向量库维度（与 embedding 模型一致）
    vector_dim: int = int(os.getenv("VECTOR_DIM", "1536"))


@dataclass
class MediaConfig:
    """多媒体合成配置：生图/TTS/视频拼接。"""
    # 是否启用合成阶段（persistence 后自动触发生图+TTS+视频）
    enable_synthesis: bool = (
        os.getenv("ENABLE_SYNTHESIS", "true").lower() == "true"
    )
    # 生图并发数（实测 image-01 在 4 并发内不排队，5 并发会被限流）
    image_concurrency: int = int(os.getenv("IMAGE_CONCURRENCY", "4"))
    # TTS 并发数（speech-02-hd 单次约 1.4s，4 并发安全）
    tts_concurrency: int = int(os.getenv("TTS_CONCURRENCY", "4"))
    # 生图模型
    image_model: str = os.getenv("IMAGE_MODEL", "image-01")
    # TTS 模型
    tts_model: str = os.getenv("TTS_MODEL", "speech-02-hd")
    # 图片宽高比
    image_aspect_ratio: str = os.getenv("IMAGE_ASPECT_RATIO", "16:9")
    # 视频分辨率（与图片比例匹配）
    video_resolution: str = os.getenv("VIDEO_RESOLUTION", "1280x720")
    # 视频帧率
    video_fps: int = int(os.getenv("VIDEO_FPS", "30"))
    # 单幅图片展示目标时长（秒），material_generator 据此估算图片数量
    # 一两句话同一场景描述为一幅，约10秒换一幅
    image_duration_target: float = float(os.getenv("IMAGE_DURATION_TARGET", "10.0"))
    # 背景音乐文件路径（None=不加BGM）。BGM 会被循环/截断对齐视频时长，音量降低做背景
    bgm_path: str = os.getenv("BGM_PATH", "./assets/bgm.mp3")
    # BGM 音量（0.0-1.0，相对TTS为1.0）。0.15 表示BGM声压约为TTS的15%
    bgm_volume: float = float(os.getenv("BGM_VOLUME", "0.15"))
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
                if hasattr(cfg.storage, k):
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
    _global_config = cfg
    return cfg
