"""
novel_pipeline.config
全局配置：模型参数、分片大小、目标集数、路径、重试上限等。
通过环境变量覆盖，便于不同部署环境。

配置分三组：
  model   —— LLM 绑定（生产/评审/支撑三层模型）
  run     —— 运行参数（分片大小、重试上限、目标集数）
  storage —— 存储路径（sqlite 断点、成品目录、记忆库、向量库开关）
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
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "8192"))


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
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

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
    cfg = load_config()
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
    _global_config = cfg
    return cfg
