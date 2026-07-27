"""
novel_pipeline.llm_factory
统一 LLM 实例工厂，绑定 MiniMax-M3（OpenAI 兼容接口）。
按角色分层实例化：生产层 / 评审层 / 支撑层。
"""
from typing import Literal
from langchain_openai import ChatOpenAI
from config import get_config


def _build(model_name: str) -> ChatOpenAI:
    """构建带请求日志的 LLM。"""
    cfg = get_config().model
    return ChatOpenAI(
        model=model_name or cfg.production_model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


def get_llm(role: Literal["production", "review", "support"] = "production") -> ChatOpenAI:
    """按角色获取 LLM。生产层高速通用、评审层强推理、支撑层高速通用。"""
    cfg = get_config().model
    model_name = {
        "production": cfg.production_model,
        "review": cfg.review_model,
        "support": cfg.support_model,
    }[role]
    return _build(model_name)


def with_structured_output(llm: ChatOpenAI, schema):
    """统一结构化输出调用入口（兼容不同 langchain 版本）。"""
    return llm.with_structured_output(schema)
