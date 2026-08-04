"""
novel_pipeline.prompts
外部化提示词加载器——从 prompts/*.md 读取各 agent 的 system prompt。

设计：
- 各 agent 的 SYSTEM_PROMPT 从对应 .md 文件加载，运行时可热更新
- material_generator.md 含 {{ART_STYLE}} 占位符，由 load_prompt 实参注入风格
  风格词从 prompts/image_style.json 的预设读取（不再硬编码 ART_STYLES）
- 文件缺失时回退到内置默认（保证不崩）
- Gradio "提示词管理" / "生图风格" 标签页直接读写这些文件
"""
import os
import json
from typing import Optional, Dict, List

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_STYLE_FILE = os.path.join(_PROMPTS_DIR, "image_style.json")

# 回退默认（image_style.json 缺失时用）
_FALLBACK_LLM_PREFIX = "anime style, ancient Chinese mythology art style"


def load_image_styles() -> Dict:
    """加载 image_style.json 全文。文件缺失返回空结构。"""
    if not os.path.exists(_IMAGE_STYLE_FILE):
        return {"presets": {}}
    try:
        with open(_IMAGE_STYLE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"presets": {}}


def save_image_styles(data: Dict) -> None:
    """保存 image_style.json。"""
    with open(_IMAGE_STYLE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_style_presets() -> List[str]:
    """列出所有预设名。"""
    data = load_image_styles()
    return sorted(data.get("presets", {}).keys())


def get_style_preset(name: str) -> Dict:
    """取指定预设的 {llm_prompt_prefix, api_prefix, api_negative}。不存在返回空。"""
    data = load_image_styles()
    return data.get("presets", {}).get(name, {})


def save_style_preset(name: str, llm_prefix: str, api_prefix: str, api_negative: str) -> None:
    """保存/更新一个预设。name 不能为空。"""
    data = load_image_styles()
    if "presets" not in data:
        data["presets"] = {}
    data["presets"][name] = {
        "llm_prompt_prefix": llm_prefix or "",
        "api_prefix": api_prefix or "",
        "api_negative": api_negative or "",
    }
    save_image_styles(data)


def delete_style_preset(name: str) -> None:
    """删除一个预设。"""
    data = load_image_styles()
    if name in data.get("presets", {}):
        del data["presets"][name]
        save_image_styles(data)


def load_prompt(name: str, art_style: Optional[str] = None) -> str:
    """加载指定 agent 的 system prompt。

    name: agent 名（对应 prompts/<name>.md）
    art_style: 生图风格预设名（仅 material_generator 生效）。
               None→读 config.json 的 art_style 值作为预设名；
               存在于 image_style.json 预设→取该预设的 llm_prompt_prefix；
               不存在→原样注入（兼容旧值 "anime"/"realistic"）。
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # 占位符替换（仅 material_generator 使用）
    if "{{ART_STYLE}}" in content:
        style_val = _resolve_llm_prefix(art_style)
        content = content.replace("{{ART_STYLE}}", style_val)
    return content


def _resolve_llm_prefix(art_style: Optional[str]) -> str:
    """把 art_style 预设名解析为 LLM 侧的 prompt 前缀。"""
    if art_style is None:
        # 从 config.json 读当前预设名
        try:
            from config import get_config
            art_style = getattr(get_config().media, "art_style", "anime")
        except Exception:
            art_style = "anime"
    # 从 image_style.json 预设取
    preset = get_style_preset(art_style)
    if preset:
        return preset.get("llm_prompt_prefix", "") or _FALLBACK_LLM_PREFIX
    # 不在预设里：旧值兼容（"anime"→默认前缀，其他→原样当字符串用）
    if art_style == "anime":
        return _FALLBACK_LLM_PREFIX
    return art_style


def save_prompt(name: str, content: str) -> None:
    """保存编辑后的 prompt（Gradio 提示词管理用）。"""
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def list_prompts() -> list:
    """列出所有可用 prompt 名。"""
    names = []
    for f in sorted(os.listdir(_PROMPTS_DIR)):
        if f.endswith(".md"):
            names.append(f[:-3])
    return names
