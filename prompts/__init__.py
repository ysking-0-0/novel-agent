"""
novel_pipeline.prompts
外部化提示词加载器——从 prompts/*.md 读取各 agent 的 system prompt。

设计：
- 各 agent 的 SYSTEM_PROMPT 从对应 .md 文件加载，运行时可热更新
- material_generator.md 含 {{ART_STYLE}} 占位符，由 load_prompt 实参注入风格
- 文件缺失时回退到 agent .py 内的内置 DEFAULT（保证不崩）
- Gradio "提示词管理"标签页直接读写这些 .md 文件
"""
import os
from typing import Optional

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# 风格预设
ART_STYLES = {
    "anime": "anime style",
    "realistic": "realistic photo style",
}


def load_prompt(name: str, art_style: Optional[str] = None) -> str:
    """加载指定 agent 的 system prompt。

    name: agent 名（对应 prompts/<name>.md）
    art_style: 生图风格，仅 material_generator 生效。
               None→用 ART_STYLES["anime"]；"anime"/"realistic"→映射到完整前缀；
               其他字符串→原样注入（支持自定义）。
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # 占位符替换（仅 material_generator 使用）
    if "{{ART_STYLE}}" in content:
        if art_style is None:
            style_val = ART_STYLES["anime"]
        elif art_style in ART_STYLES:
            style_val = ART_STYLES[art_style]
        else:
            style_val = art_style
        content = content.replace("{{ART_STYLE}}", style_val)
    return content


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
