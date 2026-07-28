"""
novel_pipeline.nodes.format_validator
格式校验节点（纯代码）—— 主循环步骤 5。
对应设计文档 5.2。

职责：
  校验三类产出的 JSON 结构、字段完整性、ID 规范性
  格式错误返回步骤 4 局部修正
  超过 max_retries 仍不通过 → 标记 manual_review 后放行归档（防止死循环）
"""
from typing import Dict
from state import NovelState
from config import get_config


REQUIRED_EPISODE_FIELDS = ["scenes", "summary", "cliffhanger"]
REQUIRED_SCENE_FIELDS = ["scene_id", "summary", "cause", "motivation", "core_action", "immediate_result", "long_term_impact", "characters", "foreshadows"]
REQUIRED_TTS_FIELDS = ["index", "text", "voice", "emotion", "speed", "pause_after"]


def format_validator_node(state: Dict) -> Dict:
    """校验格式，不直接修复。返回 format_valid + format_errors 给路由节点判断。"""
    errors = []
    episode = state.get("current_episode") or {}
    script = state.get("episode_script") or ""
    image_prompts = state.get("episode_image_prompts") or []
    tts_meta = state.get("episode_tts_meta") or []

    # 1. episode 字段
    for f in REQUIRED_EPISODE_FIELDS:
        if f not in episode or episode[f] in (None, ""):
            errors.append(f"episode 缺失字段: {f}")

    # 2. scenes 字段
    for sc in episode.get("scenes", []):
        for f in REQUIRED_SCENE_FIELDS:
            if f not in sc:
                errors.append(f"scene {sc.get('scene_id','?')} 缺失字段: {f}")
        sid = sc.get("scene_id", "")
        if not sid or not isinstance(sid, str):
            errors.append(f"scene_id 不规范: {sid}")

    # 3. script 非空
    if not script.strip():
        errors.append("episode_script 为空")

    # 4. image_prompts 非空列表（支持 List[str] 或 List[dict]）
    if not isinstance(image_prompts, list) or len(image_prompts) == 0:
        errors.append("episode_image_prompts 为空")
    else:
        for i, p in enumerate(image_prompts):
            if isinstance(p, dict):
                if not p.get("prompt") or not str(p["prompt"]).strip():
                    errors.append(f"image_prompt[{i}] prompt 为空")
            elif not isinstance(p, str) or not p.strip():
                errors.append(f"image_prompt[{i}] 非字符串或为空")

    # 5. tts_meta 非空列表且字段齐全
    if not isinstance(tts_meta, list) or len(tts_meta) == 0:
        errors.append("episode_tts_meta 为空")
    else:
        for i, t in enumerate(tts_meta):
            for f in REQUIRED_TTS_FIELDS:
                if f not in t:
                    errors.append(f"tts_meta[{i}] 缺失字段: {f}")

    valid = len(errors) == 0
    retry = state.get("retry_count", 0)
    max_retry = get_config().run.max_retries
    update: Dict = {
        "format_valid": valid,
        "format_errors": errors,
    }
    if not valid:
        if retry < max_retry:
            # 未超限：递增计数，回 material_generator 局部修正
            retry += 1
            update["retry_count"] = retry
            print(f"[格式] 校验失败（{retry}/{max_retry}），回素材生成修正")
        else:
            # 超限：标记人工复核，由路由直送 persistence 归档（绕过 review）
            print(f"[格式] 重试超限({retry}>={max_retry})，标记人工复核后归档")
            rr = state.get("review_result") or {}
            rr["manual_review"] = True
            rr["manual_review_reason"] = "format_check_exhausted"
            rr["format_errors"] = errors
            update["review_result"] = rr
    return update
