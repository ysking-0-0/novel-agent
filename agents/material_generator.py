"""
novel_pipeline.agents.material_generator
多媒体素材生成 Agent —— 三类产出统一生成入口。
对应设计文档 4.1.3。

核心职责：基于单集 Episode 因果链，一次性并行产出三类素材：
1. 整集口语化讲解文案：清晰交代因果与伏笔，适配短视频口播节奏
2. 时序化生图 Prompt 组：严格沿用全局人物设定，光影氛围匹配剧情情绪
3. 结构化 TTS 参数：朗读文本/音色/情绪/语速/停顿策略，与文案逐段对齐
"""
import json
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_llm
from agents.memory_manager import get_memory_agent
from prompts import load_prompt
from config import get_config


def _prioritize_char(c: dict) -> dict:
    """角色档案双区合并：生图用描述优先取 user_description，为空回退到 AI 维护的 appearance。
    返回扁平化的生图描述，避免 LLM 在两种字段间困惑。
    """
    user_desc = (c.get("user_description") or "").strip()
    ai_appearance = (c.get("appearance") or "").strip()
    ai_attire = (c.get("attire") or "").strip()
    out = dict(c)
    if user_desc:
        out["image_description"] = f"【用户指定·优先】{user_desc}"
    else:
        parts = [p for p in [ai_appearance, ai_attire] if p]
        out["image_description"] = "，".join(parts) if parts else "（无外貌档案，从剧情推断）"
    return out


class MaterialGeneratorAgent:
    """多媒体素材生成 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")

    def generate(self, episode: Dict, revise_instruction: Optional[Dict] = None,
                 prev_materials: Optional[Dict] = None) -> Dict:
        """生成三类素材。返回 dict 含 script / image_prompts / tts_meta。

        revise_instruction: 上一轮评审修改指令。若含 minor_revise 且提供 prev_materials
          （上一轮已生成的 script/image_prompts/tts_meta），走"局部微调"：把旧素材交给 LLM，
          只修改被点名的问题项，其余原样保留，避免全量重生成浪费 API。
        """
        # 每次生成时动态加载 prompt，确保 art_style 切换后立即生效
        # （__init__ 只调一次，若 Agent 被复用，固化 prompt 会导致旧风格残留）
        art_style = getattr(get_config().media, "art_style", "anime")
        system_prompt = load_prompt("material_generator", art_style=art_style)
        memory = get_memory_agent()
        # 取本集关联人物设定档案
        char_ids = set()
        for s in episode.get("scenes", []):
            for c in s.get("characters", []):
                if isinstance(c, dict):
                    cid = c.get("char_id") or c.get("name")
                    if cid:
                        char_ids.add(cid)
        char_profiles = [c for c in memory.get_character_profiles() if c.get("char_id") in char_ids] if char_ids else []
        # 生图优先级：用户描述（user_description）> AI 维护的 appearance+attire
        # 这里保留完整档案传入，LLM 在 prompt 引导下优先用 user_description
        char_profiles = [_prioritize_char(c) for c in char_profiles]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=self._build_user_msg(episode, char_profiles, revise_instruction, prev_materials)),
        ]
        # LLM 偶发失败/输出截断会返回空素材 → 空集归档。
        # 内部自动重试最多 3 次，并打印诊断信息；全部失败才返回空（由 persistence 防归档拦截）。
        import time
        last_content = ""
        for attempt in range(3):
            try:
                resp = self.llm.invoke(messages)
                content = resp.content
                last_content = content
                parsed = self._parse(content)
                if isinstance(parsed, dict) and not parsed.get("_parse_error") \
                        and parsed.get("image_prompts") and parsed.get("tts_meta"):
                    return parsed
                # 解析成功但关键字段为空（LLM 输出被截断/格式错）：打印诊断后重试
                content_str = str(content)
                print("    [素材生成] LLM 输出解析失败或为空 (attempt %d/3)，长度=%d，前200字: %s"
                      % (attempt + 1, len(content_str), content_str[:200].replace("\n", " ")))
            except Exception as e:
                print("    [素材生成] LLM 调用异常 (attempt %d/3): %s" % (attempt + 1, e))
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        # 三次都失败：返回最后一次解析结果（可能为空，由上层格式校验+空集防护兜底）
        return self._parse(last_content) if last_content else {
            "script": "", "image_prompts": [], "tts_meta": [], "_parse_error": True,
        }

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口。"""
        rr = state.get("review_result") or {}
        revise = rr.get("unified_revise_instruction")
        verdict = rr.get("verdict", "pass")
        # 格式校验失败也要反馈给 LLM，否则重试是盲目的（会重复同样的错误）
        format_errors = state.get("format_errors") or []
        if format_errors:
            err_instr = {"format_errors": format_errors,
                         "message": "上一轮素材未通过格式校验，必须按下述错误逐条修正后重新生成完整 JSON。特别注意：图片数量严禁超过 35 张；image_prompts/tts_meta 严禁为空。"}
            if isinstance(revise, dict):
                revise = {**revise, "format_errors": format_errors}
            else:
                revise = err_instr
        # 局部微调：仅当仲裁是 minor_revise（轻微瑕疵）且有上一轮素材时，
        # 把旧素材交给 LLM 只改被点名项；regenerate / 首次生成仍全量重生成。
        prev_materials = None
        if verdict == "minor_revise":
            prev = {
                "script": state.get("episode_script") or "",
                "image_prompts": state.get("episode_image_prompts") or [],
                "tts_meta": state.get("episode_tts_meta") or [],
            }
            if prev["image_prompts"] and prev["tts_meta"]:
                prev_materials = prev
                print("    [素材生成] minor_revise → 局部微调（只改问题项，保留其余素材）")
        result = self.generate(
            state.get("current_episode") or {},
            revise,
            prev_materials,
        )
        return {
            "episode_script": result.get("script", ""),
            "episode_image_prompts": result.get("image_prompts", []),
            "episode_tts_meta": result.get("tts_meta", []),
            "format_valid": False,  # 进入校验节点重置
            "format_errors": [],
        }

    def _build_user_msg(self, episode: Dict, char_profiles: List[Dict],
                        revise_instruction: Optional[Dict],
                        prev_materials: Optional[Dict] = None) -> str:
        from config import get_config
        cfg = get_config().media
        scenes_text = json.dumps(episode.get("scenes", []), ensure_ascii=False, indent=2)
        char_text = json.dumps(char_profiles, ensure_ascii=False, indent=2) if char_profiles else "（暂无人物档案，请从剧情中提取外貌并保持一致）"
        revise_text = ""
        if revise_instruction:
            revise_text = f"\n\n【上一轮评审修改指令（必须据此修正）】\n{json.dumps(revise_instruction, ensure_ascii=False, indent=2)}"
        prev_text = ""
        if prev_materials:
            prev_text = f"""

【上一轮已生成的素材（局部微调基础，请原样保留未点名部分）】
这是上一轮生成的三类素材。本次只做【局部微调】：
- 仅修改评审修改指令中点名的问题项（对应场景的 script / 图像 prompt / TTS 段）
- 未被点名的内容【必须原样保留】，不要改写、不要重排、不要新增
- 图片/语音数量尽量与上一轮一致，改动范围越小越好

上一轮 script：
{prev_materials.get('script', '')}

上一轮 image_prompts（JSON）：
{json.dumps(prev_materials.get('image_prompts', []), ensure_ascii=False, indent=2)}

上一轮 tts_meta（JSON）：
{json.dumps(prev_materials.get('tts_meta', []), ensure_ascii=False, indent=2)}
"""
        return f"""【本集 Episode 数据】
{scenes_text}

【本集概述】{episode.get('summary','')}
【本集悬念】{episode.get('cliffhanger','')}

【全局人物设定档案（生图Prompt必须严格沿用）】
每个角色的 image_description 字段为生图外貌依据：
- 若含"用户指定·优先"前缀 → 必须以此描述为准，忽略其他外貌字段
- 否则用 appearance+attire 组合
{char_text}{revise_text}{prev_text}

【图片节奏参数】
- 目标每图展示时长：{cfg.image_duration_target}秒
- 估算图片张数 ≈ 语音总时长 / {cfg.image_duration_target}秒
- BGM 背景音乐：{'已配置（' + cfg.bgm_path + '），音量' + str(int(cfg.bgm_volume*100)) + '%'}（如适用）

请生成三类素材，直接输出 JSON，不要输出任何思考过程、分析、解释，不要用思考标签，直接以 {{ 开头输出 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            # 规范化 image_prompts 为 List[dict]，每个含 prompt + narration_segment + start_ratio
            if isinstance(data.get("image_prompts"), list):
                normalized = []
                for p in data["image_prompts"]:
                    if isinstance(p, dict):
                        prompt_text = p.get("prompt", "")
                        narr_seg = p.get("narration_segment") or p.get("segment")
                        # narration_segment 转整数（对应 tts_meta.index）
                        try:
                            narr_seg = int(narr_seg) if narr_seg is not None else None
                        except (ValueError, TypeError):
                            narr_seg = None
                        # start_ratio: 该图在所属语音段内的出现进度位置 (0.0-1.0)
                        sr = p.get("start_ratio")
                        try:
                            sr = float(sr) if sr is not None else 0.0
                            sr = max(0.0, min(1.0, sr))
                        except (ValueError, TypeError):
                            sr = 0.0
                        # characters 字段：本图涉及的角色 char_id 列表（用于匹配定妆照）
                        chars_in_img = p.get("characters", [])
                        if not isinstance(chars_in_img, list):
                            chars_in_img = []
                        normalized.append({
                            "prompt": prompt_text,
                            "narration_segment": narr_seg,
                            "start_ratio": sr,
                            "mood": p.get("mood", ""),
                            "characters": chars_in_img,
                        })
                    else:
                        normalized.append({"prompt": str(p), "narration_segment": None, "start_ratio": 0.0, "mood": "", "characters": []})
                data["image_prompts"] = normalized
            return data
        # 兜底
        return {
            "script": content if isinstance(content, str) else "",
            "image_prompts": [],
            "tts_meta": [],
            "_parse_error": True,
        }
