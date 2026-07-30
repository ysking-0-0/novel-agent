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


class MaterialGeneratorAgent:
    """多媒体素材生成 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")
        art_style = getattr(get_config().media, "art_style", "anime")
        self.system_prompt = load_prompt("material_generator", art_style=art_style)

    def generate(self, episode: Dict, revise_instruction: Optional[Dict] = None) -> Dict:
        """生成三类素材。返回 dict 含 script / image_prompts / tts_meta。"""
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

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._build_user_msg(episode, char_profiles, revise_instruction)),
        ]
        resp = self.llm.invoke(messages)
        return self._parse(resp.content)

    def invoke(self, state: Dict) -> Dict:
        """LangGraph 节点入口。"""
        result = self.generate(
            state.get("current_episode") or {},
            (state.get("review_result") or {}).get("unified_revise_instruction"),
        )
        return {
            "episode_script": result.get("script", ""),
            "episode_image_prompts": result.get("image_prompts", []),
            "episode_tts_meta": result.get("tts_meta", []),
            "format_valid": False,  # 进入校验节点重置
            "format_errors": [],
        }

    def _build_user_msg(self, episode: Dict, char_profiles: List[Dict],
                        revise_instruction: Optional[Dict]) -> str:
        from config import get_config
        cfg = get_config().media
        scenes_text = json.dumps(episode.get("scenes", []), ensure_ascii=False, indent=2)
        char_text = json.dumps(char_profiles, ensure_ascii=False, indent=2) if char_profiles else "（暂无人物档案，请从剧情中提取外貌并保持一致）"
        revise_text = ""
        if revise_instruction:
            revise_text = f"\n\n【上一轮评审修改指令（必须据此修正）】\n{json.dumps(revise_instruction, ensure_ascii=False, indent=2)}"
        return f"""【本集 Episode 数据】
{scenes_text}

【本集概述】{episode.get('summary','')}
【本集悬念】{episode.get('cliffhanger','')}

【全局人物设定档案（生图Prompt必须严格沿用）】
{char_text}{revise_text}

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
