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


SYSTEM_PROMPT = """你是短视频多媒体素材生成专家。基于一集剧集（Episode）的剧情因果链，一次性生成三类素材，保证情绪、叙事视角统一。

【输出格式 严格 JSON】
{
  "script": "整集口语化讲解文案（纯文本，适合口播，600-1200字，用『』分隔段落，每段对应一个画面/情绪节拍）",
  "image_prompts": [
    {
      "index": 1,
      "segment": "对应 script 中第几个段落的画面",
      "prompt": "英文/中文生图Prompt，必须包含：人物外貌描述（严格沿用给定人物设定）、动作、表情、场景环境、光影氛围、镜头构图",
      "mood": "本画面情绪关键词"
    }
  ],
  "tts_meta": [
    {
      "index": 1,
      "text": "本段朗读文本（从 script 切分，逐段对齐）",
      "voice": "音色选择：narrator_male | narrator_female | character_<name>",
      "emotion": "情绪：neutral | excited | sad | tense | calm | angry | surprised",
      "speed": "语速 0.8-1.2 浮点",
      "pause_after": "段后停顿秒数 0.0-2.0"
    }
  ]
}

严格要求：
1. 讲解文案必须清晰交代本集的因果链与伏笔，不要遗漏关键事实
2. 生图 Prompt 中人物外貌必须100%沿用给定的人物档案，不得擅自改变设定
3. 光影氛围必须匹配该段落剧情情绪（紧张→冷暗、温馨→暖亮、战斗→高对比）
4. TTS 段落与 script 段落、image_prompts 时序一一对应
5. 不得新增剧情、不得篡改事实，文案是对原文的口语化转述"""


class MaterialGeneratorAgent:
    """多媒体素材生成 Agent。"""

    def __init__(self):
        self.llm = get_llm(role="production")

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
            SystemMessage(content=SYSTEM_PROMPT),
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

请生成三类素材，输出 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            # 规范化 image_prompts 为 List[str]
            if isinstance(data.get("image_prompts"), list):
                data["image_prompts"] = [
                    p.get("prompt", json.dumps(p, ensure_ascii=False))
                    if isinstance(p, dict) else str(p)
                    for p in data["image_prompts"]
                ]
            return data
        # 兜底
        return {
            "script": content if isinstance(content, str) else "",
            "image_prompts": [],
            "tts_meta": [],
            "_parse_error": True,
        }
