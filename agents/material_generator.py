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

【核心对齐规则】image_prompts 与 tts_meta 必须 1:1 严格对齐：
- 两者数量完全相同，index 从 1 起递增，一一对应
- 每段 TTS 朗读文本对应恰好一张画面，语音说到什么场景就配什么图
- 语音段落的分切以「画面切换时机」为准：剧情场景转换、情绪转折、新角色登场、关键道具出现 → 切到新画面
- 不要一段语音配多张图，也不要一张图覆盖多段语音

【输出格式 严格 JSON】
{
  "script": "整集口语化讲解文案（纯文本，用『』分隔段落，每段对应一张画面）",
  "image_prompts": [
    {
      "index": 1,
      "segment": "对应 script 中第几个段落",
      "prompt": "动漫风格生图Prompt，必须包含：(1) 'anime style'风格前缀 (2) 人物外貌（严格沿用人物档案，发色/发型/服装/体型）(3) 精确动作与表情 (4) 场景环境细节 (5) 关键视觉特征必须明确写出（如四翼金鸟必须写'four-winged golden bird with four wings spread'，不可只写'bird'）(6) 光影氛围匹配情绪 (7) 镜头构图",
      "mood": "本画面情绪关键词"
    }
  ],
  "tts_meta": [
    {
      "index": 1,
      "text": "本段朗读文本（从 script 切分，与同 index 的 image_prompt 画面一一对应）",
      "voice": "角色名（如钟岳、薪火、narrator），旁白用 narrator",
      "emotion": "情绪：neutral | excited | sad | tense | calm | angry | surprised",
      "speed": "语速 0.8-1.2 浮点",
      "pause_after": "段后停顿秒数 0.0-2.0"
    }
  ]
}

严格要求：
1. 讲解文案必须清晰交代本集的因果链与伏笔，不要遗漏关键事实
2. 生图 Prompt 中人物外貌必须100%沿用给定的人物档案，不得擅自改变设定
3. 生图 Prompt 必须以 'anime style, ' 开头，整体画面为日式动漫风格
4. 关键视觉特征必须精确描述：怪物的形态（几翼/几首/几尾）、武器外形、特殊道具、环境特征（黑霾/幽谷/火树），不可用模糊词如'bird''monster''weapon'替代
5. 光影氛围必须匹配该段落剧情情绪（紧张→冷暗色调、温馨→暖亮色调、战斗→高对比）
6. image_prompts 的 index 与 tts_meta 的 index 必须完全一致，数量相同
7. 不得新增剧情、不得篡改事实，文案是对原文的口语化转述"""


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
