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


SYSTEM_PROMPT = """你是中国古代神话题材短视频多媒体素材生成专家。基于一集剧集（Episode）的剧情因果链，一次性生成三类素材，保证情绪、叙事视角统一。

【画面风格要求（最高优先级）】
所有生图 Prompt 必须以 'ancient Chinese mythology art style, ' 开头，整体风格为：
- 中国古代神话风格，古朴、蛮荒、磅礴，具有洪荒神话感
- 男性角色：英俊帅气，五官立体，身姿挺拔，气质不凡
- 女性角色：美丽动人，容颜绝世，气质出尘
- 服饰必须符合境界与身份：外门弟子着粗麻兽皮短打，内门弟子着素色道袍，长老着华贵锦袍佩玉，神族着古朴祭祀礼服
- 发饰符合古风：束发玉冠、木簪、骨笄，不用现代发饰
- 场景环境：崇山峻岭、幽谷深壑、上古遗迹，古朴蛮荒，磅礴大气，具有神话时代感
- 光影：神秘古拙，多用幽光、灵气、神光，不用现代光源

【图片粒度规则】
- 目标节奏：每幅图展示约10秒（image_duration_target=10秒）
- 拆图原则：一两句描述同一场景/动作的画面为一幅图；当描述切换到新场景/新动作/新视角时，换下一幅图
- 参考范例（『生图频率描述.txt』节选）：
  "钟岳攀上巨石采摘五香芝（图1），忽然一阵戾啸破空袭来，只见一只四翼金鸟展开四只金色羽翅俯冲扑击（图2）！钟岳临危不乱，抄起几株五香芝纵身跃下巨石（图3）。千钧一发之际，他双手死死抓住崖壁青藤，悬于巨石下方一丈处。四翼金鸟扑了个空（图4），戾啸着冲天而起（图5），在空中盘旋蓄势，准备发动第二次攻击（图6）。钟岳咬紧牙关，沿着青藤飞速向深谷滑去（图7）"
  ——7幅图覆盖约70秒讲解，每图约10秒
- 估算数量：全集语音总时长 / 10秒 ≈ 图片张数。如全集约90秒讲解 → 约9张图
- image_prompts 的 index 独立递增，与 tts_meta 不要求数量相同
- 每个 image_prompt 必须标注 narration_segment：对应 tts_meta 的第几段（语音播放到该段时展示此图）

【输出格式 严格 JSON】
{
  "script": "整集口语化讲解文案（纯文本，用『』分隔段落，每段对应一个叙事单元）",
  "image_prompts": [
    {
      "index": 1,
      "narration_segment": "对应 tts_meta 的第几段（整数，语音播放到该段时展示此图）",
      "characters": ["本图涉及的角色 char_id 列表（如 ['钟岳','薪火']），用于匹配定妆照保证人物一致",
      "prompt": "以 'ancient Chinese mythology art style, ' 开头的生图Prompt，必须包含：(1) 风格前缀 (2) 人物固定外貌——age年龄、identity身份、appearance发色发型/眼/眉/身材/五官（100%沿用档案，不得改变）、attire穿着 (3) 精确动作与表情 (4) 场景环境（古朴蛮荒磅礴）(5) 关键视觉特征精确写出（四翼金鸟必须写'four-winged golden bird with four distinct wings spread'，不可只写'bird'）(6) 神话氛围光影 (7) 镜头构图。同一人物在所有图中外貌特征完全一致",
      "mood": "本画面情绪关键词"
    }
  ],
  "tts_meta": [
    {
      "index": 1,
      "text": "本段朗读文本（从 script 切分）",
      "voice": "角色名（如钟岳、薪火、narrator），旁白用 narrator",
      "emotion": "情绪：neutral | excited | sad | tense | calm | angry | surprised",
      "speed": "语速 0.8-1.2 浮点",
      "pause_after": "段后停顿秒数 0.0-2.0"
    }
  ]
}

严格要求：
1. 讲解文案必须清晰交代本集的因果链与伏笔，不要遗漏关键事实
2. 【人物一致性（最高优先级）】生图 Prompt 中人物外貌必须100%沿用给定的人物档案：
   - 必须包含档案中的 age（年龄）、identity（身份）、appearance（固定外貌：发色发型/眼/眉/身材/五官）、attire（穿着）
   - 不得擅自改变年龄、外貌、体型——人物不会突然变老/变年轻/换发型/换体型
   - 同一人物在所有图中必须保持相同的外貌特征（发色、发型、眼型、体型、五官）
   - 若档案中某人物缺少 appearance 字段，从 identity 和原文细节合理推断一次并全程保持
3. 生图 Prompt 必须以 'ancient Chinese mythology art style, ' 开头
4. 人物必须英俊帅气（男）/美丽动人（女），服饰符合境界身份，发饰古风
5. 关键视觉特征必须精确描述：怪物的形态（几翼/几首/几尾）、武器外形、特殊道具、环境特征，不可用模糊词
6. 图片粒度：每幅约10秒，一两句同一场景描述为一幅，场景/动作切换换下一幅（数量≈语音总时长/10秒）
7. 每个 image_prompt 必须有 narration_segment 标注对应 tts_meta 的第几段
8. 不得新增剧情、不得篡改事实，文案是对原文的口语化转述"""


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

请生成三类素材，输出 JSON。"""

    def _parse(self, content) -> Dict:
        from utils import extract_json_dict
        data = extract_json_dict(content)
        if isinstance(data, dict):
            # 规范化 image_prompts 为 List[dict]，每个含 prompt + narration_segment
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
                        # characters 字段：本图涉及的角色 char_id 列表（用于匹配定妆照）
                        chars_in_img = p.get("characters", [])
                        if not isinstance(chars_in_img, list):
                            chars_in_img = []
                        normalized.append({
                            "prompt": prompt_text,
                            "narration_segment": narr_seg,
                            "mood": p.get("mood", ""),
                            "characters": chars_in_img,
                        })
                    else:
                        normalized.append({"prompt": str(p), "narration_segment": None, "mood": "", "characters": []})
                data["image_prompts"] = normalized
            return data
        # 兜底
        return {
            "script": content if isinstance(content, str) else "",
            "image_prompts": [],
            "tts_meta": [],
            "_parse_error": True,
        }
