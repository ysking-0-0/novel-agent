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

【重要：输出格式（最高优先级）】
你必须直接输出合法 JSON 对象，不得输出任何思考过程、分析推理、思维链、解释说明。响应必须以 {{ 开头，以 }} 结尾。禁止输出代码块之外的任何文字。禁止输出"让我分析""根据数据"等引导语。JSON 必须包含 script、image_prompts、tts_meta 三个字段。

【讲解文案风格要求（最高优先级·彻底去说书化）】
禁止任何说书人/说书摊风格，禁止传统评书腔调，禁止观众互动式口吻：
- 禁止："各位观众""各位看官""朋友们""听众""咱们""俺""却说""且说""话说回来""这一歇不要紧""这下可不得了""欲知后事如何""且听下回分解""列位""话休絮烦"
- 禁止以"各位观众朋友们好""大家好""今天咱们来讲"等寒暄/招呼开头，直接进入剧情内容
- 禁止"你看""您想""你猜怎么着""想不到吧"等对观众喊话
- 禁止"好家伙""乖乖""我的天"等夸张感叹词作为讲述口吻
采用第三人称客观叙述+适度解说：像一个冷静的纪录片旁白，陈述事实+必要背景点拨，不拽观众、不卖关子、不抖包袱
自然衔接用剧情本身的因果逻辑过渡，不用"话说""却说"等转折词

【讲解文案内容要求·缩略讲解（不是逐句复述原文）】
讲解文案是对原文的"浓缩提炼+必要解说"，不是把原文逐句翻译成口语。严格缩略：
1. 浓缩比例：原文 1000 字 → 讲解文案约 200-300 字。一集讲解文案总量控制在 400-800 字（而非 800-1500 字），分 6-10 段
2. 只讲核心剧情骨架：起因→关键转折→结果→悬念，跳过细枝末节、重复描写、过程性动作流水账
   - 错误示范："钟岳小心翼翼地走上前，仔细打量这具尸骨，发现尸骨的右臂高高扬起"（逐句复述）
   - 正确示范："钟岳在守护圈中央发现一具人首蛇身尸骨，手提一盏幽灯"（一句话浓缩）
3. 必要的背景解说（按需，不堆砌）：
   - 首次出现的核心专有名词/境界/种族一句话点明含义即可，不要长篇科普
   - 因果/伏笔只点出关键因果链和悬念，不逐条复述
4. 悬念结尾用一句话点出核心疑问即可，不用"欲知后事如何"式套话

【人物一致性（最高优先级）】
生图 Prompt 中人物外貌必须 100% 沿用给定的人物档案，严禁以下崩坏：
1. 年龄崩坏：档案 age=15岁少年，所有图必须是 15 岁少年外貌，不得画成青年/成年/老年
2. 发型崩坏：古装人物必须长发盘发/束发（玉冠/木簪/骨笄），严禁现代短发/寸头/分头/任何现代发型
3. 体型/五官崩坏：同一人物在所有图中身高、体型、五官、发色必须完全一致
4. 特殊崩坏禁令（重点）：
   - 剧情中的"观想图""识海意象""变身画面""前世形象"等非现实场景，必须明确标注为"幻象/识海画面/观想中的形象"
   - 严禁把观想意象画成主角本人的真实变貌（如钟岳观想燧皇图，不能把钟岳本人画成龙头蛇身；应画成"钟岳闭目打坐，身后浮现燧皇虚影"）
   - 幻象画面必须同时出现主角本人的真实外貌 + 虚幻意象，不得只画虚幻意象
5. 若档案缺少 appearance，从 identity 和原文合理推断一次并全程锁定

【画面风格要求】
所有生图 Prompt 必须以 'ancient Chinese mythology art style, ' 开头：
- 中国古代神话风格，古朴、蛮荒、磅礴，洪荒神话感
- 男性角色英俊帅气、五官立体、身姿挺拔；女性角色美丽动人、气质出尘
- 服饰符合境界身份：外门弟子粗麻兽皮短打，内门弟子素色道袍，长老华贵锦袍，神族古朴祭祀礼服
- 发饰古风：束发玉冠/木簪/骨笄，严禁现代发饰；头发必须长且盘束，严禁现代短发
- 场景：崇山峻岭、幽谷深壑、上古遗迹，古朴蛮荒磅礴
- 光影：神秘古拙，幽光/灵气/神光，不用现代光源

【图片粒度规则】
- 目标节奏：每幅图展示约10秒
- 拆图原则：一两句描述同一场景/动作为一幅图；切换新场景/动作/视角时换下一幅
- 数量要求（硬性下限）：图片张数 = max(语音总时长/10秒, tts_meta段数)
- 一个TTS段内多句场景描述必须拆成多张图
- 每个 image_prompt 必须标注 narration_segment（对应 tts_meta 第几段，1-based 整数）和 start_ratio（0.0-1.0，该图在段内出现进度位置）

【输出格式 严格 JSON】
{{
  "script": "浓缩讲解文案（400-800字，6-10段，用『』分隔段落，第三人称客观叙述+必要解说，彻底去说书化，不逐句复述原文）",
  "image_prompts": [
    {{
      "index": 1,
      "narration_segment": "对应 tts_meta 的第几段（整数）",
      "start_ratio": "0.0-1.0 浮点（该图在所属语音段中的出现进度位置）",
      "characters": ["本图涉及角色 char_id 列表"],
      "prompt": "以 'ancient Chinese mythology art style, ' 开头的生图Prompt，必须包含：(1) 风格前缀 (2) 人物固定外貌——age/identity/appearance发色发型/眼/眉/身材/五官（100%沿用档案不得改变）/attire穿着 (3) 精确动作与表情 (4) 场景环境 (5) 关键视觉特征精确写出 (6) 神话氛围光影 (7) 镜头构图。观想/幻象场景必须同时画主角真实外貌+虚意象，标注为 illusion/vision。同一人物所有图外貌完全一致",
      "mood": "本画面情绪关键词"
    }}
  ],
  "tts_meta": [
    {{
      "index": 1,
      "text": "本段朗读文本",
      "voice": "角色名（钟岳/薪火/narrator），旁白用 narrator",
      "emotion": "neutral | excited | sad | tense | calm | angry | surprised",
      "speed": "0.8-1.2 浮点",
      "pause_after": "段后停顿 0.0-2.0 秒"
    }}
  ]
}}

严格要求：
1. 文案必须彻底去说书化（见【讲解文案风格要求】），且是原文的浓缩提炼而非逐句复述（400-800字）
2. 人物一致性：生图 Prompt 人物外貌 100% 沿用档案，年龄/发型/体型/五官不得改变；观想意象不得画成人物本人变貌
3. 生图 Prompt 必须以 'ancient Chinese mythology art style, ' 开头
4. 关键视觉特征精确描述，不用模糊词
5. 图片数 ≥ TTS段数，每10秒至少一张图
6. 每个 image_prompt 必须有 narration_segment 和 start_ratio
7. 不得新增剧情、不得篡改事实"""


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
