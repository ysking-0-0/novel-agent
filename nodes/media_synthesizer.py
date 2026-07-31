"""
novel_pipeline.nodes.media_synthesizer
多媒体合成节点 —— persistence 后自动触发。
对应设计文档扩展阶段。

核心职责（三步）：
1. 生图：遍历 image_prompts，带「人物定妆照」参考图保证人物一致性
2. TTS：遍历 tts_meta，按 voice 音色映射调 MiniMax TTS
3. 视频合成：FFmpeg 按 tts_meta 时序把图片+音频对齐拼成讲解视频

人物一致性策略：
- 每个主要角色首次出现时，用其外貌描述生成一张「定妆照」并落盘
- 后续所有含该角色的画面，生图时带 reference_image=<定妆照 base64>
- 定妆照缓存于 memory/character_portraits/<char_id>.jpg，跨集复用
"""
import os
import json
import base64
import time
import shutil
import subprocess
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import requests

from config import get_config


# ────────────── HTTP 调用（线程内复用 session） ──────────────
_session_local = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_session_local, "session", None)
    if s is None:
        s = requests.Session()
        _session_local.session = s
    return s


def _headers() -> dict:
    return {
        "Authorization": "Bearer " + get_config().model.api_key,
        "Content-Type": "application/json",
    }


def _base_url() -> str:
    return get_config().model.base_url.rstrip("/")


# ────────────── 生图 ──────────────
def _image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return "data:image/jpeg;base64," + b


def _call_image_api(prompt: str, reference_image_b64: Optional[str] = None) -> Optional[str]:
    """调用 MiniMax 生图接口，返回图片 URL 或 None。"""
    cfg = get_config()
    payload = {
        "model": cfg.media.image_model,
        "prompt": prompt,
        "aspect_ratio": cfg.media.image_aspect_ratio,
    }
    if reference_image_b64:
        payload["reference_image"] = reference_image_b64

    for attempt in range(3):
        try:
            r = _get_session().post(
                _base_url() + "/image_generation",
                headers=_headers(), json=payload, timeout=90,
            )
            d = r.json()
            if isinstance(d, dict):
                urls = (d.get("data") or {}).get("image_urls") or []
                if urls:
                    return urls[0]
                br = d.get("base_resp") or {}
                msg = (br.get("status_msg") or "")[:80]
                status_code = br.get("status_code")
                # 用量受限 / 内容审核敏感：直接放弃不重试（重试只会重复失败并占满超时）
                if "用量" in msg or "limit" in msg.lower() or status_code in (1026, 1027):
                    print("    [生图] 跳过(status=%s): %s" % (status_code, msg or "sensitive/quota"))
                    return None
        except Exception as e:
            print("    [生图] attempt %d 异常: %s" % (attempt, e))
        time.sleep(2 * (attempt + 1))
    return None


def _download(url: str, dest: str) -> bool:
    try:
        r = _get_session().get(url, timeout=60)
        r.raise_for_status()
        if r.content:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
    except Exception as e:
        print("    [下载] %s 失败: %s" % (dest, e))
    return False


def _ensure_portrait_dir() -> str:
    d = os.path.join(get_config().storage.memory_dir, "character_portraits")
    os.makedirs(d, exist_ok=True)
    return d


def get_or_create_portrait(char_id: str, appearance_desc: str, entity_type: str = "human") -> Optional[str]:
    """获取或生成人物定妆照路径。已存在则复用，否则生成并落盘。

    entity_type: "human" 走古风人物约束；"spirit" 走火焰灵体特殊画法（如薪火）。
    """
    d = _ensure_portrait_dir()
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in char_id)
    path = os.path.join(d, safe_id + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    if entity_type == "spirit":
        # 灵体/火焰化身（如薪火）：本体是火焰，特殊画法，不走古风服饰约束
        portrait_prompt = (
            "ancient Chinese mythology art style, spirit entity reference portrait, front view, "
            "neutral dark background. "
            + (appearance_desc or char_id)
            + ". "
            "FIRE SPIRIT FORM (MANDATORY): the character is a living flame spirit, body made of living fire, "
            "hair made of flames, skin has ethereal glow, surrounded by fire aura, semi-transparent flame body, "
            "dwelling in or emerging from a bronze lamp wick. "
            "The entity MUST look like a flame-made small child figure with visible fire elements - "
            "glowing flame hair, fire aura around body, semi-transparent or fiery edges, NOT a normal flesh child. "
            "NO ancient Chinese clothing, NO hairpin, NO normal skin - the body is made of fire. "
            "Mystical atmosphere, high detail, consistent character design reference sheet for image-to-image continuity."
        )
    else:
        portrait_prompt = (
            "ancient Chinese mythology art style, character reference portrait, front view, "
            "neutral background, full body visible. "
            + (appearance_desc or char_id)
            + ". Handsome and heroic if male, beautiful and ethereal if female. "
            "Ancient Chinese attire matching cultivation realm. "
            "CLOTHING (ABSOLUTE MANDATORY): 100% ancient Chinese garments only - cross-collar right-lapel robes, wide sleeves, sash-waisted, multi-layered hanfu-style robes, fabric of hemp/silk/brocade/animal hide. ABSOLUTELY NEVER modern T-shirt, shirt, trousers, skirt, zipper, buttons, modern prints. "
            "FOOTWEAR (MANDATORY): cloth shoes, straw sandals, leather boots, cloud-toe shoes only. ABSOLUTELY NEVER sneakers, leather shoes, modern shoes. "
            "ACCESSORIES: jade pendant, bone carving, bronze ring, spirit stone pendant only. ABSOLUTELY NEVER watch, glasses, modern jewelry. "
            "HAIR (ABSOLUTE MANDATORY): long black hair in ancient Chinese style, hair must be LONG and TIED UP into a visible topknot bun or coiled bun secured with jade hairpin or wooden hairpin or bone hairpin. The word 'bun' or 'topknot' MUST appear in the hairstyle description. ABSOLUTELY NEVER modern short hair, NEVER modern hairstyle, NEVER buzz cut, NEVER side part, NEVER any modern haircut, NEVER ponytail with elastic band. "
            "AGE LOCK: the character's apparent age must strictly match the age specified above, NEVER depict as older or younger. "
            "Mystical atmosphere, high detail, consistent character design reference sheet for image-to-image continuity."
        )
    print("    [定妆照] 首次生成 %s (type=%s)" % (char_id, entity_type))
    url = _call_image_api(portrait_prompt, reference_image_b64=None)
    if not url:
        return None
    if not _download(url, path):
        return None
    return path


def _collect_characters(episode: Dict) -> Dict[str, Dict]:
    """从 episode 中收集角色 ID → {desc, entity_type}（用于定妆照）。

    优先用记忆库人物档案的固定字段（age/identity/appearance/attire/personality）
    组合成完整外貌描述；档案缺失时回退到 scene 内的 appearance/state_change。
    entity_type 从档案或 scene 内推断；灵体角色（如薪火）走特殊画法。
    """
    chars: Dict[str, Dict] = {}
    # 先从 episode scenes 收集所有角色 ID（保持出场顺序）
    for sc in episode.get("scenes", []):
        for c in sc.get("characters", []):
            if isinstance(c, dict):
                cid = c.get("char_id") or c.get("name")
                if cid and cid not in chars:
                    desc = c.get("appearance") or c.get("state_change") or cid
                    et = c.get("entity_type") or ("spirit" if "灵体" in (c.get("identity","") or "") else "human")
                    chars[cid] = {"desc": desc, "entity_type": et}
    # 用记忆库人物档案的固定字段组合成完整外貌描述
    try:
        from agents import get_memory_agent
        mem = get_memory_agent()
        for cid in list(chars.keys()):
            prof = mem.get_character(cid)
            if prof:
                # 组合固定字段 → 完整外貌描述
                parts = []
                for k in ("age", "identity", "appearance", "attire", "personality"):
                    v = prof.get(k, "")
                    if v:
                        parts.append(f"{k}: {v}")
                if parts:
                    chars[cid]["desc"] = "; ".join(parts)
                elif prof.get("appearance"):
                    chars[cid]["desc"] = prof["appearance"]
                # entity_type 优先用档案值
                et_prof = prof.get("entity_type")
                if et_prof:
                    chars[cid]["entity_type"] = et_prof
                elif "灵体" in (prof.get("identity","") or "") or "薪火" in cid:
                    chars[cid]["entity_type"] = "spirit"
    except Exception:
        pass
    return chars


def generate_images(episode: Dict, prompts: List[str], ep_dir: str) -> List[Optional[str]]:
    """并发生图，返回本地图片路径列表（与 prompts 等长，失败为 None）。

    人物一致性：每张图根据 prompt 文本匹配涉及的角色，用对应角色定妆照作参考。
    """
    cfg = get_config()
    img_dir = os.path.join(ep_dir, "images")
    # 覆盖生成时清理旧图片，避免残留旧文件混入新批次（数量不一致导致索引错位）
    if os.path.isdir(img_dir):
        for old in os.listdir(img_dir):
            if old.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                try:
                    os.remove(os.path.join(img_dir, old))
                except Exception:
                    pass
    os.makedirs(img_dir, exist_ok=True)

    # 收集角色定妆照
    char_portraits: Dict[str, str] = {}
    for cid, info in _collect_characters(episode).items():
        p = get_or_create_portrait(cid, info.get("desc", cid), info.get("entity_type", "human"))
        if p:
            char_portraits[cid] = p

    # 预计算每个角色定妆照的 base64（避免每张图重复读取）
    portrait_b64: Dict[str, str] = {}
    for cid, path in char_portraits.items():
        b64 = _image_to_base64(path)
        if b64:
            portrait_b64[cid] = b64

    # 主角定妆照作 fallback 参考
    primary_ref_b64: Optional[str] = next(iter(portrait_b64.values()), None) if portrait_b64 else None

    if portrait_b64:
        print("    [生图] 使用定妆照参考: %d 张角色定妆照就绪" % len(portrait_b64))
    else:
        print("    [生图] 无定妆照，纯 prompt 生成（人物可能不一致）")

    results: List[Optional[str]] = [None] * len(prompts)

    def _match_portrait(prompt_item) -> Optional[str]:
        """根据 image_prompt 的 characters 字段匹配定妆照。
        characters 是 char_id 列表（如 ['钟岳','薪火']）。
        优先用第一个匹配到的角色定妆照；无匹配→主角 fallback。
        """
        if not portrait_b64:
            return None
        # characters 字段（List[str]）
        char_list = prompt_item.get("characters", []) if isinstance(prompt_item, dict) else []
        for cid in char_list:
            if cid in portrait_b64:
                return portrait_b64[cid]
        # 回退：prompt 文本含中文角色名
        prompt_text = prompt_item.get("prompt", "") if isinstance(prompt_item, dict) else str(prompt_item)
        for cid in portrait_b64:
            if cid in prompt_text:
                return portrait_b64[cid]
        return primary_ref_b64

    def _one(idx_prompt):
        idx, prompt = idx_prompt
        # 支持 dict（含 prompt + narration_segment + characters）或纯 str
        prompt_text = prompt.get("prompt") if isinstance(prompt, dict) else str(prompt)
        ref_b64 = _match_portrait(prompt)
        url = _call_image_api(prompt_text, reference_image_b64=ref_b64)
        if not url:
            return idx, None
        dest = os.path.join(img_dir, "%03d.png" % idx)
        ok = _download(url, dest)
        return idx, dest if ok else None

    with ThreadPoolExecutor(max_workers=cfg.media.image_concurrency) as pool:
        futs = {pool.submit(_one, (i, p)): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            idx, path = fut.result()
            results[idx] = path
            if path:
                print("    [生图] %03d/%d 完成" % (idx + 1, len(prompts)))
    ok_count = sum(1 for r in results if r)
    print("    [生图] 首轮完成: %d/%d 成功" % (ok_count, len(prompts)))

    # 补图机制：对失败（None）的图重试补全，最多两轮
    for retry_round in range(2):
        missing = [i for i, r in enumerate(results) if not r]
        if not missing:
            break
        print("    [生图] 补图第%d轮：缺 %d 张，重试" % (retry_round + 1, len(missing)))
        with ThreadPoolExecutor(max_workers=cfg.media.image_concurrency) as pool:
            futs = {pool.submit(_one, (i, prompts[i])): i for i in missing}
            for fut in as_completed(futs):
                idx, path = fut.result()
                if path:
                    results[idx] = path
                    print("    [生图] %03d/%d 补图成功" % (idx + 1, len(prompts)))
        ok_count = sum(1 for r in results if r)
        print("    [生图] 补图第%d轮后: %d/%d 成功" % (retry_round + 1, ok_count, len(prompts)))

    final_count = sum(1 for r in results if r)
    print("    [生图] 最终完成: %d/%d 成功" % (final_count, len(prompts)))
    return results


def _retry_generate_images(episode: Dict, retry_items: List, ep_dir: str) -> Dict:
    """对失败的图单独重试一轮（可能是临时网络/DNS 抖动）。
    retry_items: [(idx, prompt_item), ...]
    返回 {idx: path} 成功的。
    """
    cfg = get_config()
    img_dir = os.path.join(ep_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 收集角色定妆照（复用已有）
    char_portraits: Dict[str, str] = {}
    for cid, info in _collect_characters(episode).items():
        p = get_or_create_portrait(cid, info.get("desc", cid), info.get("entity_type", "human"))
        if p:
            char_portraits[cid] = p
    portrait_b64: Dict[str, str] = {}
    for cid, path in char_portraits.items():
        b64 = _image_to_base64(path)
        if b64:
            portrait_b64[cid] = b64
    primary_ref_b64: Optional[str] = next(iter(portrait_b64.values()), None) if portrait_b64 else None

    def _match_ref(prompt_item) -> Optional[str]:
        if not portrait_b64:
            return None
        char_list = prompt_item.get("characters", []) if isinstance(prompt_item, dict) else []
        for cid in char_list:
            if cid in portrait_b64:
                return portrait_b64[cid]
        prompt_text = prompt_item.get("prompt", "") if isinstance(prompt_item, dict) else str(prompt_item)
        for cid in portrait_b64:
            if cid in prompt_text:
                return portrait_b64[cid]
        return primary_ref_b64

    results: Dict[int, str] = {}
    def _one(idx_prompt):
        idx, prompt = idx_prompt
        prompt_text = prompt.get("prompt") if isinstance(prompt, dict) else str(prompt)
        ref_b64 = _match_ref(prompt)
        url = _call_image_api(prompt_text, reference_image_b64=ref_b64)
        if not url:
            return idx, None
        dest = os.path.join(img_dir, "%03d.png" % idx)
        if _download(url, dest):
            return idx, dest
        return idx, None

    with ThreadPoolExecutor(max_workers=cfg.media.image_concurrency) as pool:
        futs = {pool.submit(_one, item): item[0] for item in retry_items}
        for fut in as_completed(futs):
            idx, path = fut.result()
            if path:
                results[idx] = path
    return results


# ────────────── TTS ──────────────
def _decode_audio(audio_str: str) -> bytes:
    """MiniMax T2A 返回的 audio 字段可能是 hex 或 base64，自适应解码。
    hex: 字符集 [0-9a-fA-F]，解码后通常以 ID3/FF FB 开头（mp3）
    base64: 字符集 [A-Za-z0-9+/=]
    """
    if not audio_str:
        return b""
    import re
    if re.match(r'^[0-9a-fA-F]+$', audio_str) and len(audio_str) % 2 == 0:
        try:
            return bytes.fromhex(audio_str)
        except ValueError:
            pass
    try:
        return base64.b64decode(audio_str)
    except Exception:
        pass
    return audio_str.encode("utf-8", errors="ignore")


def _call_tts_api(text: str, voice_id: str, emotion: str, speed: float) -> Optional[bytes]:
    """调用 MiniMax TTS，返回 mp3 bytes 或 None。"""
    cfg = get_config()
    payload = {
        "model": cfg.media.tts_model,
        "text": text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed if speed else 1.0,
            "emotion": _resolve_emotion(emotion),
        },
        "audio_setting": {
            "sample_rate": 24000,
            "format": "mp3",
        },
    }
    switched = False
    for attempt in range(4):
        try:
            r = _get_session().post(
                _base_url() + "/t2a_v2",
                headers=_headers(), json=payload, timeout=60,
            )
            d = r.json()
            if isinstance(d, dict):
                audio_raw = (d.get("data") or {}).get("audio")
                if audio_raw:
                    return _decode_audio(audio_raw)
                br = d.get("base_resp") or {}
                msg = (br.get("status_msg") or "")[:80]
                status_code = br.get("status_code")
                # 用量受限 / 内容审核敏感：直接放弃不重试
                if "用量" in msg or "limit" in msg.lower() or status_code in (1026, 1027):
                    print("    [TTS] 跳过(status=%s): %s" % (status_code, msg or "sensitive/quota"))
                    return None
                # voice_id "not exist" 在并发时多为限流误报，退避重试而非换音色
                if "voice" in msg.lower() or "not exist" in msg.lower():
                    if attempt < 2 and not switched:
                        # 退避重试（可能是并发限流）
                        time.sleep(3 * (attempt + 1))
                        continue
                    elif not switched:
                        # 第三次仍失败，换默认音色最后试一次
                        print("    [TTS] voice_id %s 重试失败，换默认音色" % voice_id)
                        if voice_id != cfg.media.default_voice_id:
                            payload["voice_setting"]["voice_id"] = cfg.media.default_voice_id
                            switched = True
                            continue
        except Exception as e:
            print("    [TTS] attempt %d 异常: %s" % (attempt, e))
        time.sleep(2 * (attempt + 1))
    return None


def _resolve_voice_id(voice_field: str) -> str:
    """把 tts_meta.voice（角色名/旁白标识）映射到 MiniMax voice_id。
    精确匹配 voice_mapping → 模糊匹配（含关键词）→ 默认音色。
    """
    cfg = get_config()
    if not voice_field:
        return cfg.media.default_voice_id
    vf = voice_field.strip()
    # 精确匹配
    if vf in cfg.media.voice_mapping:
        return cfg.media.voice_mapping[vf]
    # 模糊匹配（voice_field 含 mapping 的某个 key）
    for key, vid in cfg.media.voice_mapping.items():
        if key in vf or vf in key:
            return vid
    # 兼容旧格式 character_male / character_female
    if "female" in vf.lower():
        return cfg.media.voice_mapping.get("narrator_female", "female-shaonv")
    # 未映射的角色名 → 默认音色
    return cfg.media.default_voice_id


# MiniMax T2A 支持的 emotion 白名单 + pipeline 产出值到白名单的映射
_TTS_EMOTION_WHITELIST = {"neutral", "happy", "sad", "angry", "disgusted", "surprised", "calm"}
_TTS_EMOTION_MAP = {
    "tense": "angry",      # 紧张 → 愤怒
    "excited": "happy",    # 兴奋 → 开心
    "afraid": "surprised", # 害怕 → 惊讶
    "fear": "surprised",
    "fearful": "surprised",
    "serious": "neutral",
    "narration": "neutral",
    "friendly": "happy",
    "relaxed": "calm",
    "intrigued": "neutral",  # 好奇 → 中性
    "horrified": "surprised",# 惊恐 → 惊讶
    "shocked": "surprised",  # 震惊 → 惊讶
    "curious": "neutral",    # 好奇 → 中性
    "nervous": "angry",      # 紧张 → 愤怒
    "anxious": "angry",      # 焦虑 → 愤怒
    "desperate": "sad",      # 绝望 → 悲伤
    "determined": "angry",   # 坚定 → 愤怒
    "mysterious": "neutral", # 神秘 → 中性
    "joyful": "happy",       # 喜悦 → 开心
    "angry": "angry",
    "sad": "sad",
    "calm": "calm",
    "happy": "happy",
    "surprised": "surprised",
}


def _resolve_emotion(emotion_field: str) -> str:
    """把 tts_meta.emotion 映射到 MiniMax 支持的 emotion 值。"""
    if not emotion_field:
        return "neutral"
    e = emotion_field.strip().lower()
    if e in _TTS_EMOTION_WHITELIST:
        return e
    return _TTS_EMOTION_MAP.get(e, "neutral")


def generate_tts(tts_meta: List[Dict], ep_dir: str) -> List[Optional[str]]:
    """并发 TTS，返回本地 mp3 路径列表。"""
    cfg = get_config()
    audio_dir = os.path.join(ep_dir, "audio")
    # 覆盖生成时清理旧音频，避免残留
    if os.path.isdir(audio_dir):
        for old in os.listdir(audio_dir):
            if old.lower().endswith((".mp3", ".wav", ".m4a")):
                try:
                    os.remove(os.path.join(audio_dir, old))
                except Exception:
                    pass
    os.makedirs(audio_dir, exist_ok=True)

    results: List[Optional[str]] = [None] * len(tts_meta)

    def _one(idx_meta):
        idx, m = idx_meta
        text = m.get("text", "")
        if not text.strip():
            return idx, None
        voice_id = _resolve_voice_id(m.get("voice", ""))
        audio = _call_tts_api(
            text, voice_id,
            m.get("emotion", "neutral"),
            float(m.get("speed", 1.08)),
        )
        if not audio:
            return idx, None
        dest = os.path.join(audio_dir, "%03d.mp3" % idx)
        with open(dest, "wb") as f:
            f.write(audio)
        return idx, dest

    with ThreadPoolExecutor(max_workers=cfg.media.tts_concurrency) as pool:
        futs = {pool.submit(_one, (i, m)): i for i, m in enumerate(tts_meta)}
        for fut in as_completed(futs):
            idx, path = fut.result()
            results[idx] = path
            if path:
                print("    [TTS] %03d/%d 完成" % (idx + 1, len(tts_meta)))
    ok_count = sum(1 for r in results if r)
    print("    [TTS] 完成: %d/%d 成功" % (ok_count, len(tts_meta)))
    return results


# ────────────── 视频合成（FFmpeg） ──────────────
def _ffmpeg_path() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _audio_duration(path: str) -> float:
    """用 ffmpeg -i 取音频时长（秒）。失败回退用文件大小估算。"""
    exe = _ffmpeg_path()
    try:
        r = subprocess.run(
            [exe, "-i", path], capture_output=True, text=True, timeout=15,
        )
        # ffmpeg 把媒体信息输出到 stderr
        for line in (r.stderr or "").splitlines():
            if "Duration:" in line:
                # 格式: Duration: 00:00:05.23, ...
                seg = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = seg.split(":")
                return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        pass
    # 回退：mp3 24kHz 单声道约 ~3KB/s，粗估
    try:
        sz = os.path.getsize(path)
        return max(1.0, sz / 3000.0)
    except Exception:
        return 5.0


def _detect_cjk_font() -> Optional[str]:
    """自动探测系统中的 CJK 字体文件路径（WSL/Windows/Linux）。"""
    candidates = [
        # WSL → Windows 字体
        "/mnt/c/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "/mnt/c/Windows/Fonts/simhei.ttf",     # 黑体
        "/mnt/c/Windows/Fonts/simsun.ttc",     # 宋体
        # Linux 常见 CJK 字体
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
        # 项目自带
        "./assets/fonts/msyh.ttc",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _format_srt_time(seconds: float) -> str:
    """秒数 → SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(tts_meta: List[Dict], audio_paths: List[Optional[str]],
                 ep_dir: str, resolution: str = "1920x1080",
                 sub_input_offset: int = 1) -> Optional[str]:
    """用 PIL 生成每段字幕的透明 PNG，返回 overlay 滤镜链字符串。

    libass (ffmpeg subtitles 滤镜) 在此环境下无法渲染 CJK 字符
    （HarfBuzz shaping 缺陷），改为 PIL 渲染字幕图片 + ffmpeg overlay 叠加。
    返回 overlay 滤镜链，由 compose_video 拼入 filter_complex。

    sub_input_offset: 字幕 PNG 在 ffmpeg 输入流中的起始序号。
        无 BGM 时输入布局 0=concat视频, 1..N=字幕PNG → offset=1；
        有 BGM 时输入布局 0=concat视频, 1=bgm, 2..N=字幕PNG → offset=2。
    """
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    import re

    W, H = resolution.split("x")
    W, H = int(W), int(H)
    font_size = 48
    margin_bottom = 40
    max_chars = 24  # 每行最大字符数

    # 字体
    font_file = _detect_cjk_font() or os.path.join("assets", "fonts", "simhei.ttf")
    if not os.path.exists(font_file):
        font_file = None
    font = None
    if font_file and os.path.exists(font_file):
        try:
            font = ImageFont.truetype(font_file, font_size)
        except Exception:
            font = None
    if font is None:
        font = ImageFont.load_default()

    sub_dir = os.path.join(ep_dir, "_subs")
    os.makedirs(sub_dir, exist_ok=True)

    # 句级拆分：按 。！？；；拆句（保留分隔符），让字幕逐句跟随音频
    # —— 无词级时间戳，按字符数比例分配各句时长（近似但足够）。
    t_cursor = 0.0
    overlays = []  # [(start, end, png_path, y_offset)]
    global_idx = 0  # 全局 PNG 序号
    for i, m in enumerate(tts_meta):
        text = m.get("text", "").strip()
        if not text:
            continue
        if i < len(audio_paths) and audio_paths[i] and os.path.exists(audio_paths[i]):
            dur = _audio_duration(audio_paths[i])
        else:
            dur = 3.0
        seg_start = t_cursor
        # 拆句：保留分隔符
        sentences = re.split(r"(?<=[。！？；])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        # 极长无标点段兜底：按 max_chars 硬切
        if len(sentences) == 1 and len(sentences[0]) > max_chars * 2:
            sentences = textwrap.wrap(sentences[0], width=max_chars)
        total_chars = sum(len(s) for s in sentences) or 1
        s_cursor = seg_start
        for si, sent in enumerate(sentences):
            # 末句吃满段尾，避免浮点误差导致缺口
            if si == len(sentences) - 1:
                s_end = seg_start + dur
            else:
                s_dur = dur * (len(sent) / total_chars)
                s_end = s_cursor + s_dur
            s_start = s_cursor
            # 折行渲染：如果换行后末行只剩几个字，合并回上一行避免孤行
            lines = textwrap.wrap(sent, width=max_chars) if len(sent) > max_chars else [sent]
            if len(lines) >= 2 and len(lines[-1]) <= 5:
                lines = lines[:-2] + [lines[-2] + lines[-1]]
            line_h = font_size + 8
            total_h = len(lines) * line_h + 20
            img = Image.new("RGBA", (W, total_h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            for li, line in enumerate(lines):
                bbox = d.textbbox((0, 0), line, font=font)
                tw = bbox[2] - bbox[0]
                x = (W - tw) // 2
                y = li * line_h + 10
                for dx, dy in [(-2,0),(2,0),(0,-2),(0,2),(-2,-2),(2,2),(-2,2),(2,-2)]:
                    d.text((x+dx, y+dy), line, fill=(0,0,0,255), font=font)
                d.text((x, y), line, fill=(255,255,255,255), font=font)
            png_path = os.path.join(sub_dir, f"sub_{global_idx:03d}.png")
            img.save(png_path)
            y_off = H - total_h - margin_bottom
            overlays.append((s_start, s_end, png_path, y_off))
            s_cursor = s_end
            global_idx += 1
        t_cursor = seg_start + dur

    # 构建 overlay 滤镜链：
    # [0:v][1:v]overlay=0:y0:enable=between(t,s,e)[v1];
    # [v1][2:v]overlay=0:y1:enable=between(t,s,e)[v2]; ...
    parts = []
    prev_label = "0:v"
    for idx, (s, e, png, y) in enumerate(overlays):
        # ffmpeg overlay 输入需要作为额外 -i 参数，这里只返回滤镜链
        # 输入序号：视频=0，每个字幕 PNG 从 sub_input_offset 开始递增
        inp = idx + sub_input_offset
        out_label = f"[vsub{idx}]" if idx < len(overlays)-1 else "[vsub]"
        parts.append(f"[{prev_label}][{inp}:v]overlay=0:{y}:enable='between(t,{s:.2f},{e:.2f})'{out_label}")
        prev_label = f"vsub{idx}"
    # 返回 (滤镜链, 字幕PNG列表)
    return ";".join(parts), [o[2] for o in overlays]


def _make_clip(image_path: Optional[str], audio_path: Optional[str],
               idx: int, tmp_dir: str, resolution: str, fps: int,
               duration: float = None, audio_seek: float = 0,
               pan_direction: str = "down") -> Optional[str]:
    """单张图片 + 音频片段合成片段 mp4，带 Ken Burns 垂直平移动效。

    动效（85%/15%设计）：图片先放大到 W x (H/0.85)≈H*1.176，
    让初始窗口(H)就展示图片85%高度，剩余15%通过滑动窗口展示。
    pan_direction: "down"=窗口从图片顶部往下滑(图片从上往下展开);
                   "up"  =窗口从图片底部往上滑(图片从下往上展开)。

    duration: 指定片段时长（用于同段音频内多图切片）。
    audio_seek: 音频起始偏移（秒），用于同段音频内多图切片。
    """
    exe = _ffmpeg_path()
    out = os.path.join(tmp_dir, "clip_%03d.mp4" % idx)

    # 图片输入
    if image_path and os.path.exists(image_path):
        img_arg = ["-loop", "1", "-framerate", str(fps), "-i", image_path]
    else:
        placeholder = os.path.join(tmp_dir, "placeholder.png")
        if not os.path.exists(placeholder):
            w, h = resolution.split("x")
            # 亮灰色占位（0x404050），与黑屏检测阈值区分，便于识别缺图位置
            subprocess.run(
                [exe, "-y", "-f", "lavfi", "-i",
                 "color=c=0x404050:s=%dx%d:d=1" % (int(w), int(h)),
                 "-frames:v", "1", placeholder],
                capture_output=True, timeout=30,
            )
        img_arg = ["-loop", "1", "-framerate", str(fps), "-i", placeholder]

    # 音频输入与时长
    if audio_path and os.path.exists(audio_path):
        # 支持 seek 偏移（同段音频多图切片）
        seek_arg = ["-ss", "%.2f" % audio_seek] if audio_seek > 0 else []
        aud_arg = seek_arg + ["-i", audio_path]
        if duration is not None:
            dur = duration
            extra = ["-t", "%.2f" % dur]
        else:
            dur = _audio_duration(audio_path)
            extra = ["-shortest"]
    else:
        aud_arg = ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=24000"]
        dur = duration if duration is not None else 3.0
        extra = ["-t", "%.2f" % dur]

    w, h = resolution.split("x")
    W = int(w)
    H = int(h)

    # ── Ken Burns 垂直平移动效（子像素平滑·crop整数舍入修复） ──
    # 设计：初始展示图片85%，剩余15%通过滑动窗口展示
    # 问题：crop 的 y 坐标被取整，小范围滑动(127px/3s≈1.4px/帧)时
    #       连续多帧 y 值相同导致画面停滞，跳到下一整数时又突跳=卡顿
    # 修复：先 scale 放大4倍，在放大域 crop 整数移动 = 原图0.25px移动，
    #       再 scale 回原始尺寸，等效4倍子像素精度，滑动平滑无卡顿
    ZOOM_RATIO = 1.176          # 1/0.85，让窗口初始占图片85%
    SUBPIXEL = 4                # 子像素放大倍数
    img_h_up = int(H * ZOOM_RATIO * SUBPIXEL)   # 放大域图片高
    win_h_up = H * SUBPIXEL                      # 放大域窗口高
    pan_total_up = img_h_up - win_h_up           # 放大域滑动总位移
    safe_dur = max(0.1, dur)
    if pan_direction == "up":
        # 窗口从底部滑到顶部（图片从下往上展开）
        # 用 clamp(.,0,pan_total) 防止 t 超出 safe_dur 时 y 越界导致纯色填充
        y_expr = "clip(%d*(1-t/%.3f),0,%d)" % (pan_total_up, safe_dur, pan_total_up)
    else:
        # 窗口从顶部滑到底部（图片从上往下展开）
        y_expr = "clip(%d*(t/%.3f),0,%d)" % (pan_total_up, safe_dur, pan_total_up)

    vf = (
        "scale=%d:%d:flags=lanczos,"           # 放大到 W x (H*1.176*4)
        "crop=%d:%d:0:'%s',"                   # 在放大域按 t 移动(整数，clamp防越界)
        "scale=%d:%d:flags=lanczos,"           # 缩回原始 W x H
        "setsar=1,"                            # 重置SAR(必须在最后一次scale后,否则残留放大比值)
        "format=yuv420p"
    ) % (W, img_h_up, W, win_h_up, y_expr, W, H)

    cmd = [exe, "-y"] + img_arg + aud_arg + [
        "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-bf", "0",                    # 禁用B帧，避免预测帧复用导致画面卡顿
        "-r", str(fps),                # 输出帧率
        "-c:a", "aac", "-b:a", "128k", "-t", "%.2f" % dur,
    ] + extra + [out]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            return out
    except Exception as e:
        print("    [片段] %d 合成失败: %s" % (idx, e))
    return None


def compose_video(image_paths: List[Optional[str]], audio_paths: List[Optional[str]],
                  tts_meta: List[Dict], ep_dir: str, eid: str,
                  image_prompts: List = None) -> Optional[str]:
    """按 tts_meta 时序拼接片段 → 完整 episode.mp4。

    支持细粒度图片：image_prompts 可能多于 tts_meta，
    每个 image_prompt 的 narration_segment 标注对应第几段语音（1-based）。
    同一语音段内的多张图按 start_ratio 精确定位出现时间（非均分）。
    每张图带 Ken Burns 垂直平移动效（方向交替）。
    """
    if not image_paths and not audio_paths:
        return None

    cfg = get_config()
    tmp_dir = os.path.join(ep_dir, "_tmp_clips")
    os.makedirs(tmp_dir, exist_ok=True)

    # 每张图收集 (path, narration_segment, start_ratio)
    seg_images: Dict[int, List[Dict]] = {}  # segment(1-based) -> [{path, start_ratio}]
    for i, img_path in enumerate(image_paths):
        narr_seg = None
        sr = 0.0
        if image_prompts and i < len(image_prompts):
            p = image_prompts[i]
            if isinstance(p, dict):
                narr_seg = p.get("narration_segment")
                try:
                    sr = float(p.get("start_ratio", 0.0) or 0.0)
                    sr = max(0.0, min(1.0, sr))
                except (ValueError, TypeError):
                    sr = 0.0
        # 没有 narration_segment 的图，按 index 顺序分配（回退：分到第 i+1 段或最后一段）
        if narr_seg is None:
            narr_seg = min(i + 1, len(tts_meta)) if tts_meta else 1
        seg_images.setdefault(narr_seg, []).append({"path": img_path, "start_ratio": sr})

    clips: List[str] = []
    clip_idx = 0
    # 遍历每段语音
    for seg_idx in range(len(tts_meta)):
        seg_no = seg_idx + 1  # 1-based
        aud = audio_paths[seg_idx] if seg_idx < len(audio_paths) else None
        # 该段的图片列表（按 start_ratio 排序，确保出现顺序正确）
        imgs = sorted(seg_images.get(seg_no, []), key=lambda x: x["start_ratio"])
        if not imgs:
            imgs = [{"path": None, "start_ratio": 0.0}]
        # 音频时长
        if aud and os.path.exists(aud):
            seg_dur = _audio_duration(aud)
        else:
            seg_dur = 3.0
        # 用 start_ratio 精确计算每张图的起点与时长
        # 该图起点 = start_ratio * seg_dur；时长 = 下张图起点 - 本张图起点（末张=段尾）
        starts = [img["start_ratio"] * seg_dur for img in imgs]
        # 保证最后一张覆盖到段尾
        for i in range(len(starts) - 1, -1, -1):
            if i == len(starts) - 1:
                starts[i] = starts[i] if starts[i] < seg_dur else max(0, seg_dur - 0.5)
            else:
                # 若相邻图起点相同（start_ratio 重合），回退到均分
                if starts[i + 1] <= starts[i]:
                    starts[i] = starts[i + 1] * (i / (len(starts)))
        # 每张图时长 = 下张起点 - 本张起点；末张 = seg_dur - 本张起点
        for i, img in enumerate(imgs):
            t_start = starts[i]
            t_end = starts[i + 1] if i + 1 < len(imgs) else seg_dur
            per_img = max(0.3, t_end - t_start)  # 最小0.3s
            # 平移方向交替：偶数 idx 从上往下，奇数从下往上
            pan_dir = "down" if clip_idx % 2 == 0 else "up"
            clip = _make_clip(img["path"], aud, clip_idx, tmp_dir,
                              cfg.media.video_resolution, cfg.media.video_fps,
                              duration=per_img, audio_seek=t_start,
                              pan_direction=pan_dir)
            if clip:
                clips.append(clip)
            clip_idx += 1

    if not clips:
        print("    [视频] 无可用片段")
        return None

    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for c in clips:
            # concat demuxer 按相对 concat.txt 解析路径，用绝对路径避免重复拼接
            f.write("file '%s'\n" % os.path.abspath(c))

    out = os.path.join(ep_dir, "%s.mp4" % eid)
    exe = _ffmpeg_path()
    # 第一步：concat 拼接 → 临时文件
    # 注意：必须重编码（不能用 -c copy），因为各片段的 zoompan 帧时间戳独立计算，
    # stream copy 会保留不连续的时间戳，导致抽帧跳到错误关键帧、动效丢失。
    concat_out = os.path.join(tmp_dir, "concat.mp4")
    concat_ok = False
    try:
        subprocess.run(
            [exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
             "-c:a", "aac",
             "-r", str(cfg.media.video_fps), concat_out],
            capture_output=True, timeout=600,
        )
        if os.path.exists(concat_out) and os.path.getsize(concat_out) > 1000:
            concat_ok = True
    except Exception as e:
        print("    [视频] concat 重编码失败: %s" % e)

    if not concat_ok:
        print("    [视频] concat 拼接失败")
        return None

    # 第二步：混入 BGM（循环播放 + volume）→ 最终输出
    # BGM 用 stream_loop=-1 无限循环，duration=first 对齐视频时长，视频在BGM就继续放
    # 注意：视频必须重编码（-c:v libx264），不能用 copy，否则 zoompan 动效帧时间戳会错乱
    cfg = get_config()
    bgm = cfg.media.bgm_path
    tts_gain = getattr(cfg.media, "tts_volume", 1.0)
    # 字幕：用 PIL 渲染 PNG + ffmpeg overlay（libass 无法渲染 CJK）
    enable_subs = getattr(cfg.media, "enable_subtitles", False)
    sub_filter = ""
    sub_pngs = []
    if enable_subs and tts_meta:
        # 输入布局：0=concat视频, 1=bgm, 2..N=字幕PNG → offset=2
        result = generate_srt(tts_meta, audio_paths, ep_dir, cfg.media.video_resolution,
                              sub_input_offset=2 if (bgm and os.path.exists(bgm)) else 1)
        if result and result[0]:
            sub_filter, sub_pngs = result
            print("    [字幕] 生成 %d 张 PNG" % len(sub_pngs))

    amix_audio = (
        "[0:a]volume=%.2f[tts];[1:a]volume=%.2f[bg];[tts][bg]amix=inputs=2:duration=first:dropout_transition=0[a]"
        % (tts_gain, cfg.media.bgm_volume)
    )
    if bgm and os.path.exists(bgm):
        # 有 BGM：filter_complex 同时做音频混音 + 视频字幕overlay
        # 输入：0=concat_out(视频), 1=bgm, 2..N=字幕PNG
        sub_inputs = [item for png in sub_pngs for item in ("-i", png)]
        if sub_filter:
            fc = "%s;%s" % (amix_audio, sub_filter)
            vmap = "[vsub]"
        else:
            fc = amix_audio
            vmap = "0:v"
        try:
            subprocess.run(
                [exe, "-y", "-i", concat_out,
                 "-stream_loop", "-1", "-i", bgm] + sub_inputs +
                ["-filter_complex", fc,
                 "-map", vmap, "-map", "[a]",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-c:a", "aac",
                 "-r", str(cfg.media.video_fps), "-shortest", out],
                capture_output=True, timeout=600,
            )
            if not (os.path.exists(out) and os.path.getsize(out) > 1000):
                print("    [视频] BGM+字幕 混入首次失败，重试")
                subprocess.run(
                    [exe, "-y", "-i", concat_out,
                     "-stream_loop", "-1", "-i", bgm] + sub_inputs +
                    ["-filter_complex", fc,
                     "-map", vmap, "-map", "[a]",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0", "-c:a", "aac",
                     "-r", str(cfg.media.video_fps), "-shortest", out],
                    capture_output=True, timeout=600,
                )
        except Exception as e:
            print("    [视频] BGM 混入失败: %s，用无BGM版本" % e)
            import shutil as _sh
            _sh.copy(concat_out, out)
    else:
        # 无 BGM：仅叠加字幕overlay
        sub_inputs = [item for png in sub_pngs for item in ("-i", png)]
        if sub_filter:
            try:
                subprocess.run(
                    [exe, "-y", "-i", concat_out] + sub_inputs +
                    ["-filter_complex", sub_filter,
                     "-map", "[vsub]",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-bf", "0",
                     "-c:a", "aac", "-r", str(cfg.media.video_fps), out],
                    capture_output=True, timeout=600,
                )
            except Exception as e:
                print("    [视频] 字幕叠加失败: %s，用无字幕版本" % e)
                import shutil as _sh
                _sh.copy(concat_out, out)
        else:
            import shutil as _sh
            _sh.copy(concat_out, out)

    # 清理临时片段
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    return None


# ────────────── LangGraph 节点入口 ──────────────
def media_synthesizer_node(state: Dict) -> Dict:
    """多媒体合成节点入口。
    从已归档的 output/ep_xxx/ 读取 episode_info/image_prompts/tts_meta，
    不依赖 state 中的临时字段（persistence 已清理）。
    """
    cfg = get_config()
    if not cfg.media.enable_synthesis:
        return {}

    eid = "ep_%03d" % state.get("completed_episode_count", 0)
    ep_dir = os.path.join(cfg.storage.output_dir, eid)
    if not os.path.isdir(ep_dir):
        return {}

    # 覆盖生成时删除旧视频，确保新合成不被旧文件干扰
    old_mp4 = os.path.join(ep_dir, f"{eid}.mp4")
    if os.path.exists(old_mp4):
        try:
            os.remove(old_mp4)
        except Exception:
            pass

    try:
        with open(os.path.join(ep_dir, "episode_info.json"), "r", encoding="utf-8") as f:
            episode = json.load(f)
        with open(os.path.join(ep_dir, "image_prompts.json"), "r", encoding="utf-8") as f:
            prompts = json.load(f)
        with open(os.path.join(ep_dir, "tts_meta.json"), "r", encoding="utf-8") as f:
            tts_meta = json.load(f)
    except Exception as e:
        print("[合成] %s 读取归档文件失败: %s" % (eid, e))
        return {}

    if not prompts and not tts_meta:
        return {}

    print("[合成] 开始 %s：生图 %d 张，TTS %d 段（并行）" % (eid, len(prompts), len(tts_meta)))

    # 生图与 TTS 并行启动（两个独立线程池同时跑，互不阻塞）
    image_paths: List = []
    audio_paths: List = []
    with ThreadPoolExecutor(max_workers=2) as parallel:
        fut_img = parallel.submit(generate_images, episode, prompts, ep_dir) if prompts else None
        fut_aud = parallel.submit(generate_tts, tts_meta, ep_dir) if tts_meta else None
        if fut_img:
            image_paths = fut_img.result()
        if fut_aud:
            audio_paths = fut_aud.result()

    # 缺图重试补全：对失败的图单独重试一轮（可能是临时网络/DNS 抖动）
    img_dir = os.path.join(ep_dir, "images")
    missing_idx = [i for i, p in enumerate(image_paths) if not p or not (p and os.path.exists(p))]
    if missing_idx and prompts:
        print("    [补图] %d 张图缺失，单独重试" % len(missing_idx))
        retry_prompts = [(i, prompts[i]) for i in missing_idx if i < len(prompts)]
        retry_results = _retry_generate_images(episode, retry_prompts, ep_dir)
        for i, path in retry_results.items():
            if path:
                image_paths[i] = path
                print("    [补图] 第%d张重试成功" % (i + 1))

    # 仍然缺失的图，用前一张成功的图占位（避免纯黑屏）
    last_ok = None
    for i, p in enumerate(image_paths):
        if p and os.path.exists(p):
            last_ok = p
        elif last_ok:
            import shutil as _sh
            placeholder = os.path.join(img_dir, "%03d.png" % i)
            try:
                _sh.copy(last_ok, placeholder)
                image_paths[i] = placeholder
                print("    [补图] 第%d张缺失，用前一张占位" % (i + 1))
            except Exception:
                pass
    still_missing = sum(1 for p in image_paths if not p)
    if still_missing:
        print("    [补图] 仍有 %d 张图无法占位（首图就失败）" % still_missing)

    print("  [阶段3] 拼接视频（Ken Burns 平移 + start_ratio 对齐）...")
    video_path = compose_video(image_paths, audio_paths, tts_meta, ep_dir, eid,
                               image_prompts=prompts)

    if video_path:
        sz_mb = os.path.getsize(video_path) / (1024 * 1024)
        print("[合成] %s 完成: %s (%.1f MB)" % (eid, video_path, sz_mb))
        # 黑屏检测：扫描视频找黑屏帧，统计黑屏占比
        black_pct = _detect_black_frames(video_path)
        if black_pct > 5.0:
            # 黑屏超 5%：找出缺失图，重试生图后重新合成
            print("    [黑屏] %.1f%% 帧为黑/暗屏，自动重试缺图 + 重合成" % black_pct)
            still_missing_idx = [i for i, p in enumerate(image_paths)
                                 if not p or (p and not os.path.exists(p))]
            # 也找出用占位图（灰色 0x404050）的项重试
            placeholder_idxs = []
            for i, p in enumerate(image_paths):
                if p and os.path.exists(p):
                    try:
                        from PIL import Image
                        import numpy as np
                        arr = np.array(Image.open(p).convert("RGB"))
                        # 纯灰占位图：所有像素接近 (64,64,80)
                        if arr.std() < 5 and abs(arr.mean() - 70) < 20:
                            placeholder_idxs.append(i)
                    except Exception:
                        pass
            retry_idxs = list(set(still_missing_idx + placeholder_idxs))
            if retry_idxs and prompts:
                print("    [黑屏修复] 重试 %d 张图" % len(retry_idxs))
                retry_items = [(i, prompts[i]) for i in retry_idxs if i < len(prompts)]
                retry_results = _retry_generate_images(episode, retry_items, ep_dir)
                replaced = 0
                for i, path in retry_results.items():
                    if path and os.path.exists(path):
                        image_paths[i] = path
                        replaced += 1
                if replaced > 0:
                    print("    [黑屏修复] 重试补回 %d 张，重新合成视频" % replaced)
                    video_path = compose_video(image_paths, audio_paths, tts_meta, ep_dir, eid,
                                               image_prompts=prompts)
                    if video_path:
                        black_pct2 = _detect_black_frames(video_path)
                        sz_mb = os.path.getsize(video_path) / (1024 * 1024)
                        print("[合成] %s 重合成完成: %s (%.1f MB) 黑屏 %.1f%%→%.1f%%"
                              % (eid, video_path, sz_mb, black_pct, black_pct2))
        elif black_pct > 0:
            print("    [黑屏] 检测到 %.1f%% 暗帧（可接受范围）" % black_pct)
    else:
        print("[合成] %s 视频未生成（图片/音频已落盘）" % eid)

    return {"video_path": video_path}


def _detect_black_frames(video_path: str, threshold: float = 0.07) -> float:
    """用 ffmpeg blackdetect 扫描视频，返回黑屏时长占比%。

    threshold: 像素亮度阈值（0~1，默认0.07≈18/255），低于此值视为黑/暗。
    纯黑帧=0，深蓝placeholder(0x1a1a2e≈26/255≈0.10)也能检出。
    """
    exe = _ffmpeg_path()
    try:
        r = subprocess.run(
            [exe, "-i", video_path,
             "-vf", "blackdetect=d=0.1:pix_th=%.2f" % threshold,
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, timeout=120,
        )
        # 总时长
        total = 0.0
        for line in (r.stderr or "").splitlines():
            if "Duration:" in line:
                seg = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = seg.split(":")
                total = int(h) * 3600 + int(m) * 60 + float(s)
                break
        if total <= 0:
            return 0.0
        # 累计黑屏时长
        black_total = 0.0
        for line in (r.stderr or "").splitlines():
            if "black_duration" in line:
                # line: [blackdetect @ ...] black_duration:1.234 start:5.6 ...
                parts = line.split("black_duration:")
                if len(parts) > 1:
                    seg = parts[1].split()[0]
                    try:
                        black_total += float(seg)
                    except ValueError:
                        pass
        return black_total / total * 100.0
    except Exception as e:
        print("    [黑屏] 检测失败: %s" % e)
        return 0.0
