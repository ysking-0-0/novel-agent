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
                if "用量" in msg or "limit" in msg.lower():
                    print("    [生图] 用量受限，跳过: %s" % msg)
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


def get_or_create_portrait(char_id: str, appearance_desc: str) -> Optional[str]:
    """获取或生成人物定妆照路径。已存在则复用，否则生成并落盘。"""
    d = _ensure_portrait_dir()
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in char_id)
    path = os.path.join(d, safe_id + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path

    portrait_prompt = (
        "ancient Chinese mythology art style, character reference portrait, front view, "
        "neutral background, full face visible. "
        + (appearance_desc or char_id)
        + ". Handsome and heroic if male, beautiful and ethereal if female. "
        "Ancient Chinese attire matching cultivation realm, jade hairpin or topknot. "
        "Mystical atmosphere, high detail, consistent character design sheet."
    )
    print("    [定妆照] 首次生成 %s" % char_id)
    url = _call_image_api(portrait_prompt, reference_image_b64=None)
    if not url:
        return None
    if not _download(url, path):
        return None
    return path


def _collect_characters(episode: Dict) -> Dict[str, str]:
    """从 episode 中收集角色 ID → 完整外貌描述（用于定妆照）。

    优先用记忆库人物档案的固定字段（age/identity/appearance/attire/personality）
    组合成完整外貌描述；档案缺失时回退到 scene 内的 appearance/state_change。
    """
    chars: Dict[str, str] = {}
    # 先从 episode scenes 收集所有角色 ID（保持出场顺序）
    for sc in episode.get("scenes", []):
        for c in sc.get("characters", []):
            if isinstance(c, dict):
                cid = c.get("char_id") or c.get("name")
                if cid and cid not in chars:
                    # scene 内可能带 appearance（plot_parser 新输出）
                    desc = c.get("appearance") or c.get("state_change") or cid
                    chars[cid] = desc
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
                    chars[cid] = "; ".join(parts)
                elif prof.get("appearance"):
                    chars[cid] = prof["appearance"]
    except Exception:
        pass
    return chars


def generate_images(episode: Dict, prompts: List[str], ep_dir: str) -> List[Optional[str]]:
    """并发生图，返回本地图片路径列表（与 prompts 等长，失败为 None）。

    人物一致性：每张图根据 prompt 文本匹配涉及的角色，用对应角色定妆照作参考。
    """
    cfg = get_config()
    img_dir = os.path.join(ep_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 收集角色定妆照
    char_portraits: Dict[str, str] = {}
    for cid, desc in _collect_characters(episode).items():
        p = get_or_create_portrait(cid, desc)
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
    print("    [生图] 完成: %d/%d 成功" % (ok_count, len(prompts)))
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
                if "用量" in msg or "limit" in msg.lower():
                    print("    [TTS] 用量受限: %s" % msg)
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
            float(m.get("speed", 1.0)),
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


def _make_clip(image_path: Optional[str], audio_path: Optional[str],
               idx: int, tmp_dir: str, resolution: str, fps: int,
               duration: float = None, audio_seek: float = 0) -> Optional[str]:
    """单张图片 + 音频片段合成片段 mp4。

    duration: 指定片段时长（用于同段音频内多图均分）。
    audio_seek: 音频起始偏移（秒），用于同段音频内多图切片。
    """
    exe = _ffmpeg_path()
    out = os.path.join(tmp_dir, "clip_%03d.mp4" % idx)

    # 图片输入
    if image_path and os.path.exists(image_path):
        img_arg = ["-loop", "1", "-i", image_path]
    else:
        placeholder = os.path.join(tmp_dir, "placeholder.png")
        if not os.path.exists(placeholder):
            w, h = resolution.split("x")
            subprocess.run(
                [exe, "-y", "-f", "lavfi", "-i",
                 "color=c=0x1a1a2e:s=%dx%d:d=1" % (int(w), int(h)),
                 "-frames:v", "1", placeholder],
                capture_output=True, timeout=30,
            )
        img_arg = ["-loop", "1", "-i", placeholder]

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
    vf = "scale=%s:%s:force_original_aspect_ratio=decrease,pad=%s:%s:(ow-iw)/2:(oh-ih)/2:black,fps=%d" % (w, h, w, h, fps)
    cmd = [exe, "-y"] + img_arg + aud_arg + [
        "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
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
    同一语音段内的多张图按音频时长均分展示。
    """
    if not image_paths and not audio_paths:
        return None

    cfg = get_config()
    tmp_dir = os.path.join(ep_dir, "_tmp_clips")
    os.makedirs(tmp_dir, exist_ok=True)

    # 按 narration_segment 分组图片到对应语音段
    # narration_segment 是 1-based（对应 tts_meta index）
    seg_images: Dict[int, List[Optional[str]]] = {}  # segment(1-based) -> [image_path]
    for i, img_path in enumerate(image_paths):
        narr_seg = None
        if image_prompts and i < len(image_prompts):
            p = image_prompts[i]
            narr_seg = p.get("narration_segment") if isinstance(p, dict) else None
        # 没有 narration_segment 的图，按 index 顺序分配（回退：分到第 i+1 段或最后一段）
        if narr_seg is None:
            narr_seg = min(i + 1, len(tts_meta)) if tts_meta else 1
        seg_images.setdefault(narr_seg, []).append(img_path)

    clips: List[str] = []
    clip_idx = 0
    # 遍历每段语音
    for seg_idx in range(len(tts_meta)):
        seg_no = seg_idx + 1  # 1-based
        aud = audio_paths[seg_idx] if seg_idx < len(audio_paths) else None
        # 该段的图片列表
        imgs = seg_images.get(seg_no, [])
        if not imgs:
            # 无图，用占位
            imgs = [None]
        # 音频时长
        if aud and os.path.exists(aud):
            seg_dur = _audio_duration(aud)
        else:
            seg_dur = 3.0
        # 均分时长给每张图，每张图取音频的对应片段
        per_img = seg_dur / len(imgs)
        seek_acc = 0.0
        for img in imgs:
            clip = _make_clip(img, aud, clip_idx, tmp_dir,
                              cfg.media.video_resolution, cfg.media.video_fps,
                              duration=per_img, audio_seek=seek_acc)
            if clip:
                clips.append(clip)
            clip_idx += 1
            seek_acc += per_img

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
    # 第一步：concat 拼接（可能 stream copy 或重编码）→ 临时文件
    concat_out = os.path.join(tmp_dir, "concat.mp4")
    concat_ok = False
    try:
        subprocess.run(
            [exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", concat_out],
            capture_output=True, timeout=300,
        )
        if os.path.exists(concat_out) and os.path.getsize(concat_out) > 1000:
            concat_ok = True
    except Exception as e:
        print("    [视频] stream copy 失败: %s，尝试重编码" % e)
    if not concat_ok:
        try:
            subprocess.run(
                [exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", concat_out],
                capture_output=True, timeout=600,
            )
            if os.path.exists(concat_out) and os.path.getsize(concat_out) > 1000:
                concat_ok = True
        except Exception as e2:
            print("    [视频] 重编码也失败: %s" % e2)

    if not concat_ok:
        print("    [视频] concat 拼接失败")
        return None

    # 第二步：混入 BGM（循环播放 + volume）→ 最终输出
    # BGM 用 stream_loop=-1 无限循环，duration=first 对齐视频时长，视频在BGM就继续放
    cfg = get_config()
    bgm = cfg.media.bgm_path
    if bgm and os.path.exists(bgm):
        try:
            subprocess.run(
                [exe, "-y", "-i", concat_out,
                 "-stream_loop", "-1", "-i", bgm,
                 "-filter_complex",
                 "[1:a]volume=%.2f[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]" % cfg.media.bgm_volume,
                 "-map", "0:v", "-map", "[a]",
                 "-c:v", "copy", "-c:a", "aac", "-shortest", out],
                capture_output=True, timeout=600,
            )
            if not (os.path.exists(out) and os.path.getsize(out) > 1000):
                # copy v 失败，重编码视频
                subprocess.run(
                    [exe, "-y", "-i", concat_out,
                     "-stream_loop", "-1", "-i", bgm,
                     "-filter_complex",
                     "[1:a]volume=%.2f[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[a]" % cfg.media.bgm_volume,
                     "-map", "0:v", "-map", "[a]",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
                    capture_output=True, timeout=600,
                )
        except Exception as e:
            print("    [视频] BGM 混入失败: %s，用无BGM版本" % e)
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

    print("[合成] 开始 %s：生图 %d 张，TTS %d 段" % (eid, len(prompts), len(tts_meta)))

    if prompts:
        print("  [阶段1] 生成图片...")
        image_paths = generate_images(episode, prompts, ep_dir)
    else:
        image_paths = []

    if tts_meta:
        print("  [阶段2] 生成语音...")
        audio_paths = generate_tts(tts_meta, ep_dir)
    else:
        audio_paths = []

    print("  [阶段3] 拼接视频...")
    video_path = compose_video(image_paths, audio_paths, tts_meta, ep_dir, eid,
                               image_prompts=prompts)

    if video_path:
        sz_mb = os.path.getsize(video_path) / (1024 * 1024)
        print("[合成] %s 完成: %s (%.1f MB)" % (eid, video_path, sz_mb))
    else:
        print("[合成] %s 视频未生成（图片/音频已落盘）" % eid)

    return {"video_path": video_path}
