"""批量重跑 ep_082~ep_093：复用图片、重新生成 TTS（新静音压缩参数）+ 重合成视频。"""
import os, sys, time
sys.path.insert(0, "/mnt/d/多agent拆分/novel_pipeline")
os.chdir("/mnt/d/多agent拆分/novel_pipeline")

from config import set_config, apply_book, get_config
from nodes.media_synthesizer import (
    generate_tts, _retry_generate_tts, compose_video, _load_ep_meta,
    _list_image_paths, _list_audio_paths,
)

set_config("config.json")
apply_book("人道至尊")
cfg = get_config()
OUT = cfg.storage.output_dir
print(f"[批量] 输出目录: {OUT}", flush=True)
print(f"[批量] 压缩参数: max={cfg.media.silence_max_pause}s, target={cfg.media.silence_target_pause}s", flush=True)

EPISODES = ["ep_%03d" % n for n in range(82, 94)]
for EP in EPISODES:
    ep_dir = os.path.join(OUT, EP)
    t0 = time.time()
    print(f"\n{'='*50}\n[批量] 开始 {EP}", flush=True)
    if not os.path.isdir(ep_dir):
        print(f"[批量] {EP} 不存在，跳过", flush=True)
        continue
    try:
        episode, prompts, tts_meta = _load_ep_meta(ep_dir)
    except Exception as e:
        print(f"[批量] {EP} 读取归档失败: {e}", flush=True)
        continue
    if not tts_meta:
        print(f"[批量] {EP} 无 tts_meta，跳过", flush=True)
        continue
    print(f"[批量] {EP}: 图片prompt {len(prompts)} 个, TTS {len(tts_meta)} 段", flush=True)

    # 阶段1: 重新生成 TTS（清旧音频 + 新参数压缩）
    audio_paths = generate_tts(tts_meta, ep_dir)
    ok = sum(1 for p in audio_paths if p)
    if ok < len(tts_meta):
        missing = [i for i, p in enumerate(audio_paths) if not p]
        print(f"[批量] {EP} 缺失 {len(missing)} 段，重试补全...", flush=True)
        retry = _retry_generate_tts(tts_meta, missing, ep_dir)
        for i, p in retry.items():
            if p:
                audio_paths[i] = p
        ok = sum(1 for p in audio_paths if p)
    print(f"[批量] {EP} TTS: {ok}/{len(tts_meta)}", flush=True)

    # 阶段2: 复用图片重合成视频
    image_paths = [p if os.path.exists(p) else None
                   for p in _list_image_paths(ep_dir, len(prompts))]
    audio_paths = [p if os.path.exists(p) else None
                   for p in _list_audio_paths(ep_dir, len(tts_meta))]
    n_img = sum(1 for p in image_paths if p)
    print(f"[批量] {EP} 可用图片: {n_img}/{len(image_paths)}", flush=True)
    if n_img == 0:
        print(f"[批量] {EP} 无可用图片，跳过合成", flush=True)
        continue
    old_mp4 = os.path.join(ep_dir, f"{EP}.mp4")
    if os.path.exists(old_mp4):
        os.remove(old_mp4)
    video_path = compose_video(image_paths, audio_paths, tts_meta, ep_dir, EP,
                               image_prompts=prompts)
    dt = time.time() - t0
    if video_path:
        print(f"[批量] {EP} 成功: {video_path} ({dt:.0f}s)", flush=True)
    else:
        print(f"[批量] {EP} 合成失败 ({dt:.0f}s)", flush=True)

print("\n[批量] 全部完成", flush=True)
