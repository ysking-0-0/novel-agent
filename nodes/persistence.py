"""
novel_pipeline.nodes.persistence
持久化归档节点（纯代码）—— 主循环步骤 9。
对应设计文档 5.2 / 6.2。

职责：
  按 episode_id 存储整集素材 → completed_episode_count + 1
  目录结构：
    output/ep_xxx/episode_info.json
                    /script.txt
                    /image_prompts.json
                    /tts_meta.json
                    /original_snippet.txt
"""
import json
import os
from typing import Dict
from config import get_config
from agents import get_memory_agent


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def persistence_node(state: Dict) -> Dict:
    """归档当前集素材到本地文件系统，递增已完成集数，并触发记忆增量更新（步骤 9→10）。"""
    cfg = get_config()
    out_dir = cfg.storage.output_dir
    _ensure_dir(out_dir)

    episode = dict(state.get("current_episode") or {})
    # 赋 episode_id（聚合阶段为 None，此处按完成序号赋值 ep_xxx）
    completed = state.get("completed_episode_count", 0) + 1
    eid = episode.get("episode_id") or f"ep_{completed:03d}"
    episode["episode_id"] = eid
    # 写回每个 scene 的 episode_id，便于记忆更新
    for sc in episode.get("scenes", []):
        sc["episode_id"] = eid

    ep_dir = os.path.join(out_dir, eid)
    _ensure_dir(ep_dir)

    # episode_info.json
    info = {
        "episode_id": eid,
        "summary": episode.get("summary", ""),
        "cliffhanger": episode.get("cliffhanger", ""),
        "scene_count": len(episode.get("scenes", [])),
        "scenes": episode.get("scenes", []),
        "linked_foreshadow_ids": episode.get("linked_foreshadow_ids", []),
        "source_offset_range": episode.get("source_offset_range", []),
        "is_final_episode": episode.get("is_final_episode", False),
        "review_result": state.get("review_result"),
    }
    _write_json(os.path.join(ep_dir, "episode_info.json"), info)

    # script.txt
    _write_text(os.path.join(ep_dir, "script.txt"), state.get("episode_script", "") or "")

    # image_prompts.json
    _write_json(os.path.join(ep_dir, "image_prompts.json"), state.get("episode_image_prompts", []))

    # tts_meta.json
    _write_json(os.path.join(ep_dir, "tts_meta.json"), state.get("episode_tts_meta", []))

    # original_snippet.txt（对应原著原文片段：取本集 scene 摘要拼接作为备查）
    snippet = "\n\n".join(s.get("summary", "") for s in episode.get("scenes", []))
    _write_text(os.path.join(ep_dir, "original_snippet.txt"), snippet)

    print(f"[归档] {eid} 完成 → {ep_dir}  (累计 {completed} 集)")

    # 步骤 10：记忆增量更新（本集场景、伏笔、人物变化入库）
    mem = get_memory_agent()
    try:
        mem.update_from_episode(episode)
        mem.save()
    except Exception as e:
        print(f"[记忆] 更新失败但不阻塞: {type(e).__name__}: {e}")

    return {
        "completed_episode_count": completed,
        "current_episode": None,
        "episode_script": None,
        "episode_image_prompts": [],
        "episode_tts_meta": [],
        "review_result": None,
        "retry_count": 0,
        "format_valid": False,
        "format_errors": [],
        "prefetched_memory": None,
        "review_reports": [],
    }
