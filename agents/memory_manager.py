"""
novel_pipeline.agents.memory_manager
全局记忆管理 Agent —— 全系统唯一记忆读写入口。
维护三大结构化知识库：人物档案库、时序事件库、伏笔台账。
提供 FAISS 向量检索服务，召回历史事件 / 人物信息 / 关联伏笔。
对应设计文档 4.3 支撑服务 Agent 层。
"""
import os
import json
import uuid
import threading
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

from config import get_config
from llm_factory import get_llm


# ───────── 内存数据结构（单例，进程内共享） ─────────
class _MemoryStore:
    """进程内单例记忆库，三大知识库 + FAISS 索引。线程安全。"""

    def __init__(self, memory_dir: str, enable_vector: bool = True):
        self._lock = threading.RLock()
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        # 三大结构化知识库
        self.character_profiles: Dict[str, Dict] = {}   # 人物ID -> 档案
        self.timeline_events: Dict[str, Dict] = {}      # 事件ID -> 事件
        self.foreshadow_ledger: Dict[str, Dict] = {}    # 伏笔ID -> 伏笔记录

        # 向量索引（FAISS）：存储历史场景向量，用于长线伏笔召回
        self.enable_vector = enable_vector and _FAISS_AVAILABLE
        self._faiss_index = None
        self._vector_meta: List[Dict] = []   # 与索引向量一一对应
        self._dim = 768  # 默认嵌入维度，按实际模型调整

        self._load()

    # ---------- 持久化 ----------
    def _path(self, name: str) -> str:
        return os.path.join(self.memory_dir, f"{name}.json")

    def _load(self):
        with self._lock:
            for attr, fname in [
                ("character_profiles", "characters"),
                ("timeline_events", "events"),
                ("foreshadow_ledger", "foreshadows"),
            ]:
                p = self._path(fname)
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
            # 加载向量元数据
            vp = self._path("vector_meta")
            if os.path.exists(vp):
                with open(vp, "r", encoding="utf-8") as f:
                    self._vector_meta = json.load(f)
                if self.enable_vector:
                    self._rebuild_faiss_from_meta()

    def save(self):
        with self._lock:
            for attr, fname in [
                ("character_profiles", "characters"),
                ("timeline_events", "events"),
                ("foreshadow_ledger", "foreshadows"),
            ]:
                with open(self._path(fname), "w", encoding="utf-8") as f:
                    json.dump(getattr(self, attr), f, ensure_ascii=False, indent=2)
            if self.enable_vector:
                with open(self._path("vector_meta"), "w", encoding="utf-8") as f:
                    json.dump(self._vector_meta, f, ensure_ascii=False, indent=2)

    # ---------- 人物档案 ----------
    # 固定外貌字段：一旦建立就不被后续场景覆盖（保证人物一致性）
    _FIXED_PROFILE_FIELDS = ("appearance", "age", "identity", "personality")
    # 用户专属字段：AI 永不覆盖/不改写，仅用户通过 UI 编辑
    _USER_FIELDS = ("user_description",)

    def upsert_character(self, char_id: str, profile: Dict):
        with self._lock:
            existing = self.character_profiles.get(char_id, {})
            # appearance_override=true：变形/易容/化妆/法术变身，允许更新固定外貌字段
            override = profile.get("appearance_override") is True
            if override:
                # 变形场景：更新所有传入的固定字段（appearance/age 等）
                for k in self._FIXED_PROFILE_FIELDS:
                    v = profile.get(k)
                    if v:
                        existing[k] = v
            else:
                # 正常场景：固定字段只在档案中尚无该值时写入；已有则保留原值
                for k in self._FIXED_PROFILE_FIELDS:
                    v = profile.get(k)
                    if v and not existing.get(k):
                        existing[k] = v
            # attire（穿着）：与 identity 绑定。identity 不变 → attire 保持原值
            # 只有 identity 发生明确变化（如外门弟子→内门弟子）时才允许更新 attire 和 identity
            new_attire = profile.get("attire")
            new_identity = profile.get("identity")
            old_identity = existing.get("identity")
            if new_attire:
                if not existing.get("attire"):
                    # 首次写入
                    existing["attire"] = new_attire
                elif new_identity and old_identity and new_identity != old_identity:
                    # identity 变化才更新 attire
                    existing["attire"] = new_attire
                elif override:
                    # 变形场景也允许更新 attire
                    existing["attire"] = new_attire
                # 否则保持原 attire（identity 未变，穿着不变）
            # identity：身份明确变化时更新（如外门弟子→内门弟子）
            if new_identity and old_identity and new_identity != old_identity:
                existing["identity"] = new_identity
            # 可变字段（state_change/is_new/appearance_override 等）常规更新
            for k, v in profile.items():
                if k in self._FIXED_PROFILE_FIELDS or k == "attire":
                    continue  # 已上面处理
                if k in self._USER_FIELDS:
                    continue  # 用户专属字段，AI 不得覆盖
                if v is not None:
                    existing[k] = v
            existing["char_id"] = char_id
            self.character_profiles[char_id] = existing

    def save_user_description(self, char_id: str, description: str):
        """用户通过 UI 编辑角色描述，写回 characters.json。AI 永不覆盖此字段。
        description 为空串则清除。"""
        with self._lock:
            existing = self.character_profiles.get(char_id, {})
            existing["char_id"] = char_id
            existing["user_description"] = description or ""
            self.character_profiles[char_id] = existing
        self.save()

    def get_character(self, char_id: str) -> Optional[Dict]:
        with self._lock:
            return self.character_profiles.get(char_id)

    def query_characters(self, char_ids: List[str]) -> List[Dict]:
        with self._lock:
            return [self.character_profiles[c] for c in char_ids if c in self.character_profiles]

    def all_characters(self) -> List[Dict]:
        with self._lock:
            return list(self.character_profiles.values())

    # ---------- 时序事件库 ----------
    def add_event(self, event: Dict) -> str:
        with self._lock:
            eid = event.get("event_id") or f"evt_{uuid.uuid4().hex[:8]}"
            event["event_id"] = eid
            self.timeline_events[eid] = event
            return eid

    def get_event(self, eid: str) -> Optional[Dict]:
        with self._lock:
            return self.timeline_events.get(eid)

    def recent_events(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            evts = list(self.timeline_events.values())
            return evts[-limit:]

    def all_events(self) -> List[Dict]:
        with self._lock:
            return list(self.timeline_events.values())

    # ---------- 伏笔台账 ----------
    def upsert_foreshadow(self, fid: str, record: Dict):
        with self._lock:
            existing = self.foreshadow_ledger.get(fid, {})
            existing.update(record)
            existing["foreshadow_id"] = fid
            self.foreshadow_ledger[fid] = existing

    def get_foreshadow(self, fid: str) -> Optional[Dict]:
        with self._lock:
            return self.foreshadow_ledger.get(fid)

    def all_foreshadows(self) -> List[Dict]:
        with self._lock:
            return list(self.foreshadow_ledger.values())

    def unresolved_foreshadows(self) -> List[Dict]:
        with self._lock:
            return [f for f in self.foreshadow_ledger.values() if f.get("status") != "resolved"]

    # ---------- 向量检索（FAISS） ----------
    def _rebuild_faiss_from_meta(self):
        """从 _vector_meta 重建 FAISS 索引（仅启动时）。"""
        if not self.enable_vector or not self._vector_meta:
            return
        dim = len(self._vector_meta[0]["vector"])
        self._dim = dim
        self._faiss_index = faiss.IndexFlatIP(dim)
        mat = np.array([m["vector"] for m in self._vector_meta], dtype=np.float32)
        faiss.normalize_L2(mat)
        self._faiss_index.add(mat)

    def _embed(self, text: str) -> Optional[List[float]]:
        """调用 LLM 获取文本嵌入向量。这里用简单哈希降维作为可运行兜底；
        生产环境应替换为真实 Embedding 模型调用。"""
        if not self.enable_vector:
            return None
        # 真实嵌入：通过 OpenAI 兼容 embeddings 接口
        try:
            from openai import OpenAI
            cfg = get_config().model
            client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            resp = client.embeddings.create(
                model="text-embedding-1",  # MiniMax 嵌入模型名按实际替换
                input=text,
            )
            return list(resp.data[0].embedding)
        except Exception:
            # 兜底：确定性伪向量，保证流程可跑（无真实语义）
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            return rng.normal(0, 1, self._dim).astype(np.float32).tolist()

    def add_scene_vector(self, scene_text: str, meta: Dict):
        """将历史场景文本向量化入库，供长线伏笔召回。"""
        with self._lock:
            if not self.enable_vector:
                return
            vec = self._embed(scene_text)
            if vec is None:
                return
            if self._faiss_index is None:
                self._dim = len(vec)
                self._faiss_index = faiss.IndexFlatIP(self._dim)
            v = np.array([vec], dtype=np.float32)
            faiss.normalize_L2(v)
            self._faiss_index.add(v)
            self._vector_meta.append({"vector": vec, "meta": meta})

    def search_similar(self, query_text: str, k: int = 5) -> List[Dict]:
        """向量检索召回相关历史场景。"""
        with self._lock:
            if not self.enable_vector or self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []
            vec = self._embed(query_text)
            if vec is None:
                return []
            q = np.array([vec], dtype=np.float32)
            faiss.normalize_L2(q)
            scores, idxs = self._faiss_index.search(q, min(k, self._faiss_index.ntotal))
            return [
                {"score": float(scores[0][i]), "meta": self._vector_meta[idx]["meta"]}
                for i, idx in enumerate(idxs[0]) if idx >= 0 and idx < len(self._vector_meta)
            ]


# 进程内单例
_store: Optional[_MemoryStore] = None
_store_lock = threading.Lock()


def _get_store() -> _MemoryStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                cfg = get_config().storage
                _store = _MemoryStore(cfg.memory_dir, cfg.enable_vector_retrieval)
    return _store


def reset_memory_store():
    """重置进程内单例——apply_book 切书后调用，让下次 get_memory_agent 重新加载新书目录。"""
    global _store, _agent
    with _store_lock:
        _store = None
    _agent = None


# ───────── LLM 包装：记忆更新 / 召回的语义化处理 ─────────
class MemoryManagerAgent:
    """
    全局记忆管理 Agent。
    所有 Agent 不得直接修改记忆库，统一通过本类。
    LLM 用于：从场景数据中抽取结构化人物/事件/伏笔信息。
    """

    def __init__(self):
        self.llm = get_llm(role="support")
        self.store = _get_store()

    # ---------- 召回 ----------
    def recall_context(self, query: str, char_ids: Optional[List[str]] = None) -> Dict:
        """根据查询请求召回关联历史事件、人物信息、关联伏笔。"""
        result: Dict[str, Any] = {}
        # 人物档案
        if char_ids:
            result["characters"] = self.store.query_characters(char_ids)
        else:
            result["characters"] = self.store.all_characters()
        # 向量检索相关历史场景
        result["similar_scenes"] = self.store.search_similar(query, k=5)
        # 未解伏笔（潜在关联）
        result["unresolved_foreshadows"] = self.store.unresolved_foreshadows()
        # 最近事件（提供近期因果上下文）
        result["recent_events"] = self.store.recent_events(limit=20)
        return result

    def recall_for_episode(self, episode: Dict) -> Dict:
        """为单集评审预检索关联记忆，供所有评审共享。"""
        # 取本集关联人物与伏笔ID
        char_ids = [s.get("char_id") for s in episode.get("scenes", []) if s.get("char_id")]
        # 去重
        seen = set()
        uniq_ids = [c for c in char_ids if not (c in seen or seen.add(c))]
        query = episode.get("summary", "") or json.dumps(
            episode.get("scenes", []), ensure_ascii=False)[:500]
        return self.recall_context(query, uniq_ids)

    # ---------- 增量更新 ----------
    def update_from_scenes(self, scenes: List[Dict]):
        """本集场景、新增伏笔、人物变化增量入库。"""
        for scene in scenes:
            # 入时序事件库
            self.store.add_event({
                "event_id": scene.get("scene_id"),
                "summary": scene.get("summary", ""),
                "cause": scene.get("cause", ""),
                "effect": scene.get("effect", ""),
                "characters": scene.get("characters", []),
                "episode_id": scene.get("episode_id"),
                "raw_order": scene.get("raw_order"),
            })
            # 入向量库
            scene_text = json.dumps(scene, ensure_ascii=False)
            self.store.add_scene_vector(scene_text, {
                "scene_id": scene.get("scene_id"),
                "episode_id": scene.get("episode_id"),
                "summary": scene.get("summary", ""),
            })
            # 人物变化增量
            for char in scene.get("characters", []):
                if isinstance(char, dict):
                    cid = char.get("char_id") or char.get("name")
                    if cid:
                        self.store.upsert_character(cid, char)
            # 伏笔更新
            for f in scene.get("foreshadows", []):
                fid = f.get("foreshadow_id")
                if fid:
                    self.store.upsert_foreshadow(fid, f)

    def update_from_episode(self, episode: Dict):
        """归档后增量更新本集全部记忆。"""
        self.update_from_scenes(episode.get("scenes", []))

    def save(self):
        self.store.save()

    # ---------- 查询辅助（评审 Agent 用） ----------
    def get_character_profiles(self) -> List[Dict]:
        return self.store.all_characters()

    def get_character(self, char_id: str):
        """转发到 store，供 UI 按角色ID加载单条档案。"""
        return self.store.get_character(char_id)

    def save_user_description(self, char_id: str, description: str):
        """转发到 store，供 UI 调用。"""
        self.store.save_user_description(char_id, description)

    def get_foreshadow_ledger(self) -> List[Dict]:
        return self.store.all_foreshadows()

    def get_timeline_events(self) -> List[Dict]:
        return self.store.all_events()

    def get_original_snippet_context(self, episode: Dict) -> str:
        """取本集关联历史事件，拼接为「原著前情」上下文文本，供评审对照。"""
        evts = self.store.recent_events(limit=30)
        lines = []
        for e in evts:
            lines.append(f"- [{e.get('event_id')}] {e.get('summary','')}")
        return "\n".join(lines)


# 模块级单例
_agent: Optional[MemoryManagerAgent] = None


def get_memory_agent() -> MemoryManagerAgent:
    global _agent
    if _agent is None:
        _agent = MemoryManagerAgent()
    return _agent
