"""扫描各集音频完整性：audio 文件数 vs tts_meta 段数。"""
import json, os, re

OUT = "/mnt/d/多agent拆分/novel_pipeline/novels/人道至尊/output"

def scan(ep_dir):
    ep = os.path.basename(ep_dir)
    audio_dir = os.path.join(ep_dir, "audio")
    tts_path = os.path.join(ep_dir, "tts_meta.json")
    if not os.path.isdir(audio_dir) or not os.path.isfile(tts_path):
        return None
    try:
        with open(tts_path, encoding="utf-8") as f:
            tts = json.load(f)
    except Exception as e:
        return {"ep": ep, "error": f"tts_meta 读取失败: {e}"}
    expected = len(tts) if isinstance(tts, list) else 0
    files = sorted(os.listdir(audio_dir))
    mp3s = [f for f in files if f.endswith(".mp3")]
    idxs = []
    for f in mp3s:
        m = re.match(r"^(\d+)\.mp3$", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs_sorted = sorted(idxs)
    missing = []
    if expected > 0:
        for i in range(expected):
            if i not in set(idxs):
                missing.append(i)
    # 检查损坏（0 字节）
    zero = [f for f in mp3s if os.path.getsize(os.path.join(audio_dir, f)) == 0]
    return {
        "ep": ep,
        "expected": expected,
        "audio_count": len(mp3s),
        "idxs": idxs_sorted,
        "missing": missing,
        "zero_size": zero,
        "mp4": os.path.isfile(os.path.join(ep_dir, f"{ep}.mp4")),
    }

rows = []
for d in sorted(os.listdir(OUT), key=lambda x: int(re.sub(r"\D", "", x) or 0)):
    p = os.path.join(OUT, d)
    if not os.path.isdir(p):
        continue
    r = scan(p)
    if r:
        rows.append(r)

print(f"{'集':<8}{'tts段数':<8}{'音频数':<8}{'缺失':<20}{'0字节':<8}{'mp4'}")
print("-" * 80)
bad = []
for r in rows:
    missing_str = ",".join(str(x) for x in r["missing"]) if r["missing"] else "-"
    zero_str = ",".join(r["zero_size"]) if r["zero_size"] else "-"
    flag = ""
    if r["missing"] or r["zero_size"]:
        flag = "  <== 异常"
        bad.append(r)
    print(f"{r['ep']:<8}{r['expected']:<8}{r['audio_count']:<8}{missing_str:<20}{zero_str:<8}{str(r.get('mp4')):<6}{flag}")

print(f"\n共扫描 {len(rows)} 集，异常 {len(bad)} 集")
for r in bad:
    print(json.dumps(r, ensure_ascii=False))
