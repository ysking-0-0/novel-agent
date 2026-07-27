"""
novel_pipeline.nodes.retry_counter
重试计数节点（纯代码）—— 主循环步骤 8 的子节点。
对应设计文档 5.2。

职责：
  仲裁判定为 regenerate 时，递增 retry_count，
  清理上一轮的评审产物，回到素材生成整集重生成。
"""
from typing import Dict


def retry_counter_node(state: Dict) -> Dict:
    retry = state.get("retry_count", 0) + 1
    print(f"[重试] 整集重生成，retry_count={retry}")
    return {
        "retry_count": retry,
        # 清理上一轮产物
        "review_result": None,
        "review_reports": [],
        "format_valid": False,
        "format_errors": [],
        "prefetched_memory": None,
    }
