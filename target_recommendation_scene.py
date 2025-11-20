"""
侦察场景使用标签模块
"""

from __future__ import annotations

from typing import Optional


def derive_scene_usage_label(
    task_scene: Optional[str],
    scout_type: Optional[str],
    task_type: Optional[str],
    is_precise: bool,
) -> str:
    """根据任务场景、侦察类型、任务类型以及是否精确需求构建标签。"""

    scene = (task_scene or "未知场景").strip()
    scout = (scout_type or "未知侦察").strip()
    task = (task_type or "未知类型").strip()

    precision_tag = "精准" if is_precise else "常规"

    mode_tag = ""
    if "电子" in scout:
        mode_tag = "电侦"
    elif "光学" in scout or "红外" in scout:
        mode_tag = "光电"
    elif "雷达" in scout:
        mode_tag = "雷达"
    elif "通信" in scout:
        mode_tag = "通信"
    else:
        mode_tag = scout

    task_tag = {
        "1": "监视",
        "2": "跟踪",
        "3": "压制",
        "4": "评估",
        "5": "综合",
    }.get(task, task)

    return f"{precision_tag}{mode_tag}-{scene}-{task_tag}"


__all__ = ["derive_scene_usage_label"]


