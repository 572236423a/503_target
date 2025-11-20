"""
目标推荐算法模块
----------------
核心逻辑拆分为多个独立子模块：
- `target_recommendation_models.py`：数据结构
- `target_recommendation_frequency.py`：侦察周期 / 频率标签
- `target_recommendation_spatial.py`：空间密度聚类
- `target_recommendation_scene.py`：侦察场景使用标签

此文件聚合各模块能力，并提供统一的对外接口。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from mission import Mission
from target_info import TargetInfo
from target_recommendation_frequency import ScoutFrequencyLabels, build_scout_frequency_labels
from target_recommendation_models import TargetRecommendation
from target_recommendation_spatial import compute_spatial_density_labels


def _build_top_label_stats(counter: Counter[str], *, key_name: str, top_n: int = 3) -> Dict[str, object]:
    """根据标签计数生成 TopN 统计。"""

    total = sum(counter.values())
    if total == 0:
        return {"top_n": top_n, "top_combinations": [], "total_combinations": 0}

    top_items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:top_n]
    combinations = [
        {key_name: label, "count": count, "percentage": round(count / total, 2)}
        for label, count in top_items
    ]
    return {"top_n": top_n, "top_combinations": combinations, "total_combinations": total}


def _build_preferred_scout_scenario(
    counter: Counter[tuple], top_n: int = 3
) -> Dict[str, object]:
    """根据 (task_type, scout_type, task_scene, is_precise) 组合生成 TopN 列表。"""

    total = sum(counter.values())
    if total == 0:
        return {"top_n": top_n, "top_scenarios": [], "total_combinations": 0}

    top_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    top_scenarios = [
        {
            "task_type": task_type,
            "scout_type": scout_type,
            "task_scene": task_scene,
            "is_precise": is_precise,
            "count": count,
            "percentage": round(count / total, 2),
        }
        for (task_type, scout_type, task_scene, is_precise), count in top_items
    ]
    return {"top_n": top_n, "top_scenarios": top_scenarios, "total_combinations": total}


def generate_target_recommendations(
    target_info_list: List[TargetInfo],
    mission_list: List[Mission],
    *,
    spatial_eps_km: float = 60.0,
    spatial_min_samples: int = 4,
    spatial_auto_tune: bool = True,
    spatial_min_clusters: int = 7,
) -> List[TargetRecommendation]:
    """
    综合生成标签，返回 `TargetRecommendation` 列表。

    输出字段说明：
    - scout_cycle_label: 同一 target_id 的侦察周期型标签 Top3 统计
    - scout_frequency_label: 同一 target_id 的侦察频次标签 Top3 统计
    - preferred_scout_scenario: 同一 target_id 的任务场景组合 Top3 统计
    - spatial_density_label: 空间密度簇编号（-1 表示孤立点）
    - target_type_label: 目标类型
    - target_priority_label: 目标优先级
    """

    target_by_id: Dict[str, TargetInfo] = {target.target_id: target for target in target_info_list}
    spatial_labels = compute_spatial_density_labels(
        target_info_list,
        eps_km=spatial_eps_km,
        min_samples=spatial_min_samples,
        auto_tune=spatial_auto_tune,
        desired_min_clusters=spatial_min_clusters,
    )

    cycle_counters: Dict[str, Counter[str]] = defaultdict(Counter)
    frequency_counters: Dict[str, Counter[str]] = defaultdict(Counter)
    scenario_counters: Dict[str, Counter[tuple]] = defaultdict(Counter)
    spatial_counters: Dict[str, Counter[int]] = defaultdict(Counter)
    type_counters: Dict[str, Counter[str]] = defaultdict(Counter)
    priority_counters: Dict[str, Counter[float]] = defaultdict(Counter)
    pending_records: List[Mission] = []

    for mission in mission_list:
        target = target_by_id.get(mission.target_id)

        frequency_labels = build_scout_frequency_labels(
            mission.req_cycle, mission.req_cycle_time, mission.req_times
        )
        spatial_density_label = spatial_labels.get(mission.target_id, -1)
        target_type_label = target.target_type if target else "未知类型"
        target_priority_label = mission.target_priority

        cycle_counters[mission.target_id][frequency_labels.cycle_label] += 1
        frequency_counters[mission.target_id][frequency_labels.frequency_label] += 1
        scenario_key = (
            mission.task_type or "未知类型",
            mission.scout_type or "未知侦察",
            mission.task_scene or "未知场景",
            mission.is_precise,
        )
        scenario_counters[mission.target_id][scenario_key] += 1

        spatial_counters[mission.target_id][spatial_density_label] += 1
        type_counters[mission.target_id][target_type_label] += 1
        priority_counters[mission.target_id][target_priority_label] += 1

        pending_records.append(mission)

    cycle_stats = {
        target_id: _build_top_label_stats(counter, key_name="cycle_label") for target_id, counter in cycle_counters.items()
    }
    frequency_stats = {
        target_id: _build_top_label_stats(counter, key_name="scout_frequency_label")
        for target_id, counter in frequency_counters.items()
    }
    scenario_stats = {
        target_id: _build_preferred_scout_scenario(counter) for target_id, counter in scenario_counters.items()
    }
    spatial_stats = {
        target_id: _build_top_label_stats(counter, key_name="spatial_density_label")
        for target_id, counter in spatial_counters.items()
    }
    type_stats = {
        target_id: _build_top_label_stats(counter, key_name="target_type_label") for target_id, counter in type_counters.items()
    }
    priority_stats = {
        target_id: _build_top_label_stats(counter, key_name="target_priority_label") for target_id, counter in priority_counters.items()
    }

    recommendations: List[TargetRecommendation] = []
    for mission in pending_records:
        recommendations.append(
            TargetRecommendation(
                req_id=mission.req_id,
                target_id=mission.target_id,
                scout_cycle_label=cycle_stats.get(
                    mission.target_id, {"top_n": 3, "top_combinations": [], "total_combinations": 0}
                ),
                scout_frequency_label=frequency_stats.get(
                    mission.target_id, {"top_n": 3, "top_combinations": [], "total_combinations": 0}
                ),
                spatial_density_label=spatial_stats.get(
                    mission.target_id, {"top_n": 3, "top_combinations": [], "total_combinations": 0}
                ),
                target_type_label=type_stats.get(
                    mission.target_id, {"top_n": 3, "top_combinations": [], "total_combinations": 0}
                ),
                target_priority_label=priority_stats.get(
                    mission.target_id, {"top_n": 3, "top_combinations": [], "total_combinations": 0}
                ),
                preferred_scout_scenario=scenario_stats.get(
                    mission.target_id, {"top_n": 3, "top_scenarios": [], "total_combinations": 0}
                ),
            )
        )

    return recommendations


__all__ = [
    "TargetRecommendation",
    "compute_spatial_density_labels",
    "generate_target_recommendations",
]

