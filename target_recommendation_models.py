from dataclasses import dataclass


@dataclass
class TargetRecommendation:
    """单条任务的目标推荐标签汇总。"""

    req_id: str
    target_id: str
    scout_cycle_label: dict
    scout_frequency_label: dict
    spatial_density_label: dict
    target_type_label: dict
    target_priority_label: dict
    preferred_scout_scenario: dict


__all__ = ["TargetRecommendation"]


