"""
一键执行数据生成 + 目标推荐标签输出的示例脚本。

使用示例：
    python run_pipeline.py --num-targets 10 --num-missions 1000 --enable-rf \
        --save-raw --targets-file outputs/targets.txt \
        --missions-file outputs/missions.txt \
        --recommendations-file outputs/recommendations.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from data_generator import generate_smart_data, save_data_to_files
from target_recommendation import generate_target_recommendations
from target_recommendation_models import TargetRecommendation


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _dump_recommendations(recommendations: List[TargetRecommendation], output_path: Path) -> None:
    _ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fp:
        for record in recommendations:
            json.dump(asdict(record), fp, ensure_ascii=False)
            fp.write("\n")


def _preview_recommendations(
    recommendations: Iterable[TargetRecommendation], *, limit: int = 5
) -> None:
    print("\n=== 推荐结果预览 ===")
    count = 0
    for record in recommendations:
        if count >= limit:
            break
        print(json.dumps(asdict(record), ensure_ascii=False, indent=2))
        print()
        count += 1
    if count == 0:
        print("(暂无推荐结果可供预览)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成测试数据并输出目标推荐标签")
    parser.add_argument("--num-targets", type=int, default=10, help="生成的目标数量")
    parser.add_argument("--num-missions", type=int, default=1000, help="生成的任务数量")
    parser.add_argument("--enable-rf", action="store_true", help="启用随机森林用户模式")
    parser.add_argument("--cluster-spread-deg", type=float, default=5, help="簇内经纬度扰动幅度（度）")
    parser.add_argument("--save-raw", action="store_true", help="是否同时保存原始目标/任务数据")
    parser.add_argument("--targets-file", type=Path, default=Path("outputs/targets.txt"), help="目标信息输出文件")
    parser.add_argument("--missions-file", type=Path, default=Path("outputs/missions.txt"), help="任务信息输出文件")
    parser.add_argument("--recommendations-file", type=Path, default=Path("outputs/recommendations.jsonl"), help="推荐标签输出文件")
    parser.add_argument("--spatial-eps-km", type=float, default=600.0, help="DBSCAN 邻域半径（公里）")
    parser.add_argument("--spatial-min-samples", type=int, default=5, help="DBSCAN 最小样本数")
    parser.add_argument("--spatial-min-clusters", type=int, default=7, help="自动调参期望的最小簇数量")
    parser.add_argument("--disable-spatial-auto-tune", action="store_true", help="关闭空间聚类自动调参")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    targets, missions = generate_smart_data(
        num_targets=args.num_targets,
        num_missions=args.num_missions,
        enable_rf_users=args.enable_rf,
        cluster_spread_deg=max(0.01, args.cluster_spread_deg),
    )

    if args.save_raw:
        _ensure_parent(args.targets_file)
        _ensure_parent(args.missions_file)
        save_data_to_files(targets, missions, str(args.targets_file), str(args.missions_file))

    recommendations = generate_target_recommendations(
        targets,
        missions,
        spatial_eps_km=args.spatial_eps_km,
        spatial_min_samples=max(1, args.spatial_min_samples),
        spatial_min_clusters=max(1, args.spatial_min_clusters),
        spatial_auto_tune=not args.disable_spatial_auto_tune,
    )
    _dump_recommendations(recommendations, args.recommendations_file)
    _preview_recommendations(recommendations)

    print("\n=== 运行完成 ===")
    print(f"生成目标：{len(targets):,} 个")
    print(f"生成任务：{len(missions):,} 条")
    print(f"推荐标签：{len(recommendations):,} 条 -> {args.recommendations_file}")
    if args.save_raw:
        print(f"目标数据输出：{args.targets_file}")
        print(f"任务数据输出：{args.missions_file}")


if __name__ == "__main__":
    main()



