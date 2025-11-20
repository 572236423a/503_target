#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制空间聚类散点图（模拟世界地图）。

示例：
    # 方式1：从 recommendations.jsonl 读取（推荐）
    python plot_spatial_clusters.py \
        --recommendations-file outputs/recommendations.jsonl \
        --num-targets 400 --cluster-spread-deg 0.05 \
        --output outputs/spatial_clusters.png
    
    # 方式2：重新生成数据并绘图
    python plot_spatial_clusters.py \
        --num-targets 400 --num-missions 2000 \
        --cluster-spread-deg 0.05 \
        --spatial-eps-km 60 --spatial-min-samples 4 \
        --spatial-min-clusters 7 --output outputs/spatial_clusters.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib import font_manager

from data_generator import generate_smart_data
from target_info import TargetInfo, Trajectory
from target_recommendation_spatial import compute_spatial_density_labels


def _extract_lon_lat_pairs(targets: List[TargetInfo]) -> Dict[str, Tuple[float, float]]:
    result: Dict[str, Tuple[float, float]] = {}
    for target in targets:
        coord = None
        for trajectory in target.trajectory_list:
            if isinstance(trajectory, Trajectory):
                try:
                    lon = float(trajectory.lon)
                    lat = float(trajectory.lat)
                    coord = (lon, lat)
                    break
                except (TypeError, ValueError):
                    continue
        if coord:
            result[target.target_id] = coord
    return result


def _get_geographic_name(lon: float, lat: float) -> str:
    """
    根据经纬度坐标返回地理位置名称（中文）
    """
    # 东亚
    if 100 <= lon <= 140 and 20 <= lat <= 50:
        if 110 <= lon <= 125 and 30 <= lat <= 45:
            return "中国东部"
        elif 125 <= lon <= 140 and 30 <= lat <= 45:
            return "日本"
        elif 100 <= lon <= 110 and 20 <= lat <= 30:
            return "东南亚"
        return "东亚"
    
    # 南亚
    if 60 <= lon <= 100 and 5 <= lat <= 40:
        if 70 <= lon <= 90 and 20 <= lat <= 35:
            return "印度"
        elif 60 <= lon <= 75 and 20 <= lat <= 35:
            return "巴基斯坦"
        return "南亚"
    
    # 中东
    if 30 <= lon <= 60 and 10 <= lat <= 45:
        if 35 <= lon <= 50 and 25 <= lat <= 40:
            return "波斯湾"
        return "中东"
    
    # 欧洲
    if -10 <= lon <= 40 and 35 <= lat <= 70:
        if 2 <= lon <= 15 and 45 <= lat <= 55:
            return "中欧"
        elif -10 <= lon <= 10 and 45 <= lat <= 55:
            return "西欧"
        elif 10 <= lon <= 30 and 45 <= lat <= 60:
            return "东欧"
        elif -5 <= lon <= 5 and 40 <= lat <= 45:
            return "伊比利亚"
        return "欧洲"
    
    # 北美
    if -130 <= lon <= -60 and 25 <= lat <= 70:
        if -80 <= lon <= -70 and 35 <= lat <= 45:
            return "美国东海岸"
        elif -125 <= lon <= -115 and 32 <= lat <= 42:
            return "美国西海岸"
        elif -100 <= lon <= -85 and 35 <= lat <= 50:
            return "美国中部"
        return "北美"
    
    # 南美
    if -80 <= lon <= -30 and -60 <= lat <= 15:
        return "南美"
    
    # 非洲
    if -20 <= lon <= 50 and -35 <= lat <= 40:
        if 20 <= lon <= 40 and 20 <= lat <= 35:
            return "北非"
        return "非洲"
    
    # 澳大利亚
    if 110 <= lon <= 155 and -45 <= lat <= -10:
        return "澳大利亚"
    
    # 俄罗斯
    if 30 <= lon <= 180 and 40 <= lat <= 80:
        if 100 <= lon <= 140 and 50 <= lat <= 70:
            return "西伯利亚"
        return "俄罗斯"
    
    # 其他区域
    if -180 <= lon <= -130 and 50 <= lat <= 70:
        return "阿拉斯加"
    elif 140 <= lon <= 180 and 40 <= lat <= 70:
        return "俄罗斯远东"
    elif -60 <= lon <= -30 and 50 <= lat <= 70:
        return "格陵兰"
    
    return "未知区域"


def plot_clusters(
    coords: Dict[str, Tuple[float, float]],
    labels: Dict[str, int],
    output_path: Path,
    title: str,
    show_labels: bool = False,
    point_size: int = 50,
) -> None:
    if not coords:
        raise ValueError("没有可用的坐标数据，无法绘图。")

    font = font_manager.FontProperties(family="Microsoft YaHei")
    label_values = sorted({label for label in labels.values() if label >= 0})
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", max(len(label_values), 1))

    plt.figure(figsize=(12, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("#f4f6fb")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("经度", fontproperties=font)
    ax.set_ylabel("纬度", fontproperties=font)
    ax.set_title(title, fontproperties=font)
    ax.grid(color="#d0d7e3", linestyle="--", linewidth=0.5, alpha=0.7)

    # 绘制所有目标点
    plotted_count = 0
    for target_id, (lon, lat) in coords.items():
        label = labels.get(target_id, -1)
        if label >= 0:
            color = cmap(label % cmap.N)
            ax.scatter(lon, lat, c=[color], s=point_size, alpha=0.9, edgecolors="none")
            plotted_count += 1
            # 可选：显示目标ID标签
            if show_labels:
                ax.text(lon, lat, target_id, fontsize=6, ha="left", va="bottom", 
                       fontproperties=font, alpha=0.7)
        else:
            ax.scatter(lon, lat, c=["#bbbbbb"], s=point_size * 0.6, alpha=0.6, edgecolors="none")
            plotted_count += 1
            if show_labels:
                ax.text(lon, lat, target_id, fontsize=6, ha="left", va="bottom", 
                       fontproperties=font, alpha=0.5)
    
    print(f"[DEBUG] 实际绘制了 {plotted_count} 个目标点")

    # 计算每个簇的地理位置名称
    print(f"[DEBUG] 簇标签值: {sorted(label_values)}")
    cluster_names: Dict[int, str] = {}
    for cluster_label in label_values:
        cluster_points = [
            coords[tid]
            for tid, lab in labels.items()
            if lab == cluster_label and tid in coords
        ]
        if not cluster_points:
            continue
        # 计算簇的中心坐标
        avg_lon = sum(p[0] for p in cluster_points) / len(cluster_points)
        avg_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
        # 根据中心坐标获取地理位置名称
        geo_name = _get_geographic_name(avg_lon, avg_lat)
        cluster_names[cluster_label] = geo_name

    # 添加图例，只显示中文地理位置名称
    legend_items = []
    for label in label_values:
        color = cmap(label % cmap.N)
        geo_name = cluster_names.get(label, f"未知区域")
        legend_items.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=geo_name,
                markerfacecolor=color,
                markersize=8,
                markeredgecolor="none",
            )
        )
    if any(l == -1 for l in labels.values()):
        legend_items.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="噪声",
                markerfacecolor="#bbbbbb",
                markersize=6,
                markeredgecolor="none",
            )
        )
    if legend_items:
        print(f"[DEBUG] 图例项数量: {len(legend_items)}")
        ax.legend(
            handles=legend_items,
            loc="upper right",
            fontsize=9,
            framealpha=0.95,
            facecolor="#ffffff",
            edgecolor="#d0d7e3",
            prop=font,
            ncol=1 if len(legend_items) <= 10 else 2,  # 如果簇太多，分两列显示
        )

    # 可选：为簇中心标注地理位置名称
    # cluster_names = {0: "东亚", 1: "南亚", ...}
    # for cluster_label in label_values:
    #     if cluster_label not in cluster_names:
    #         continue
    #     cluster_points = [
    #         coords[tid]
    #         for tid, lab in labels.items()
    #         if lab == cluster_label and tid in coords
    #     ]
    #     if not cluster_points:
    #         continue
    #     avg_lon = sum(p[0] for p in cluster_points) / len(cluster_points)
    #     avg_lat = sum(p[1] for p in cluster_points) / len(cluster_points)
    #     ax.text(
    #         avg_lon,
    #         avg_lat,
    #         cluster_names[cluster_label],
    #         fontproperties=font,
    #         fontsize=9,
    #         ha="center",
    #         va="center",
    #         color="#0d1c2c",
    #         bbox=dict(
    #             boxstyle="round,pad=0.2",
    #             facecolor="#ffffffcc",
    #             edgecolor="#d0d7e3",
    #             linewidth=0.5,
    #         ),
    #     )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor="white")
    plt.close()


def _load_all_recommendations(recommendations_file: Path) -> Tuple[List[Dict], Dict[str, int]]:
    """
    从 recommendations.jsonl 读取所有记录，返回：
    - 所有记录的列表（包含 target_id, spatial_density_label 等）
    - target_id -> spatial_density_label 的映射（用于去重）
    """
    all_records: List[Dict] = []
    labels: Dict[str, int] = {}
    unique_target_ids: set = set()
    
    if not recommendations_file.exists():
        raise FileNotFoundError(f"推荐文件不存在: {recommendations_file}")
    
    with recommendations_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            target_id = record.get("target_id")
            spatial_label = record.get("spatial_density_label")
            if target_id is not None and spatial_label is not None:
                all_records.append(record)
                unique_target_ids.add(target_id)
                # 如果同一个 target_id 出现多次，保留最后一次的标签
                labels[target_id] = int(spatial_label)
    
    return all_records, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据聚类结果绘制世界地图散点图")
    parser.add_argument("--recommendations-file", type=Path, help="从 recommendations.jsonl 读取标签（推荐）")
    parser.add_argument("--num-targets", type=int, default=400, help="生成的目标数量（用于获取坐标）")
    parser.add_argument("--num-missions", type=int, default=2000, help="生成的任务数量（仅在不使用 --recommendations-file 时使用）")
    parser.add_argument("--enable-rf", action="store_true", help="启用随机森林用户")
    parser.add_argument("--cluster-spread-deg", type=float, default=0.05, help="簇内经纬度扰动幅度（度）")
    parser.add_argument("--spatial-eps-km", type=float, default=500.0, help="DBSCAN 邻域半径（公里）")
    parser.add_argument("--spatial-min-samples", type=int, default=4, help="DBSCAN 最小样本数")
    parser.add_argument("--spatial-min-clusters", type=int, default=7, help="期望至少的簇数量")
    parser.add_argument("--disable-spatial-auto-tune", action="store_true", help="关闭聚类自动调参（不推荐）")
    parser.add_argument("--output", type=Path, default=Path("outputs/spatial_clusters.png"), help="输出图片路径")
    parser.add_argument("--show-labels", action="store_true", help="显示目标ID标签（目标多时可能拥挤）")
    parser.add_argument("--point-size", type=int, default=2, help="散点图点的大小（默认2，可增大以更好区分）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.recommendations_file:
        # 从 recommendations.jsonl 读取聚类标签（spatial_density_label）
        print(f"[INFO] 从 {args.recommendations_file} 读取聚类标签...")
        all_records, labels = _load_all_recommendations(args.recommendations_file)
        unique_target_ids = set(labels.keys())
        print(f"[OK] 读取了 {len(all_records)} 条记录，共 {len(unique_target_ids)} 个唯一目标")
        
        # 从 target_id 中提取最大编号，确保生成足够的目标
        max_target_num = 0
        for tid in unique_target_ids:
            if tid.startswith("TGT"):
                try:
                    num = int(tid[3:])
                    max_target_num = max(max_target_num, num)
                except ValueError:
                    pass
        
        # 生成足够数量的目标（至少是最大编号，或者用户指定的数量）
        num_targets_to_generate = max(max_target_num, args.num_targets, len(unique_target_ids))
        print(f"[INFO] 生成 {num_targets_to_generate} 个目标以获取坐标...")
        targets, _ = generate_smart_data(
            num_targets=num_targets_to_generate,
            num_missions=0,  # 不生成任务
            enable_rf_users=args.enable_rf,
            cluster_spread_deg=max(0.005, args.cluster_spread_deg),
        )
        coords = _extract_lon_lat_pairs(targets)
        
        # 只保留在 labels 中存在的目标
        coords = {tid: coord for tid, coord in coords.items() if tid in labels}
        targets_filtered = [t for t in targets if t.target_id in coords]
        
        print(f"[INFO] 成功获取 {len(coords)} 个目标的坐标")
        
        # 使用新生成的坐标重新计算聚类，使标签与当前坐标分布一致
        print(f"[INFO] 基于新坐标重新计算空间聚类...")
        labels = compute_spatial_density_labels(
            targets_filtered,
            eps_km=args.spatial_eps_km,
            min_samples=max(1, args.spatial_min_samples),
            auto_tune=not args.disable_spatial_auto_tune,
            desired_min_clusters=max(1, args.spatial_min_clusters),
        )
        print(f"[INFO] 聚类计算完成，标签与当前坐标分布一致")
        
        # 更新 recommendations.jsonl 中的 spatial_density_label
        print(f"[INFO] 更新 {args.recommendations_file} 中的 spatial_density_label...")
        updated_count = 0
        for record in all_records:
            target_id = record.get("target_id")
            if target_id and target_id in labels:
                old_label = record.get("spatial_density_label")
                new_label = labels[target_id]
                if old_label != new_label:
                    record["spatial_density_label"] = new_label
                    updated_count += 1
        
        # 写回文件
        output_path = args.recommendations_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in all_records:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        print(f"[OK] 已更新 {updated_count} 条记录的 spatial_density_label，文件已保存")
        
        print(f"[INFO] 将在世界地图上绘制 {len(coords)} 个点，每个点代表一个目标")
        
        # 统计簇数量
        unique_clusters = {label for label in labels.values() if label >= 0}
        title = f"目标空间聚类分布图（目标数={len(coords)}, 簇数={len(unique_clusters)}）"
    else:
        # 原来的方式：重新生成数据并计算聚类
        targets, missions = generate_smart_data(
            num_targets=args.num_targets,
            num_missions=args.num_missions,
            enable_rf_users=args.enable_rf,
            cluster_spread_deg=max(0.005, args.cluster_spread_deg),
        )

        labels = compute_spatial_density_labels(
            targets,
            eps_km=args.spatial_eps_km,
            min_samples=max(1, args.spatial_min_samples),
            auto_tune=not args.disable_spatial_auto_tune,
            desired_min_clusters=max(1, args.spatial_min_clusters),
        )

        coords = _extract_lon_lat_pairs(targets)
        unique_clusters = {label for label in labels.values() if label >= 0}
        title = f"目标空间聚类（targets={len(targets)}, clusters≥{args.spatial_min_clusters}）"

    plot_clusters(
        coords,
        labels,
        output_path=args.output,
        title=title,
        show_labels=args.show_labels,
        point_size=args.point_size,
    )
    print(f"[OK] 已保存空间聚类散点图：{args.output}")
    print(f"[提示] 如果点看起来重叠，可以尝试：")
    print(f"  1. 增大 --point-size 参数（当前：{args.point_size}）")
    print(f"  2. 增大 --cluster-spread-deg 参数（当前：{args.cluster_spread_deg}）")
    print(f"  3. 使用 --show-labels 显示目标ID标签")


if __name__ == "__main__":
    main()


