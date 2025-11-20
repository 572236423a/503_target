#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版智能数据生成器
支持从小规模(100条)到超大规模(500,000+条)的灵活数据生成
包含智能用户分配策略以测试不同算法选择场景
"""

import random
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from src.models.mission import Mission
from src.models.target_info import TargetInfo, Group, Trajectory


def generate_target_info(
    num_targets: int,
    cluster_centers: Optional[List[Tuple[float, float]]] = None,
    cluster_spread_deg: float = 0.05,
) -> List[TargetInfo]:
    """
    生成目标信息数据
    :param num_targets: 生成目标数量
    :param cluster_centers: 预设簇中心列表 (lon, lat)，若为 None 则使用默认5簇
    :param cluster_spread_deg: 每个簇内的经纬度扰动幅度（单位：度）
    :return: 目标信息列表
    """
    target_info = []

    if cluster_centers is None or len(cluster_centers) == 0:
        # 默认构造 8 个分布更广的坐标簇，覆盖全球多个区域
        cluster_centers = [
            (118.0, 24.0),    # 东亚
            (77.0, 28.0),     # 南亚
            (55.0, 25.0),     # 中东
            (2.0, 48.0),      # 西欧
            (-3.0, 40.0),     # 南欧
            (-74.0, 40.7),    # 北美东海岸
            (-122.0, 37.8),   # 北美西海岸
            (151.0, -33.8),   # 澳大利亚东海岸
        ]
    
    # 根据目标数量选择不同的数据丰富度
    if num_targets <= 10:
        # 小规模：基础类型
        target_types = ["军事基地", "港口", "机场", "通信设施", "工业设施"]
        target_categories = ["重要目标", "次要目标", "一般目标"]
        area_types = ["城区", "郊区", "山区", "沿海", "内陆"]
        sources = ["电子侦察", "光学侦察", "雷达侦察"]
        statuses = ["活跃", "待命", "维护"]
    else:
        # 大规模：扩展类型
        target_types = ["军事基地", "港口", "机场", "通信设施", "工业设施", "雷达站", "指挥中心", "导弹基地", "核设施"]
        target_categories = ["重要目标", "次要目标", "一般目标", "关键目标", "战略目标"]
        area_types = ["城区", "郊区", "山区", "沿海", "内陆", "边境", "岛屿", "沙漠", "高原"]
        sources = ["电子侦察", "光学侦察", "雷达侦察", "红外侦察", "通信侦察", "信号情报"]
        statuses = ["活跃", "待命", "维护", "升级", "测试"]
    
    for i in range(num_targets):
        center_lon, center_lat = random.choice(cluster_centers)
        lon = center_lon + random.uniform(-cluster_spread_deg, cluster_spread_deg)
        lat = center_lat + random.uniform(-cluster_spread_deg, cluster_spread_deg)

        target = TargetInfo(
            target_id=f"TGT{i+1:03d}",
            target_name=f"目标{i+1}",
            target_type=random.choice(target_types),
            target_category=random.choice(target_categories),
            target_priority=round(random.uniform(0.1, 1.0), 1),
            target_area_type=random.choice(area_types),
            group_list=[
                Group(
                    group_name=f"技术组{chr(65+(i%26))}",
                    source=random.choice(sources),
                    status=random.choice(statuses)
                )
            ],
            trajectory_list=[
                Trajectory(
                    lon=str(round(lon, 4)),
                    lat=str(round(lat, 4)),
                    alt=str(random.randint(10, 200)),
                    point_time=f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d} {random.randint(0,23):02d}:00:00",
                    speed=str(random.randint(10, 80)),
                    heading=str(random.randint(0, 359)),
                    seq=str(i+1),
                    elect_silence=random.choice(["是", "否"])
                )
            ]
        )
        target_info.append(target)
    
    return target_info


def generate_smart_data(
    num_targets: int = 2,
    num_missions: int = 100,
    enable_rf_users: bool = False,
    cluster_centers: Optional[List[Tuple[float, float]]] = None,
    cluster_spread_deg: float = 0.05,
) -> Tuple[List[TargetInfo], List[Mission]]:
    """
    智能数据生成器 - 支持小规模到超大规模的灵活生成
    :param num_targets: 目标数量
    :param num_missions: 任务数量
    :param enable_rf_users: 是否启用随机森林用户（创建>5000任务的用户）
    :param cluster_centers: 目标空间簇中心 (lon, lat) 列表
    :param cluster_spread_deg: 每簇扰动范围
    :return: (目标信息列表, 任务列表)
    """
    scale = "超大规模" if num_missions >= 100000 else "大规模" if num_missions >= 10000 else "中规模" if num_missions >= 1000 else "小规模"
    print(f"=== 生成{scale}数据 ({num_missions:,}条) ===\n")
    
    if num_missions >= 10000:
        print("[INFO] 开始生成数据，这可能需要几分钟时间...")
    else:
        print("[INFO] 开始生成数据...")
    
    start_time = time.time()
    
    # 生成目标信息
    print(f"[STEP] 生成目标信息 ({num_targets}个)...")
    target_info = generate_target_info(
        num_targets,
        cluster_centers=cluster_centers,
        cluster_spread_deg=cluster_spread_deg,
    )
    print(f"[OK] 生成了 {len(target_info)} 个目标信息")
    
    # 定义基础数据
    if num_missions <= 1000:
        # 小规模：基础配置
        units = ["第一情报部", "第二技术部", "第三作战部", "第四指挥部", "第五后勤部"]
        groups = ["华北区组", "华东区组", "华南区组", "华西区组", "东北区组", "西北区组"]
        scout_types = ["电子侦察", "光学侦察", "雷达侦察", "通信侦察", "红外侦察", "多光谱侦察"]
        countries = ["目标国A", "目标国B", "目标国C", "目标国D", "目标国E", "目标国F"]
        task_types = ["1", "2", "3", "4", "5"]
        task_scenes = ["海上", "陆地", "空中", "太空", "网络"]
        req_cycles = ["一周", "两周", "三周", "一个月", "三个月", "半年", "一年"]
        mission_play_types = ["自动筹划", "半自动筹划", "人工筹划"]
    else:
        # 大规模：扩展配置
        units = ["第一情报部", "第二技术部", "第三作战部", "第四指挥部", "第五后勤部", "第六通信部", "第七装备部"]
        groups = ["华北区组", "华东区组", "华南区组", "华西区组", "东北区组", "西北区组", "华中区组", "西南区组"]
        scout_types = ["电子侦察", "光学侦察", "雷达侦察", "通信侦察", "红外侦察", "多光谱侦察", "合成孔径雷达", "信号情报"]
        countries = ["目标国A", "目标国B", "目标国C", "目标国D", "目标国E", "目标国F", "目标国G", "目标国H"]
        task_types = ["1", "2", "3", "4", "5"]
        task_scenes = ["海上", "陆地", "空中", "太空", "网络", "联合", "多域"]
        req_cycles = ["一周", "两周", "三周", "一个月", "两个月", "三个月", "半年", "一年", "两年"]
        mission_play_types = ["自动筹划", "半自动筹划", "人工筹划", "智能筹划"]
    
    emcon_options = ["是", "否"]
    
    # 智能用户分配策略
    print("[STEP] 设计用户任务分配方案...")
    user_allocation = []
    
    if enable_rf_users and num_missions >= 10000:
        # 大规模数据：创建超高频用户以触发随机森林
        print("   启用随机森林用户模式")
        
        if num_missions >= 100000:
            # 超大规模：多个超高频用户
            user_allocation.extend([
                ("第一情报部", "华北区组", min(50000, num_missions // 10)),
                ("第二技术部", "华东区组", min(40000, num_missions // 12)),
                ("第一情报部", "华南区组", min(30000, num_missions // 16)),
                ("第二技术部", "华南区组", min(25000, num_missions // 20)),
                ("第三作战部", "华北区组", min(20000, num_missions // 25)),
            ])
            
            # 高频用户
            user_allocation.extend([
                ("第一情报部", "华东区组", min(8000, num_missions // 60)),
                ("第二技术部", "华北区组", min(7500, num_missions // 65)),
                ("第四指挥部", "华南区组", min(7000, num_missions // 70)),
                ("第五后勤部", "华东区组", min(6500, num_missions // 75)),
                ("第三作战部", "华南区组", min(6000, num_missions // 80)),
            ])
        else:
            # 大规模：少量超高频用户
            user_allocation.extend([
                ("第一情报部", "华北区组", min(6000, num_missions // 3)),
                ("第二技术部", "华东区组", min(5500, num_missions // 3)),
            ])
    
    # 分配剩余任务给其他用户
    allocated_tasks = sum(allocation[2] for allocation in user_allocation)
    remaining_tasks = num_missions - allocated_tasks
    
    # 创建剩余用户列表
    remaining_users = []
    for unit in units:
        for group in groups:
            user_key = (unit, group)
            if user_key not in [(u[0], u[1]) for u in user_allocation]:
                remaining_users.append(user_key)
    
    # 为剩余用户分配任务
    if remaining_users and remaining_tasks > 0:
        if num_missions <= 1000:
            # 小规模：均匀分配
            avg_tasks = remaining_tasks // len(remaining_users)
            for i, (unit, group) in enumerate(remaining_users):
                if i == len(remaining_users) - 1:
                    tasks = remaining_tasks - avg_tasks * (len(remaining_users) - 1)
                else:
                    tasks = avg_tasks + random.randint(-10, 10)
                user_allocation.append((unit, group, max(1, tasks)))
        else:
            # 大规模：随机分配
            for i, (unit, group) in enumerate(remaining_users):
                if i == len(remaining_users) - 1:
                    tasks = remaining_tasks - sum(allocation[2] for allocation in user_allocation[len(user_allocation):])
                else:
                    max_tasks = max(10, min(4000, remaining_tasks // (len(remaining_users) - i)))
                    min_tasks = min(100, max_tasks)
                    if max_tasks < min_tasks:
                        min_tasks = max_tasks
                    tasks = random.randint(min_tasks, max_tasks)
                    remaining_tasks -= tasks
                user_allocation.append((unit, group, max(10, tasks)))
    
    # 显示分配统计
    super_users = sum(1 for _, _, count in user_allocation if count > 10000)
    high_users = sum(1 for _, _, count in user_allocation if 5000 < count <= 10000)
    rf_users = sum(1 for _, _, count in user_allocation if count > 5000)
    
    print(f"[STAT] 用户分配统计:")
    print(f"   - 总用户数: {len(user_allocation)}")
    if super_users > 0:
        print(f"   - 超高频用户 (>10000): {super_users} 个")
    if high_users > 0:
        print(f"   - 高频用户 (5000-10000): {high_users} 个")
    print(f"   - 将使用随机森林的用户: {rf_users} 个")
    print(f"   - 将使用决策树的用户: {len(user_allocation) - rf_users} 个")
    
    # 显示最活跃用户
    top_users = sorted(user_allocation, key=lambda x: x[2], reverse=True)[:min(10, len(user_allocation))]
    print(f"\n[TOP] 最活跃用户 (Top {len(top_users)}):")
    for i, (unit, group, count) in enumerate(top_users, 1):
        algo = "RF(>5000)" if count > 5000 else "DT(<=5000)"
        print(f"   {i:2d}. {unit}_{group}: {count:,} 条任务 → {algo}")
    
    # 生成任务数据
    print(f"\n[STEP] 开始生成 {num_missions:,} 条任务数据...")
    missions = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    batch_size = max(1000, num_missions // 100)  # 动态批次大小
    total_generated = 0
    
    for unit, group, task_count in user_allocation:
        if num_missions >= 10000:
            print(f"   生成 {unit}_{group} 的 {task_count:,} 条任务...")
        
        for i in range(task_count):
            # 生成时间（分布在一年内）
            days_offset = random.randint(0, 365)
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            req_time = base_time + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
            
            # 生成新字段数据
            req_cycle_val = random.choice(req_cycles)
            cycle_time = random.randint(1, 3)
            req_times_val = random.randint(1, 10)
            
            mission = Mission(
                req_id=f"REQ{len(missions)+1:06d}",
                topic_id=f"TP{len(missions)+1:06d}",
                req_unit=unit,
                req_group=group,
                req_start_time=req_time.strftime("%Y-%m-%d %H:%M:%S"),
                req_end_time=(req_time + timedelta(hours=random.randint(1, 24))).strftime("%Y-%m-%d %H:%M:%S"),
                task_type=random.choice(task_types),
                target_id=f"TGT{random.randint(1, num_targets):03d}",
                country_name=random.choice(countries),
                target_priority=round(random.uniform(0.1, 1.0), 1),
                is_emcon=random.choice(emcon_options),
                is_precise=random.choice([True, False]),
                scout_type=random.choice(scout_types),
                task_scene=random.choice(task_scenes),
                resolution=round(random.uniform(0.5, 1.0), 2),
                req_cycle=req_cycle_val,
                req_cycle_time=str(cycle_time),
                req_times=req_times_val,
                mission_play_type=random.choice(mission_play_types)
            )
            missions.append(mission)
            total_generated += 1
            
            # 显示进度（仅大规模数据）
            if num_missions >= 10000 and total_generated % batch_size == 0:
                elapsed = time.time() - start_time
                progress = (total_generated / num_missions) * 100
                print(f"     进度: {total_generated:,}/{num_missions:,} ({progress:.1f}%) - 用时: {elapsed:.1f}秒")
    
    elapsed_time = time.time() - start_time
    print(f"\n[OK] 数据生成完成！")
    print(f"   - 总计: {len(missions):,} 条任务")
    print(f"   - 用时: {elapsed_time:.1f} 秒")
    if elapsed_time > 0:
        print(f"   - 速度: {len(missions)/elapsed_time:.0f} 条/秒")
    
    return target_info, missions


def save_data_to_files(target_info: List[TargetInfo], missions: List[Mission], 
                      target_file: str = "targets.txt", 
                      mission_file: str = "missions.txt"):
    """
    保存数据到文件
    :param target_info: 目标信息列表
    :param missions: 任务列表
    :param target_file: 目标信息文件名
    :param mission_file: 任务信息文件名
    """
    print(f"\n[STEP] 保存数据到文件...")
    save_start = time.time()
    
    # 保存目标信息
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write("目标ID\t目标名称\t目标类型\t目标种类\t目标优先级\t区域类型\n")
        for target in target_info:
            f.write(f"{target.target_id}\t{target.target_name}\t{target.target_type}\t"
                   f"{target.target_category}\t{target.target_priority}\t{target.target_area_type}\n")
    
    # 保存任务信息
    with open(mission_file, 'w', encoding='utf-8') as f:
        f.write("需求ID\t专题ID\t部门\t区组\t开始时间\t结束时间\t任务类型\t目标ID\t"
               f"国家\t优先级\t电磁管制\t是否精确\t侦察类型\t任务场景\t分辨率\t"
               f"需求周期\t周期次数\t需求次数\t筹划方式\n")
        
        if len(missions) >= 10000:
            # 大数据：批量写入
            batch_size = 50000
            for i in range(0, len(missions), batch_size):
                batch = missions[i:i+batch_size]
                for mission in batch:
                    f.write(f"{mission.req_id}\t{mission.topic_id}\t{mission.req_unit}\t"
                           f"{mission.req_group}\t{mission.req_start_time}\t{mission.req_end_time}\t"
                           f"{mission.task_type}\t{mission.target_id}\t{mission.country_name}\t"
                           f"{mission.target_priority}\t{mission.is_emcon}\t{mission.is_precise}\t"
                           f"{mission.scout_type}\t{mission.task_scene}\t{mission.resolution}\t"
                           f"{mission.req_cycle}\t{mission.req_cycle_time}\t{mission.req_times}\t"
                           f"{mission.mission_play_type}\n")
                
                progress = ((i + len(batch)) / len(missions)) * 100
                if len(missions) >= 50000:
                    print(f"   保存进度: {progress:.1f}%")
        else:
            # 小数据：直接写入
            for mission in missions:
                f.write(f"{mission.req_id}\t{mission.topic_id}\t{mission.req_unit}\t"
                       f"{mission.req_group}\t{mission.req_start_time}\t{mission.req_end_time}\t"
                       f"{mission.task_type}\t{mission.target_id}\t{mission.country_name}\t"
                       f"{mission.target_priority}\t{mission.is_emcon}\t{mission.is_precise}\t"
                       f"{mission.scout_type}\t{mission.task_scene}\t{mission.resolution}\t"
                       f"{mission.req_cycle}\t{mission.req_cycle_time}\t{mission.req_times}\t"
                       f"{mission.mission_play_type}\n")
    
    save_time = time.time() - save_start
    print(f"[OK] 文件保存完成！用时: {save_time:.1f} 秒")
    print(f"   - 目标信息: {target_file}")
    print(f"   - 任务信息: {mission_file}")


def print_data_statistics(target_info: List[TargetInfo], missions: List[Mission]):
    """
    打印数据统计信息
    :param target_info: 目标信息列表
    :param missions: 任务列表
    """
    print(f"\n📊 数据统计分析:")
    print(f"   - 目标信息: {len(target_info):,} 个")
    print(f"   - 历史需求: {len(missions):,} 条")
    
    # 统计用户分布
    user_stats = {}
    for mission in missions:
        user_key = f"{mission.req_unit}_{mission.req_group}"
        user_stats[user_key] = user_stats.get(user_key, 0) + 1
    
    print(f"   - 涉及用户: {len(user_stats):,} 个")
    print(f"   - 平均每用户任务数: {len(missions) / len(user_stats):,.1f} 条")
    
    # 算法选择统计
    rf_users = [(user, count) for user, count in user_stats.items() if count > 5000]
    dt_users = [(user, count) for user, count in user_stats.items() if count <= 5000]
    
    print(f"\n🤖 算法选择预测:")
    print(f"   - 随机森林用户: {len(rf_users):,} 个")
    print(f"   - 决策树用户: {len(dt_users):,} 个")
    
    # 显示最活跃的用户
    top_users = sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n🏆 最活跃用户 (Top {len(top_users)}):")
    for i, (user, count) in enumerate(top_users, 1):
        algo = "RF" if count > 5000 else "DT"
        print(f"   {i:2d}. {user}: {count:,} 条任务 {algo}")
    
    # 部门统计
    unit_stats = {}
    for mission in missions:
        unit_stats[mission.req_unit] = unit_stats.get(mission.req_unit, 0) + 1
    
    print(f"\n🏢 部门分布:")
    for unit, count in sorted(unit_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(missions)) * 100
        print(f"   - {unit}: {count:,} 条 ({percentage:.1f}%)")


# 兼容性函数 - 保持与原有代码的兼容性
def generate_sample_data(num_targets: int = 2, num_missions: int = 100) -> Tuple[List[TargetInfo], List[Mission]]:
    """
    兼容性函数 - 生成基础示例数据
    :param num_targets: 目标数量
    :param num_missions: 任务数量
    :return: (目标信息列表, 任务列表)
    """
    return generate_smart_data(num_targets, num_missions, enable_rf_users=False)


def generate_500k_data() -> Tuple[List[TargetInfo], List[Mission]]:
    """
    兼容性函数 - 生成500K数据
    :return: (目标信息列表, 任务列表)
    """
    return generate_smart_data(num_targets=50, num_missions=500000, enable_rf_users=True)


def main():
    """主函数 - 演示不同规模的数据生成"""
    print("=== 增强版智能数据生成器 ===\n")
    print("支持的生成模式:")
    print("1. 小规模数据 (100条) - 测试基础功能")
    print("2. 中规模数据 (1,000条) - 测试中等负载")
    print("3. 大规模数据 (10,000条) - 测试高负载")
    print("4. 超大规模数据 (500,000条) - 测试极限性能")
    print()
    
    # 默认生成500K数据
    target_info, missions = generate_smart_data(
        num_targets=50, 
        num_missions=500000, 
        enable_rf_users=True
    )
    
    # 统计分析
    print_data_statistics(target_info, missions)
    
    # 保存数据
    save_data_to_files(target_info, missions, "targets_500k.txt", "missions_500k.txt")
    
    print(f"\n🎉 数据生成完成！现在可以测试用户画像算法了！")


if __name__ == "__main__":
    main()



