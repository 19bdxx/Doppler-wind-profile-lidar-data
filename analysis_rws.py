#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多普勒风廓线激光雷达 RWS (径向风速) 分析脚本
==========================================

本脚本提供基于 RWS (Radial Wind Speed) 的完整分析功能：
1. 单角度组合分析（Azimuth + Elevation）
2. 多角度组合对比分析
3. CNR 质量控制
4. 完整的可视化输出

作者：GitHub Copilot
日期：2025年10月
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 忽略警告
warnings.filterwarnings('ignore')

# ===========================
# 一、数据加载与预处理
# ===========================

def load_lidar_data(file_path):
    """
    加载激光雷达数据
    
    参数:
        file_path: CSV 文件路径
        
    返回:
        DataFrame: 加载的数据
    """
    print(f"正在加载数据: {file_path}")
    
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 转换时间戳
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    print(f"数据加载完成！共 {len(df)} 行数据")
    print(f"时间范围: {df['Timestamp'].min()} 到 {df['Timestamp'].max()}")
    
    return df


def get_data_overview(df):
    """
    获取数据概览
    
    参数:
        df: DataFrame
    """
    print("\n" + "="*60)
    print("数据概览")
    print("="*60)
    
    # 基本统计
    print(f"总数据点数: {len(df)}")
    print(f"唯一时间戳数: {df['Timestamp'].nunique()}")
    
    # 角度信息
    print(f"\n方位角 (Azimuth) 范围: {df['Azimuth(deg)'].min():.3f}° ~ {df['Azimuth(deg)'].max():.3f}°")
    print(f"唯一方位角数: {df['Azimuth(deg)'].nunique()}")
    
    print(f"\n仰角 (Elevation) 范围: {df['Elevation(deg)'].min():.3f}° ~ {df['Elevation(deg)'].max():.3f}°")
    print(f"唯一仰角数: {df['Elevation(deg)'].nunique()}")
    
    # 距离信息
    print(f"\n距离 (Distance) 范围: {df['Distance(m)'].min():.1f} m ~ {df['Distance(m)'].max():.1f} m")
    print(f"距离门数量: {df['Distance(m)'].nunique()}")
    
    # RWS 信息
    print(f"\nRWS 范围: {df['RWS(m/s)'].min():.3f} m/s ~ {df['RWS(m/s)'].max():.3f} m/s")
    print(f"RWS 均值: {df['RWS(m/s)'].mean():.3f} m/s")
    print(f"RWS 标准差: {df['RWS(m/s)'].std():.3f} m/s")
    
    # CNR 信息
    print(f"\nCNR 范围: {df['CNR(dB)'].min():.3f} dB ~ {df['CNR(dB)'].max():.3f} dB")
    print(f"CNR 均值: {df['CNR(dB)'].mean():.3f} dB")
    
    # 角度组合
    df['AngleComb'] = df['Azimuth(deg)'].round(2).astype(str) + '_' + df['Elevation(deg)'].round(2).astype(str)
    print(f"\n唯一角度组合数: {df['AngleComb'].nunique()}")
    
    return df


def apply_cnr_filter(df, cnr_threshold=-20):
    """
    应用 CNR 阈值筛选
    
    参数:
        df: DataFrame
        cnr_threshold: CNR 阈值 (dB)，默认 -20 dB
        
    返回:
        DataFrame: 筛选后的数据
    """
    print(f"\n应用 CNR 筛选，阈值: {cnr_threshold} dB")
    
    before_count = len(df)
    df_filtered = df[df['CNR(dB)'] >= cnr_threshold].copy()
    after_count = len(df_filtered)
    
    print(f"筛选前数据量: {before_count}")
    print(f"筛选后数据量: {after_count}")
    print(f"保留比例: {after_count/before_count*100:.2f}%")
    
    return df_filtered


# ===========================
# 二、单角度组合分析
# ===========================

def single_angle_statistics(df, azimuth, elevation):
    """
    计算单个角度组合的统计指标
    
    参数:
        df: DataFrame
        azimuth: 方位角
        elevation: 仰角
        
    返回:
        dict: 统计结果
    """
    # 筛选特定角度组合的数据
    mask = (np.abs(df['Azimuth(deg)'] - azimuth) < 0.1) & \
           (np.abs(df['Elevation(deg)'] - elevation) < 0.1)
    data = df[mask]['RWS(m/s)']
    
    if len(data) == 0:
        print(f"警告: 未找到角度组合 ({azimuth}°, {elevation}°) 的数据")
        return None
    
    # 计算统计量
    stats = {
        '数据点数': len(data),
        '均值': data.mean(),
        '中位数': data.median(),
        '标准差': data.std(),
        '最小值': data.min(),
        '最大值': data.max(),
        'P10分位数': data.quantile(0.1),
        'P50分位数': data.quantile(0.5),
        'P90分位数': data.quantile(0.9)
    }
    
    return stats


def print_angle_statistics(stats, azimuth, elevation):
    """
    打印角度统计结果
    """
    print("\n" + "="*60)
    print(f"角度组合: 方位角={azimuth}°, 仰角={elevation}°")
    print("="*60)
    
    if stats is None:
        print("无数据")
        return
    
    for key, value in stats.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.3f}")


def analyze_rws_by_distance(df, azimuth, elevation, output_dir='output'):
    """
    分析 RWS 随距离的变化趋势
    
    参数:
        df: DataFrame
        azimuth: 方位角
        elevation: 仰角
        output_dir: 输出目录
    """
    # 筛选数据
    mask = (np.abs(df['Azimuth(deg)'] - azimuth) < 0.1) & \
           (np.abs(df['Elevation(deg)'] - elevation) < 0.1)
    data = df[mask].copy()
    
    if len(data) == 0:
        print(f"警告: 未找到角度组合 ({azimuth}°, {elevation}°) 的数据")
        return
    
    # 按距离分组统计
    distance_stats = data.groupby('Distance(m)')['RWS(m/s)'].agg([
        ('均值', 'mean'),
        ('标准差', 'std'),
        ('最小值', 'min'),
        ('最大值', 'max'),
        ('数据点数', 'count')
    ]).reset_index()
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 子图1: RWS 均值随距离变化
    ax1 = axes[0]
    ax1.plot(distance_stats['Distance(m)'], distance_stats['均值'], 
             'b-', linewidth=2, label='均值')
    ax1.fill_between(distance_stats['Distance(m)'],
                      distance_stats['均值'] - distance_stats['标准差'],
                      distance_stats['均值'] + distance_stats['标准差'],
                      alpha=0.3, label='±1标准差')
    ax1.set_xlabel('距离 (m)', fontsize=12)
    ax1.set_ylabel('RWS (m/s)', fontsize=12)
    ax1.set_title(f'RWS 随距离变化 - 方位角={azimuth}°, 仰角={elevation}°', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 子图2: RWS 范围（最小值-最大值）
    ax2 = axes[1]
    ax2.fill_between(distance_stats['Distance(m)'],
                      distance_stats['最小值'],
                      distance_stats['最大值'],
                      alpha=0.5, color='green', label='最小值-最大值范围')
    ax2.plot(distance_stats['Distance(m)'], distance_stats['均值'],
             'r-', linewidth=2, label='均值')
    ax2.set_xlabel('距离 (m)', fontsize=12)
    ax2.set_ylabel('RWS (m/s)', fontsize=12)
    ax2.set_title(f'RWS 变化范围 - 方位角={azimuth}°, 仰角={elevation}°', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/rws_distance_az{azimuth:.1f}_el{elevation:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()
    
    return distance_stats


def plot_rws_distribution(df, azimuth, elevation, output_dir='output'):
    """
    绘制 RWS 分布特征（直方图、箱线图、分位数曲线）
    
    参数:
        df: DataFrame
        azimuth: 方位角
        elevation: 仰角
        output_dir: 输出目录
    """
    # 筛选数据
    mask = (np.abs(df['Azimuth(deg)'] - azimuth) < 0.1) & \
           (np.abs(df['Elevation(deg)'] - elevation) < 0.1)
    data = df[mask]['RWS(m/s)'].dropna()
    
    if len(data) == 0:
        print(f"警告: 未找到角度组合 ({azimuth}°, {elevation}°) 的数据")
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1: 直方图
    ax1 = axes[0]
    ax1.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'均值: {data.mean():.2f}')
    ax1.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'中位数: {data.median():.2f}')
    ax1.set_xlabel('RWS (m/s)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('RWS 直方图分布', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 箱线图
    ax2 = axes[1]
    bp = ax2.boxplot([data], vert=True, patch_artist=True, 
                      widths=0.5, showmeans=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    ax2.set_ylabel('RWS (m/s)', fontsize=12)
    ax2.set_title('RWS 箱线图', fontsize=14, fontweight='bold')
    ax2.set_xticklabels([f'Az={azimuth}°\nEl={elevation}°'])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 分位数曲线
    ax3 = axes[2]
    quantiles = np.linspace(0, 1, 101)
    quantile_values = [data.quantile(q) for q in quantiles]
    ax3.plot(quantiles * 100, quantile_values, 'b-', linewidth=2)
    ax3.axhline(data.quantile(0.1), color='orange', linestyle='--', linewidth=1.5, 
                label=f'P10: {data.quantile(0.1):.2f}')
    ax3.axhline(data.quantile(0.5), color='green', linestyle='--', linewidth=1.5,
                label=f'P50: {data.quantile(0.5):.2f}')
    ax3.axhline(data.quantile(0.9), color='red', linestyle='--', linewidth=1.5,
                label=f'P90: {data.quantile(0.9):.2f}')
    ax3.set_xlabel('分位数 (%)', fontsize=12)
    ax3.set_ylabel('RWS (m/s)', fontsize=12)
    ax3.set_title('RWS 分位数曲线', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(f'RWS 分布特征 - 方位角={azimuth}°, 仰角={elevation}°', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/rws_distribution_az{azimuth:.1f}_el{elevation:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


def compare_cnr_filtering(df, azimuth, elevation, cnr_threshold=-20, output_dir='output'):
    """
    对比 CNR 筛选前后的统计差异
    
    参数:
        df: DataFrame
        azimuth: 方位角
        elevation: 仰角
        cnr_threshold: CNR 阈值
        output_dir: 输出目录
    """
    # 筛选角度数据
    mask = (np.abs(df['Azimuth(deg)'] - azimuth) < 0.1) & \
           (np.abs(df['Elevation(deg)'] - elevation) < 0.1)
    data_all = df[mask].copy()
    
    if len(data_all) == 0:
        print(f"警告: 未找到角度组合 ({azimuth}°, {elevation}°) 的数据")
        return
    
    # 应用 CNR 筛选
    data_filtered = data_all[data_all['CNR(dB)'] >= cnr_threshold].copy()
    
    print("\n" + "="*60)
    print(f"CNR 筛选对比 - 方位角={azimuth}°, 仰角={elevation}°")
    print(f"CNR 阈值: {cnr_threshold} dB")
    print("="*60)
    
    # 统计对比
    stats_before = {
        '数据点数': len(data_all),
        'RWS均值': data_all['RWS(m/s)'].mean(),
        'RWS标准差': data_all['RWS(m/s)'].std(),
        'RWS最小值': data_all['RWS(m/s)'].min(),
        'RWS最大值': data_all['RWS(m/s)'].max()
    }
    
    stats_after = {
        '数据点数': len(data_filtered),
        'RWS均值': data_filtered['RWS(m/s)'].mean(),
        'RWS标准差': data_filtered['RWS(m/s)'].std(),
        'RWS最小值': data_filtered['RWS(m/s)'].min(),
        'RWS最大值': data_filtered['RWS(m/s)'].max()
    }
    
    print("\n筛选前:")
    for key, value in stats_before.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.3f}")
    
    print("\n筛选后:")
    for key, value in stats_after.items():
        if isinstance(value, (int, np.integer)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.3f}")
    
    print(f"\n数据保留率: {len(data_filtered)/len(data_all)*100:.2f}%")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 直方图对比
    ax1 = axes[0]
    ax1.hist(data_all['RWS(m/s)'], bins=30, alpha=0.5, color='blue', 
             label=f'筛选前 (n={len(data_all)})', edgecolor='black')
    ax1.hist(data_filtered['RWS(m/s)'], bins=30, alpha=0.5, color='red',
             label=f'筛选后 (n={len(data_filtered)})', edgecolor='black')
    ax1.set_xlabel('RWS (m/s)', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('CNR 筛选前后 RWS 分布对比', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 箱线图对比
    ax2 = axes[1]
    bp = ax2.boxplot([data_all['RWS(m/s)'], data_filtered['RWS(m/s)']], 
                      labels=['筛选前', '筛选后'],
                      patch_artist=True, showmeans=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    ax2.set_ylabel('RWS (m/s)', fontsize=12)
    ax2.set_title('CNR 筛选前后 RWS 箱线图对比', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'CNR 筛选效果对比 - 方位角={azimuth}°, 仰角={elevation}°, 阈值={cnr_threshold}dB',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/cnr_filter_comparison_az{azimuth:.1f}_el{elevation:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


# ===========================
# 三、多角度组合对比分析
# ===========================

def compare_azimuths(df, elevation, output_dir='output'):
    """
    对比不同方位角的 RWS 分布
    
    参数:
        df: DataFrame
        elevation: 固定仰角
        output_dir: 输出目录
    """
    # 筛选特定仰角的数据
    mask = np.abs(df['Elevation(deg)'] - elevation) < 0.1
    data = df[mask].copy()
    
    if len(data) == 0:
        print(f"警告: 未找到仰角 {elevation}° 的数据")
        return
    
    # 获取所有方位角
    azimuths = sorted(data['Azimuth(deg)'].unique())
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 箱线图
    ax1 = axes[0]
    data_list = [data[np.abs(data['Azimuth(deg)'] - az) < 0.1]['RWS(m/s)'].values 
                 for az in azimuths]
    labels = [f'{az:.1f}°' for az in azimuths]
    bp = ax1.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=6))
    ax1.set_xlabel('方位角', fontsize=12)
    ax1.set_ylabel('RWS (m/s)', fontsize=12)
    ax1.set_title(f'不同方位角 RWS 分布对比 (仰角={elevation}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # 子图2: 小提琴图
    ax2 = axes[1]
    # 准备数据用于小提琴图
    plot_data = []
    for az in azimuths:
        az_data = data[np.abs(data['Azimuth(deg)'] - az) < 0.1]['RWS(m/s)'].values
        for val in az_data:
            plot_data.append({'Azimuth': f'{az:.1f}°', 'RWS': val})
    plot_df = pd.DataFrame(plot_data)
    
    sns.violinplot(data=plot_df, x='Azimuth', y='RWS', ax=ax2, palette='Set2')
    ax2.set_xlabel('方位角', fontsize=12)
    ax2.set_ylabel('RWS (m/s)', fontsize=12)
    ax2.set_title(f'不同方位角 RWS 分布（小提琴图）(仰角={elevation}°)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/azimuth_comparison_el{elevation:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


def compare_elevations(df, azimuth, output_dir='output'):
    """
    对比不同仰角的 RWS 分布
    
    参数:
        df: DataFrame
        azimuth: 固定方位角
        output_dir: 输出目录
    """
    # 筛选特定方位角的数据
    mask = np.abs(df['Azimuth(deg)'] - azimuth) < 0.1
    data = df[mask].copy()
    
    if len(data) == 0:
        print(f"警告: 未找到方位角 {azimuth}° 的数据")
        return
    
    # 获取所有仰角
    elevations = sorted(data['Elevation(deg)'].unique())
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1: 箱线图
    ax1 = axes[0]
    data_list = [data[np.abs(data['Elevation(deg)'] - el) < 0.1]['RWS(m/s)'].values 
                 for el in elevations]
    labels = [f'{el:.2f}°' for el in elevations]
    bp = ax1.boxplot(data_list, labels=labels, patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='lightgreen', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='blue', markersize=8))
    ax1.set_xlabel('仰角', fontsize=12)
    ax1.set_ylabel('RWS (m/s)', fontsize=12)
    ax1.set_title(f'不同仰角 RWS 分布对比 (方位角={azimuth}°)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 子图2: 均值和标准差对比
    ax2 = axes[1]
    means = [data[np.abs(data['Elevation(deg)'] - el) < 0.1]['RWS(m/s)'].mean() 
             for el in elevations]
    stds = [data[np.abs(data['Elevation(deg)'] - el) < 0.1]['RWS(m/s)'].std() 
            for el in elevations]
    
    x_pos = np.arange(len(elevations))
    ax2.bar(x_pos, means, yerr=stds, alpha=0.7, color='skyblue', 
            capsize=5, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('仰角', fontsize=12)
    ax2.set_ylabel('RWS (m/s)', fontsize=12)
    ax2.set_title(f'不同仰角 RWS 均值±标准差 (方位角={azimuth}°)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/elevation_comparison_az{azimuth:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


def plot_azimuth_distance_heatmap(df, elevation, output_dir='output'):
    """
    绘制方位角 × 距离的二维热力图
    
    参数:
        df: DataFrame
        elevation: 固定仰角
        output_dir: 输出目录
    """
    # 筛选特定仰角的数据
    mask = np.abs(df['Elevation(deg)'] - elevation) < 0.1
    data = df[mask].copy()
    
    if len(data) == 0:
        print(f"警告: 未找到仰角 {elevation}° 的数据")
        return
    
    # 创建透视表
    pivot_table = data.pivot_table(
        values='RWS(m/s)',
        index='Distance(m)',
        columns='Azimuth(deg)',
        aggfunc='mean'
    )
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    im = ax.imshow(pivot_table.values, aspect='auto', cmap='coolwarm', 
                   origin='lower', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels([f'{az:.1f}' for az in pivot_table.columns], rotation=45)
    ax.set_yticks(np.arange(0, len(pivot_table.index), 20))
    ax.set_yticklabels([f'{pivot_table.index[i]:.0f}' for i in range(0, len(pivot_table.index), 20)])
    
    ax.set_xlabel('方位角 (°)', fontsize=12)
    ax.set_ylabel('距离 (m)', fontsize=12)
    ax.set_title(f'RWS 热力图：方位角 × 距离 (仰角={elevation}°)', 
                 fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RWS (m/s)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/heatmap_azimuth_distance_el{elevation:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


def plot_elevation_distance_heatmap(df, azimuth, output_dir='output'):
    """
    绘制仰角 × 距离的二维热力图
    
    参数:
        df: DataFrame
        azimuth: 固定方位角
        output_dir: 输出目录
    """
    # 筛选特定方位角的数据
    mask = np.abs(df['Azimuth(deg)'] - azimuth) < 0.1
    data = df[mask].copy()
    
    if len(data) == 0:
        print(f"警告: 未找到方位角 {azimuth}° 的数据")
        return
    
    # 创建透视表
    pivot_table = data.pivot_table(
        values='RWS(m/s)',
        index='Distance(m)',
        columns='Elevation(deg)',
        aggfunc='mean'
    )
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(pivot_table.values, aspect='auto', cmap='coolwarm',
                   origin='lower', interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_xticklabels([f'{el:.2f}' for el in pivot_table.columns])
    ax.set_yticks(np.arange(0, len(pivot_table.index), 20))
    ax.set_yticklabels([f'{pivot_table.index[i]:.0f}' for i in range(0, len(pivot_table.index), 20)])
    
    ax.set_xlabel('仰角 (°)', fontsize=12)
    ax.set_ylabel('距离 (m)', fontsize=12)
    ax.set_title(f'RWS 热力图：仰角 × 距离 (方位角={azimuth}°)', 
                 fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RWS (m/s)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/heatmap_elevation_distance_az{azimuth:.1f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


def plot_wind_rose(df, output_dir='output'):
    """
    绘制风速玫瑰图（基于 RWS）
    
    参数:
        df: DataFrame
        output_dir: 输出目录
    """
    # 计算每个方位角的平均 RWS
    azimuth_stats = df.groupby('Azimuth(deg)')['RWS(m/s)'].agg([
        ('均值', 'mean'),
        ('绝对值均值', lambda x: np.abs(x).mean())
    ]).reset_index()
    
    # 转换为极坐标（方位角需要转换为弧度）
    theta = np.deg2rad(azimuth_stats['Azimuth(deg)'])
    r = azimuth_stats['绝对值均值']
    
    # 创建极坐标图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # 设置0度在正北方向
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)  # 顺时针
    
    # 绘制玫瑰图
    bars = ax.bar(theta, r, width=np.deg2rad(5), 
                  color='skyblue', alpha=0.7, edgecolor='blue', linewidth=1.5)
    
    # 添加数值标签
    for angle, radius in zip(theta, r):
        ax.text(angle, radius, f'{radius:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    ax.set_title('RWS 方位玫瑰图（显示 |RWS| 均值）', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('|RWS| 均值 (m/s)', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    Path(output_dir).mkdir(exist_ok=True)
    filename = f'{output_dir}/wind_rose.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {filename}")
    plt.close()


# ===========================
# 四、主程序
# ===========================

def main():
    """
    主程序入口
    """
    import sys
    
    print("="*60)
    print("多普勒风廓线激光雷达 RWS 分析系统")
    print("="*60)
    
    # 配置参数
    # 支持命令行参数指定数据文件
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'Molas3D_00941_RealTime_20251005_前5000行.csv'
    
    output_dir = 'output_rws_analysis'
    cnr_threshold = -20  # CNR 阈值 (dB)
    
    print(f"\n数据文件: {data_file}")
    print(f"输出目录: {output_dir}")
    print(f"CNR 阈值: {cnr_threshold} dB\n")
    
    # 1. 加载数据
    df = load_lidar_data(data_file)
    
    # 2. 数据概览
    df = get_data_overview(df)
    
    # 3. 获取唯一的角度组合
    azimuths = sorted(df['Azimuth(deg)'].unique())
    elevations = sorted(df['Elevation(deg)'].unique())
    
    print(f"\n找到 {len(azimuths)} 个唯一方位角")
    print(f"找到 {len(elevations)} 个唯一仰角")
    
    # 4. 单角度组合分析（选择第一个角度组合）
    if len(azimuths) > 0 and len(elevations) > 0:
        test_azimuth = azimuths[0]
        test_elevation = elevations[0]
        
        print("\n" + "="*60)
        print("执行单角度组合分析")
        print("="*60)
        
        # 统计分析
        stats = single_angle_statistics(df, test_azimuth, test_elevation)
        print_angle_statistics(stats, test_azimuth, test_elevation)
        
        # 距离维度分析
        print("\n分析 RWS 随距离的变化...")
        distance_stats = analyze_rws_by_distance(df, test_azimuth, test_elevation, output_dir)
        
        # 分布特征可视化
        print("\n绘制 RWS 分布特征图...")
        plot_rws_distribution(df, test_azimuth, test_elevation, output_dir)
        
        # CNR 筛选对比
        print("\n对比 CNR 筛选前后的差异...")
        compare_cnr_filtering(df, test_azimuth, test_elevation, cnr_threshold, output_dir)
    
    # 5. 多角度组合对比分析
    print("\n" + "="*60)
    print("执行多角度组合对比分析")
    print("="*60)
    
    # 不同方位角对比（选择第一个仰角）
    if len(elevations) > 0:
        print(f"\n对比不同方位角（仰角={elevations[0]}°）...")
        compare_azimuths(df, elevations[0], output_dir)
    
    # 不同仰角对比（选择第一个方位角）
    if len(azimuths) > 0:
        print(f"\n对比不同仰角（方位角={azimuths[0]}°）...")
        compare_elevations(df, azimuths[0], output_dir)
    
    # 二维热力图
    if len(elevations) > 0:
        print(f"\n绘制方位角×距离热力图（仰角={elevations[0]}°）...")
        plot_azimuth_distance_heatmap(df, elevations[0], output_dir)
    
    if len(azimuths) > 0:
        print(f"\n绘制仰角×距离热力图（方位角={azimuths[0]}°）...")
        plot_elevation_distance_heatmap(df, azimuths[0], output_dir)
    
    # 风速玫瑰图
    print("\n绘制风速玫瑰图...")
    plot_wind_rose(df, output_dir)
    
    print("\n" + "="*60)
    print("分析完成！所有图表已保存到目录:", output_dir)
    print("="*60)


if __name__ == '__main__':
    main()
    
    print("\n" + "="*60)
    print("使用说明:")
    print("  python analysis_rws.py [数据文件路径]")
    print("  ")
    print("示例:")
    print("  python analysis_rws.py Molas3D_00941_RealTime_20251005_前5000行.csv")
    print("  python analysis_rws.py Molas3D_00943_RealTime_20251005_前5000行.csv")
    print("="*60)
