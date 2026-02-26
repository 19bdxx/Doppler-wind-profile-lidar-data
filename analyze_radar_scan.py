#!/usr/bin/env python3
"""
analyze_radar_scan.py
---------------------
自动分析多普勒风廓线激光雷达（Molas3D）长时间扫描模式与调度策略。

功能：
  1. 读取一个或多个 Molas3D CSV 数据文件。
  2. 提取每束扫描的时间戳、方位角（Azimuth）、仰角（Elevation）。
  3. 识别仰角层、扫描周期（体扫）、模式切换等调度规律。
  4. 输出统计摘要（控制台）及图表（PNG 文件）。

用法：
  python3 analyze_radar_scan.py [CSV文件1] [CSV文件2] ...

  若不指定文件，自动搜索当前目录下所有 *_RealTime_*.csv 文件。

输出文件（保存至脚本所在目录）：
  <设备ID>_elevation_vs_time.png      — 仰角随时间变化
  <设备ID>_azimuth_vs_time.png        — 方位角随时间变化
  <设备ID>_dwell_time_distribution.png — 每层持续时长分布
  <设备ID>_cycle_period_histogram.png  — 体扫周期分布
"""

import sys
import os
import glob
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合无显示器环境
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as _fm

# ── 自动选择支持中文的字体 ──────────────────────────────────────────────────
def _find_cjk_font() -> str | None:
    """返回系统中第一个可用的 CJK 字体名称，找不到则返回 None。"""
    candidates = [
        "Noto Sans CJK SC", "Noto Serif CJK SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "SimHei", "Microsoft YaHei", "PingFang SC", "Heiti SC",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    # 备选：在文件路径中匹配关键词
    keywords = ["NotoSansCJK", "NotoSerifCJK", "wqy", "simhei", "simsun"]
    for f in _fm.findSystemFonts():
        for kw in keywords:
            if kw.lower() in f.lower():
                prop = _fm.FontProperties(fname=f)
                return prop.get_name()
    return None

_cjk_font = _find_cjk_font()
if _cjk_font:
    matplotlib.rcParams["font.family"] = _cjk_font
# 确保负号正常显示
matplotlib.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """读取 Molas3D CSV，返回按时间排序的 DataFrame（只保留每束第一行）。"""
    df = pd.read_csv(path, low_memory=False)
    # 解析时间戳
    df["_time"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M:%S.%f",
                                 errors="coerce")
    df = df.dropna(subset=["_time"])
    # 每个时间戳对应一束扫描（多个距离门），只取第一行代表该束
    beam_df = df.drop_duplicates(subset=["Timestamp"]).copy()
    beam_df = beam_df.sort_values("_time").reset_index(drop=True)
    beam_df["az"] = pd.to_numeric(beam_df["Azimuth(deg)"], errors="coerce")
    beam_df["el"] = pd.to_numeric(beam_df["Elevation(deg)"], errors="coerce")
    return beam_df


def round_el(series: pd.Series, tol: float = 0.05) -> pd.Series:
    """将仰角四舍五入到离散层（容差 tol 度内视为同一层）。"""
    vals = series.values
    rounded = vals.copy()
    unique_raw = np.unique(vals[~np.isnan(vals)])
    # 聚类：贪心合并相邻值
    layers = []
    for v in sorted(unique_raw):
        if not layers or v - layers[-1] > tol:
            layers.append(v)
        else:
            layers[-1] = (layers[-1] + v) / 2  # 更新中心
    mapping = {}
    for v in unique_raw:
        diffs = [abs(v - l) for l in layers]
        mapping[v] = layers[np.argmin(diffs)]
    rounded = series.map(mapping)
    return rounded.round(3)


def detect_cycles(beam_df: pd.DataFrame) -> pd.DataFrame:
    """
    检测体扫周期：每当仰角从较高层回落到最低层时，标记为新周期开始。

    返回 beam_df 新增列：
      el_layer  — 离散化仰角层编号
      cycle_id  — 体扫周期编号（从 0 开始）
    """
    beam_df = beam_df.copy()
    beam_df["el_rounded"] = round_el(beam_df["el"])

    el_layers = sorted(beam_df["el_rounded"].dropna().unique())
    el_to_layer = {e: i for i, e in enumerate(el_layers)}
    beam_df["el_layer"] = beam_df["el_rounded"].map(el_to_layer)

    # 检测周期：仰角层序列出现"从高层跳回低层"时视为新周期
    min_layer = 0
    cycle_ids = [0]
    prev_layer = beam_df["el_layer"].iloc[0]
    cid = 0
    for cur_layer in beam_df["el_layer"].iloc[1:]:
        if cur_layer == min_layer and prev_layer != min_layer:
            cid += 1
        cycle_ids.append(cid)
        prev_layer = cur_layer
    beam_df["cycle_id"] = cycle_ids
    return beam_df, el_layers


def compute_layer_dwell(beam_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每次连续停留在同一仰角层的持续时长（秒）。

    返回 DataFrame：el_rounded, dwell_seconds, cycle_id
    """
    records = []
    groups = []
    current_el = beam_df["el_rounded"].iloc[0]
    current_cid = beam_df["cycle_id"].iloc[0]
    start_time = beam_df["_time"].iloc[0]
    end_time = beam_df["_time"].iloc[0]

    for _, row in beam_df.iloc[1:].iterrows():
        if row["el_rounded"] == current_el and row["cycle_id"] == current_cid:
            end_time = row["_time"]
        else:
            dwell = (end_time - start_time).total_seconds()
            records.append({
                "el_rounded": current_el,
                "cycle_id": current_cid,
                "dwell_seconds": max(dwell, 0),
            })
            current_el = row["el_rounded"]
            current_cid = row["cycle_id"]
            start_time = row["_time"]
            end_time = row["_time"]

    # 最后一段
    dwell = (end_time - start_time).total_seconds()
    records.append({
        "el_rounded": current_el,
        "cycle_id": current_cid,
        "dwell_seconds": max(dwell, 0),
    })
    return pd.DataFrame(records)


def compute_cycle_periods(beam_df: pd.DataFrame) -> pd.Series:
    """
    计算每个体扫周期的总时长（秒）。
    """
    periods = {}
    for cid, grp in beam_df.groupby("cycle_id"):
        t0 = grp["_time"].min()
        t1 = grp["_time"].max()
        periods[cid] = (t1 - t0).total_seconds()
    return pd.Series(periods).rename("period_seconds")


# ─────────────────────────────────────────────────────────────────────────────
# 绘图函数
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_el_vs_time(beam_df: pd.DataFrame, el_layers, device_id: str,
                    out_dir: str) -> str:
    """仰角随时间变化图。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    sc = ax.scatter(beam_df["_time"], beam_df["el"], c=beam_df["el_layer"],
                    cmap="tab10", s=10, zorder=3)
    ax.set_xlabel("时间 (UTC)")
    ax.set_ylabel("仰角 Elevation (°)")
    ax.set_title(f"[{device_id}] 仰角随时间变化")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_ticks(range(len(el_layers)))
    cbar.set_ticklabels([f"{e:.3f}°" for e in el_layers])
    cbar.set_label("仰角层")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{device_id}_elevation_vs_time.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_az_vs_time(beam_df: pd.DataFrame, el_layers, device_id: str,
                    out_dir: str) -> str:
    """方位角随时间变化图（按仰角层着色）。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, el in enumerate(el_layers):
        mask = beam_df["el_rounded"] == el
        sub = beam_df[mask]
        ax.scatter(sub["_time"], sub["az"], label=f"El={el:.3f}°",
                   color=COLORS[i % len(COLORS)], s=10, zorder=3)
    ax.set_xlabel("时间 (UTC)")
    ax.set_ylabel("方位角 Azimuth (°)")
    ax.set_title(f"[{device_id}] 方位角随时间变化（按仰角层着色）")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    ax.legend(loc="upper right", markerscale=3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{device_id}_azimuth_vs_time.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_dwell_distribution(dwell_df: pd.DataFrame, el_layers,
                             device_id: str, out_dir: str) -> str:
    """每层持续时长分布（箱线图）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [dwell_df[dwell_df["el_rounded"] == el]["dwell_seconds"].values
            for el in el_layers]
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
    ax.set_xticks(range(1, len(el_layers) + 1))
    ax.set_xticklabels([f"{e:.3f}°" for e in el_layers], rotation=30)
    ax.set_xlabel("仰角层 Elevation Layer (°)")
    ax.set_ylabel("持续时长 Dwell Time (s)")
    ax.set_title(f"[{device_id}] 每层持续时长分布")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{device_id}_dwell_time_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cycle_histogram(periods: pd.Series, device_id: str,
                          out_dir: str) -> str:
    """体扫周期时长直方图。"""
    if len(periods) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(periods.values, bins=max(5, len(periods) // 2),
            color="steelblue", edgecolor="white")
    ax.set_xlabel("体扫周期时长 Cycle Duration (s)")
    ax.set_ylabel("频次 Count")
    ax.set_title(f"[{device_id}] 体扫周期时长分布")
    ax.axvline(periods.mean(), color="red", linestyle="--",
               label=f"均值 {periods.mean():.1f} s")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, f"{device_id}_cycle_period_histogram.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(csv_path: str, out_dir: str) -> None:
    """分析单个 CSV 文件，输出统计摘要和图表。"""
    fname = os.path.basename(csv_path)
    # 从文件名提取设备 ID（例如 Molas3D_00941_...）
    parts = fname.split("_")
    device_id = parts[1] if len(parts) >= 2 else fname.replace(".csv", "")

    print(f"\n{'=' * 60}")
    print(f"文件: {fname}")
    print(f"设备 ID: {device_id}")
    print("=" * 60)

    # 加载数据
    beam_df = load_csv(csv_path)
    if beam_df.empty:
        print("  [警告] 无有效数据，跳过。")
        return

    total_beams = len(beam_df)
    t_start = beam_df["_time"].min()
    t_end = beam_df["_time"].max()
    duration_s = (t_end - t_start).total_seconds()

    print(f"\n【基本信息】")
    print(f"  扫描束数量       : {total_beams}")
    print(f"  时间跨度         : {t_start}  →  {t_end}")
    print(f"  总时长           : {duration_s:.1f} 秒 ({duration_s/60:.2f} 分钟)")

    # 仰角层分析
    beam_df, el_layers = detect_cycles(beam_df)
    print(f"\n【仰角层】")
    print(f"  发现 {len(el_layers)} 个仰角层: {[f'{e:.3f}°' for e in el_layers]}")
    for el in el_layers:
        cnt = (beam_df["el_rounded"] == el).sum()
        az_vals = beam_df[beam_df["el_rounded"] == el]["az"]
        print(f"  El={el:.3f}°  → {cnt} 束，"
              f"Az 范围 [{az_vals.min():.3f}°, {az_vals.max():.3f}°]，"
              f"Az 跨度 {az_vals.max() - az_vals.min():.3f}°")

    # 方位角扫描步进分析
    print(f"\n【方位角扫描步进（各仰角层）】")
    for el in el_layers:
        az_series = beam_df[beam_df["el_rounded"] == el]["az"].dropna()
        if len(az_series) >= 2:
            steps = az_series.diff().dropna()
            pos_steps = steps[steps > 0]
            if not pos_steps.empty:
                print(f"  El={el:.3f}°: 平均步进 {pos_steps.mean():.3f}°，"
                      f"中位步进 {pos_steps.median():.3f}°，"
                      f"步进范围 [{pos_steps.min():.3f}°, {pos_steps.max():.3f}°]")

    # 体扫周期分析
    n_cycles = beam_df["cycle_id"].max() + 1
    print(f"\n【体扫周期】")
    print(f"  检测到 {n_cycles} 个体扫周期")

    if n_cycles >= 2:
        periods = compute_cycle_periods(beam_df)
        valid_periods = periods[periods > 0]
        print(f"  周期时长统计（秒）: "
              f"均值={valid_periods.mean():.1f}, "
              f"中位={valid_periods.median():.1f}, "
              f"最短={valid_periods.min():.1f}, "
              f"最长={valid_periods.max():.1f}")
        # 检测异常周期（>1.5×中位值可能为模式切换/插入扫描）
        median_p = valid_periods.median()
        outliers = valid_periods[valid_periods > 1.5 * median_p]
        if not outliers.empty:
            print(f"  [注意] 发现 {len(outliers)} 个异常长周期（>1.5×中位={median_p:.1f}s），"
                  f"可能为插入扫描或模式切换：")
            for cid, p in outliers.items():
                t0 = beam_df[beam_df["cycle_id"] == cid]["_time"].min()
                print(f"    周期 {cid}: {p:.1f}s，起始时间 {t0}")
    else:
        periods = pd.Series(dtype=float)

    # 每层持续时长
    dwell_df = compute_layer_dwell(beam_df)
    print(f"\n【每层持续时长（秒）】")
    for el in el_layers:
        sub = dwell_df[dwell_df["el_rounded"] == el]["dwell_seconds"]
        if not sub.empty:
            print(f"  El={el:.3f}°: 均值={sub.mean():.1f}s, "
                  f"中位={sub.median():.1f}s, "
                  f"范围=[{sub.min():.1f}, {sub.max():.1f}]s")

    # 采样频率
    if total_beams >= 2:
        intervals = beam_df["_time"].diff().dropna().dt.total_seconds()
        print(f"\n【采样间隔（束间，秒）】")
        print(f"  均值={intervals.mean():.3f}s, "
              f"中位={intervals.median():.3f}s, "
              f"最短={intervals.min():.3f}s, "
              f"最长={intervals.max():.3f}s")
        print(f"  等效采样频率: {1/intervals.median():.2f} Hz")

    # 生成图表
    print(f"\n【生成图表】")
    out_paths = []
    out_paths.append(plot_el_vs_time(beam_df, el_layers, device_id, out_dir))
    out_paths.append(plot_az_vs_time(beam_df, el_layers, device_id, out_dir))
    out_paths.append(plot_dwell_distribution(dwell_df, el_layers, device_id, out_dir))
    if not periods.empty:
        p = plot_cycle_histogram(periods, device_id, out_dir)
        if p:
            out_paths.append(p)

    for p in out_paths:
        if p:
            print(f"  已保存: {p}")

    print(f"\n【扫描模式判断】")
    if len(el_layers) == 1:
        print("  → 单仰角扇形扫描（Single-Elevation Sector Scan）")
    elif len(el_layers) == 2:
        print("  → 双仰角扇形 DBS 扫描（Two-Elevation Sector DBS Scan）")
    else:
        print(f"  → 多仰角扇形扫描（{len(el_layers)}-Elevation Sector Scan）")
    if n_cycles >= 2:
        print(f"  → 存在 {n_cycles} 个体扫周期，平均周期约 "
              f"{periods.mean():.1f} 秒")
    else:
        print("  → 数据量不足以判断完整体扫周期（建议使用完整数据文件）")


def main():
    # 确定输出目录（脚本所在目录）
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 确定输入文件
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
    else:
        pattern = os.path.join(out_dir, "*_RealTime_*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print("未找到 CSV 文件。请指定文件路径，或将 CSV 文件放在脚本所在目录。")
            sys.exit(1)

    print(f"共发现 {len(csv_files)} 个 CSV 文件")
    for f in csv_files:
        if not os.path.isfile(f):
            print(f"[警告] 文件不存在，跳过: {f}")
            continue
        analyze_file(f, out_dir)

    print(f"\n{'=' * 60}")
    print("分析完成。")


if __name__ == "__main__":
    main()
