#!/usr/bin/env python3
"""
analyze_radar_scan.py
---------------------
自动分析多普勒风廓线激光雷达（Molas3D）长时间扫描模式与调度策略。

功能：
  1. 自动扫描 process_data/ 目录（或命令行指定文件）中的所有 CSV。
  2. 提取每束扫描的时间戳、方位角（Azimuth）、仰角（Elevation）。
  3. 识别仰角层、方位圈（sweep）、扫描周期（体扫 volume）。
  4. 输出每个 sweep 的摘要表（CSV）、稀疏轨迹（CSV）及图表（PNG）。
  5. 汇总所有文件生成分析报告（analysis_report.md）。

用法：
  python3 analyze_radar_scan.py [CSV文件1] [CSV文件2] ...

  若不指定文件，自动搜索 process_data/ 子目录下所有 *.csv 文件。

输出文件（保存至 output/ 子目录）：
  <前缀>_sweep_summary.csv            — 每个 sweep 的摘要表
  <前缀>_sparse_trajectory.csv        — 0.5s 采样的稀疏轨迹
  <前缀>_elevation_vs_time.png        — 仰角随时间变化
  <前缀>_azimuth_vs_time.png          — 方位角随时间变化
  <前缀>_dwell_time_distribution.png  — 每层持续时长分布
  <前缀>_cycle_period_histogram.png   — 体扫周期分布
  analysis_report.md                  — 全部文件汇总分析报告
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


def detect_sweeps(beam_df: pd.DataFrame, az_jump_tol: float = 30.0) -> pd.DataFrame:
    """
    在 detect_cycles() 结果基础上进一步检测方位圈（sweep）。

    判定规则：当仰角层切换 **或** 方位角出现显著负跳变（< -az_jump_tol°，
    即如 350°→10° 的绕圈）时，标记为新 sweep 开始。

    新增列：sweep_id（全局连续编号，从 0 开始）
    """
    beam_df = beam_df.copy()
    az = beam_df["az"].values
    el_layer = beam_df["el_layer"].values

    sweep_ids = [0]
    sid = 0
    for i in range(1, len(beam_df)):
        az_diff = az[i] - az[i - 1]
        el_changed = el_layer[i] != el_layer[i - 1]
        az_jumped = az_diff < -az_jump_tol
        if el_changed or az_jumped:
            sid += 1
        sweep_ids.append(sid)
    beam_df["sweep_id"] = sweep_ids
    return beam_df


def compute_sweep_summary(beam_df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总每个 sweep 的统计摘要。

    返回列：
      sweep_id, cycle_id, el_layer, el_rounded_deg,
      start_time, end_time, duration_s,
      az_min_deg, az_max_deg, az_span_deg, az_arc_deg,
      n_beams, scan_type (full_circle / sector)

    az_arc_deg：累加所有正向方位步进的总弧度（比 az_span 更能反映
    绕圈覆盖范围，不受越 0° 截断影响）。
    """
    records = []
    for sid, grp in beam_df.groupby("sweep_id", sort=True):
        grp = grp.sort_values("_time")
        el_val = grp["el_rounded"].mode().iloc[0]
        az_arr = grp["az"].values
        az_span = float(grp["az"].max() - grp["az"].min())
        az_diffs = np.diff(az_arr)
        az_arc = float(az_diffs[az_diffs > 0].sum())
        scan_type = "full_circle" if az_arc >= 330.0 else "sector"
        t0 = grp["_time"].min()
        t1 = grp["_time"].max()
        records.append({
            "sweep_id": int(sid),
            "cycle_id": int(grp["cycle_id"].iloc[0]),
            "el_layer": int(grp["el_layer"].iloc[0]),
            "el_rounded_deg": round(float(el_val), 3),
            "start_time": t0.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
            "end_time": t1.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
            "duration_s": round((t1 - t0).total_seconds(), 3),
            "az_min_deg": round(float(grp["az"].min()), 3),
            "az_max_deg": round(float(grp["az"].max()), 3),
            "az_span_deg": round(az_span, 3),
            "az_arc_deg": round(az_arc, 3),
            "n_beams": len(grp),
            "scan_type": scan_type,
        })
    return pd.DataFrame(records)


def compute_sparse_trajectory(beam_df: pd.DataFrame,
                              interval_s: float = 0.5) -> pd.DataFrame:
    """
    按时间间隔 interval_s 稀疏采样扫描轨迹，便于可视化动画。

    返回列：sample_time, az_deg, el_deg, el_rounded_deg, sweep_id, cycle_id
    """
    t_start = beam_df["_time"].min()
    t_end = beam_df["_time"].max()
    t_range = pd.date_range(start=t_start, end=t_end,
                            freq=f"{interval_s}s")
    # 将时间序列与采样点统一为 float64（秒级精度）进行最近邻查找
    beam_sec = (beam_df["_time"].values.astype("datetime64[ms]")
                .astype("int64") / 1000.0)  # 毫秒→秒
    sample_sec = np.array([t.timestamp() for t in t_range])

    # 使用 searchsorted 高效查找每个采样点的最近 beam
    ins = np.searchsorted(beam_sec, sample_sec, side="left").clip(0, len(beam_sec) - 1)
    # 比较左右邻居，取更近者
    ins_prev = (ins - 1).clip(0, len(beam_sec) - 1)
    diff_right = np.abs(beam_sec[ins] - sample_sec)
    diff_left = np.abs(beam_sec[ins_prev] - sample_sec)
    best_idx = np.where(diff_left < diff_right, ins_prev, ins)

    rows = []
    for i, idx in enumerate(best_idx):
        row = beam_df.iloc[int(idx)]
        rows.append({
            "sample_time": t_range[i].strftime("%Y/%m/%d %H:%M:%S.%f")[:-3],
            "az_deg": round(float(row["az"]), 3),
            "el_deg": round(float(row["el"]), 3),
            "el_rounded_deg": round(float(row["el_rounded"]), 3),
            "sweep_id": int(row["sweep_id"]),
            "cycle_id": int(row["cycle_id"]),
        })
    return pd.DataFrame(rows)


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


def generate_markdown_report(file_summaries: list, out_path: str) -> None:
    """
    生成汇总所有 CSV 文件分析结果的 Markdown 报告。

    Parameters
    ----------
    file_summaries : list of dict
        每个 dict 对应一个文件的分析摘要，由 analyze_file() 返回。
    out_path : str
        报告输出路径（.md）。
    """
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 激光雷达扫描动态分析报告",
        "",
        f"**生成时间**：{now_str}  ",
        f"**数据来源**：process_data/ 目录下所有 CSV 文件  ",
        f"**处理文件数**：{len(file_summaries)}",
        "",
        "---",
        "",
    ]

    for s in file_summaries:
        lines += [
            f"## {s['fname']}",
            "",
            "### 基本信息",
            "",
            f"| 项目 | 值 |",
            f"|------|----|",
            f"| 设备 ID | {s['device_id']} |",
            f"| 扫描束总数 | {s['total_beams']} |",
            f"| 开始时间 | {s['t_start']} |",
            f"| 结束时间 | {s['t_end']} |",
            f"| 总时长 | {s['duration_s']:.1f} 秒（{s['duration_s']/60:.2f} 分钟）|",
            f"| 仰角层数 | {s['n_el_layers']} |",
            f"| 体扫周期数 | {s['n_cycles']} |",
            f"| Sweep 总数 | {s['n_sweeps']} |",
            f"| 扫描模式 | {s['scan_mode']} |",
            "",
        ]

        if s['el_layers']:
            lines += [
                "### 仰角层概览",
                "",
                "| 仰角层 (°) | 束数 | Az 最小 (°) | Az 最大 (°) | Az 跨度 (°) |",
                "|-----------|------|------------|------------|------------|",
            ]
            for el_info in s['el_layers']:
                lines.append(
                    f"| {el_info['el']:.3f} | {el_info['cnt']} "
                    f"| {el_info['az_min']:.3f} | {el_info['az_max']:.3f} "
                    f"| {el_info['az_span']:.3f} |"
                )
            lines.append("")

        if s['cycle_stats']:
            cs = s['cycle_stats']
            lines += [
                "### 体扫周期统计",
                "",
                f"| 均值 (s) | 中位数 (s) | 最短 (s) | 最长 (s) |",
                f"|---------|-----------|---------|---------|",
                f"| {cs['mean']:.1f} | {cs['median']:.1f} "
                f"| {cs['min']:.1f} | {cs['max']:.1f} |",
                "",
            ]

        lines += [
            "### Sweep 摘要（前 20 条）",
            "",
        ]
        if s.get("sweep_head"):
            lines += [
                "| sweep_id | cycle_id | el(°) | 开始时间 | 结束时间 | 时长(s) "
                "| Az最小(°) | Az最大(°) | Az跨度(°) | Az弧度(°) | 束数 | 类型 |",
                "|----------|----------|-------|---------|---------|--------|"
                "----------|----------|---------|---------|------|------|",
            ]
            for r in s["sweep_head"]:
                lines.append(
                    f"| {r['sweep_id']} | {r['cycle_id']} "
                    f"| {r['el_rounded_deg']:.3f} "
                    f"| {r['start_time']} | {r['end_time']} "
                    f"| {r['duration_s']:.1f} "
                    f"| {r['az_min_deg']:.3f} | {r['az_max_deg']:.3f} "
                    f"| {r['az_span_deg']:.3f} | {r['az_arc_deg']:.3f} "
                    f"| {r['n_beams']} | {r['scan_type']} |"
                )
        lines += ["", "---", ""]

    lines += [
        "## 附录：输出文件说明",
        "",
        "| 文件名模式 | 说明 |",
        "|-----------|------|",
        "| `*_sweep_summary.csv` | 每个 sweep 的详细摘要（仰角、起止时间、Az 范围等）|",
        "| `*_sparse_trajectory.csv` | 每 0.5 秒采样一次的扫描轨迹，可用于动画可视化 |",
        "| `*_elevation_vs_time.png` | 仰角随时间变化散点图 |",
        "| `*_azimuth_vs_time.png` | 方位角随时间变化图（按仰角层着色）|",
        "| `*_dwell_time_distribution.png` | 各仰角层停留时长箱线图 |",
        "| `*_cycle_period_histogram.png` | 体扫周期时长分布直方图 |",
        "",
        "> 所有输出文件保存在 `output/` 子目录中。",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  已生成分析报告: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主分析函数
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(csv_path: str, out_dir: str) -> dict:
    """
    分析单个 CSV 文件，输出统计摘要、图表、sweep 摘要 CSV 及稀疏轨迹 CSV。

    Returns
    -------
    dict : 文件分析摘要，用于生成 Markdown 报告。返回 None 表示文件无有效数据。
    """
    fname = os.path.basename(csv_path)
    # 从文件名提取前缀（去掉 .csv），用于输出文件命名
    stem = Path(fname).stem
    # 从文件名提取设备 ID（例如 Molas3D_00941_...）
    parts = fname.split("_")
    device_id = "_".join(parts[:2]) if len(parts) >= 2 else stem

    print(f"\n{'=' * 60}")
    print(f"文件: {fname}")
    print(f"设备 ID: {device_id}")
    print("=" * 60)

    # 加载数据
    beam_df = load_csv(csv_path)
    if beam_df.empty:
        print("  [警告] 无有效数据，跳过。")
        return None

    total_beams = len(beam_df)
    t_start = beam_df["_time"].min()
    t_end = beam_df["_time"].max()
    duration_s = (t_end - t_start).total_seconds()

    print(f"\n【基本信息】")
    print(f"  扫描束数量       : {total_beams}")
    print(f"  时间跨度         : {t_start}  →  {t_end}")
    print(f"  总时长           : {duration_s:.1f} 秒 ({duration_s/60:.2f} 分钟)")

    # 仰角层 & 体扫周期分析
    beam_df, el_layers = detect_cycles(beam_df)
    print(f"\n【仰角层】")
    print(f"  发现 {len(el_layers)} 个仰角层: {[f'{e:.3f}°' for e in el_layers]}")

    el_layer_infos = []
    for el in el_layers:
        cnt = (beam_df["el_rounded"] == el).sum()
        az_vals = beam_df[beam_df["el_rounded"] == el]["az"]
        az_min = az_vals.min()
        az_max = az_vals.max()
        az_span = az_max - az_min
        el_layer_infos.append({"el": el, "cnt": cnt,
                                "az_min": az_min, "az_max": az_max,
                                "az_span": az_span})
        print(f"  El={el:.3f}°  → {cnt} 束，"
              f"Az 范围 [{az_min:.3f}°, {az_max:.3f}°]，"
              f"Az 跨度 {az_span:.3f}°")

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
    n_cycles = int(beam_df["cycle_id"].max()) + 1
    print(f"\n【体扫周期】")
    print(f"  检测到 {n_cycles} 个体扫周期")

    cycle_stats = None
    if n_cycles >= 2:
        periods = compute_cycle_periods(beam_df)
        valid_periods = periods[periods > 0]
        cycle_stats = {
            "mean": valid_periods.mean(),
            "median": valid_periods.median(),
            "min": valid_periods.min(),
            "max": valid_periods.max(),
        }
        print(f"  周期时长统计（秒）: "
              f"均值={cycle_stats['mean']:.1f}, "
              f"中位={cycle_stats['median']:.1f}, "
              f"最短={cycle_stats['min']:.1f}, "
              f"最长={cycle_stats['max']:.1f}")
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

    # Sweep 检测
    beam_df = detect_sweeps(beam_df)
    sweep_df = compute_sweep_summary(beam_df)
    n_sweeps = len(sweep_df)
    print(f"\n【Sweep（方位圈）】")
    print(f"  检测到 {n_sweeps} 个 sweep")
    full_circles = (sweep_df["scan_type"] == "full_circle").sum()
    sectors = n_sweeps - full_circles
    print(f"  全圈扫描: {full_circles}，扇形扫描: {sectors}")

    # 保存 sweep 摘要 CSV
    sweep_csv_path = os.path.join(out_dir, f"{stem}_sweep_summary.csv")
    sweep_df.to_csv(sweep_csv_path, index=False, encoding="utf-8-sig")
    print(f"  已保存 sweep 摘要: {sweep_csv_path}")

    # 保存稀疏轨迹 CSV
    traj_df = compute_sparse_trajectory(beam_df, interval_s=0.5)
    traj_csv_path = os.path.join(out_dir, f"{stem}_sparse_trajectory.csv")
    traj_df.to_csv(traj_csv_path, index=False, encoding="utf-8-sig")
    print(f"  已保存稀疏轨迹: {traj_csv_path}")

    # 生成图表
    print(f"\n【生成图表】")
    out_paths = []
    out_paths.append(plot_el_vs_time(beam_df, el_layers, stem, out_dir))
    out_paths.append(plot_az_vs_time(beam_df, el_layers, stem, out_dir))
    out_paths.append(plot_dwell_distribution(dwell_df, el_layers, stem, out_dir))
    if not periods.empty:
        p = plot_cycle_histogram(periods, stem, out_dir)
        if p:
            out_paths.append(p)
    for p in out_paths:
        if p:
            print(f"  已保存: {p}")

    # 扫描模式判断
    print(f"\n【扫描模式判断】")
    if len(el_layers) == 1:
        scan_mode = "单仰角扇形扫描（Single-Elevation Sector Scan）"
    elif len(el_layers) == 2:
        scan_mode = "双仰角扇形 DBS 扫描（Two-Elevation Sector DBS Scan）"
    else:
        scan_mode = f"多仰角体扫（{len(el_layers)}-Elevation Volume Scan）"
    print(f"  → {scan_mode}")
    if n_cycles >= 2:
        print(f"  → 存在 {n_cycles} 个体扫周期，平均周期约 "
              f"{periods.mean():.1f} 秒")
    else:
        print("  → 数据量不足以判断完整体扫周期（建议使用完整数据文件）")

    return {
        "fname": fname,
        "device_id": device_id,
        "total_beams": total_beams,
        "t_start": str(t_start),
        "t_end": str(t_end),
        "duration_s": duration_s,
        "n_el_layers": len(el_layers),
        "n_cycles": n_cycles,
        "n_sweeps": n_sweeps,
        "scan_mode": scan_mode,
        "el_layers": el_layer_infos,
        "cycle_stats": cycle_stats,
        "sweep_head": sweep_df.head(20).to_dict(orient="records"),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 确定输入文件
    if len(sys.argv) > 1:
        csv_files = sys.argv[1:]
        out_dir = script_dir
    else:
        # 默认扫描 process_data/ 子目录
        data_dir = os.path.join(script_dir, "process_data")
        if not os.path.isdir(data_dir):
            data_dir = script_dir
        pattern = os.path.join(data_dir, "*.csv")
        csv_files = sorted(glob.glob(pattern))
        if not csv_files:
            print("未在 process_data/ 目录下找到 CSV 文件。"
                  "请指定文件路径，或将 CSV 文件放入 process_data/ 目录。")
            sys.exit(1)
        # 输出到 output/ 子目录
        out_dir = os.path.join(script_dir, "output")

    os.makedirs(out_dir, exist_ok=True)

    print(f"共发现 {len(csv_files)} 个 CSV 文件")
    print(f"输出目录: {out_dir}")

    file_summaries = []
    for f in csv_files:
        if not os.path.isfile(f):
            print(f"[警告] 文件不存在，跳过: {f}")
            continue
        result = analyze_file(f, out_dir)
        if result is not None:
            file_summaries.append(result)

    # 生成汇总 Markdown 报告
    if file_summaries:
        report_path = os.path.join(out_dir, "analysis_report.md")
        generate_markdown_report(file_summaries, report_path)

    print(f"\n{'=' * 60}")
    print(f"分析完成。共处理 {len(file_summaries)} 个文件，"
          f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
