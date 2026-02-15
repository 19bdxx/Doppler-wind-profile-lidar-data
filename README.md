# Doppler Wind Profile Lidar Data

## 多普勒风廓线激光雷达数据

This repository contains Doppler wind profile lidar data from Molas3D lidar systems.

本仓库包含来自 Molas3D 激光雷达系统的多普勒风廓线数据。

## Documentation / 文档

For detailed information about the data structure, field descriptions, and scanning modes, please refer to:

有关数据结构、字段说明和扫描模式的详细信息，请参阅：

📄 **[雷达数据说明文档.md](./雷达数据说明文档.md)** - Comprehensive Chinese documentation

📊 **[风场风速分析建议报告.md](./风场风速分析建议报告.md)** - Wind field analysis recommendation report / 风场风速分析建议报告

📈 **[RWS分析报告.md](./RWS分析报告.md)** - RWS (Radial Wind Speed) comprehensive analysis report / RWS 径向风速完整分析报告

## Data Files / 数据文件

- `Molas3D_00941_RealTime_20251005_前5000行.csv` - Device 00941 real-time data
- `Molas3D_00943_RealTime_20251005_前5000行.csv` - Device 00943 real-time data

## Key Features / 主要特点

- **Device Type / 设备类型**: Molas3D Doppler Wind Profile Lidar
- **Temporal Resolution / 时间分辨率**: ~1 second per scan
- **Range Resolution / 距离分辨率**: 17 meters
- **Detection Range / 探测范围**: 100 - 5166 meters
- **Scan Mode / 扫描模式**: Sectoral DBS (Doppler Beam Swinging)

## Data Structure / 数据结构

The data contains 29 fields including:
- Radial wind speed (RWS)
- Carrier-to-Noise Ratio (CNR)
- Azimuth and elevation angles
- Distance gates
- Meteorological parameters
- Atmospheric boundary layer height

数据包含 29 个字段，包括：
- 径向风速 (RWS)
- 载噪比 (CNR)
- 方位角和仰角
- 距离门
- 气象参数
- 大气边界层高度

## Usage Notes / 使用说明

This is **raw measurement data** without wind field retrieval. To obtain wind speed and direction, further processing using algorithms such as VAD (Velocity Azimuth Display) is required.

这是**原始测量数据**，未进行风场反演。要获得风速和风向信息，需要使用 VAD（速度方位显示）等算法进行进一步处理。

## Date / 日期

Data collection date: October 5, 2025 (00:00:00 UTC+8)

数据采集日期：2025年10月5日（00:00:00 UTC+8）

## Analysis Tools / 分析工具

### RWS Analysis Script / RWS 分析脚本

A comprehensive Python script for analyzing Radial Wind Speed (RWS) data is provided: **`analysis_rws.py`**

提供了完整的 Python 径向风速（RWS）分析脚本：**`analysis_rws.py`**

**Features / 功能:**

- **Single angle combination analysis / 单角度组合分析**
  - Statistical indicators (mean, median, std, quantiles)
  - Distance-based trend analysis
  - Distribution visualization (histogram, boxplot, quantile curves)
  
- **Multi-angle comparison / 多角度对比分析**
  - Azimuth comparison
  - Elevation comparison
  - 2D heatmaps (azimuth×distance, elevation×distance)
  - Wind rose diagram
  
- **Quality control / 质量控制**
  - CNR threshold filtering
  - Before/after comparison

**Usage / 使用方法:**

```bash
# Install dependencies / 安装依赖
pip install pandas numpy matplotlib seaborn

# Run analysis with default data file / 使用默认数据文件运行分析
python analysis_rws.py

# Or specify a data file / 或指定数据文件
python analysis_rws.py Molas3D_00941_RealTime_20251005_前5000行.csv
python analysis_rws.py Molas3D_00943_RealTime_20251005_前5000行.csv

# Output / 输出
# Results will be saved in output_rws_analysis/ directory
# 结果将保存在 output_rws_analysis/ 目录中
```

**Jupyter Notebook / 交互式 Notebook:**

For interactive analysis, use **`analysis_rws.ipynb`**. It provides the same functionality with step-by-step execution and inline visualization.

交互式分析请使用 **`analysis_rws.ipynb`**，提供相同功能并支持逐步执行和内联可视化。

**Analysis Report / 分析报告:**

See **[RWS分析报告.md](./RWS分析报告.md)** for detailed analysis results, visualizations, and interpretations.

详细的分析结果、可视化图表和解读请参见 **[RWS分析报告.md](./RWS分析报告.md)**。
