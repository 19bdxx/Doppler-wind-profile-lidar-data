# Doppler Wind Profile Lidar Data

## 多普勒风廓线激光雷达数据

This repository contains Doppler wind profile lidar data from Molas3D lidar systems.

本仓库包含来自 Molas3D 激光雷达系统的多普勒风廓线数据。

## Documentation / 文档

For detailed information about the data structure, field descriptions, and scanning modes, please refer to:

有关数据结构、字段说明和扫描模式的详细信息，请参阅：

📄 **[雷达数据说明文档.md](./雷达数据说明文档.md)** - Comprehensive Chinese documentation

📊 **[风场风速分析建议报告.md](./风场风速分析建议报告.md)** - Wind field analysis recommendation report / 风场风速分析建议报告

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
