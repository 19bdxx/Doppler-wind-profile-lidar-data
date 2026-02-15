#!/bin/bash
# Example script to run RWS analysis on both data files

echo "========================================"
echo "RWS Analysis Example"
echo "========================================"
echo ""

# Clean output directory
rm -rf output_rws_analysis

echo "Analyzing Device 00941 data..."
python3 analysis_rws.py Molas3D_00941_RealTime_20251005_前5000行.csv

echo ""
echo "========================================"
echo ""
echo "Analysis complete! Check the following:"
echo "  - output_rws_analysis/ directory for visualizations"
echo "  - RWS分析报告.md for detailed report"
echo ""
echo "To analyze Device 00943 data, run:"
echo "  python3 analysis_rws.py Molas3D_00943_RealTime_20251005_前5000行.csv"
echo ""
echo "========================================"
