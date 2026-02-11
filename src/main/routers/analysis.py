"""
Analysis API router.

Shift, lap, gear, power, and full report analysis endpoints.
"""

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.features import (
    ShiftAnalyzer, LapAnalysis, GearAnalysis,
    PowerAnalysis, SessionReportGenerator
)
from src.features.lap_analysis import compare_laps_detailed
from src.utils.dataframe_helpers import (
    find_column,
    sanitize_for_json,
    SPEED_MS_TO_MPH,
)
from ..deps import find_parquet_file

router = APIRouter()


@router.get("/api/analyze/shifts/{filename:path}")
async def analyze_shifts(filename: str, trace: bool = False):
    """Run shift analysis on a Parquet file"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        if trace:
            analyzer = ShiftAnalyzer()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        df = pd.read_parquet(file_path)

        time_data = df.index.values
        rpm_data = find_column(df, ['RPM', 'rpm'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])

        if rpm_data is None:
            raise HTTPException(status_code=400, detail="RPM data not found in file")
        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH

        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_session(rpm_data, speed_data, time_data, filename)

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shift analysis failed: {str(e)}")


@router.get("/api/analyze/laps/compare/{filename:path}")
async def compare_laps(filename: str, lap_a: int, lap_b: int, segments: int = 10):
    """Compare two laps showing where time/speed was gained or lost"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        result = compare_laps_detailed(str(file_path), lap_a, lap_b, segments)
        if result is None:
            raise HTTPException(status_code=400, detail=f"Could not compare laps {lap_a} and {lap_b}")
        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lap comparison failed: {str(e)}")


@router.get("/api/analyze/laps/{filename:path}")
async def analyze_laps(filename: str, trace: bool = False):
    """Run lap analysis on a Parquet file"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        if trace:
            analyzer = LapAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        df = pd.read_parquet(file_path)

        time_data = df.index.values
        lat_data = find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude'])
        rpm_data = find_column(df, ['RPM', 'rpm'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS data not found in file")

        if speed_data is None:
            raise HTTPException(status_code=422, detail="Speed data not found in file - required for lap analysis")
        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH
        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))

        analyzer = LapAnalysis()
        report = analyzer.analyze_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, filename
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lap analysis failed: {str(e)}")


@router.get("/api/analyze/gears/{filename:path}")
async def analyze_gears(filename: str, trace: bool = False):
    """Run gear usage analysis on a Parquet file"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        if trace:
            analyzer = GearAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        df = pd.read_parquet(file_path)

        time_data = df.index.values
        rpm_data = find_column(df, ['RPM', 'rpm'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
        lat_data = find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude'])

        if rpm_data is None:
            raise HTTPException(status_code=400, detail="RPM data not found in file")
        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH

        analyzer = GearAnalysis()
        report = analyzer.analyze_from_arrays(
            time_data, rpm_data, speed_data, lat_data, lon_data, filename
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gear analysis failed: {str(e)}")


@router.get("/api/analyze/power/{filename:path}")
async def analyze_power(filename: str, trace: bool = False):
    """Run power/acceleration analysis on a Parquet file"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        if trace:
            analyzer = PowerAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        df = pd.read_parquet(file_path)

        time_data = df.index.values
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
        rpm_data = find_column(df, ['RPM', 'rpm'])

        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH

        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_arrays(time_data, speed_data, rpm_data, filename)

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Power analysis failed: {str(e)}")


@router.get("/api/analyze/report/{filename:path}")
async def analyze_full_report(filename: str, trace: bool = False):
    """Run full session analysis and return combined report"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        if trace:
            generator = SessionReportGenerator()
            report = generator.generate_from_parquet(str(file_path), include_trace=True)
            return sanitize_for_json(report.to_dict())

        df = pd.read_parquet(file_path)

        time_data = df.index.values
        lat_data = find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude'])
        rpm_data = find_column(df, ['RPM', 'rpm'])
        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=422, detail="GPS data (Latitude/Longitude) not found - required for session report")
        if speed_data is None:
            raise HTTPException(status_code=422, detail="Speed data not found - required for session report")
        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH
        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))

        generator = SessionReportGenerator()
        report = generator.generate_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, filename
        )

        return sanitize_for_json(report.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
