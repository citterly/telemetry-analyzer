"""
Analysis API router.

Shift, lap, gear, power, and full report analysis endpoints.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from src.features import (
    ShiftAnalyzer, LapAnalysis, GearAnalysis,
    PowerAnalysis, SessionReportGenerator
)
from src.features.lap_analysis import compare_laps_detailed
from src.utils.dataframe_helpers import sanitize_for_json
from ..deps import find_parquet_file, load_session

router = APIRouter()


@router.get("/api/analyze/shifts/{filename:path}")
async def analyze_shifts(filename: str, trace: bool = False):
    """Run shift analysis on a Parquet file"""
    try:
        if trace:
            file_path = find_parquet_file(filename)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")
            analyzer = ShiftAnalyzer()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        channels = load_session(filename, required=["speed", "rpm"])
        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_session(channels.rpm, channels.speed_mph, channels.time, filename)
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
    try:
        if trace:
            file_path = find_parquet_file(filename)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")
            analyzer = LapAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        channels = load_session(filename, required=["speed", "latitude", "longitude"])
        rpm_data = channels.rpm if channels.rpm is not None else np.zeros(channels.sample_count)

        analyzer = LapAnalysis()
        report = analyzer.analyze_from_arrays(
            channels.time, channels.latitude, channels.longitude,
            rpm_data, channels.speed_mph, filename
        )
        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lap analysis failed: {str(e)}")


@router.get("/api/analyze/gears/{filename:path}")
async def analyze_gears(filename: str, trace: bool = False):
    """Run gear usage analysis on a Parquet file"""
    try:
        if trace:
            file_path = find_parquet_file(filename)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")
            analyzer = GearAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        channels = load_session(filename, required=["speed", "rpm"])

        analyzer = GearAnalysis()
        report = analyzer.analyze_from_arrays(
            channels.time, channels.rpm, channels.speed_mph,
            channels.latitude, channels.longitude, filename
        )
        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gear analysis failed: {str(e)}")


@router.get("/api/analyze/power/{filename:path}")
async def analyze_power(filename: str, trace: bool = False):
    """Run power/acceleration analysis on a Parquet file"""
    try:
        if trace:
            file_path = find_parquet_file(filename)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")
            analyzer = PowerAnalysis()
            report = analyzer.analyze_from_parquet(str(file_path), include_trace=True)
            return report.to_dict()

        channels = load_session(filename, required=["speed"])

        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_arrays(
            channels.time, channels.speed_mph, channels.rpm, filename
        )
        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Power analysis failed: {str(e)}")


@router.get("/api/analyze/report/{filename:path}")
async def analyze_full_report(filename: str, trace: bool = False):
    """Run full session analysis and return combined report"""
    try:
        if trace:
            file_path = find_parquet_file(filename)
            if not file_path:
                raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")
            generator = SessionReportGenerator()
            report = generator.generate_from_parquet(str(file_path), include_trace=True)
            return sanitize_for_json(report.to_dict())

        channels = load_session(filename, required=["speed", "latitude", "longitude"])
        rpm_data = channels.rpm if channels.rpm is not None else np.zeros(channels.sample_count)

        generator = SessionReportGenerator()
        report = generator.generate_from_arrays(
            channels.time, channels.latitude, channels.longitude,
            rpm_data, channels.speed_mph, filename
        )
        return sanitize_for_json(report.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
