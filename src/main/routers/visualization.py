"""
Visualization API router.

Track maps, G-G diagrams, and corner analysis endpoints.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import HTMLResponse

from src.features.gg_analysis import GGAnalyzer
from src.features.corner_analysis import CornerAnalyzer
from src.visualization.track_map import TrackMap
from src.visualization.gg_diagram import GGDiagram
from src.config.vehicles import get_active_vehicle
from src.analysis.lap_analyzer import LapAnalyzer
from src.features.lap_analysis import compare_laps_detailed
from src.utils.dataframe_helpers import (
    find_column,
    sanitize_for_json,
    SPEED_MS_TO_MPH,
)
from ..deps import find_parquet_file

router = APIRouter()


@router.get("/api/track-map/delta/{filename:path}")
async def get_delta_track_map(
    filename: str,
    lap_a: int,
    lap_b: int,
    segments: int = 10,
    format: str = "svg"
):
    """
    Generate delta track map showing time gained/lost between two laps.

    Green sections = lap_a faster (gaining time)
    Red sections = lap_b faster (losing time)
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        comparison = compare_laps_detailed(str(file_path), lap_a, lap_b, segments)
        if comparison is None:
            raise HTTPException(status_code=400, detail=f"Could not compare laps {lap_a} and {lap_b}")

        # Load parquet and detect laps to get GPS data for each lap
        df = pd.read_parquet(file_path)

        time_data = df.index.values
        lat_col = None
        lon_col = None
        speed_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'latitude' in col_lower:
                lat_col = col
            elif 'longitude' in col_lower:
                lon_col = col
            elif 'speed' in col_lower and speed_col is None:
                speed_col = col

        if lat_col is None or lon_col is None:
            raise HTTPException(status_code=400, detail="GPS data not found in file")

        lat_data = df[lat_col].values
        lon_data = df[lon_col].values
        if speed_col is None:
            raise HTTPException(status_code=422, detail="Speed data not found - required for lap comparison")
        speed_data = df[speed_col].values

        if speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH

        # Detect laps
        session_data = {
            'time': time_data,
            'latitude': lat_data,
            'longitude': lon_data,
            'rpm': np.zeros(len(time_data)),
            'speed_mph': speed_data,
            'speed_ms': speed_data / SPEED_MS_TO_MPH
        }

        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps()

        # Find the two laps
        lap_a_info = None
        lap_b_info = None
        for lap in laps:
            if lap.lap_number == lap_a:
                lap_a_info = lap
            if lap.lap_number == lap_b:
                lap_b_info = lap

        if lap_a_info is None or lap_b_info is None:
            raise HTTPException(status_code=400, detail=f"Lap {lap_a} or {lap_b} not found")

        # Get GPS data for each lap
        lap_a_data = analyzer.get_lap_data(lap_a_info)
        lap_b_data = analyzer.get_lap_data(lap_b_info)

        ref_lat = np.array(lap_a_data['latitude'])
        ref_lon = np.array(lap_a_data['longitude'])
        comp_lat = np.array(lap_b_data['latitude'])
        comp_lon = np.array(lap_b_data['longitude'])

        # Generate track map
        track_map = TrackMap()

        if format == 'json':
            return {
                "lap_a": lap_a,
                "lap_b": lap_b,
                "segments": comparison.segments,
                "time_delta": comparison.time_delta,
                "summary": comparison.summary
            }
        else:
            svg = track_map.render_delta_svg(
                ref_lat, ref_lon,
                comp_lat, comp_lon,
                comparison.segments,
                title=f"Delta: Lap {lap_a} vs Lap {lap_b}",
                ref_label=f"Lap {lap_a}",
                comp_label=f"Lap {lap_b}"
            )
            return HTMLResponse(content=svg, media_type="image/svg+xml")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delta track map generation failed: {str(e)}")


@router.get("/api/track-map/{filename:path}")
async def get_track_map(
    filename: str,
    color_by: str = "speed",
    format: str = "svg",
    discrete: bool = True,
    low_threshold: float = 33.0,
    high_threshold: float = 66.0
):
    """Generate track map visualization"""
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        lat_data = find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'longitude'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS data not found in file")

        # Get color data based on selection
        color_data = None
        if color_by == 'speed':
            color_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
            if color_data is not None and color_data.max() < 100:
                color_data = color_data * SPEED_MS_TO_MPH
        elif color_by == 'rpm':
            color_data = find_column(df, ['RPM', 'rpm'])

        track_map = TrackMap()

        if format == 'html':
            return HTMLResponse(
                content=track_map.render_html(
                    lat_data, lon_data, color_data, color_by, f"Track Map - {filename}"
                )
            )
        elif format == 'json':
            return track_map.to_dict(lat_data, lon_data, color_data, color_by)
        else:
            return HTMLResponse(
                content=track_map.render_svg(
                    lat_data, lon_data, color_data, color_by, f"Track Map - {filename}",
                    discrete_mode=discrete,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold
                ),
                media_type="image/svg+xml"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track map generation failed: {str(e)}")


@router.get("/api/gg-diagram/{filename:path}")
async def get_gg_diagram(
    filename: str,
    format: str = "json",
    color_by: str = "speed",
    lap: str = None
):
    """
    Generate G-G diagram data or visualization.

    Args:
        filename: Parquet file path
        format: Output format ('json', 'svg')
        color_by: Color scheme ('speed', 'throttle', 'lap')
        lap: Filter to specific lap number (optional)
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        # Get vehicle config values for accurate per-quadrant reference
        vehicle = get_active_vehicle()
        max_lateral_g = getattr(vehicle, 'max_lateral_g', 1.3)
        max_braking_g = getattr(vehicle, 'max_braking_g', max_lateral_g * 1.1)
        power_limited_accel_g = getattr(vehicle, 'power_limited_accel_g', 0.4)

        # Parse lap filter
        lap_filter = None
        if lap and lap != 'all':
            try:
                lap_filter = int(lap)
            except ValueError:
                pass

        # Run analysis using analyze_from_parquet for full feature support
        analyzer = GGAnalyzer(
            max_g_reference=max_lateral_g,
            max_braking_g=max_braking_g,
            power_limited_accel_g=power_limited_accel_g
        )
        result = analyzer.analyze_from_parquet(file_path, session_id=filename, lap_filter=lap_filter)

        # Get arrays for SVG rendering
        df = pd.read_parquet(file_path)
        lat_acc = find_column(df, ['GPS LatAcc', 'LatAcc'])
        lon_acc = find_column(df, ['GPS LonAcc', 'LonAcc'])

        if lat_acc is None or lon_acc is None:
            raise HTTPException(status_code=400, detail="Acceleration data (GPS LatAcc/LonAcc) not found")

        speed_data = find_column(df, ['GPS Speed', 'speed', 'Speed'])
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * SPEED_MS_TO_MPH

        throttle_data = find_column(df, ['PedalPos', 'throttle', 'Throttle'])

        # Filter SVG arrays by lap if specified
        if lap_filter is not None and result.lap_numbers:
            lat_gps = find_column(df, ['GPS Latitude', 'latitude', 'gps_lat'])
            lon_gps = find_column(df, ['GPS Longitude', 'longitude', 'gps_lon'])

            if lat_gps is not None and lon_gps is not None:
                try:
                    time_data = df.index.values
                    session_data = {
                        'time': time_data,
                        'latitude': lat_gps,
                        'longitude': lon_gps,
                        'rpm': np.zeros(len(time_data)),
                        'speed_mph': speed_data if speed_data is not None else np.zeros(len(time_data)),
                        'speed_ms': (speed_data / SPEED_MS_TO_MPH) if speed_data is not None else np.zeros(len(time_data))
                    }
                    lap_analyzer = LapAnalyzer(session_data)
                    laps = lap_analyzer.detect_laps()
                    if laps:
                        lap_data_arr = np.zeros(len(time_data))
                        for l in laps:
                            lap_data_arr[l.start_index:l.end_index+1] = l.lap_number
                        lap_mask = lap_data_arr == lap_filter
                        if np.sum(lap_mask) > 0:
                            lat_acc = lat_acc[lap_mask]
                            lon_acc = lon_acc[lap_mask]
                            if speed_data is not None:
                                speed_data = speed_data[lap_mask]
                            if throttle_data is not None:
                                throttle_data = throttle_data[lap_mask]
                except Exception as e:
                    logging.warning(f"Lap filtering for SVG arrays failed: {e}")

        if format == 'json':
            return result.to_dict()
        else:
            # Generate SVG
            diagram = GGDiagram()

            color_data = None
            if color_by == 'speed' and speed_data is not None:
                color_data = speed_data
            elif color_by == 'throttle' and throttle_data is not None:
                color_data = throttle_data

            svg = diagram.render_svg(
                lat_acc, lon_acc,
                color_data=color_data,
                color_scheme=color_by,
                reference_max_g=result.reference_max_g,
                data_max_g=result.stats.data_derived_max_g,
                title=f"G-G Diagram - {Path(filename).stem}" + (f" (Lap {lap_filter})" if lap_filter else "")
            )
            return HTMLResponse(content=svg, media_type="image/svg+xml")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G-G diagram generation failed: {str(e)}")


@router.get("/api/corner-analysis/{filename:path}")
async def get_corner_analysis(
    filename: str,
    track_name: str = "Unknown Track"
):
    """
    Analyze corners in a session.

    Detects corners and calculates per-corner metrics:
    - Entry/apex/exit speeds
    - Time in corner
    - Throttle pickup point
    - Lift detection
    - Trail braking
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        lat_data = find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS latitude/longitude data not found")

        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_parquet(
            str(file_path),
            session_id=Path(filename).stem,
            track_name=track_name
        )

        return sanitize_for_json(result.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner analysis failed: {str(e)}")


@router.get("/api/corner-track-map/{filename:path}")
async def get_corner_track_map(
    filename: str,
    track_name: str = "Unknown Track",
    color_scheme: str = "speed",
    selected_corner: Optional[str] = None
):
    """
    Generate track map SVG with corner overlay.

    Shows corner markers (numbered circles) at apex positions,
    highlighted corner boundaries, and track colored by speed/gear/etc.
    """
    file_path = find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        lat_data = find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS latitude/longitude data not found")

        # Get color data based on scheme
        if color_scheme == 'speed':
            color_data = find_column(df, ['GPS Speed', 'gps_speed', 'speed'])
            if color_data is not None and color_data.max() < 100:
                color_data = color_data * SPEED_MS_TO_MPH
        elif color_scheme == 'rpm':
            color_data = find_column(df, ['RPM', 'engine_rpm', 'rpm'])
        elif color_scheme == 'gear':
            color_data = find_column(df, ['Gear', 'gear'])
        elif color_scheme == 'throttle':
            color_data = find_column(df, ['PedalPos', 'throttle', 'Throttle'])
        else:
            color_data = None

        # Detect corners
        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_parquet(
            str(file_path),
            session_id=Path(filename).stem,
            track_name=track_name
        )

        # Convert corner zones to dicts for rendering
        corners = []
        for zone in result.corner_zones:
            corners.append({
                'name': zone.name,
                'alias': zone.alias,
                'apex_lat': zone.apex_lat,
                'apex_lon': zone.apex_lon,
                'entry_idx': zone.entry_idx,
                'apex_idx': zone.apex_idx,
                'exit_idx': zone.exit_idx,
                'corner_type': zone.corner_type,
                'direction': zone.direction
            })

        # Render track map with corner overlay
        track_map = TrackMap()
        svg = track_map.render_corner_overlay_svg(
            lat_data,
            lon_data,
            corners,
            color_data=color_data,
            color_scheme=color_scheme,
            title=f"{track_name} - Corner Map",
            selected_corner=selected_corner
        )

        return Response(content=svg, media_type="image/svg+xml")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track map generation failed: {str(e)}")
