"""
Shared DataFrame utilities for telemetry analysis.

Consolidates duplicated helpers from 9+ modules:
- find_column(): Column discovery with fuzzy matching
- find_column_name(): Returns column name instead of values
- SPEED_MS_TO_MPH: Conversion constant (replaces bare 2.237 literals)
- ensure_speed_mph(): Detect m/s and convert to mph
- sanitize_for_json(): NaN/Inf/numpy type cleanup for JSON serialization
- safe_float(): Single-value NaN/Inf guard
- KNOWN_COLUMNS: Standard column name mappings
"""

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# --- Constants ---

SPEED_MS_TO_MPH = 2.237
"""Meters per second to miles per hour conversion factor."""

KNOWN_COLUMNS: Dict[str, List[str]] = {
    "latitude": ["GPS Latitude", "latitude", "gps_lat", "Latitude"],
    "longitude": ["GPS Longitude", "longitude", "gps_lon", "Longitude"],
    "speed": ["GPS Speed", "speed", "Speed", "gps_speed"],
    "rpm": ["RPM", "rpm", "Engine RPM"],
    "lat_acc": ["GPS LatAcc", "LatAcc", "lateral_acc", "Lateral Acceleration"],
    "lon_acc": ["GPS LonAcc", "LonAcc", "longitudinal_acc", "Longitudinal Acceleration"],
    "throttle": ["PedalPos", "throttle", "Throttle", "TPS"],
    "gear": ["Gear", "gear", "CurrentGear"],
    "water_temp": ["Water Temp", "WaterTemp", "Coolant Temp", "water_temp"],
    "oil_temp": ["Oil Temp", "OilTemp", "oil_temp"],
    "oil_pressure": ["Oil Press", "OilPress", "oil_pressure"],
}
"""Standard column name candidates for common telemetry channels."""


# --- Column Discovery ---

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[np.ndarray]:
    """
    Find a column by trying multiple names, preferring columns with actual data.

    Search order:
    1. Exact name match (with data quality check)
    2. Case-insensitive exact match
    3. Partial/contains match (best data wins)
    4. Fallback: first exact match even without data

    Args:
        df: DataFrame to search
        candidates: List of candidate column names, in priority order

    Returns:
        Column values as numpy array, or None if not found
    """
    def has_data(col_data):
        if col_data is None or len(col_data) == 0:
            return False
        non_null = np.sum(
            ~np.isnan(col_data) if np.issubdtype(col_data.dtype, np.floating)
            else np.ones(len(col_data), dtype=bool)
        )
        non_zero = np.sum(col_data != 0)
        return non_null > 0 and non_zero > len(col_data) * 0.1

    # Exact matches with data
    for col in candidates:
        if col in df.columns:
            data = df[col].values
            if has_data(data):
                return data

    # Case-insensitive exact matches
    for col in candidates:
        for actual_col in df.columns:
            if actual_col.lower() == col.lower():
                data = df[actual_col].values
                if has_data(data):
                    return data

    # Partial/contains matches — pick the one with most non-zero data
    matching_cols = []
    for col in candidates:
        for actual_col in df.columns:
            if col.lower() in actual_col.lower():
                matching_cols.append(actual_col)

    best_col = None
    best_score = 0
    for col in matching_cols:
        data = df[col].values
        non_zero = np.sum(data != 0)
        if non_zero > best_score:
            best_score = non_zero
            best_col = col

    if best_col is not None:
        return df[best_col].values

    # Fallback: first exact match even if no data
    for col in candidates:
        if col in df.columns:
            return df[col].values

    return None


def find_column_name(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find a column name by trying multiple candidates.

    Like find_column() but returns the column name string instead of values.
    Useful when you need to reference the column later.

    Args:
        df: DataFrame to search
        candidates: List of candidate column names

    Returns:
        Matching column name, or None if not found
    """
    # Exact match
    for name in candidates:
        if name in df.columns:
            return name

    # Case-insensitive exact match
    columns_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        actual = columns_lower.get(name.lower())
        if actual:
            return actual

    # Partial/contains match
    for name in candidates:
        name_lower = name.lower()
        for col_lower, col_actual in columns_lower.items():
            if name_lower in col_lower:
                return col_actual

    return None


# --- Speed Conversion ---

def ensure_speed_mph(speed_data: np.ndarray) -> np.ndarray:
    """
    Detect if speed is in m/s and convert to mph if needed.

    Heuristic: if max speed < 100, assume m/s and convert.
    This matches the pattern used across all endpoints.

    Args:
        speed_data: Speed array (may be m/s or mph)

    Returns:
        Speed array in mph
    """
    if speed_data.max() < 100:
        return speed_data * SPEED_MS_TO_MPH
    return speed_data


# --- JSON Sanitization ---

def sanitize_for_json(obj):
    """
    Recursively replace NaN/Inf with None and numpy types with native Python types.

    Handles dicts, lists, numpy arrays, numpy scalars, and Python floats.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj


def safe_float(value: float, default: float = 0.0) -> float:
    """
    Convert a float to a JSON-safe value, replacing NaN/inf with default.

    Args:
        value: Float value to check
        default: Value to return if NaN/Inf

    Returns:
        The value if valid, otherwise default
    """
    if np.isnan(value) or np.isinf(value):
        return default
    return float(value)
