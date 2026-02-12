"""
Session import pipeline.

Orchestrates validation, track detection, lap detection, classification,
and database storage for importing Parquet sessions.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.dataframe_helpers import SPEED_MS_TO_MPH

from src.analysis.lap_analyzer import LapAnalyzer, LapInfo
from src.config.tracks import TrackDatabase, get_track_database
from src.session.classifier import LapClassifier
from src.session.models import (
    ImportStatus,
    Lap,
    LapClassification,
    Session,
    SessionType,
    Stint,
)
from src.session.session_database import SessionDatabase
from src.session.validator import ParquetValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of importing a session."""

    session_id: Optional[int] = None
    parquet_path: str = ""
    is_valid: bool = False
    validation_warnings: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    detected_track: Optional[str] = None
    track_confidence: float = 0.0
    lap_count: int = 0
    stint_count: int = 0
    lap_classifications: List[dict] = field(default_factory=list)
    best_lap_time: Optional[float] = None
    duplicate: bool = False
    error: Optional[str] = None
    total_duration: Optional[float] = None
    sample_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "parquet_path": self.parquet_path,
            "is_valid": self.is_valid,
            "validation_warnings": self.validation_warnings,
            "validation_errors": self.validation_errors,
            "detected_track": self.detected_track,
            "track_confidence": self.track_confidence,
            "lap_count": self.lap_count,
            "stint_count": self.stint_count,
            "lap_classifications": self.lap_classifications,
            "best_lap_time": self.best_lap_time,
            "duplicate": self.duplicate,
            "error": self.error,
        }


class SessionImporter:
    """
    Imports Parquet files into the session database.

    Pipeline: validate -> detect track -> detect laps -> classify -> store.
    """

    def __init__(self, session_db: SessionDatabase):
        self.db = session_db
        self.validator = ParquetValidator()
        self.classifier = LapClassifier()
        self.track_db = get_track_database()

    def import_session(
        self,
        parquet_path: str,
        vehicle_id: Optional[str] = None,
        session_type: Optional[str] = None,
        notes: str = "",
    ) -> ImportResult:
        """
        Import a single Parquet session.

        Args:
            parquet_path: Path to Parquet file.
            vehicle_id: Optional vehicle identifier.
            session_type: Optional session type (practice/qualifying/race/test).
            notes: Optional notes.

        Returns:
            ImportResult with session_id and classification data.
        """
        result = ImportResult(parquet_path=parquet_path)

        # Step 1: Validate
        validation = self.validator.validate(parquet_path)
        result.validation_warnings = validation.warnings
        result.validation_errors = validation.errors

        if not validation.is_valid:
            result.error = "; ".join(validation.errors)
            return result

        result.is_valid = True

        # Step 2: Check for duplicates
        existing = self.db.get_session_by_hash(validation.file_hash)
        if existing:
            result.duplicate = True
            result.session_id = existing.id
            result.detected_track = existing.track_name
            result.lap_count = existing.total_laps
            result.best_lap_time = existing.best_lap_time
            return result

        # Step 3: Load DataFrame
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            result.error = f"Failed to read Parquet: {e}"
            return result

        # Populate sample count and duration
        result.sample_count = len(df)
        if hasattr(df.index, 'values') and len(df.index) > 0:
            # Duration from index (time in seconds)
            result.total_duration = float(df.index[-1] - df.index[0])

        # Step 4: Detect track from GPS
        detected_track = None
        track_confidence = 0.0

        lat_col = validation.channel_map.get("latitude")
        lon_col = validation.channel_map.get("longitude")

        if lat_col and lon_col:
            lat_data = df[lat_col].values
            lon_data = df[lon_col].values

            detected_track = self.track_db.detect_track(lat_data, lon_data, threshold=0.5)
            if detected_track:
                result.detected_track = detected_track.name
                # Estimate confidence from bounding box overlap
                track_confidence = self._estimate_track_confidence(
                    lat_data, lon_data, detected_track
                )
                result.track_confidence = track_confidence

        # Step 5: Detect laps
        session_data = self._build_session_data(df, validation)
        analyzer = LapAnalyzer(session_data)
        laps = analyzer.detect_laps(track=detected_track)

        if not laps:
            logger.warning("No laps detected in %s", parquet_path)

        result.lap_count = len(laps)

        # Step 6: Classify laps
        classifications, stint_infos = self.classifier.classify(laps)
        result.stint_count = len(stint_infos)
        result.lap_classifications = [c.to_dict() for c in classifications]

        # Best lap time
        if laps:
            result.best_lap_time = min(l.lap_time for l in laps)

        # Step 7: Determine session type
        s_type = SessionType.UNKNOWN
        if session_type:
            try:
                s_type = SessionType(session_type)
            except ValueError:
                pass

        # Step 8: Build session date from file name or parquet metadata
        session_date = self._extract_session_date(parquet_path, df)

        # Step 9: Store in database
        session = Session(
            parquet_path=str(Path(parquet_path).resolve()),
            file_hash=validation.file_hash,
            track_id=detected_track.id if detected_track else None,
            track_name=detected_track.name if detected_track else None,
            track_confidence=track_confidence,
            vehicle_id=vehicle_id,
            session_date=session_date,
            session_type=s_type,
            import_status=ImportStatus.PENDING,
            total_laps=len(laps),
            best_lap_time=result.best_lap_time,
            total_duration=validation.duration_seconds,
            notes=notes,
        )
        session = self.db.create_session(session)
        result.session_id = session.id

        # Store laps
        if laps and classifications:
            # Build classification lookup
            class_lookup = {c.lap_number: c for c in classifications}

            lap_models = []
            for lap_info in laps:
                cls_result = class_lookup.get(lap_info.lap_number)
                lap_model = Lap(
                    session_id=session.id,
                    lap_number=lap_info.lap_number,
                    stint_number=cls_result.stint_number if cls_result else 0,
                    lap_time=lap_info.lap_time,
                    start_time=lap_info.start_time,
                    end_time=lap_info.end_time,
                    start_index=lap_info.start_index,
                    end_index=lap_info.end_index,
                    classification=cls_result.classification if cls_result else LapClassification.NORMAL,
                    classification_confidence=cls_result.confidence if cls_result else 0.0,
                    max_speed_mph=lap_info.max_speed_mph,
                    max_rpm=lap_info.max_rpm,
                    avg_rpm=lap_info.avg_rpm,
                    sample_count=lap_info.sample_count,
                )
                lap_models.append(lap_model)

            self.db.bulk_create_laps(lap_models)

        # Store stints
        if stint_infos:
            stint_models = self.classifier.build_stint_models(laps, stint_infos)
            for s in stint_models:
                s.session_id = session.id
            self.db.bulk_create_stints(stint_models)

        return result

    def import_batch(self, directory_path: str, **kwargs) -> List[ImportResult]:
        """
        Import all Parquet files in a directory.

        Args:
            directory_path: Path to directory containing Parquet files.
            **kwargs: Additional arguments passed to import_session.

        Returns:
            List of ImportResult for each file.
        """
        results = []
        directory = Path(directory_path)

        if not directory.is_dir():
            return [ImportResult(
                parquet_path=directory_path,
                error=f"Not a directory: {directory_path}",
            )]

        for pq_file in sorted(directory.rglob("*.parquet")):
            result = self.import_session(str(pq_file), **kwargs)
            results.append(result)

        return results

    def _build_session_data(
        self, df: pd.DataFrame, validation: ValidationResult
    ) -> Dict:
        """Build the session_data dict expected by LapAnalyzer."""
        cm = validation.channel_map
        time_data = df.index.values.astype(float)

        lat_data = df[cm["latitude"]].values if "latitude" in cm else np.zeros(len(df))
        lon_data = df[cm["longitude"]].values if "longitude" in cm else np.zeros(len(df))

        rpm_data = df[cm["rpm"]].values if "rpm" in cm else np.zeros(len(df))

        speed_data = np.zeros(len(df))
        if "speed" in cm:
            speed_data = df[cm["speed"]].values.copy()
            # Convert to mph if needed
            if validation.detected_speed_unit == "m/s":
                speed_mph = speed_data * SPEED_MS_TO_MPH
            elif validation.detected_speed_unit == "km/h":
                speed_mph = speed_data / 1.609
            else:
                speed_mph = speed_data
        else:
            speed_mph = speed_data

        return {
            "time": time_data,
            "latitude": lat_data,
            "longitude": lon_data,
            "rpm": rpm_data,
            "speed_mph": speed_mph,
            "speed_ms": speed_mph / SPEED_MS_TO_MPH,
        }

    def _estimate_track_confidence(self, lat_data, lon_data, track) -> float:
        """Estimate confidence of track detection from GPS overlap."""
        bounds = track.gps_bounds
        if not bounds:
            return 0.5

        min_lat = np.nanmin(lat_data)
        max_lat = np.nanmax(lat_data)
        min_lon = np.nanmin(lon_data)
        max_lon = np.nanmax(lon_data)

        data_area = (max_lat - min_lat) * (max_lon - min_lon)
        if data_area <= 0:
            return 0.0

        overlap_lat = max(0, min(max_lat, bounds["max_lat"]) - max(min_lat, bounds["min_lat"]))
        overlap_lon = max(0, min(max_lon, bounds["max_lon"]) - max(min_lon, bounds["min_lon"]))
        overlap_area = overlap_lat * overlap_lon

        return min(1.0, overlap_area / data_area)

    def _extract_session_date(self, parquet_path: str, df: pd.DataFrame) -> Optional[str]:
        """Try to extract session date from filename or DataFrame attrs."""
        # Try df.attrs first
        session_date = df.attrs.get("session_date")
        if session_date:
            return str(session_date)

        # Try to parse from filename pattern: YYYYMMDD_HHMMSS_...
        name = Path(parquet_path).stem
        parts = name.split("_")
        if len(parts) >= 2 and len(parts[0]) == 8 and parts[0].isdigit():
            try:
                date_str = parts[0]
                return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            except (ValueError, IndexError):
                pass

        return None
