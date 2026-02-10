"""
Tests for ParquetValidator module
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.session.validator import ParquetValidator, ValidationResult


class TestParquetValidator:
    """Tests for ParquetValidator"""

    @pytest.fixture
    def validator(self):
        """Create a validator instance"""
        return ParquetValidator(min_rows=10)

    @pytest.fixture
    def temp_parquet(self):
        """Create a temporary parquet file path"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name
        yield path
        # Cleanup
        try:
            os.unlink(path)
        except:
            pass

    def _create_valid_df(self, rows=100, speed_max=150, speed_col="GPS Speed"):
        """Helper to create a valid test DataFrame"""
        time = np.linspace(0, 100, rows)
        df = pd.DataFrame({
            "GPS Latitude": np.random.uniform(43.78, 43.82, rows),
            "GPS Longitude": np.random.uniform(-87.99, -87.95, rows),
            speed_col: np.random.uniform(0, speed_max, rows),
            "RPM": np.random.uniform(3000, 7000, rows),
        }, index=time)
        return df

    def test_validate_nonexistent_file(self, validator):
        """Test validation of nonexistent file returns errors"""
        result = validator.validate("/nonexistent/file.parquet")
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_validate_valid_parquet(self, validator, temp_parquet):
        """Test validation of valid parquet with all required channels"""
        df = self._create_valid_df()
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True
        assert "latitude" in result.channel_map
        assert "longitude" in result.channel_map
        assert "speed" in result.channel_map
        assert "rpm" in result.channel_map
        assert result.row_count == 100
        assert result.column_count == 4
        assert result.file_hash != ""
        assert result.duration_seconds > 0

    def test_validate_missing_gps(self, validator, temp_parquet):
        """Test validation fails when GPS coordinates are missing"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "GPS Speed": np.random.uniform(0, 150, 100),
            "RPM": np.random.uniform(3000, 7000, 100),
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # Missing latitude and longitude
        assert any("latitude" in err.lower() for err in result.errors)
        assert any("longitude" in err.lower() for err in result.errors)

    def test_validate_empty_file(self, validator, temp_parquet):
        """Test validation fails for file with too few rows"""
        time = np.linspace(0, 5, 5)
        df = pd.DataFrame({
            "GPS Latitude": [43.8] * 5,
            "GPS Longitude": [-87.97] * 5,
            "GPS Speed": [100.0] * 5,
            "RPM": [5000.0] * 5,
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "too few rows" in result.errors[0].lower()
        assert "5" in result.errors[0]
        assert "10" in result.errors[0]

    def test_channel_map_case_insensitive(self, validator, temp_parquet):
        """Test channel mapping works with lowercase column names"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "gps latitude": np.random.uniform(43.78, 43.82, 100),
            "gps longitude": np.random.uniform(-87.99, -87.95, 100),
            "gps speed": np.random.uniform(0, 150, 100),
            "rpm": np.random.uniform(3000, 7000, 100),
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True
        assert "latitude" in result.channel_map
        assert "longitude" in result.channel_map
        assert result.channel_map["latitude"] == "gps latitude"
        assert result.channel_map["longitude"] == "gps longitude"

    def test_speed_unit_detection_ms(self, validator, temp_parquet):
        """Test speed unit detection for m/s range (max ~50)"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "GPS Latitude": np.random.uniform(43.78, 43.82, 100),
            "GPS Longitude": np.random.uniform(-87.99, -87.95, 100),
            "GPS Speed": np.random.uniform(0, 50, 100),  # m/s range
            "RPM": np.random.uniform(3000, 7000, 100),
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True
        assert result.detected_speed_unit == "m/s"

    def test_speed_unit_detection_mph(self, validator, temp_parquet):
        """Test speed unit detection for mph range (max ~150)"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "GPS Latitude": np.random.uniform(43.78, 43.82, 100),
            "GPS Longitude": np.random.uniform(-87.99, -87.95, 100),
            "GPS Speed": np.random.uniform(0, 150, 100),  # mph range
            "RPM": np.random.uniform(3000, 7000, 100),
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True
        assert result.detected_speed_unit == "mph"

    def test_speed_unit_detection_stored_units(self, validator, temp_parquet):
        """Test speed unit detection uses stored units in df.attrs"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "GPS Latitude": np.random.uniform(43.78, 43.82, 100),
            "GPS Longitude": np.random.uniform(-87.99, -87.95, 100),
            "GPS Speed": np.random.uniform(0, 50, 100),  # Would be detected as m/s
            "RPM": np.random.uniform(3000, 7000, 100),
        }, index=time)
        # But we explicitly set it as mph in attrs
        df.attrs["units"] = {"GPS Speed": "mph"}
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True
        assert result.detected_speed_unit == "mph"  # Should use stored units
        assert "GPS Speed" in result.units
        assert result.units["GPS Speed"] == "mph"

    def test_file_hash_consistency(self, validator, temp_parquet):
        """Test that same file produces same hash, different files produce different hashes"""
        df1 = self._create_valid_df(rows=100)
        df1.to_parquet(temp_parquet)

        # Validate twice
        result1 = validator.validate(temp_parquet)
        result2 = validator.validate(temp_parquet)

        # Same file should have same hash
        assert result1.file_hash == result2.file_hash
        assert result1.file_hash != ""

        # Different file should have different hash
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_parquet2 = f.name

        try:
            df2 = self._create_valid_df(rows=150)  # Different size
            df2.to_parquet(temp_parquet2)

            result3 = validator.validate(temp_parquet2)
            assert result3.file_hash != result1.file_hash
        finally:
            os.unlink(temp_parquet2)

    def test_file_hash_different_content(self, validator):
        """Test that files with different content produce different hashes"""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f1:
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2:
            path2 = f2.name

        try:
            # Create two different dataframes
            df1 = self._create_valid_df(rows=100, speed_max=100)
            df2 = self._create_valid_df(rows=100, speed_max=150)

            df1.to_parquet(path1)
            df2.to_parquet(path2)

            result1 = validator.validate(path1)
            result2 = validator.validate(path2)

            assert result1.file_hash != result2.file_hash
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_validate_warnings_for_optional_channels(self, validator, temp_parquet):
        """Test that warnings are generated for missing optional channels"""
        time = np.linspace(0, 100, 100)
        df = pd.DataFrame({
            "GPS Latitude": np.random.uniform(43.78, 43.82, 100),
            "GPS Longitude": np.random.uniform(-87.99, -87.95, 100),
            # Missing speed, rpm, throttle, lat_acc, lon_acc
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True  # Still valid, just warnings
        assert len(result.warnings) > 0
        # Should have warnings for missing optional channels
        warning_text = " ".join(result.warnings).lower()
        assert "rpm" in warning_text or "throttle" in warning_text

    def test_validate_invalid_file_type(self, validator):
        """Test validation fails for non-parquet file"""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
            f.write(b"not a parquet file")

        try:
            result = validator.validate(path)
            assert result.is_valid is False
            assert len(result.errors) > 0
            assert "parquet" in result.errors[0].lower()
        finally:
            os.unlink(path)

    def test_validate_result_to_dict(self, validator, temp_parquet):
        """Test ValidationResult serialization to dict"""
        df = self._create_valid_df()
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "is_valid" in result_dict
        assert "channel_map" in result_dict
        assert "units" in result_dict
        assert "warnings" in result_dict
        assert "errors" in result_dict
        assert "row_count" in result_dict
        assert "column_count" in result_dict
        assert "file_hash" in result_dict
        assert result_dict["is_valid"] is True

    def test_validate_partial_nan_data(self, validator, temp_parquet):
        """Test validation handles partial NaN data with warning"""
        time = np.linspace(0, 100, 100)
        lat = np.random.uniform(43.78, 43.82, 100)
        lon = np.random.uniform(-87.99, -87.95, 100)

        # Set 60% of lat/lon to NaN
        lat[:60] = np.nan
        lon[:60] = np.nan

        df = pd.DataFrame({
            "GPS Latitude": lat,
            "GPS Longitude": lon,
            "GPS Speed": np.random.uniform(0, 150, 100),
            "RPM": np.random.uniform(3000, 7000, 100),
        }, index=time)
        df.to_parquet(temp_parquet)

        result = validator.validate(temp_parquet)
        assert result.is_valid is True  # Has required columns
        assert len(result.warnings) > 0
        # Should warn about low valid value count
        warning_text = " ".join(result.warnings).lower()
        assert "valid" in warning_text or "latitude" in warning_text or "longitude" in warning_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
