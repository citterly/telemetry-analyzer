import pytest
import pandas as pd
from pathlib import Path

from src.session.session_builder import extract_full_session, export_session,_extract_all_channels

from src.io.file_manager import FileManager
from src.config.vehicles import DEFAULT_SESSION
from src.config.config import EXPORTS_PATH

@pytest.mark.smoke
def test_canonical_export(tmp_path):
    """
    Verify canonical export produces Parquet + metadata pointer.
    """

    # 1. Extract DataFrame
    df = extract_full_session(DEFAULT_SESSION, resample_hz=10)
    assert isinstance(df, pd.DataFrame)
    assert df.attrs.get("units") is not None
    assert isinstance(df.attrs["units"], dict)

    # 2. Export to Parquet
    session_id = Path(DEFAULT_SESSION).stem
    out_path = export_session(df, session_id)

    assert out_path.exists()
    assert out_path.suffix == ".parquet"

    # 3. Reload Parquet round-trip
    df2 = pd.read_parquet(out_path)
    assert df2.shape == df.shape

    # 4. Check FileManager metadata
    fm = FileManager()
    meta = fm.get_file_metadata(f"{session_id}.xrk")
    assert meta is not None

    # Check canonical fields exist
    assert meta.parquet_path == str(out_path)
    assert isinstance(meta.channel_list, list)
    assert isinstance(meta.units_map, dict)



def test_dataframe_alignment_and_channels():
    df = extract_full_session(DEFAULT_SESSION, resample_hz=10)

    # Existing checks...
    assert df.shape[1] > 4
    diffs = df.index.to_series().diff().dropna()
    assert (diffs.round(6) == 0.1).all()

    # Units should exist for all columns
    units = df.attrs.get("units", {})
    assert set(units.keys()) == set(df.columns)
    for col, unit in units.items():
        assert unit is not None






