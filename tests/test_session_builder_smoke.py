import pytest
import pandas as pd
from pathlib import Path

from src.session import session_builder
from src.io.file_manager import FileManager

SAMPLE_FILE = "data/uploads/20250712_104619_Road America_a_0394.xrk"
SESSION_ID  = Path(SAMPLE_FILE).stem

def test_canonical_export(tmp_path):
    # --- Load + build canonical dataframe ---
    df = session_builder.extract_full_session(SAMPLE_FILE, resample_hz=10)
    assert not df.empty, "DataFrame should not be empty"

    # --- Check metadata attrs ---
    assert "units" in df.attrs, "Units must be attached"
    assert isinstance(df.attrs["units"], dict)
    assert "unit_sources" in df.attrs, "Unit sources must be attached"
    assert "base_rate_hz" in df.attrs, "Base rate must be recorded"
    assert df.attrs["base_rate_hz"] == 10

    # --- Export to Parquet ---
    parquet_path = session_builder.export_session(df, SESSION_ID)
    assert parquet_path.exists(), "Parquet file must be written"

    # --- Reload Parquet ---
    df2 = pd.read_parquet(parquet_path)
    assert not df2.empty, "Reloaded DataFrame should not be empty"
    assert set(df.columns) == set(df2.columns), "Columns must survive reload"

    # --- Metadata check ---
    fm = FileManager()
    meta = fm.get_file_metadata(f"{SESSION_ID}.xrk")
    assert meta is not None, "Metadata JSON must exist"
    assert meta.parquet_path.endswith(".parquet"), "Metadata must include parquet_path"
    assert isinstance(meta.units_map, dict), "Metadata must include units_map"

    print("✅ Smoke test passed: canonical export & metadata verified")
