"""
Parquet viewer API router.

List, view, and summarize Parquet data files.
"""

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..deps import config

router = APIRouter()


@router.get("/api/parquet/list")
async def list_parquet_files():
    """List all available Parquet files"""
    exports_dir = Path(config.EXPORTS_DIR)
    uploads_dir = Path(config.UPLOAD_DIR)

    parquet_files = []

    # Check exports directory (processed files)
    if exports_dir.exists():
        for pq_file in exports_dir.rglob("*.parquet"):
            try:
                df = pd.read_parquet(pq_file)
                parquet_files.append({
                    "filename": pq_file.name,
                    "path": str(pq_file.relative_to(Path(config.DATA_DIR))),
                    "full_path": str(pq_file),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": round(pq_file.stat().st_size / (1024 * 1024), 2),
                    "source": "exports"
                })
            except Exception as e:
                parquet_files.append({
                    "filename": pq_file.name,
                    "path": str(pq_file),
                    "error": str(e),
                    "source": "exports"
                })

    # Check uploads directory (may have parquet alongside xrk)
    if uploads_dir.exists():
        for pq_file in uploads_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(pq_file)
                parquet_files.append({
                    "filename": pq_file.name,
                    "path": str(pq_file.relative_to(Path(config.DATA_DIR))),
                    "full_path": str(pq_file),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size_mb": round(pq_file.stat().st_size / (1024 * 1024), 2),
                    "source": "uploads"
                })
            except Exception as e:
                pass

    return {"parquet_files": parquet_files}


@router.get("/api/parquet/view/{filename:path}")
async def view_parquet_file(filename: str, limit: int = 100, offset: int = 0):
    """View contents of a Parquet file"""
    data_dir = Path(config.DATA_DIR)
    file_path = data_dir / filename

    if not file_path.exists():
        # Try exports subdirectories
        for pq in data_dir.rglob(filename):
            file_path = pq
            break

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        # Get subset of data
        subset = df.iloc[offset:offset + limit]

        # Convert to records, handling NaN
        records = subset.fillna("NaN").to_dict(orient="records")

        # Get column stats
        col_stats = {}
        for col in df.columns:
            valid = df[col].notna().sum()
            col_stats[col] = {
                "valid_count": int(valid),
                "valid_pct": round(100 * valid / len(df), 1),
                "min": float(df[col].min()) if valid > 0 and df[col].dtype.kind in 'fi' else None,
                "max": float(df[col].max()) if valid > 0 and df[col].dtype.kind in 'fi' else None,
            }

        return {
            "filename": filename,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "column_stats": col_stats,
            "offset": offset,
            "limit": limit,
            "data": records,
            "index_range": [float(df.index.min()), float(df.index.max())] if len(df) > 0 else [0, 0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read Parquet: {str(e)}")


@router.get("/api/parquet/summary/{filename:path}")
async def parquet_summary(filename: str):
    """Get summary statistics for a Parquet file"""
    data_dir = Path(config.DATA_DIR)
    file_path = data_dir / filename

    if not file_path.exists():
        for pq in data_dir.rglob(filename):
            file_path = pq
            break

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        summary = {
            "filename": filename,
            "rows": len(df),
            "columns": len(df.columns),
            "duration_seconds": float(df.index.max() - df.index.min()) if len(df) > 0 else 0,
            "sample_rate_hz": round(len(df) / (df.index.max() - df.index.min()), 1) if len(df) > 1 else 0,
            "channels": {}
        }

        for col in df.columns:
            valid = df[col].notna().sum()
            if valid > 0 and df[col].dtype.kind in 'fi':
                summary["channels"][col] = {
                    "min": round(float(df[col].min()), 2),
                    "max": round(float(df[col].max()), 2),
                    "mean": round(float(df[col].mean()), 2),
                    "valid_pct": round(100 * valid / len(df), 1)
                }
            else:
                summary["channels"][col] = {"valid_pct": round(100 * valid / len(df), 1)}

        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {str(e)}")
