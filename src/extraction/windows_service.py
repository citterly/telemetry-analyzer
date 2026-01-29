"""
Windows Extraction Service
FastAPI service that processes XRK files via the AIM DLL and returns Parquet.

This service runs ONLY on Windows where the AIM DLL is available.
It is called by the queue worker running on the Linux server.

Usage:
    uvicorn src.extraction.windows_service:app --host 0.0.0.0 --port 8001
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Check if we're on Windows
IS_WINDOWS = sys.platform == "win32"

app = FastAPI(
    title="XRK Extraction Service",
    description="Windows-only service for extracting XRK telemetry data to Parquet",
    version="1.0.0"
)

# Configuration
TEMP_DIR = Path(tempfile.gettempdir()) / "xrk_extraction"
TEMP_DIR.mkdir(exist_ok=True)


class ExtractionRequest(BaseModel):
    """Request body for extraction endpoint"""
    filename: str
    resample_hz: int = 10


class ExtractionResponse(BaseModel):
    """Response from extraction endpoint"""
    success: bool
    message: str
    output_filename: Optional[str] = None
    rows: Optional[int] = None
    columns: Optional[int] = None
    duration_seconds: Optional[float] = None
    channels: Optional[list] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    platform: str
    dll_available: bool
    temp_dir: str
    timestamp: str


def check_dll_available() -> bool:
    """Check if the AIM DLL is available"""
    if not IS_WINDOWS:
        return False
    try:
        from src.io.dll_interface import AIMDLL
        dll = AIMDLL()
        return dll.setup()
    except Exception:
        return False


def cleanup_temp_file(filepath: Path):
    """Background task to clean up temporary files"""
    try:
        if filepath.exists():
            filepath.unlink()
    except Exception:
        pass


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Returns service status and DLL availability.
    """
    return HealthResponse(
        status="healthy",
        platform=sys.platform,
        dll_available=check_dll_available() if IS_WINDOWS else False,
        temp_dir=str(TEMP_DIR),
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_xrk(
    file: UploadFile = File(...),
    resample_hz: int = 10
):
    """
    Extract XRK file to Parquet.

    Accepts an uploaded XRK file, processes it via the AIM DLL,
    and returns metadata about the extracted data.

    The actual Parquet file can be downloaded via /download/{filename}
    """
    if not IS_WINDOWS:
        raise HTTPException(
            status_code=503,
            detail="Extraction service only available on Windows"
        )

    if not file.filename or not file.filename.lower().endswith('.xrk'):
        raise HTTPException(
            status_code=400,
            detail="File must be an XRK file"
        )

    # Save uploaded file to temp location
    temp_xrk = TEMP_DIR / f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file.filename}"

    try:
        with open(temp_xrk, "wb") as f:
            content = await file.read()
            f.write(content)

        # Import session builder (only works on Windows with DLL)
        try:
            from src.session.session_builder import extract_full_session
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Session builder not available: {e}"
            )

        # Extract session data
        try:
            df = extract_full_session(str(temp_xrk), resample_hz=resample_hz)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Extraction failed: {e}"
            )

        # Save to Parquet
        output_filename = f"{Path(file.filename).stem}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
        output_path = TEMP_DIR / output_filename
        df.to_parquet(output_path, engine="pyarrow", index=True)

        # Calculate duration
        duration = float(df.index.max() - df.index.min()) if len(df) > 0 else 0

        return ExtractionResponse(
            success=True,
            message="Extraction completed successfully",
            output_filename=output_filename,
            rows=len(df),
            columns=len(df.columns),
            duration_seconds=round(duration, 2),
            channels=list(df.columns)[:50]  # Limit to first 50 for response size
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {e}"
        )
    finally:
        # Clean up uploaded XRK file
        if temp_xrk.exists():
            temp_xrk.unlink()


@app.get("/download/{filename}")
async def download_parquet(filename: str, background_tasks: BackgroundTasks):
    """
    Download an extracted Parquet file.

    The file is deleted after download.
    """
    filepath = TEMP_DIR / filename

    if not filepath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )

    if not filename.endswith('.parquet'):
        raise HTTPException(
            status_code=400,
            detail="Only Parquet files can be downloaded"
        )

    # Schedule cleanup after response is sent
    background_tasks.add_task(cleanup_temp_file, filepath)

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream"
    )


@app.get("/files")
async def list_temp_files():
    """
    List temporary files available for download.
    Useful for debugging and recovery.
    """
    files = []
    for f in TEMP_DIR.glob("*.parquet"):
        files.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
        })
    return {"files": files}


@app.delete("/files/{filename}")
async def delete_temp_file(filename: str):
    """Delete a temporary file"""
    filepath = TEMP_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")

    filepath.unlink()
    return {"status": "deleted", "filename": filename}


@app.delete("/files")
async def clear_temp_files():
    """Clear all temporary files"""
    count = 0
    for f in TEMP_DIR.glob("*"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return {"status": "cleared", "files_deleted": count}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    print(f"XRK Extraction Service starting...")
    print(f"  Platform: {sys.platform}")
    print(f"  Windows: {IS_WINDOWS}")
    print(f"  Temp directory: {TEMP_DIR}")
    if IS_WINDOWS:
        dll_ok = check_dll_available()
        print(f"  DLL available: {dll_ok}")
        if not dll_ok:
            print("  WARNING: AIM DLL not available, extraction will fail")
    else:
        print("  WARNING: Not running on Windows, extraction endpoints disabled")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
