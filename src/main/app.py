"""
Main FastAPI application for Telemetry Analyzer
Minimal web interface for trackside use
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
import shutil
import json
from pathlib import Path
from typing import List, Optional
import traceback

from src.config.config import get_config
from src.io.file_manager import FileManager

# Initialize configuration
config = get_config()
config.init_app()

# Initialize FastAPI app
app = FastAPI(
    title="Telemetry Analyzer",
    description="Portable racing telemetry analysis for trackside use",
    version="1.0.0",
    debug=config.DEBUG
)

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates for HTML responses
templates = Jinja2Templates(directory="templates")

# Initialize file manager
file_manager = FileManager(config.DATA_DIR)

# Global stats for the dashboard
def get_dashboard_stats():
    """Get statistics for the main dashboard"""
    try:
        stats = file_manager.get_stats()
        recent_files = file_manager.get_file_list()[:5]  # Last 5 files
        
        return {
            "stats": stats,
            "recent_files": [
                {
                    "filename": f.filename,
                    "processed": f.processed,
                    "import_date": f.import_date,
                    "fastest_lap_time": f.fastest_lap_time,
                    "total_laps": f.total_laps
                }
                for f in recent_files
            ]
        }
    except Exception as e:
        return {"stats": {}, "recent_files": [], "error": str(e)}


# Routes

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    dashboard_data = get_dashboard_stats()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Telemetry Analyzer - Dashboard",
        **dashboard_data
    })


@app.get("/upload", response_class=HTMLResponse) 
async def upload_page(request: Request):
    """File upload page"""
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "title": "Upload XRK File",
        "max_size_mb": config.MAX_UPLOAD_SIZE_MB
    })


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and import XRK file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not any(file.filename.lower().endswith(ext) for ext in config.ALLOWED_EXTENSIONS):
            raise HTTPException(status_code=400, detail="Only XRK files are supported")
        
        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > config.MAX_UPLOAD_SIZE_MB:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {config.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Save temporary file
        temp_path = config.get_upload_path(f"temp_{file.filename}")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(file_content)
        
        # Import using file manager
        try:
            metadata = file_manager.import_file(
                str(temp_path),
                custom_attributes={
                    "uploaded_via": "web_interface",
                    "original_filename": file.filename
                }
            )
            
            # Clean up temp file
            temp_path.unlink()
            
            return {
                "status": "success",
                "message": f"File {file.filename} imported successfully",
                "filename": metadata.filename,
                "file_size_mb": round(metadata.file_size_bytes / (1024 * 1024), 2),
                "session_duration": metadata.session_duration_seconds,
                "sample_count": metadata.sample_count
            }
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/files")
async def list_files(processed: Optional[bool] = None):
    """Get list of uploaded files"""
    try:
        files = file_manager.get_file_list(filter_processed=processed)
        
        return {
            "files": [
                {
                    "filename": f.filename,
                    "file_size_mb": round(f.file_size_bytes / (1024 * 1024), 2),
                    "import_date": f.import_date,
                    "session_date": f.session_date,
                    "track_name": f.track_name,
                    "processed": f.processed,
                    "total_laps": f.total_laps,
                    "fastest_lap_time": f.fastest_lap_time,
                    "max_speed_mph": f.max_speed_mph,
                    "max_rpm": f.max_rpm,
                    "session_duration": f.session_duration_seconds
                }
                for f in files
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


@app.post("/api/process/{filename}")
async def process_file(filename: str):
    """Process a file and run analysis"""
    try:
        # Check if file exists
        metadata = file_manager.get_file_metadata(filename)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Process the file
        results = file_manager.process_file(filename)
        
        # Return processing results
        response_data = {
            "status": "success",
            "message": f"Processed {filename}",
            "filename": filename,
            "total_laps": len(results['laps']),
            "session_duration": results['session_data']['session_duration'],
            "sample_count": results['session_data']['sample_count']
        }
        
        # Add fastest lap info if available
        if results['fastest_lap_data']:
            lap_info = results['fastest_lap_data']['lap_info']
            response_data.update({
                "fastest_lap": {
                    "lap_number": lap_info.lap_number,
                    "lap_time": lap_info.lap_time,
                    "max_speed_mph": lap_info.max_speed_mph,
                    "max_rpm": lap_info.max_rpm
                }
            })
        
        # Add lap summary
        if results['laps']:
            lap_times = [lap.lap_time for lap in results['laps']]
            response_data["lap_summary"] = {
                "lap_count": len(lap_times),
                "fastest_time": min(lap_times),
                "slowest_time": max(lap_times),
                "average_time": sum(lap_times) / len(lap_times)
            }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"Processing failed: {str(e)}"
        if config.DEBUG:
            error_detail += f"\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/api/file/{filename}")
async def get_file_details(filename: str):
    """Get detailed information about a specific file"""
    try:
        metadata = file_manager.get_file_metadata(filename)
        if not metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "filename": metadata.filename,
            "file_size_mb": round(metadata.file_size_bytes / (1024 * 1024), 2),
            "file_hash": metadata.file_hash[:16] + "...",  # Shortened for display
            "import_date": metadata.import_date,
            "session_date": metadata.session_date,
            "track_name": metadata.track_name,
            "session_duration": metadata.session_duration_seconds,
            "sample_count": metadata.sample_count,
            "processed": metadata.processed,
            "total_laps": metadata.total_laps,
            "fastest_lap_time": metadata.fastest_lap_time,
            "max_speed_mph": metadata.max_speed_mph,
            "max_rpm": metadata.max_rpm,
            "custom_attributes": metadata.custom_attributes
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get file details: {str(e)}")


@app.delete("/api/file/{filename}")
async def delete_file(filename: str):
    """Delete a file and its metadata"""
    try:
        success = file_manager.delete_file(filename)
        if not success:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {"status": "success", "message": f"File {filename} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    try:
        stats = file_manager.get_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/files", response_class=HTMLResponse)
async def files_page(request: Request):
    """File management page"""
    try:
        files = file_manager.get_file_list()
        return templates.TemplateResponse("files.html", {
            "request": request,
            "title": "File Management",
            "files": files
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "title": "Error",
            "error": str(e)
        })


# ============================================================
# Parquet Viewer (Linux-compatible, no DLL required)
# ============================================================

@app.get("/api/parquet/list")
async def list_parquet_files():
    """List all available Parquet files"""
    import pandas as pd
    from pathlib import Path

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


@app.get("/api/parquet/view/{filename:path}")
async def view_parquet_file(filename: str, limit: int = 100, offset: int = 0):
    """View contents of a Parquet file"""
    import pandas as pd
    from pathlib import Path

    # Try to find the file
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
                "min": float(df[col].min()) if valid > 0 and df[col].dtype in ['float64', 'int64'] else None,
                "max": float(df[col].max()) if valid > 0 and df[col].dtype in ['float64', 'int64'] else None,
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


@app.get("/api/parquet/summary/{filename:path}")
async def parquet_summary(filename: str):
    """Get summary statistics for a Parquet file"""
    import pandas as pd
    from pathlib import Path

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
            if valid > 0 and df[col].dtype in ['float64', 'int64']:
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


@app.get("/parquet", response_class=HTMLResponse)
async def parquet_viewer_page(request: Request):
    """Parquet file viewer page"""
    return templates.TemplateResponse("parquet.html", {
        "request": request,
        "title": "Session Data Viewer"
    })


# ============================================================
# Analysis API Endpoints
# ============================================================

from src.features import (
    ShiftAnalyzer, LapAnalysis, GearAnalysis,
    PowerAnalysis, SessionReportGenerator
)
from src.visualization.track_map import TrackMap


@app.get("/api/analyze/shifts/{filename:path}")
async def analyze_shifts(filename: str):
    """Run shift analysis on a Parquet file"""
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        # Find required columns
        time_data = df.index.values
        rpm_data = _find_column(df, ['RPM', 'rpm'])
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])

        if rpm_data is None:
            raise HTTPException(status_code=400, detail="RPM data not found in file")
        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        # Convert speed if needed
        if speed_data.max() < 100:
            speed_data = speed_data * 2.237

        analyzer = ShiftAnalyzer()
        report = analyzer.analyze_session(rpm_data, speed_data, time_data, filename)

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shift analysis failed: {str(e)}")


@app.get("/api/analyze/laps/{filename:path}")
async def analyze_laps(filename: str):
    """Run lap analysis on a Parquet file"""
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        time_data = df.index.values
        lat_data = _find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = _find_column(df, ['GPS Longitude', 'longitude'])
        rpm_data = _find_column(df, ['RPM', 'rpm'])
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS data not found in file")

        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))
        if speed_data is None:
            speed_data = np.zeros(len(time_data))
        elif speed_data.max() < 100:
            speed_data = speed_data * 2.237

        analyzer = LapAnalysis()
        report = analyzer.analyze_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, filename
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lap analysis failed: {str(e)}")


@app.get("/api/analyze/gears/{filename:path}")
async def analyze_gears(filename: str):
    """Run gear usage analysis on a Parquet file"""
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        time_data = df.index.values
        rpm_data = _find_column(df, ['RPM', 'rpm'])
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])
        lat_data = _find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = _find_column(df, ['GPS Longitude', 'longitude'])

        if rpm_data is None:
            raise HTTPException(status_code=400, detail="RPM data not found in file")
        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        if speed_data.max() < 100:
            speed_data = speed_data * 2.237

        analyzer = GearAnalysis()
        report = analyzer.analyze_from_arrays(
            time_data, rpm_data, speed_data, lat_data, lon_data, filename
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gear analysis failed: {str(e)}")


@app.get("/api/analyze/power/{filename:path}")
async def analyze_power(filename: str):
    """Run power/acceleration analysis on a Parquet file"""
    import pandas as pd

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        time_data = df.index.values
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])
        rpm_data = _find_column(df, ['RPM', 'rpm'])

        if speed_data is None:
            raise HTTPException(status_code=400, detail="Speed data not found in file")

        if speed_data.max() < 100:
            speed_data = speed_data * 2.237

        analyzer = PowerAnalysis()
        report = analyzer.analyze_from_arrays(time_data, speed_data, rpm_data, filename)

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Power analysis failed: {str(e)}")


@app.get("/api/analyze/report/{filename:path}")
async def analyze_full_report(filename: str):
    """Run full session analysis and return combined report"""
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        time_data = df.index.values
        lat_data = _find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = _find_column(df, ['GPS Longitude', 'longitude'])
        rpm_data = _find_column(df, ['RPM', 'rpm'])
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])

        # Use zeros for missing data
        if lat_data is None:
            lat_data = np.zeros(len(time_data))
        if lon_data is None:
            lon_data = np.zeros(len(time_data))
        if rpm_data is None:
            rpm_data = np.zeros(len(time_data))
        if speed_data is None:
            speed_data = np.zeros(len(time_data))
        elif speed_data.max() < 100:
            speed_data = speed_data * 2.237

        generator = SessionReportGenerator()
        report = generator.generate_from_arrays(
            time_data, lat_data, lon_data, rpm_data, speed_data, filename
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/track-map/{filename:path}")
async def get_track_map(
    filename: str,
    color_by: str = "speed",
    format: str = "svg"
):
    """Generate track map visualization"""
    import pandas as pd

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        lat_data = _find_column(df, ['GPS Latitude', 'latitude'])
        lon_data = _find_column(df, ['GPS Longitude', 'longitude'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS data not found in file")

        # Get color data based on selection
        color_data = None
        if color_by == 'speed':
            color_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])
            if color_data is not None and color_data.max() < 100:
                color_data = color_data * 2.237
        elif color_by == 'rpm':
            color_data = _find_column(df, ['RPM', 'rpm'])

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
                    lat_data, lon_data, color_data, color_by, f"Track Map - {filename}"
                ),
                media_type="image/svg+xml"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Track map generation failed: {str(e)}")


def _find_parquet_file(filename: str) -> Optional[Path]:
    """Find a Parquet file by name in the data directories"""
    data_dir = Path(config.DATA_DIR)

    # Try direct path
    file_path = data_dir / filename
    if file_path.exists():
        return file_path

    # Try with .parquet extension
    if not filename.endswith('.parquet'):
        file_path = data_dir / f"{filename}.parquet"
        if file_path.exists():
            return file_path

    # Search recursively
    for pq in data_dir.rglob(f"*{filename}*"):
        if pq.suffix == '.parquet':
            return pq

    return None


def _find_column(df, candidates: List[str]):
    """Find a column by trying multiple names"""
    import numpy as np
    for col in candidates:
        if col in df.columns:
            return df[col].values
        for actual_col in df.columns:
            if actual_col.lower() == col.lower():
                return df[actual_col].values
    return None


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "title": "Page Not Found",
        "error": "The requested page was not found"
    }, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "title": "Server Error", 
        "error": "An internal server error occurred"
    }, status_code=500)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for deployment monitoring"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "data_dir": config.DATA_DIR,
        "debug": config.DEBUG
    }


if __name__ == "__main__":
    import uvicorn
    
    print(f"Starting Telemetry Analyzer on {config.HOST}:{config.PORT}")
    print(f"Debug mode: {config.DEBUG}")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Open browser to: http://{config.HOST}:{config.PORT}")
    
    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info" if not config.DEBUG else "debug"
    )