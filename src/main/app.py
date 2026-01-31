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


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """Analysis results viewer page"""
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "title": "Session Analysis"
    })


# ============================================================
# Analysis API Endpoints
# ============================================================

from src.features import (
    ShiftAnalyzer, LapAnalysis, GearAnalysis,
    PowerAnalysis, SessionReportGenerator
)
from src.features.lap_analysis import compare_laps_detailed
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


@app.get("/api/analyze/laps/compare/{filename:path}")
async def compare_laps(filename: str, lap_a: int, lap_b: int, segments: int = 10):
    """Compare two laps showing where time/speed was gained or lost"""
    file_path = _find_parquet_file(filename)
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
    format: str = "svg",
    discrete: bool = True,
    low_threshold: float = 33.0,
    high_threshold: float = 66.0
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


@app.get("/api/track-map/delta/{filename:path}")
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
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        # Get lap comparison data
        from src.features.lap_analysis import compare_laps_detailed
        from src.analysis.lap_analyzer import LapAnalyzer

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
        speed_data = df[speed_col].values if speed_col else np.zeros(len(time_data))

        if speed_data.max() < 100:
            speed_data = speed_data * 2.237

        # Detect laps
        session_data = {
            'time': time_data,
            'latitude': lat_data,
            'longitude': lon_data,
            'rpm': np.zeros(len(time_data)),
            'speed_mph': speed_data,
            'speed_ms': speed_data / 2.237
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


# ============================================================
# G-G Diagram Endpoints
# ============================================================

from src.features.gg_analysis import GGAnalyzer
from src.visualization.gg_diagram import GGDiagram
from src.features.corner_analysis import CornerAnalyzer


@app.get("/gg-diagram")
async def gg_diagram_page(request: Request):
    """G-G Diagram page"""
    return templates.TemplateResponse("gg_diagram.html", {"request": request})


@app.get("/api/gg-diagram/{filename:path}")
async def get_gg_diagram(
    filename: str,
    format: str = "json",
    color_by: str = "speed",
    max_g: float = 1.3
):
    """
    Generate G-G diagram data or visualization.

    Args:
        filename: Parquet file path
        format: Output format ('json', 'svg')
        color_by: Color scheme ('speed', 'throttle', 'lap')
        max_g: Reference max g from vehicle config
    """
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        # Find acceleration columns
        lat_acc = _find_column(df, ['GPS LatAcc', 'LatAcc'])
        lon_acc = _find_column(df, ['GPS LonAcc', 'LonAcc'])

        if lat_acc is None or lon_acc is None:
            raise HTTPException(status_code=400, detail="Acceleration data (GPS LatAcc/LonAcc) not found")

        # Find optional columns for coloring
        speed_data = _find_column(df, ['GPS Speed', 'speed', 'Speed'])
        if speed_data is not None and speed_data.max() < 100:
            speed_data = speed_data * 2.237

        throttle_data = _find_column(df, ['PedalPos', 'throttle', 'Throttle'])

        # Run analysis
        analyzer = GGAnalyzer(max_g_reference=max_g)
        result = analyzer.analyze_from_arrays(
            df.index.values, lat_acc, lon_acc,
            speed_data=speed_data,
            throttle_data=throttle_data,
            session_id=filename
        )

        if format == 'json':
            return result.to_dict()
        else:
            # Generate SVG
            diagram = GGDiagram()

            # Get color data based on selection
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
                title=f"G-G Diagram - {Path(filename).stem}"
            )
            return HTMLResponse(content=svg, media_type="image/svg+xml")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"G-G diagram generation failed: {str(e)}")


# ============================================================
# Corner Analysis Endpoints
# ============================================================


@app.get("/corner-analysis")
async def corner_analysis_page(request: Request):
    """Corner analysis page"""
    return templates.TemplateResponse("corner_analysis.html", {"request": request})


@app.get("/api/corner-analysis/{filename:path}")
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
    import pandas as pd
    import numpy as np

    file_path = _find_parquet_file(filename)
    if not file_path:
        raise HTTPException(status_code=404, detail=f"Parquet file not found: {filename}")

    try:
        df = pd.read_parquet(file_path)

        # Check for required GPS data
        lat_data = _find_column(df, ['GPS Latitude', 'gps_lat', 'latitude'])
        lon_data = _find_column(df, ['GPS Longitude', 'gps_lon', 'longitude'])

        if lat_data is None or lon_data is None:
            raise HTTPException(status_code=400, detail="GPS latitude/longitude data not found")

        # Run corner analysis
        analyzer = CornerAnalyzer()
        result = analyzer.analyze_from_parquet(
            str(file_path),
            session_id=Path(filename).stem,
            track_name=track_name
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner analysis failed: {str(e)}")


# ============================================================
# Queue Dashboard Endpoints
# ============================================================

from src.extraction.queue import ExtractionQueue
from src.extraction.models import JobStatus

# Initialize queue (uses default db path)
_extraction_queue = None

def get_queue() -> ExtractionQueue:
    """Get or create the extraction queue singleton"""
    global _extraction_queue
    if _extraction_queue is None:
        db_path = Path(config.DATA_DIR) / "extraction_queue.db"
        _extraction_queue = ExtractionQueue(str(db_path))
    return _extraction_queue


@app.get("/queue", response_class=HTMLResponse)
async def queue_dashboard_page(request: Request):
    """Queue status dashboard page"""
    return templates.TemplateResponse("queue.html", {
        "request": request,
        "title": "Extraction Queue"
    })


@app.get("/api/queue/stats")
async def get_queue_stats():
    """Get queue statistics"""
    queue = get_queue()
    return queue.get_stats()


@app.get("/api/queue/jobs")
async def list_queue_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List jobs in the queue"""
    queue = get_queue()

    job_status = None
    if status and status != "all":
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = queue.list_jobs(status=job_status, limit=limit, offset=offset)
    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": queue.count(job_status),
        "status_filter": status
    }


@app.get("/api/queue/jobs/{job_id}")
async def get_queue_job(job_id: int):
    """Get a specific job"""
    queue = get_queue()
    job = queue.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job.to_dict()


@app.post("/api/queue/jobs/{job_id}/retry")
async def retry_queue_job(job_id: int):
    """Retry a failed job"""
    queue = get_queue()
    job = queue.retry(job_id)
    if not job:
        raise HTTPException(status_code=400, detail=f"Cannot retry job {job_id} - not found or not eligible")
    return {"success": True, "job": job.to_dict()}


@app.post("/api/queue/retry-all")
async def retry_all_failed_jobs():
    """Retry all failed jobs that are eligible"""
    queue = get_queue()
    count = queue.retry_all_failed()
    return {"success": True, "retried_count": count}


@app.delete("/api/queue/jobs/{job_id}")
async def delete_queue_job(job_id: int):
    """Delete a job from the queue"""
    queue = get_queue()
    success = queue.delete(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return {"success": True}


@app.post("/api/queue/clear-completed")
async def clear_completed_jobs():
    """Remove all completed jobs from the queue"""
    queue = get_queue()
    count = queue.clear_completed()
    return {"success": True, "cleared_count": count}


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
    """Find a column by trying multiple names, preferring columns with actual data"""
    import numpy as np

    def has_data(col_data):
        """Check if column has meaningful non-zero data"""
        if col_data is None or len(col_data) == 0:
            return False
        non_null = np.sum(~np.isnan(col_data) if np.issubdtype(col_data.dtype, np.floating) else np.ones(len(col_data), dtype=bool))
        non_zero = np.sum(col_data != 0)
        return non_null > 0 and non_zero > len(col_data) * 0.1  # At least 10% non-zero

    # First try exact matches
    for col in candidates:
        if col in df.columns:
            data = df[col].values
            if has_data(data):
                return data

    # Then try case-insensitive exact matches
    for col in candidates:
        for actual_col in df.columns:
            if actual_col.lower() == col.lower():
                data = df[actual_col].values
                if has_data(data):
                    return data

    # Then try partial/contains matches, collecting all candidates
    matching_cols = []
    for col in candidates:
        for actual_col in df.columns:
            if col.lower() in actual_col.lower():
                matching_cols.append(actual_col)

    # Return the matching column with the most non-zero data
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

    # Fallback: return first exact match even if no data
    for col in candidates:
        if col in df.columns:
            return df[col].values

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