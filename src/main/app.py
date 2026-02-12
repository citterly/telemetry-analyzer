"""
Main FastAPI application for Telemetry Analyzer
Minimal web interface for trackside use
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import Optional
import traceback

from src.config.config import get_config
from src.io.file_manager import FileManager

# Import routers
from src.main.routers import (
    analysis,
    context,
    parquet,
    queue,
    sessions,
    vehicles,
    visualization,
)


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

# Include API routers
app.include_router(analysis.router)
app.include_router(context.router)
app.include_router(parquet.router)
app.include_router(queue.router)
app.include_router(sessions.router)
app.include_router(vehicles.router)
app.include_router(visualization.router)


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


# ============================================================
# HTML Page Routes
# ============================================================

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


@app.get("/gg-diagram")
async def gg_diagram_page(request: Request):
    """G-G Diagram page"""
    return templates.TemplateResponse("gg_diagram.html", {"request": request})


@app.get("/corner-analysis")
async def corner_analysis_page(request: Request):
    """Corner analysis page"""
    return templates.TemplateResponse("corner_analysis.html", {"request": request})


@app.get("/track-map")
async def track_map_page(request: Request):
    """Track map visualization page"""
    return templates.TemplateResponse("track_map.html", {
        "request": request,
        "title": "Track Map",
        "svg_content": "",
        "laps": [],
        "stats": {},
    })


@app.get("/queue", response_class=HTMLResponse)
async def queue_dashboard_page(request: Request):
    """Queue status dashboard page"""
    return templates.TemplateResponse("queue.html", {
        "request": request,
        "title": "Extraction Queue"
    })


@app.get("/vehicles")
async def vehicles_page(request: Request):
    """Vehicle settings page"""
    return templates.TemplateResponse("vehicles.html", {"request": request})


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    """Session browser page"""
    return templates.TemplateResponse("sessions.html", {
        "request": request,
        "title": "Sessions",
    })


@app.get("/sessions/import", response_class=HTMLResponse)
async def session_import_page(request: Request):
    """Session import wizard page"""
    return templates.TemplateResponse("session_import.html", {
        "request": request,
        "title": "Import Session",
    })


@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail_page(request: Request, session_id: int):
    """Session detail page"""
    db = sessions.get_session_db()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return templates.TemplateResponse("session_detail.html", {
        "request": request,
        "title": f"Session: {session.track_name or 'Unknown'} - {session.session_date or 'Unknown'}",
        "session": session.to_dict(),
        "session_id": session_id,
    })


# ============================================================
# File Management API Endpoints
# ============================================================

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
            # Log traceback server-side only, never expose to client
            import sys
            print(f"ERROR: {traceback.format_exc()}", file=sys.stderr)
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


# ============================================================
# Error Handlers & Health Check
# ============================================================

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
