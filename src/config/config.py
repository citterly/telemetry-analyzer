"""
Configuration module for Telemetry Analyzer
Environment-based configuration for portability across different machines
"""

import os
from pathlib import Path
from typing import Optional

# --- Test Defaults ---
BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_PATH / "data" / "uploads"

# Used by tests/test_file_manager.py and test_session_builder.py
SAMPLE_FILES_PATH = DATA_PATH
DEFAULT_SESSION = str(DATA_PATH / "20250712_104619_Road America_a_0394.xrk")

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # go up two levels from src/config
THIRD_PARTY_ROOT = PROJECT_ROOT / "third-party" / "AIM"
SRC_ROOT         = PROJECT_ROOT / "src"
DATA_ROOT        = PROJECT_ROOT / "data"
SAMPLE_FILES_PATH = DATA_ROOT / "uploads"

DLL_PATH        = THIRD_PARTY_ROOT / "DLL-2022" / "MatLabXRK-2022-64-ReleaseU.dll"
DEPENDENCY_PATH = THIRD_PARTY_ROOT / "64"
UNITS_XML_PATH  = SRC_ROOT / "analysis" / "units.xml"
UPLOADS_PATH    = DATA_ROOT / "uploads"
EXPORTS_PATH    = DATA_ROOT / "exports"


class Config:
    """Main configuration class with environment variable overrides"""
    
    # Data storage paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = os.getenv('DATA_DIR', str(BASE_DIR / 'data'))
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', f'{DATA_DIR}/uploads')
    METADATA_DIR = os.getenv('METADATA_DIR', f'{DATA_DIR}/metadata')
    CACHE_DIR = os.getenv('CACHE_DIR', f'{DATA_DIR}/cache')
    EXPORTS_DIR = os.getenv('EXPORTS_DIR', f'{DATA_DIR}/exports')
    
    # Database (SQLite for portability, PostgreSQL for production)
    DATABASE_URL = os.getenv('DATABASE_URL', f'sqlite:///{DATA_DIR}/sessions.db')
    
    # Web application settings
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    HOST = os.getenv('HOST', 'localhost')
    PORT = int(os.getenv('PORT', 8000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # File upload limits
    MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 100))
    ALLOWED_EXTENSIONS = {'.xrk'}
    
    # Analysis settings (from your existing analysis/config.py)
    TIRE_CIRCUMFERENCE_METERS = 2.026
    ENGINE_SAFE_RPM_LIMIT = 7000
    ENGINE_MAX_RPM = 8000
    
    # Track settings
    DEFAULT_TRACK_NAME = os.getenv('TRACK_NAME', 'Road America')
    
    # Third-party paths (for AIM DLL - adjust as needed)
    THIRD_PARTY_ROOT = BASE_DIR.parent / "third-party" / "AIM"
    DLL_PATH = THIRD_PARTY_ROOT / "DLL-2022" / "MatLabXRK-2022-64-ReleaseU.dll"
    DEPENDENCY_PATH = THIRD_PARTY_ROOT / "64"
    
    @classmethod
    def init_app(cls):
        """Initialize application directories and settings"""
        # Create data directories
        for directory in [cls.DATA_DIR, cls.UPLOAD_DIR, cls.METADATA_DIR, 
                         cls.CACHE_DIR, cls.EXPORTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print(f"Telemetry Analyzer Configuration:")
        print(f"  Data directory: {cls.DATA_DIR}")
        print(f"  Debug mode: {cls.DEBUG}")
        print(f"  Web server: {cls.HOST}:{cls.PORT}")
        print(f"  Max upload: {cls.MAX_UPLOAD_SIZE_MB}MB")
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path for a file in the data directory"""
        return Path(cls.DATA_DIR) / filename
    
    @classmethod
    def get_upload_path(cls, filename: str) -> Path:
        """Get full path for an uploaded file"""
        return Path(cls.UPLOAD_DIR) / filename


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True
    HOST = 'localhost'
    PORT = 8000


class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = int(os.getenv('PORT', 80))
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set in production
    
    @classmethod
    def init_app(cls):
        super().init_app()
        
        # Production validations
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            raise ValueError("SECRET_KEY must be set in production")


class TestConfig(Config):
    """Test-specific configuration"""
    DEBUG = True
    DATA_DIR = '/tmp/telemetry_test_data'
    DATABASE_URL = 'sqlite:///:memory:'


# Configuration selection
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'default')
    
    return config_map.get(config_name, DevelopmentConfig)


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    config.init_app()
    
    print("\nTesting configuration paths:")
    print(f"Upload path for 'test.xrk': {config.get_upload_path('test.xrk')}")
    print(f"Data path for 'cache/results.json': {config.get_data_path('cache/results.json')}")
    
    print(f"\nEnvironment variables that can override settings:")
    print(f"  DATA_DIR - Data storage location (current: {config.DATA_DIR})")
    print(f"  HOST - Web server host (current: {config.HOST})")
    print(f"  PORT - Web server port (current: {config.PORT})")
    print(f"  DEBUG - Debug mode (current: {config.DEBUG})")
    print(f"  TRACK_NAME - Default track name (current: {config.DEFAULT_TRACK_NAME})")