# If analysis needs runtime paths, use config.py
from src.config.config import DLL_PATH, DEPENDENCY_PATH, SAMPLE_FILES_PATH, UNITS_XML_PATH
from src.config.vehicles import get_processing_config
from src.config.tracks import get_track_config


# If analysis needs vehicle/track constants, import from vehicles.py and tracks.py
# from src.config.vehicles import get_engine_specs, get_current_setup, ...
# from src.config.tracks import get_track_config, get_track_database, ...
