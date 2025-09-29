# If analysis needs runtime paths, use config.py
from src.config.config import DLL_PATH, DEPENDENCY_PATH, SAMPLE_FILES_PATH, UNITS_XML_PATH
from src.config.vehicle_config import PROCESSING_CONFIG, TRACK_CONFIG


# If analysis needs vehicle/track constants, import from vehicle_config instead
# from src.config.vehicle_config import PROCESSING_CONFIG, TRACK_CONFIG, ENGINE_SPECS, ...
