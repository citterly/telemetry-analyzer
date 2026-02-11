"""
Features module for telemetry analysis
Contains high-level analysis features built on core modules
"""

from .base_analyzer import BaseAnalyzer, BaseAnalysisReport
from .shift_analysis import ShiftAnalyzer, ShiftReport
from .transmission_comparison import TransmissionComparison, TransmissionComparisonReport
from .lap_analysis import LapAnalysis, LapAnalysisReport
from .gear_analysis import GearAnalysis, GearAnalysisReport
from .power_analysis import PowerAnalysis, PowerAnalysisReport
from .gg_analysis import GGAnalyzer, GGAnalysisResult
from .corner_analysis import CornerAnalyzer, CornerAnalysisResult
from .session_report import SessionReportGenerator, SessionReport
from .registry import AnalyzerRegistry, AnalyzerRegistration, analyzer_registry

__all__ = [
    'BaseAnalyzer',
    'BaseAnalysisReport',
    'ShiftAnalyzer',
    'ShiftReport',
    'TransmissionComparison',
    'TransmissionComparisonReport',
    'LapAnalysis',
    'LapAnalysisReport',
    'GearAnalysis',
    'GearAnalysisReport',
    'PowerAnalysis',
    'PowerAnalysisReport',
    'GGAnalyzer',
    'GGAnalysisResult',
    'CornerAnalyzer',
    'CornerAnalysisResult',
    'SessionReportGenerator',
    'SessionReport',
    'AnalyzerRegistry',
    'AnalyzerRegistration',
    'analyzer_registry',
]
