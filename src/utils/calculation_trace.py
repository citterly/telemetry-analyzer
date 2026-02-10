"""
Calculation traceability and sanity check infrastructure.

Provides CalculationTrace and SanityCheck dataclasses for recording
what inputs, constants, and intermediate values produced each analysis
result, and for validating results against physical constraints.

Usage:
    trace = CalculationTrace(analyzer_name="PowerAnalysis", ...)
    trace.record_input("speed_column", "GPS Speed")
    trace.record_config("vehicle_mass_kg", 1565.0)
    trace.record_intermediate("max_raw_power_hp", 342.5)
    trace.add_check("max_power_plausible", "pass", "342 HP is reasonable")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class SanityCheck:
    """A single validation check on a calculation result."""
    name: str
    status: str  # "pass", "warn", "fail"
    message: str
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    severity: str = "warning"  # "info", "warning", "error"

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "severity": self.severity,
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.actual is not None:
            result["actual"] = self.actual
        return result


@dataclass
class CalculationTrace:
    """Records the full trace of a calculation for debugging.

    Attached to analysis reports when include_trace=True is passed
    to an analyzer. Zero overhead when not requested.
    """
    analyzer_name: str
    timestamp: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Any] = field(default_factory=dict)
    sanity_checks: List[SanityCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """True if any sanity check has status 'fail'."""
        return any(c.status == "fail" for c in self.sanity_checks)

    @property
    def has_warnings(self) -> bool:
        """True if any sanity check has status 'warn'."""
        return any(c.status == "warn" for c in self.sanity_checks)

    def add_check(self, name: str, status: str, message: str,
                  expected: Optional[Any] = None, actual: Optional[Any] = None,
                  severity: str = "warning") -> None:
        """Append a SanityCheck."""
        self.sanity_checks.append(SanityCheck(
            name=name, status=status, message=message,
            expected=expected, actual=actual, severity=severity,
        ))

    def record_input(self, key: str, value: Any) -> None:
        """Record an input used in the calculation."""
        self.inputs[key] = value

    def record_config(self, key: str, value: Any) -> None:
        """Record a config value applied."""
        self.config[key] = value

    def record_intermediate(self, key: str, value: Any) -> None:
        """Record a key intermediate value."""
        self.intermediates[key] = value

    def to_dict(self) -> dict:
        return {
            "analyzer_name": self.analyzer_name,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "config": self.config,
            "intermediates": self.intermediates,
            "sanity_checks": [c.to_dict() for c in self.sanity_checks],
            "warnings": self.warnings,
            "has_failures": self.has_failures,
            "has_warnings": self.has_warnings,
        }
