"""
health_monitor.py
=================
VTVL Rocket — Engine & Vehicle Health Monitoring System
Continuous real-time monitoring of all 9 engines + vehicle systems.

Monitors:
  - Chamber pressure (over/under pressure)
  - Turbopump RPM (over-speed / stall)
  - Nozzle temperature
  - Propellant flow rate
  - Igniter continuity
  - Vibration / accelerometer (anomaly detection)
  - Vehicle structural loads

Outputs:
  - Health scores per engine (0.0–1.0)
  - Go/No-Go decisions
  - Fault isolation messages
  - AI anomaly flag (simple threshold + rate-of-change)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
import math


class HealthLevel(Enum):
    NOMINAL  = "NOMINAL"
    CAUTION  = "CAUTION"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class SensorReading:
    """One sensor sample at one time."""
    name: str
    value: float
    unit: str
    timestamp_s: float
    limit_low: float
    limit_high: float
    prev_value: Optional[float] = None

    def in_limits(self) -> bool:
        return self.limit_low <= self.value <= self.limit_high

    def rate_of_change(self, dt: float = 1.0) -> float:
        if self.prev_value is None or dt == 0:
            return 0.0
        return (self.value - self.prev_value) / dt

    def margin(self) -> float:
        """How far from nearest limit (0 = at limit, 1 = midpoint)."""
        span = self.limit_high - self.limit_low
        if span == 0:
            return 1.0
        dist_low  = abs(self.value - self.limit_low)
        dist_high = abs(self.value - self.limit_high)
        return min(dist_low, dist_high) / (span / 2)


@dataclass
class EngineHealthPacket:
    """Complete health picture for one engine at one moment."""
    engine_id: int
    timestamp_s: float
    chamber_pressure_bar: float
    turbopump_rpm_lox: float
    turbopump_rpm_lch4: float
    nozzle_temp_C: float
    propellant_flow_kgs: float
    vibration_g: float
    igniter_ok: bool
    health_score: float = 1.0
    level: HealthLevel = HealthLevel.NOMINAL
    faults: List[str] = field(default_factory=list)

    # Nominal limits
    P_LOW, P_HIGH   = 200.0, 275.0     # bar
    RPM_LOW, RPM_HIGH = 20_000, 42_000
    T_LOW, T_HIGH   = 500.0, 1250.0    # °C
    FLOW_LOW, FLOW_HIGH = 15.0, 30.0   # kg/s per engine
    VIB_HIGH        = 8.0              # g

    def evaluate(self) -> "EngineHealthPacket":
        """Run all limit checks and compute composite health score."""
        score = 1.0
        faults = []

        # Chamber pressure
        if not (self.P_LOW <= self.chamber_pressure_bar <= self.P_HIGH):
            score *= 0.4
            faults.append(
                f"P_chamber={self.chamber_pressure_bar:.0f}bar "
                f"[limit {self.P_LOW}–{self.P_HIGH}]"
            )

        # Turbopump RPM
        for rpm, label in [(self.turbopump_rpm_lox, "LOX"), (self.turbopump_rpm_lch4, "LCH4")]:
            if not (self.RPM_LOW <= rpm <= self.RPM_HIGH):
                score *= 0.5
                faults.append(f"RPM_{label}={rpm:.0f} out of range")

        # Nozzle temperature
        if not (self.T_LOW <= self.nozzle_temp_C <= self.T_HIGH):
            score *= 0.6
            faults.append(f"T_nozzle={self.nozzle_temp_C:.0f}°C [limit {self.T_HIGH}°C]")

        # Propellant flow
        if not (self.FLOW_LOW <= self.propellant_flow_kgs <= self.FLOW_HIGH):
            score *= 0.7
            faults.append(f"Flow={self.propellant_flow_kgs:.1f}kg/s out of range")

        # Vibration
        if self.vibration_g > self.VIB_HIGH:
            score *= max(0.1, 1.0 - (self.vibration_g - self.VIB_HIGH) / 10.0)
            faults.append(f"Vibration={self.vibration_g:.1f}g [limit {self.VIB_HIGH}g]")

        # Igniter
        if not self.igniter_ok:
            score *= 0.3
            faults.append("Igniter continuity FAILED")

        self.health_score = max(0.0, min(1.0, score))
        self.faults = faults

        if score >= 0.85:
            self.level = HealthLevel.NOMINAL
        elif score >= 0.65:
            self.level = HealthLevel.CAUTION
        elif score >= 0.45:
            self.level = HealthLevel.WARNING
        else:
            self.level = HealthLevel.CRITICAL

        return self

    def __str__(self):
        fault_str = "; ".join(self.faults) if self.faults else "none"
        return (
            f"E{self.engine_id} [{self.level.value}] H={self.health_score:.2f} "
            f"| P={self.chamber_pressure_bar:.0f}bar "
            f"T={self.nozzle_temp_C:.0f}°C "
            f"V={self.vibration_g:.1f}g "
            f"| Faults: {fault_str}"
        )


@dataclass
class VehicleHealth:
    """Top-level vehicle health aggregation."""
    timestamp_s: float
    engine_packets: Dict[int, EngineHealthPacket] = field(default_factory=dict)
    structural_load_fraction: float = 0.0   # 0–1 (1 = design limit)
    tps_temp_max_C: float = 20.0
    propellant_remaining_kg: float = 10_000.0
    gps_signal_ok: bool = True
    imu_ok: bool = True
    lidar_ok: bool = True

    def overall_go(self) -> Tuple[bool, str]:
        """Returns (go, reason). Must be GO for launch/burn authorisation."""
        for eid, pkt in self.engine_packets.items():
            if pkt.level == HealthLevel.CRITICAL:
                return False, f"E{eid} CRITICAL: {'; '.join(pkt.faults)}"
        if not self.imu_ok:
            return False, "IMU failure — no attitude reference"
        if self.structural_load_fraction > 0.95:
            return False, f"Structural load at {self.structural_load_fraction:.0%} — HOLD"
        if self.tps_temp_max_C > 1300:
            return False, f"TPS over-temperature: {self.tps_temp_max_C:.0f}°C"
        return True, "GO"

    def min_engine_health(self) -> float:
        if not self.engine_packets:
            return 0.0
        return min(p.health_score for p in self.engine_packets.values())

    def report(self) -> str:
        lines = [
            f"{'='*60}",
            f"  VEHICLE HEALTH REPORT  T+{self.timestamp_s:.1f}s",
            f"{'='*60}",
        ]
        for eid, pkt in sorted(self.engine_packets.items()):
            lines.append(f"  {pkt}")
        go, reason = self.overall_go()
        lines += [
            f"{'─'*60}",
            f"  IMU: {'OK' if self.imu_ok else 'FAIL'}  "
            f"GPS: {'OK' if self.gps_signal_ok else 'FAIL'}  "
            f"LIDAR: {'OK' if self.lidar_ok else 'FAIL'}",
            f"  TPS peak: {self.tps_temp_max_C:.0f}°C  "
            f"  Struct load: {self.structural_load_fraction:.0%}",
            f"  Min engine health: {self.min_engine_health():.2f}",
            f"  OVERALL: {'>>> GO <<<' if go else '>>> NO-GO: ' + reason + ' <<<'}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ── Health Monitor ─────────────────────────────────────────────────────────

class HealthMonitor:
    """
    Continuous monitoring loop.
    Call ingest_engine() every sample tick (target 1 kHz in hardware, 10 Hz in sim).
    Raises anomaly callbacks to FlightPhaseController.
    """

    def __init__(self, anomaly_callback=None):
        self._history: Dict[int, List[EngineHealthPacket]] = {i: [] for i in range(9)}
        self._anomaly_cb = anomaly_callback or (lambda eid, fault: None)
        self._rate_window = 5   # samples for rate-of-change check

    def ingest_engine(self, pkt: EngineHealthPacket) -> EngineHealthPacket:
        pkt.evaluate()
        self._history[pkt.engine_id].append(pkt)

        # AI-style rate-of-change anomaly detection
        history = self._history[pkt.engine_id]
        if len(history) >= self._rate_window:
            pressures = [h.chamber_pressure_bar for h in history[-self._rate_window:]]
            dP = pressures[-1] - pressures[0]
            if abs(dP) > 30:   # >30 bar change in window = anomaly
                fault = f"P_rate_anomaly: delta={dP:+.0f}bar in {self._rate_window} samples"
                pkt.faults.append(fault)
                pkt.health_score *= 0.6
                pkt.level = HealthLevel.WARNING
                self._anomaly_cb(pkt.engine_id, fault)

        if pkt.level in (HealthLevel.CRITICAL, HealthLevel.WARNING):
            self._anomaly_cb(pkt.engine_id, f"{pkt.level.value}: {'; '.join(pkt.faults)}")

        return pkt

    def build_vehicle_health(self, t: float) -> VehicleHealth:
        latest = {}
        for eid, hist in self._history.items():
            if hist:
                latest[eid] = hist[-1]
        return VehicleHealth(timestamp_s=t, engine_packets=latest)

    def preflight_check(self, packets: Dict[int, EngineHealthPacket]) -> Tuple[bool, List[str]]:
        """
        Pre-launch GO/NO-GO — cold (standby) engine checks only.
        Engines are NOT lit; checks igniter, vibration, and sensors.
        Returns (all_go, list_of_issues).
        """
        issues = []
        for eid, pkt in packets.items():
            faults = []
            if not pkt.igniter_ok:
                faults.append("Igniter continuity FAILED")
            if pkt.vibration_g > EngineHealthPacket.VIB_HIGH:
                faults.append(f"Vibration={pkt.vibration_g:.1f}g > limit {EngineHealthPacket.VIB_HIGH}g")
            if faults:
                issues.append(f"E{eid}: NO-GO — {'; '.join(faults)}")
        return (len(issues) == 0, issues)


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    def my_anomaly_cb(eid, fault):
        print(f"  [ANOMALY CALLBACK] Engine {eid}: {fault}")

    monitor = HealthMonitor(anomaly_callback=my_anomaly_cb)

    print("="*60)
    print("  HEALTH MONITOR — PREFLIGHT CHECK")
    print("="*60)

    preflight_packets = {}
    for i in range(9):
        preflight_packets[i] = EngineHealthPacket(
            engine_id=i, timestamp_s=0.0,
            chamber_pressure_bar=0.0,   # Not ignited
            turbopump_rpm_lox=0.0,
            turbopump_rpm_lch4=0.0,
            nozzle_temp_C=20.0,
            propellant_flow_kgs=0.0,
            vibration_g=0.1,
            igniter_ok=True
        )
    # Simulate E6 igniter failure
    preflight_packets[6].igniter_ok = False

    go, issues = monitor.preflight_check(preflight_packets)
    print(f"\nPreflight result: {'GO' if go else 'NO-GO'}")
    for issue in issues:
        print(f"  ISSUE: {issue}")

    print("\n" + "="*60)
    print("  IN-FLIGHT MONITORING (10 sample ticks)")
    print("="*60)

    for tick in range(10):
        t = tick * 0.1
        for i in range(9):
            # Simulate nominal + slight noise
            vib = random.uniform(0.5, 2.0)
            if i == 3 and tick >= 7:
                # Simulate E3 developing a fault
                vib = 12.0
            pkt = EngineHealthPacket(
                engine_id=i, timestamp_s=t,
                chamber_pressure_bar=250 + random.uniform(-5, 5),
                turbopump_rpm_lox=35_000 + random.uniform(-500, 500),
                turbopump_rpm_lch4=40_000 + random.uniform(-500, 500),
                nozzle_temp_C=1050 + random.uniform(-20, 20),
                propellant_flow_kgs=25.0 + random.uniform(-0.5, 0.5),
                vibration_g=vib,
                igniter_ok=True
            )
            monitor.ingest_engine(pkt)

    vh = monitor.build_vehicle_health(1.0)
    print(vh.report())
