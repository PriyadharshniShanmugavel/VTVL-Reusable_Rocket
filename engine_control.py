"""
engine_control.py
=================
VTVL Hybrid Rocket — Engine Control Logic
Implements core rules:
  - Opposite-engine shutdown balance law
  - Phase-based engine selection (9 → 3 → 1)
  - Engine failure detection & compensating shutdown
  - Landing engine selection
  - Health-gated ignition
  - State machine: STANDBY → IGNITION → RUNNING → SHUTDOWN → FAILED
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time


# ── Engine State Machine ────────────────────────────────────────────────────

class EngineState(Enum):
    STANDBY   = auto()   # Ready, not lit
    IGNITION  = auto()   # Starting sequence in progress
    RUNNING   = auto()   # Nominal combustion
    THROTTLING = auto()  # Changing thrust level
    SHUTDOWN  = auto()   # Commanded shutdown (safe)
    FAILED    = auto()   # Anomaly detected — do not restart


# ── Per-Engine Data ─────────────────────────────────────────────────────────

@dataclass
class EngineUnit:
    """Represents one physical engine in the cluster."""
    id: int                          # 0 = centre; 1-8 = outer ring
    state: EngineState = EngineState.STANDBY
    throttle: float = 0.0            # 0.0 – 1.0
    chamber_pressure_bar: float = 0.0
    turbopump_rpm: float = 0.0
    nozzle_temp_C: float = 20.0
    health_score: float = 1.0        # 1.0 = perfect; <0.5 = critical
    cycles: int = 0                  # Total hot-fire cycles on this unit
    fault_codes: List[str] = field(default_factory=list)

    # Physical limits
    MAX_CHAMBER_PRESSURE = 275       # bar  (nominal 250 bar)
    MIN_THROTTLE = 0.40              # 40 % — below this: combustion instability
    MAX_THROTTLE = 1.00
    MAX_NOZZLE_TEMP_C = 1250
    NOMINAL_RPM_LOX = 35_000
    NOMINAL_RPM_LCH4 = 40_000

    def is_healthy(self) -> bool:
        return (
            self.health_score >= 0.5
            and self.state != EngineState.FAILED
            and self.chamber_pressure_bar <= self.MAX_CHAMBER_PRESSURE
            and self.nozzle_temp_C <= self.MAX_NOZZLE_TEMP_C
        )

    def thrust_kN(self) -> float:
        """Vacuum thrust produced by this engine right now."""
        if self.state != EngineState.RUNNING:
            return 0.0
        return 100.0 * self.throttle   # 100 kN at full throttle

    def __repr__(self):
        return (f"Engine E{self.id} [{self.state.name}] "
                f"T={self.throttle:.0%} P={self.chamber_pressure_bar:.0f}bar "
                f"H={self.health_score:.2f}")


# ── Cluster Geometry ─────────────────────────────────────────────────────────
# Opposite-engine map: for any outer engine i, its structural opposite
# is the engine 4 positions away in the ring (180° apart).
OPPOSITE: Dict[int, int] = {
    1: 5, 5: 1,
    2: 6, 6: 2,
    3: 7, 7: 3,
    4: 8, 8: 4,
    0: 0,          # Centre has no opposite — shutdown is asymmetric-forbidden
}

# Phase → set of engine IDs that should be RUNNING
PHASE_ENGINES = {
    "LIFTOFF":   {0, 1, 2, 3, 4, 5, 6, 7, 8},   # All 9
    "MAX_Q":     {0, 1, 2, 3, 4, 5, 6, 7, 8},   # All 9
    "MECO_PREP": {0, 1, 2, 3, 4, 5, 6, 7, 8},   # Throttle-down, still all 9
    "BOOSTBACK": {0, 2, 6},                       # 3 — symmetric opposite pair
    "ENTRY":     {0, 2, 6},                       # 3 — same subset
    "LANDING":   {0, 1, 5},                       # 3 — symmetric trio
    "HOVSLAM":   {0},                             # 1 — centre only
    "ABORT":     {0, 1, 2, 3, 4, 5, 6, 7, 8},   # Max thrust for abort
}

# Phase → throttle level (fraction)
PHASE_THROTTLE = {
    "LIFTOFF":   1.00,
    "MAX_Q":     0.95,
    "MECO_PREP": 0.70,
    "BOOSTBACK": 0.85,
    "ENTRY":     0.65,
    "LANDING":   0.55,
    "HOVSLAM":   0.42,
    "ABORT":     1.00,
}


# ── Engine Controller ────────────────────────────────────────────────────────

class EngineController:
    """
    Central authority for all 9 engines.
    Enforces:
      1. Opposite-pair shutdown balance law
      2. Minimum 3-engine symmetric configuration (when >1 engine)
      3. Health-gated ignition (won't light a failed engine)
      4. Failure compensation (shuts opposite partner on detected failure)
      5. Centre-engine protection (E0 is last to shut, first to light)
    """

    def __init__(self):
        self.engines: Dict[int, EngineUnit] = {
            i: EngineUnit(id=i) for i in range(9)
        }
        self.current_phase: str = "STANDBY"
        self.events: List[str] = []
        self._t = 0.0   # Simulation time (s)

    # ── Public API ───────────────────────────────────────────────────────────

    def transition_phase(self, new_phase: str) -> List[str]:
        """
        Command a flight-phase transition.
        Returns list of action strings taken.
        """
        if new_phase not in PHASE_ENGINES:
            raise ValueError(f"Unknown phase: {new_phase}")

        actions = []
        target_ids = PHASE_ENGINES[new_phase]
        target_throttle = PHASE_THROTTLE[new_phase]

        # --- Step 1: Light engines that should be on ---
        for eid in sorted(target_ids):
            eng = self.engines[eid]
            if eng.state in (EngineState.STANDBY, EngineState.SHUTDOWN):
                if eng.is_healthy():
                    result = self._ignite(eid, target_throttle)
                    actions.append(result)
                else:
                    actions.append(f"  [WARN] E{eid} unhealthy — skip ignition (H={eng.health_score:.2f})")
                    # If unhealthy engine needed, escalate to abort
                    if eid == 0:
                        actions += self._abort_engine_out(eid)
            elif eng.state == EngineState.RUNNING:
                self._set_throttle(eid, target_throttle)
                actions.append(f"  E{eid} throttle → {target_throttle:.0%}")

        # --- Step 2: Shut down engines not in target set ---
        for eid in range(9):
            if eid not in target_ids:
                eng = self.engines[eid]
                if eng.state == EngineState.RUNNING:
                    result = self._shutdown(eid, reason="phase-transition")
                    actions.append(result)

        self.current_phase = new_phase
        self._log(f"Phase → {new_phase} | Active: {self.active_engine_ids()}")
        return actions

    def report_sensor_anomaly(self, engine_id: int, fault: str) -> List[str]:
        """
        Health monitor calls this when a sensor breach is detected.
        Implements failure isolation + balance compensation.
        """
        eng = self.engines[engine_id]
        eng.fault_codes.append(fault)
        eng.health_score *= 0.5
        actions = [f"  [FAULT] E{engine_id}: {fault} (H→{eng.health_score:.2f})"]

        if eng.health_score < 0.5 and eng.state == EngineState.RUNNING:
            actions += self._emergency_shutdown(engine_id)

        return actions

    def active_engine_ids(self) -> List[int]:
        return [i for i, e in self.engines.items() if e.state == EngineState.RUNNING]

    def total_thrust_kN(self) -> float:
        return sum(e.thrust_kN() for e in self.engines.values())

    def cluster_balance_error(self) -> Tuple[float, float]:
        """
        Returns (x_offset, y_offset) of thrust centroid from vehicle axis.
        Zero = perfectly balanced.
        Engine positions (normalised unit circle, 0-indexed outer ring):
        """
        import math
        positions = {
            0: (0.0, 0.0),
            1: (0.0, -1.0),
            2: (math.sin(math.pi/4),  -math.cos(math.pi/4)),
            3: (1.0, 0.0),
            4: (math.sin(math.pi/4),   math.cos(math.pi/4)),
            5: (0.0, 1.0),
            6: (-math.sin(math.pi/4),  math.cos(math.pi/4)),
            7: (-1.0, 0.0),
            8: (-math.sin(math.pi/4), -math.cos(math.pi/4)),
        }
        total_thrust = self.total_thrust_kN()
        if total_thrust == 0:
            return (0.0, 0.0)

        cx = sum(self.engines[i].thrust_kN() * positions[i][0] for i in range(9))
        cy = sum(self.engines[i].thrust_kN() * positions[i][1] for i in range(9))
        return (cx / total_thrust, cy / total_thrust)

    def status_report(self) -> str:
        lines = [
            f"{'='*60}",
            f"  ENGINE CLUSTER STATUS  |  Phase: {self.current_phase}",
            f"{'='*60}",
        ]
        for i, eng in self.engines.items():
            label = "CTR" if i == 0 else f"OUT"
            star = "*" if eng.state == EngineState.RUNNING else " "
            lines.append(
                f"  {star}E{i} [{label}] {eng.state.name:<12} "
                f"T={eng.throttle:.0%}  P={eng.chamber_pressure_bar:>5.0f}bar  "
                f"H={eng.health_score:.2f}  RPM={eng.turbopump_rpm:>6.0f}"
            )
        bx, by = self.cluster_balance_error()
        lines += [
            f"{'─'*60}",
            f"  Total thrust : {self.total_thrust_kN():>7.1f} kN",
            f"  Active engines: {len(self.active_engine_ids())} / 9",
            f"  Balance error : ({bx:+.4f}, {by:+.4f})  (target 0,0)",
            f"  Faults logged : {sum(len(e.fault_codes) for e in self.engines.values())}",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ignite(self, eid: int, throttle: float) -> str:
        eng = self.engines[eid]
        eng.state = EngineState.IGNITION
        time.sleep(0.001)   # Simulate 50 ms ignition sequence
        eng.state = EngineState.RUNNING
        eng.throttle = max(eng.MIN_THROTTLE, throttle)
        eng.chamber_pressure_bar = 250.0 * eng.throttle
        eng.turbopump_rpm = eng.NOMINAL_RPM_LOX * eng.throttle
        eng.nozzle_temp_C = 800.0 + 400.0 * eng.throttle
        eng.cycles += 1
        return f"  E{eid} IGNITED at {throttle:.0%} throttle"

    def _set_throttle(self, eid: int, throttle: float):
        eng = self.engines[eid]
        if eng.state != EngineState.RUNNING:
            return
        eng.throttle = max(eng.MIN_THROTTLE, min(eng.MAX_THROTTLE, throttle))
        eng.chamber_pressure_bar = 250.0 * eng.throttle
        eng.turbopump_rpm = eng.NOMINAL_RPM_LOX * eng.throttle
        eng.nozzle_temp_C = 800.0 + 400.0 * eng.throttle
        eng.state = EngineState.THROTTLING
        eng.state = EngineState.RUNNING

    def _shutdown(self, eid: int, reason: str = "commanded") -> str:
        eng = self.engines[eid]
        eng.state = EngineState.SHUTDOWN
        eng.throttle = 0.0
        eng.chamber_pressure_bar = 0.0
        eng.turbopump_rpm = 0.0
        eng.nozzle_temp_C = max(20.0, eng.nozzle_temp_C - 200.0)
        return f"  E{eid} SHUTDOWN ({reason})"

    def _emergency_shutdown(self, eid: int) -> List[str]:
        """
        RULE 1 — Opposite-pair balance law:
        If an outer engine fails, immediately shut its diametrically
        opposite partner to maintain thrust symmetry.
        """
        actions = []
        eng = self.engines[eid]
        eng.state = EngineState.FAILED

        result = self._shutdown(eid, reason="FAILURE")
        actions.append(f"  [EMERG] {result}")

        opp = OPPOSITE.get(eid)
        if opp and opp != eid and opp != 0:
            opp_eng = self.engines[opp]
            if opp_eng.state == EngineState.RUNNING:
                result2 = self._shutdown(opp, reason="balance-compensation")
                actions.append(f"  [BALANCE] {result2}")
        elif eid == 0:
            # Centre engine failure during landing = critical abort
            actions += self._abort_engine_out(eid)

        self._log(f"ENGINE FAILURE E{eid} — balance compensation applied")
        return actions

    def _abort_engine_out(self, failed_eid: int) -> List[str]:
        """Critical abort: E0 failure near landing has no recovery."""
        actions = [f"  [CRITICAL ABORT] E0 failure — FTS arm, emergency protocol"]
        for i in range(9):
            if i != failed_eid and self.engines[i].state == EngineState.RUNNING:
                actions.append(self._shutdown(i, reason="abort-safing"))
        return actions

    def _log(self, msg: str):
        self.events.append(f"T+{self._t:.1f}s | {msg}")

    def advance_time(self, dt: float):
        self._t += dt


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctrl = EngineController()

    print("\n" + "="*60)
    print("  VTVL ENGINE CONTROLLER — FULL MISSION SEQUENCE")
    print("="*60)

    phases = [
        ("LIFTOFF",   2.0),
        ("MAX_Q",     35.0),
        ("MECO_PREP", 5.0),
        ("BOOSTBACK", 40.0),
        ("ENTRY",     60.0),
        ("LANDING",   20.0),
        ("HOVSLAM",   15.0),
    ]

    for phase, duration in phases:
        print(f"\n>>> Entering phase: {phase}")
        actions = ctrl.transition_phase(phase)
        for a in actions:
            print(a)
        ctrl.advance_time(duration)

    print("\n" + ctrl.status_report())

    print("\n>>> Simulating engine failure on E3 during ENTRY burn...")
    fault_actions = ctrl.report_sensor_anomaly(3, "chamber_pressure_spike: 290bar")
    for a in fault_actions:
        print(a)

    print("\n>>> Cluster balance after failure:")
    bx, by = ctrl.cluster_balance_error()
    print(f"  Balance error: ({bx:+.4f}, {by:+.4f})")
    print("\n" + ctrl.status_report())
