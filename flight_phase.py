"""
flight_phase.py
===============
VTVL Hybrid Rocket — Flight Phase Controller
Manages the full mission state machine from pre-launch to touchdown.

States:
  PRE_LAUNCH → LIFTOFF → MAX_Q → MECO → FLIP → BOOSTBACK →
  COAST → ENTRY → DESCENT → LANDING → HOVSLAM → TOUCHDOWN → SAFED

Triggers:
  - Altitude thresholds
  - Velocity thresholds
  - Timer overrides
  - Sensor inputs (IMU, altimeter, LIDAR)
  - Engine health status
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from engine_control import EngineController, PHASE_THROTTLE


# ── Mission State Machine ────────────────────────────────────────────────────

class FlightPhase(Enum):
    PRE_LAUNCH   = auto()
    LIFTOFF      = auto()
    MAX_Q        = auto()
    MECO_PREP    = auto()   # Throttle-down before cutoff
    MECO         = auto()   # Main engine cutoff
    FLIP         = auto()   # 180° attitude flip using cold-gas RCS
    BOOSTBACK    = auto()   # Engine relight, trajectory reversal
    COAST        = auto()   # Ballistic coast (no thrust)
    ENTRY        = auto()   # Re-entry deceleration burn
    SUBSONIC     = auto()   # Below Mach 1, grid fins primary
    LANDING      = auto()   # 3-engine landing burn
    HOVSLAM      = auto()   # Single engine final descent
    TOUCHDOWN    = auto()   # Contact
    SAFED        = auto()   # Post-landing safe state
    ABORT        = auto()   # Emergency abort at any phase


# ── Flight Conditions (from sensors) ────────────────────────────────────────

@dataclass
class FlightConditions:
    altitude_m: float = 0.0
    velocity_ms: float = 0.0          # Positive = upward
    mach: float = 0.0
    dynamic_pressure_Pa: float = 0.0
    pitch_deg: float = 90.0           # 90° = vertical
    roll_deg: float = 0.0
    propellant_remaining_kg: float = 10_000.0
    propellant_total_kg: float = 10_000.0
    time_s: float = 0.0
    # Landing zone
    lateral_offset_m: float = 0.0
    lidar_altitude_m: float = 0.0

    def propellant_fraction(self) -> float:
        return self.propellant_remaining_kg / self.propellant_total_kg

    def is_landing_approach(self) -> bool:
        return self.altitude_m < 1500 and self.velocity_ms > -500

    def __str__(self):
        return (
            f"T+{self.time_s:.1f}s | Alt={self.altitude_m/1000:.1f}km "
            f"| V={self.velocity_ms:.0f}m/s | M={self.mach:.2f} "
            f"| Pitch={self.pitch_deg:.1f}° | Prop={self.propellant_fraction():.1%}"
        )


# ── Phase Transition Rules ────────────────────────────────────────────────────

@dataclass
class TransitionRule:
    """One guard condition for leaving a phase."""
    description: str
    condition: Callable[[FlightConditions], bool]
    next_phase: FlightPhase


PHASE_TRANSITIONS = {
    FlightPhase.PRE_LAUNCH: [
        TransitionRule(
            "Ignition command + all engines healthy",
            lambda c: c.time_s >= 0.0,
            FlightPhase.LIFTOFF
        ),
    ],
    FlightPhase.LIFTOFF: [
        TransitionRule(
            "Altitude > 1 km",
            lambda c: c.altitude_m > 1_000,
            FlightPhase.MAX_Q
        ),
    ],
    FlightPhase.MAX_Q: [
        TransitionRule(
            "Dynamic pressure falling (alt > 15 km)",
            lambda c: c.altitude_m > 15_000,
            FlightPhase.MECO_PREP
        ),
    ],
    FlightPhase.MECO_PREP: [
        TransitionRule(
            "Target velocity reached OR propellant at MECO level",
            lambda c: c.velocity_ms > 3_000 or c.propellant_fraction() < 0.20,
            FlightPhase.MECO
        ),
    ],
    FlightPhase.MECO: [
        TransitionRule(
            "Engines shut — begin flip",
            lambda c: c.time_s > 0.5,
            FlightPhase.FLIP
        ),
    ],
    FlightPhase.FLIP: [
        TransitionRule(
            "Pitch ≤ -85° (inverted) — flip complete",
            lambda c: c.pitch_deg <= -85.0,
            FlightPhase.BOOSTBACK
        ),
    ],
    FlightPhase.BOOSTBACK: [
        TransitionRule(
            "Velocity reversed toward landing site",
            lambda c: c.velocity_ms < 200 and c.lateral_offset_m < 2_000,
            FlightPhase.COAST
        ),
    ],
    FlightPhase.COAST: [
        TransitionRule(
            "Altitude < 70 km descending — begin entry burn",
            lambda c: c.altitude_m < 70_000 and c.velocity_ms < 0,
            FlightPhase.ENTRY
        ),
    ],
    FlightPhase.ENTRY: [
        TransitionRule(
            "Mach < 1.5 — entry burn complete",
            lambda c: c.mach < 1.5,
            FlightPhase.SUBSONIC
        ),
    ],
    FlightPhase.SUBSONIC: [
        TransitionRule(
            "Altitude < 2 km — initiate landing burn",
            lambda c: c.altitude_m < 2_000,
            FlightPhase.LANDING
        ),
    ],
    FlightPhase.LANDING: [
        TransitionRule(
            "Altitude < 200 m — hover-slam",
            lambda c: c.altitude_m < 200,
            FlightPhase.HOVSLAM
        ),
    ],
    FlightPhase.HOVSLAM: [
        TransitionRule(
            "Altitude = 0 — touchdown",
            lambda c: c.altitude_m <= 0.5,
            FlightPhase.TOUCHDOWN
        ),
    ],
    FlightPhase.TOUCHDOWN: [
        TransitionRule(
            "Vehicle stable",
            lambda c: c.time_s > 2.0,
            FlightPhase.SAFED
        ),
    ],
}

# Map flight phases to engine controller phase names
PHASE_ENGINE_MAP = {
    FlightPhase.LIFTOFF:   "LIFTOFF",
    FlightPhase.MAX_Q:     "MAX_Q",
    FlightPhase.MECO_PREP: "MECO_PREP",
    FlightPhase.MECO:      "LIFTOFF",    # will be shut by controller
    FlightPhase.FLIP:      "LIFTOFF",
    FlightPhase.BOOSTBACK: "BOOSTBACK",
    FlightPhase.ENTRY:     "ENTRY",
    FlightPhase.SUBSONIC:  "ENTRY",
    FlightPhase.LANDING:   "LANDING",
    FlightPhase.HOVSLAM:   "HOVSLAM",
    FlightPhase.TOUCHDOWN: "HOVSLAM",
    FlightPhase.SAFED:     "HOVSLAM",
}


# ── Flight Phase Controller ──────────────────────────────────────────────────

class FlightPhaseController:
    """
    Manages phase transitions, calls EngineController for each phase,
    logs all events, and exposes sensor inputs for real-time update.
    """

    def __init__(self, engine_ctrl: EngineController):
        self.ctrl = engine_ctrl
        self.phase = FlightPhase.PRE_LAUNCH
        self.phase_entry_time: float = 0.0
        self.conditions = FlightConditions()
        self.log: List[str] = []
        self.phase_history: List[tuple] = []

    def update(self, conditions: FlightConditions) -> Optional[FlightPhase]:
        """
        Call every simulation tick.
        Checks transition guards, fires transitions if conditions met.
        Returns new phase if transition occurred, else None.
        """
        self.conditions = conditions
        rules = PHASE_TRANSITIONS.get(self.phase, [])

        for rule in rules:
            local_conds = FlightConditions(**{
                k: v for k, v in conditions.__dict__.items()
            })
            # Time is relative to phase entry for some guards
            local_conds.time_s = conditions.time_s - self.phase_entry_time

            if rule.condition(local_conds):
                return self._transition(rule.next_phase, rule.description, conditions)

        # Check abort conditions at any phase
        if self._abort_condition(conditions) and self.phase != FlightPhase.ABORT:
            return self._transition(FlightPhase.ABORT, "Abort condition triggered", conditions)

        return None

    def _transition(self, new_phase: FlightPhase, reason: str,
                    conds: FlightConditions) -> FlightPhase:
        old = self.phase
        self.phase_history.append((old, conds.time_s, conds.altitude_m))
        self.phase = new_phase
        self.phase_entry_time = conds.time_s

        msg = (f"T+{conds.time_s:.1f}s | {old.name} → {new_phase.name} "
               f"| {reason} | Alt={conds.altitude_m/1000:.1f}km")
        self.log.append(msg)
        print(f"  [PHASE] {msg}")

        # Command engine controller
        if new_phase in PHASE_ENGINE_MAP:
            eng_phase = PHASE_ENGINE_MAP[new_phase]
            if new_phase == FlightPhase.MECO:
                # Shut all engines
                for i in range(9):
                    if self.ctrl.engines[i].state.name == "RUNNING":
                        self.ctrl._shutdown(i, reason="MECO")
            elif new_phase == FlightPhase.FLIP:
                pass   # RCS only — engine controller idles
            else:
                self.ctrl.transition_phase(eng_phase)

        return new_phase

    def _abort_condition(self, c: FlightConditions) -> bool:
        """
        Hard abort triggers (any phase):
        - Propellant below minimum reserve during active burn
        - All engines failed
        - Structural / sensor loss
        """
        if c.propellant_fraction() < 0.03 and self.phase not in (
                FlightPhase.TOUCHDOWN, FlightPhase.SAFED, FlightPhase.MECO,
                FlightPhase.COAST, FlightPhase.FLIP):
            return True
        return False

    def status(self) -> str:
        c = self.conditions
        return (
            f"Phase: {self.phase.name:15s} | {c}"
        )

    def full_log(self) -> str:
        return "\n".join(self.log)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from engine_control import EngineController

    ctrl = EngineController()
    fpc = FlightPhaseController(ctrl)

    # Simulate a simplified ascent + return trajectory
    print("\n" + "="*65)
    print("  FLIGHT PHASE CONTROLLER — MISSION SIMULATION")
    print("="*65)

    # (time_s, alt_m, vel_ms, mach, pitch, prop_frac, lat_offset)
    waypoints = [
        (0,     0,       0,     0.00,  90,   1.00, 0),
        (5,     500,     80,    0.24,  90,   0.98, 0),
        (30,    5_000,   400,   1.20,  89,   0.90, 0),   # LIFTOFF→MAX_Q
        (90,    20_000,  1500,  4.50,  88,   0.75, 0),   # MAX_Q→MECO_PREP
        (130,   60_000,  3_100, 9.20,  87,   0.19, 0),   # →MECO
        (133,   61_000,  3_050, 9.10,  87,   0.19, 0),   # →FLIP
        (148,   65_000,  2_800, 8.50, -86,   0.18, 1500),# →BOOSTBACK
        (200,   80_000,  180,   0.54, -88,   0.13, 800), # →COAST
        (260,   68_000, -200,   0.62, -88,   0.13, 600), # →ENTRY
        (330,   15_000, -1500,  1.30, -88,   0.09, 200), # entry
        (360,   5_000,  -900,   0.90, -88,   0.08, 100), # →SUBSONIC
        (380,   1_800,  -300,   0.28, -88,   0.07, 50),  # →LANDING
        (395,   180,    -80,    0.07, -89,   0.055, 5),  # →HOVSLAM
        (400,   0.3,    -1.8,   0.00, -90,   0.05, 0),   # →TOUCHDOWN
        (402,   0,      0,      0.00, -90,   0.05, 0),   # →SAFED
    ]

    prev_t = 0
    for (t, alt, vel, mach, pitch, pfrac, lat) in waypoints:
        conds = FlightConditions(
            altitude_m=alt, velocity_ms=vel, mach=mach,
            pitch_deg=pitch, propellant_remaining_kg=pfrac * 10_000,
            propellant_total_kg=10_000, time_s=t,
            lateral_offset_m=lat, lidar_altitude_m=alt
        )
        result = fpc.update(conds)

    print("\n\n>>> PHASE TRANSITION LOG:")
    print(fpc.full_log())
    print("\n>>> FINAL ENGINE STATUS:")
    print(ctrl.status_report())
