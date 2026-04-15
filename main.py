"""
main.py
=======
VTVL Hybrid Rocket — Master Run Script
Executes:
  1. Preflight health check
  2. Engine controller demo (all phases)
  3. Full flight phase mission
  4. Sensor fusion landing approach
  5. Full simulation + plot generation
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from engine_control   import EngineController
from flight_phase     import FlightPhaseController, FlightConditions
from health_monitor   import HealthMonitor, EngineHealthPacket
from sensors          import SensorFusion, LandingTrigger
from simulation       import MissionSimulator


BANNER = r"""
  ██╗   ██╗████████╗██╗   ██╗██╗     
  ██║   ██║╚══██╔══╝██║   ██║██║     
  ██║   ██║   ██║   ██║   ██║██║     
  ╚██╗ ██╔╝   ██║   ╚██╗ ██╔╝██║     
   ╚████╔╝    ██║    ╚████╔╝ ███████╗
    ╚═══╝     ╚═╝     ╚═══╝  ╚══════╝
  VTVL Hybrid Rocket — Propulsion Control System
  LOX/LCH4  |  9-Engine Cluster  |  Full-Flow Staged Combustion
"""


def section(title: str):
    w = 65
    print(f"\n{'═'*w}")
    print(f"  {title}")
    print(f"{'═'*w}")


def run_preflight():
    section("1. PREFLIGHT HEALTH CHECK")
    monitor = HealthMonitor()
    packets = {}
    for i in range(9):
        packets[i] = EngineHealthPacket(
            engine_id=i, timestamp_s=0.0,
            chamber_pressure_bar=0.0, turbopump_rpm_lox=0.0,
            turbopump_rpm_lch4=0.0, nozzle_temp_C=22.0,
            propellant_flow_kgs=0.0, vibration_g=0.08, igniter_ok=True
        )
    # Introduce a fault to show detection
    packets[4].igniter_ok = False
    packets[4].vibration_g = 9.5

    go, issues = monitor.preflight_check(packets)
    print(f"\n  Preflight result: {'▶ GO' if go else '✗ NO-GO'}")
    if issues:
        for i in issues:
            print(f"  ⚠  {i}")
    else:
        print("  All 9 engines nominal.")

    # Fix the fault and recheck
    print("\n  Fixing E4 (replacing igniter, clearing vibration fault)...")
    packets[4].igniter_ok = True
    packets[4].vibration_g = 0.9
    go2, issues2 = monitor.preflight_check(packets)
    print(f"  Recheck result: {'▶ GO' if go2 else '✗ NO-GO'}")


def run_engine_controller():
    section("2. ENGINE CONTROLLER — PHASE TRANSITIONS")
    ctrl = EngineController()

    phases_seq = [
        ("LIFTOFF",   2.0),
        ("MAX_Q",     35.0),
        ("MECO_PREP", 5.0),
        ("BOOSTBACK", 45.0),
        ("ENTRY",     60.0),
        ("LANDING",   20.0),
        ("HOVSLAM",   15.0),
    ]

    for phase, duration in phases_seq:
        print(f"\n  Phase → {phase}")
        actions = ctrl.transition_phase(phase)
        for a in actions:
            print(a)
        ctrl.advance_time(duration)

    print("\n" + ctrl.status_report())

    # Demonstrate engine failure handling
    section("2b. ENGINE FAILURE SIMULATION")
    print("  Injecting fault on E2 during BOOSTBACK burn...")
    fault_actions = ctrl.report_sensor_anomaly(2, "chamber_pressure_spike: 288bar")
    for a in fault_actions:
        print(a)
    bx, by = ctrl.cluster_balance_error()
    print(f"\n  Post-failure balance error: ({bx:+.4f}, {by:+.4f})")
    print(f"  (target: 0,0 — balance {'OK' if abs(bx)+abs(by) < 0.05 else 'COMPROMISED'})")


def run_sensor_demo():
    section("3. SENSOR FUSION — LANDING APPROACH")
    fusion  = SensorFusion()
    trigger = LandingTrigger()

    print(f"\n  {'Time':>5}  {'TrueAlt':>8}  {'FusedAlt':>9}  {'Source':<10}  {'Conf':>5}  Commands")
    print(f"  {'-'*65}")

    profile = [
        (0,  3000, -300), (5, 1480, -250), (9, 990, -200),
        (12, 475,  -60),  (14, 185, -30),  (16, 95, -15),
        (18, 18,   -4),   (19, 0.3, -1.6),
    ]
    for (t, alt, vel) in profile:
        nav = fusion.update(
            true_alt=alt, true_vel=vel,
            true_pitch=-90, true_roll=0, true_yaw=0,
            true_pitch_rate=0, true_roll_rate=0, true_yaw_rate=0,
            true_lateral=max(0, alt * 0.008), t=float(t), dt=1.0
        )
        cmds = trigger.evaluate(nav)
        cmd_str = ", ".join(cmds) if cmds else "—"
        print(f"  {t:>4}s  {alt:>8.1f}  {nav.altitude_m:>9.1f}  "
              f"{nav.source:<10}  {nav.confidence:>5.2f}  {cmd_str}")


def run_simulation():
    section("4. FULL MISSION SIMULATION + PLOTS")
    sim = MissionSimulator()
    sim.run(dt=0.5, t_max=450)
    sim.plot(save_path="/mnt/user-data/outputs/simulation_plots.png")


if __name__ == "__main__":
    print(BANNER)

    run_preflight()
    run_engine_controller()
    run_sensor_demo()
    run_simulation()

    section("MISSION COMPLETE")
    print("\n  Outputs:")
    print("  ✓  simulation_plots.png — 8-panel mission analysis")
    print("  ✓  All subsystem demos complete\n")
