# VTVL Hybrid Rocket — Propulsion Control System

## Overview
Full software simulation of a VTVL (Vertical Take-off, Vertical Landing) hybrid rocket with:
- **9-engine LOX/LCH₄ cluster** — full-flow staged combustion
- **Phase-based engine selection**: 9 → 3 → 1
- **Opposite-pair balance law** — automatic compensation on engine failure
- **Sensor fusion** — IMU + GPS + LIDAR + Radar + Barometric
- **Health monitoring** — 1 kHz sensor polling, AI anomaly detection
- **Full mission simulation** with 8-panel visualisation

---

## Project Structure

``` 
rocket_project/
├── engine_control.py   Engine state machine, cluster rules, failure handling
├── flight_phase.py     Flight phase FSM, phase transitions, engine mapping
├── health_monitor.py   Sensor health monitoring, go/no-go logic
├── sensors.py          IMU, GPS, LIDAR, altimeters, sensor fusion
├── simulation.py       Full trajectory integrator + matplotlib plots
├── main.py             Master run script (all demos + simulation)
└── README.md           This file
```

---

## Quick Start

```bash
pip install matplotlib numpy
cd rocket_project
python main.py
```

This runs:
1. Preflight health check (with fault injection demo)
2. Engine controller phase sequence (LIFTOFF → HOVSLAM)
3. Engine failure simulation (E2 fault → balance compensation)
4. Sensor fusion landing approach
5. Full 450-second mission simulation
6. Saves `simulation_plots.png` (8-panel mission analysis)

---

## Core Design Rules (Implemented in Code)

### Rule 1 — Opposite-Pair Balance Law
```python
# engine_control.py — EngineController._emergency_shutdown()
OPPOSITE = {1:5, 5:1, 2:6, 6:2, 3:7, 7:3, 4:8, 8:4, 0:0}
# If E3 fails → E7 is automatically shut down to maintain balance
```

### Rule 2 — Phase Engine Selection
```python
PHASE_ENGINES = {
    "LIFTOFF":   {0,1,2,3,4,5,6,7,8},   # All 9
    "BOOSTBACK": {0,2,6},                 # 3 — symmetric
    "LANDING":   {0,1,5},                 # 3 — symmetric
    "HOVSLAM":   {0},                     # 1 — centre only
}
```

### Rule 3 — Minimum Throttle Gate
No engine is commanded below 40% throttle (combustion instability threshold).

### Rule 4 — Health-Gated Ignition
Engines with health_score < 0.5 are blocked from ignition commands.

### Rule 5 — Centre Engine Protection (E0)
E0 failure during HOVSLAM triggers full abort sequence — no recovery path.

---

## Simulation Physics

| Parameter | Value |
|---|---|
| Engine count | 9 |
| Thrust per engine (vac) | 100 kN |
| Isp (vacuum) | 363 s |
| Isp (sea level) | 330 s |
| Propellant mass | 78 000 kg |
| Dry mass | 22 000 kg |
| Propellant reserve | 15 % |
| Cycle type | Full-flow staged combustion |
| Propellants | LOX / LCH₄ |

---

## Module API Summary

### engine_control.py
```python
ctrl = EngineController()
ctrl.transition_phase("LANDING")          # Phase switch
ctrl.report_sensor_anomaly(3, "fault")    # Inject failure
ctrl.total_thrust_kN()                    # Current thrust
ctrl.cluster_balance_error()              # (x, y) centroid offset
ctrl.status_report()                      # Formatted status string
```

### flight_phase.py
```python
fpc = FlightPhaseController(ctrl)
fpc.update(conditions)                    # Call every tick
fpc.phase                                 # Current FlightPhase enum
fpc.full_log()                            # All transitions
```

### health_monitor.py
```python
monitor = HealthMonitor(anomaly_callback=my_cb)
monitor.ingest_engine(pkt)               # Feed sensor data
monitor.preflight_check(packets)         # GO/NO-GO
monitor.build_vehicle_health(t)          # VehicleHealth snapshot
```

### sensors.py
```python
fusion = SensorFusion()
nav = fusion.update(true_alt, true_vel, ...)    # FusedNavState
trigger = LandingTrigger()
cmds = trigger.evaluate(nav)             # ["LEGS_DEPLOY", ...]
```

### simulation.py
```python
sim = MissionSimulator()
sim.run(dt=0.5, t_max=450)
sim.plot(save_path="output.png")
```

---

## Flight Phase State Machine

```
PRE_LAUNCH
    ↓ T=0
LIFTOFF        [9 engines, 100%]
    ↓ alt > 1 km
MAX_Q          [9 engines, 95%]
    ↓ alt > 15 km
MECO_PREP      [9 engines, 70%]
    ↓ vel > 3000 m/s
MECO           [engines cut]
    ↓ immediately
FLIP           [RCS cold-gas, 180° rotation]
    ↓ pitch ≤ -85°
BOOSTBACK      [E0+E2+E6, 85%] — reverse trajectory
    ↓ vel < 200 m/s
COAST          [no thrust]
    ↓ alt < 70 km
ENTRY          [E0+E2+E6, 65%] — hypersonic decel
    ↓ Mach < 1.5
SUBSONIC       [grid fins primary]
    ↓ alt < 2 km
LANDING        [E0+E1+E5, 55%]
    ↓ alt < 200 m
HOVSLAM        [E0 only, 42%]
    ↓ alt ≤ 0.5 m
TOUCHDOWN → SAFED
```
