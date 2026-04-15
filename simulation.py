"""
simulation.py
=============
VTVL Rocket — Full Mission Simulation
Integrates all subsystems and generates plots:
  1. Altitude vs time
  2. Velocity vs time
  3. Thrust vs time (with phase shading)
  4. Active engine count vs time
  5. Fuel usage vs time
  6. IMU pitch angle vs time
  7. Health score (min engine) vs time
  8. Phase timeline (Gantt-style)
"""

import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

from engine_control import EngineController, PHASE_ENGINES, PHASE_THROTTLE
from flight_phase import FlightPhaseController, FlightPhase, FlightConditions
from health_monitor import HealthMonitor, EngineHealthPacket
from sensors import SensorFusion, LandingTrigger

random.seed(42)
np.random.seed(42)


# ── Physics Constants ─────────────────────────────────────────────────────────

G0         = 9.80665        # m/s²
ISP_VAC    = 363.0          # s  vacuum specific impulse
ISP_SL     = 330.0          # s  sea level
MASS_DRY   = 22_000         # kg  dry mass (stage)
MASS_PROP  = 78_000         # kg  propellant
MASS_TOTAL = MASS_DRY + MASS_PROP
THRUST_PER_ENGINE_kN = 100.0
NUM_ENGINES = 9
DRAG_COEFF  = 0.35
REF_AREA    = 10.0          # m²  reference cross-section
RHO_SL      = 1.225         # kg/m³  sea-level air density


def air_density(alt_m: float) -> float:
    """Simple exponential atmosphere model."""
    if alt_m > 85_000:
        return 0.0
    return RHO_SL * math.exp(-alt_m / 8_500)


def isp(alt_m: float) -> float:
    """Linearly interpolate Isp from SL to vacuum."""
    return ISP_SL + (ISP_VAC - ISP_SL) * min(1.0, alt_m / 30_000)


# ── Simplified Trajectory Integrator ─────────────────────────────────────────

class MissionSimulator:
    def __init__(self):
        self.ctrl = EngineController()
        self.fpc  = FlightPhaseController(self.ctrl)
        self.monitor = HealthMonitor()
        self.fusion  = SensorFusion()
        self.trigger = LandingTrigger()

        # State
        self.t      = 0.0
        self.alt    = 0.0
        self.vel    = 0.0     # m/s positive upward
        self.mass   = MASS_TOTAL
        self.pitch  = 90.0    # degrees

        # Data logs
        self.log_t       = []
        self.log_alt     = []
        self.log_vel     = []
        self.log_thrust  = []
        self.log_mass    = []
        self.log_engines = []
        self.log_health  = []
        self.log_pitch   = []
        self.log_phase   = []
        self.log_mach    = []

        # Phase colours for plotting
        self.phase_colours = {
            "PRE_LAUNCH": "#D3D1C7",
            "LIFTOFF":    "#378ADD",
            "MAX_Q":      "#0C447C",
            "MECO_PREP":  "#1D9E75",
            "MECO":       "#5F5E5A",
            "FLIP":       "#7F77DD",
            "BOOSTBACK":  "#534AB7",
            "COAST":      "#888780",
            "ENTRY":      "#D85A30",
            "SUBSONIC":   "#BA7517",
            "LANDING":    "#E24B4A",
            "HOVSLAM":    "#993C1D",
            "TOUCHDOWN":  "#1D9E75",
            "SAFED":      "#3B6D11",
            "ABORT":      "#501313",
        }

    def _active_engines(self) -> int:
        return len(self.ctrl.active_engine_ids())

    def _total_thrust(self) -> float:
        return self.ctrl.total_thrust_kN() * 1000  # → N

    def _phase_name(self) -> str:
        return self.fpc.phase.name

    def run(self, dt: float = 0.5, t_max: float = 450.0):
        """Run full mission simulation."""
        print("\n" + "="*65)
        print("  VTVL MISSION SIMULATION — FULL TRAJECTORY")
        print("="*65)

        # Scripted phase transitions keyed by (alt_m, vel_ms, time_s)
        meco_done    = False
        flip_done    = False
        boostback_done = False

        while self.t <= t_max:
            # ── Update flight conditions ───────────────────────────────────
            mach = abs(self.vel) / 340.0
            dyn_q = 0.5 * air_density(self.alt) * self.vel**2
            prop_rem = self.mass - MASS_DRY
            pfrac = max(0, prop_rem / MASS_PROP)

            conds = FlightConditions(
                altitude_m=max(0, self.alt),
                velocity_ms=self.vel,
                mach=mach,
                dynamic_pressure_Pa=dyn_q,
                pitch_deg=self.pitch,
                propellant_remaining_kg=max(0, prop_rem),
                propellant_total_kg=MASS_PROP,
                time_s=self.t,
                lateral_offset_m=max(0, 500 - self.t * 1.5) if self.t > 200 else 2000,
                lidar_altitude_m=max(0, self.alt)
            )

            # ── Trigger phase transitions ──────────────────────────────────
            self.fpc.update(conds)

            # Manual phase guards for phases that need explicit scripting
            phase = self._phase_name()

            if phase == "MECO" and not meco_done:
                meco_done = True
                for i in range(9):
                    self.ctrl._shutdown(i, "MECO")

            if phase == "FLIP":
                # Simulate pitch rotating from +90 to -90 over 15s
                if self.pitch > -85:
                    self.pitch -= 12.0 * dt
                else:
                    self.pitch = -88.0
                    flip_done = True

            if phase == "BOOSTBACK" and self.vel < 100 and not boostback_done:
                boostback_done = True

            # ── Physics integration ────────────────────────────────────────
            thrust_N = self._total_thrust()
            mdot = 0.0
            if thrust_N > 0 and prop_rem > 0:
                isp_curr = isp(self.alt)
                mdot = thrust_N / (isp_curr * G0)   # kg/s propellant flow
                mdot = min(mdot, prop_rem / dt)
                self.mass = max(MASS_DRY, self.mass - mdot * dt)

            # Aerodynamic drag
            drag_N = 0.5 * air_density(self.alt) * DRAG_COEFF * REF_AREA * self.vel * abs(self.vel)

            # Net force  (sign: thrust opposes velocity direction during decel burns)
            if phase in ("BOOSTBACK", "ENTRY", "LANDING", "HOVSLAM"):
                # Engine pointing down (inverted) — thrust acts upward (opposing descent)
                thrust_sign = +1
            else:
                thrust_sign = +1

            # Gravity always downward
            F_net = thrust_sign * thrust_N - drag_N - self.mass * G0
            accel = F_net / self.mass

            self.vel += accel * dt
            self.alt += self.vel * dt
            self.alt = max(0.0, self.alt)

            # TOUCHDOWN stop
            if self.alt <= 0.0 and self.t > 50:
                self.vel = 0.0
                if phase not in ("TOUCHDOWN", "SAFED"):
                    self.fpc.update(FlightConditions(
                        altitude_m=0, velocity_ms=0, mach=0,
                        pitch_deg=self.pitch, time_s=self.t,
                        propellant_remaining_kg=max(0, self.mass - MASS_DRY),
                        propellant_total_kg=MASS_PROP
                    ))

            # ── Ingest fake health packets ─────────────────────────────────
            for i in range(9):
                eng = self.ctrl.engines[i]
                if eng.state.name == "RUNNING":
                    pkt = EngineHealthPacket(
                        engine_id=i, timestamp_s=self.t,
                        chamber_pressure_bar=250 * eng.throttle + random.gauss(0, 2),
                        turbopump_rpm_lox=35_000 * eng.throttle + random.gauss(0, 200),
                        turbopump_rpm_lch4=40_000 * eng.throttle + random.gauss(0, 200),
                        nozzle_temp_C=800 + 400 * eng.throttle + random.gauss(0, 10),
                        propellant_flow_kgs=25.0 * eng.throttle,
                        vibration_g=abs(random.gauss(1.0, 0.5)),
                        igniter_ok=True
                    )
                    self.monitor.ingest_engine(pkt)

            # ── Log ───────────────────────────────────────────────────────
            self.log_t.append(self.t)
            self.log_alt.append(self.alt / 1000)         # km
            self.log_vel.append(self.vel)                 # m/s
            self.log_thrust.append(thrust_N / 1000)       # kN
            self.log_mass.append(self.mass)
            self.log_engines.append(self._active_engines())
            self.log_health.append(
                self.monitor.build_vehicle_health(self.t).min_engine_health()
            )
            self.log_pitch.append(self.pitch)
            self.log_phase.append(self._phase_name())
            self.log_mach.append(mach)

            self.t += dt

            if self.alt <= 0.0 and self.t > 100:
                break

        print(f"  Simulation complete. Total time: {self.t:.0f}s")
        print(f"  Peak altitude: {max(self.log_alt):.1f} km")
        print(f"  Propellant used: {(MASS_TOTAL - self.mass):.0f} kg")
        print(f"  Remaining: {max(0, self.mass - MASS_DRY):.0f} kg")

    def plot(self, save_path: str = "/mnt/user-data/outputs/simulation_plots.png"):
        """Generate comprehensive 8-panel plot."""
        t = np.array(self.log_t)
        phases_arr = self.log_phase
        unique_phases = list(dict.fromkeys(phases_arr))

        # Build phase time spans for shading
        phase_spans = []
        cur = phases_arr[0]
        start = t[0]
        for i, p in enumerate(phases_arr):
            if p != cur or i == len(phases_arr) - 1:
                phase_spans.append((cur, start, t[i - 1]))
                cur = p
                start = t[i]

        fig = plt.figure(figsize=(18, 20))
        fig.patch.set_facecolor("#0D1117")
        gs = GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.3)

        panel_style = dict(facecolor="#161B22")
        txt_col   = "#E6EDF3"
        grid_col  = "#21262D"
        accent1   = "#378ADD"    # blue
        accent2   = "#D85A30"    # coral/orange
        accent3   = "#1D9E75"    # teal
        accent4   = "#7F77DD"    # purple
        accent5   = "#BA7517"    # amber

        def shade_phases(ax):
            for (pname, ts, te) in phase_spans:
                col = self.phase_colours.get(pname, "#888780")
                ax.axvspan(ts, te, alpha=0.12, color=col, linewidth=0)

        # ─ 1. Altitude ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0], **panel_style)
        shade_phases(ax1)
        ax1.plot(t, self.log_alt, color=accent1, linewidth=2)
        ax1.set_title("Altitude vs Time", color=txt_col, fontsize=12, pad=8)
        ax1.set_ylabel("Altitude (km)", color=txt_col)
        ax1.tick_params(colors=txt_col); ax1.grid(color=grid_col)
        ax1.spines[:].set_color(grid_col)
        for spine in ax1.spines.values(): spine.set_color(grid_col)

        # ─ 2. Velocity ───────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1], **panel_style)
        shade_phases(ax2)
        ax2.plot(t, self.log_vel, color=accent2, linewidth=2)
        ax2.axhline(0, color="#5F5E5A", linewidth=0.8, linestyle="--")
        ax2.set_title("Velocity vs Time", color=txt_col, fontsize=12, pad=8)
        ax2.set_ylabel("Velocity (m/s)", color=txt_col)
        ax2.tick_params(colors=txt_col); ax2.grid(color=grid_col)
        for spine in ax2.spines.values(): spine.set_color(grid_col)

        # ─ 3. Thrust ─────────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0], **panel_style)
        shade_phases(ax3)
        ax3.fill_between(t, self.log_thrust, alpha=0.35, color=accent2)
        ax3.plot(t, self.log_thrust, color=accent2, linewidth=1.5)
        ax3.set_title("Thrust vs Time", color=txt_col, fontsize=12, pad=8)
        ax3.set_ylabel("Thrust (kN)", color=txt_col)
        ax3.tick_params(colors=txt_col); ax3.grid(color=grid_col)
        for spine in ax3.spines.values(): spine.set_color(grid_col)

        # ─ 4. Active Engines ─────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1], **panel_style)
        shade_phases(ax4)
        ax4.step(t, self.log_engines, color=accent4, linewidth=2, where="post")
        ax4.fill_between(t, self.log_engines, step="post", alpha=0.25, color=accent4)
        ax4.set_yticks([0, 1, 3, 9])
        ax4.set_ylim(-0.3, 10)
        ax4.axhline(9, color=accent4, alpha=0.25, linewidth=0.8, linestyle=":")
        ax4.axhline(3, color=accent5, alpha=0.25, linewidth=0.8, linestyle=":")
        ax4.axhline(1, color=accent2, alpha=0.25, linewidth=0.8, linestyle=":")
        ax4.set_title("Active Engine Count", color=txt_col, fontsize=12, pad=8)
        ax4.set_ylabel("Engines", color=txt_col)
        ax4.tick_params(colors=txt_col); ax4.grid(color=grid_col)
        for spine in ax4.spines.values(): spine.set_color(grid_col)

        # ─ 5. Propellant ─────────────────────────────────────────────────
        prop = [(m - MASS_DRY) / 1000 for m in self.log_mass]
        ax5 = fig.add_subplot(gs[2, 0], **panel_style)
        shade_phases(ax5)
        ax5.fill_between(t, prop, alpha=0.3, color=accent3)
        ax5.plot(t, prop, color=accent3, linewidth=2)
        ax5.axhline(MASS_PROP * 0.15 / 1000, color=accent5, linewidth=1,
                    linestyle="--", label="15% reserve")
        ax5.legend(facecolor="#21262D", labelcolor=txt_col, fontsize=9)
        ax5.set_title("Propellant Remaining", color=txt_col, fontsize=12, pad=8)
        ax5.set_ylabel("Propellant (tonnes)", color=txt_col)
        ax5.tick_params(colors=txt_col); ax5.grid(color=grid_col)
        for spine in ax5.spines.values(): spine.set_color(grid_col)

        # ─ 6. IMU Pitch ──────────────────────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 1], **panel_style)
        shade_phases(ax6)
        ax6.plot(t, self.log_pitch, color=accent5, linewidth=2)
        ax6.axhline(90, color="#5F5E5A", linewidth=0.7, linestyle=":")
        ax6.axhline(-90, color="#5F5E5A", linewidth=0.7, linestyle=":")
        ax6.set_title("IMU Pitch Angle", color=txt_col, fontsize=12, pad=8)
        ax6.set_ylabel("Pitch (degrees)", color=txt_col)
        ax6.tick_params(colors=txt_col); ax6.grid(color=grid_col)
        for spine in ax6.spines.values(): spine.set_color(grid_col)

        # ─ 7. Engine Health ───────────────────────────────────────────────
        ax7 = fig.add_subplot(gs[3, 0], **panel_style)
        shade_phases(ax7)
        ax7.plot(t, self.log_health, color=accent3, linewidth=2)
        ax7.axhline(0.5, color=accent2, linewidth=1, linestyle="--", label="Critical (0.5)")
        ax7.axhline(0.85, color=accent3, linewidth=0.8, linestyle=":", label="Nominal (0.85)")
        ax7.set_ylim(0, 1.05)
        ax7.legend(facecolor="#21262D", labelcolor=txt_col, fontsize=9)
        ax7.set_title("Min Engine Health Score", color=txt_col, fontsize=12, pad=8)
        ax7.set_ylabel("Health (0–1)", color=txt_col)
        ax7.tick_params(colors=txt_col); ax7.grid(color=grid_col)
        for spine in ax7.spines.values(): spine.set_color(grid_col)

        # ─ 8. Phase Timeline (Gantt) ──────────────────────────────────────
        ax8 = fig.add_subplot(gs[3, 1], **panel_style)
        ax8.set_facecolor("#161B22")
        phase_order = ["LIFTOFF","MAX_Q","MECO_PREP","MECO","FLIP",
                       "BOOSTBACK","COAST","ENTRY","SUBSONIC",
                       "LANDING","HOVSLAM","TOUCHDOWN"]
        for idx, (pname, ts, te) in enumerate(phase_spans):
            if pname not in phase_order:
                continue
            y = phase_order.index(pname) if pname in phase_order else 0
            col = self.phase_colours.get(pname, "#888780")
            ax8.barh(y, te - ts, left=ts, color=col, alpha=0.85, height=0.7)
        ax8.set_yticks(range(len(phase_order)))
        ax8.set_yticklabels(phase_order, color=txt_col, fontsize=8)
        ax8.set_title("Mission Phase Timeline", color=txt_col, fontsize=12, pad=8)
        ax8.set_xlabel("Time (s)", color=txt_col)
        ax8.tick_params(colors=txt_col, axis="x"); ax8.grid(color=grid_col, axis="x")
        for spine in ax8.spines.values(): spine.set_color(grid_col)

        # Global title
        fig.suptitle(
            "VTVL Hybrid Rocket — Full Mission Simulation",
            color=txt_col, fontsize=16, fontweight="bold", y=0.98
        )

        # X-label on bottom panels
        for ax in [ax7, ax8]:
            ax.set_xlabel("Time (s)", color=txt_col)

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.tick_params(axis="x", colors=txt_col)

        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  Plot saved → {save_path}")
        plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sim = MissionSimulator()
    sim.run(dt=0.5, t_max=450)
    sim.plot()
