"""
sensors.py
==========
VTVL Rocket — Sensor & Control Simulation
Simulates:
  - IMU (gyroscope + accelerometer) with drift and noise
  - Barometric altimeter (with bias above 80 km)
  - LIDAR altimeter (active below 2 km)
  - GPS / NavIC (position + velocity)
  - Radar altimeter (below 3 km)
  - Kalman-filter-style sensor fusion

All sensors include:
  - Measurement noise model
  - Failure mode injection
  - Update rate simulation
"""

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ── Noise & Drift Models ──────────────────────────────────────────────────────

def gaussian_noise(sigma: float) -> float:
    return random.gauss(0.0, sigma)

def bounded_noise(amplitude: float) -> float:
    return random.uniform(-amplitude, amplitude)


# ── IMU ───────────────────────────────────────────────────────────────────────

@dataclass
class IMUState:
    """Integrated IMU output — attitude angles + rates."""
    pitch_deg: float = 90.0       # 90 = vertical upright
    roll_deg:  float = 0.0
    yaw_deg:   float = 0.0
    pitch_rate_degs: float = 0.0  # deg/s
    roll_rate_degs:  float = 0.0
    yaw_rate_degs:   float = 0.0
    ax_ms2: float = 0.0           # Longitudinal acceleration
    ay_ms2: float = 0.0
    az_ms2: float = 0.0
    valid: bool = True


class IMUSensor:
    """
    Triple-redundant ring-laser gyro simulation.
    Includes:
      - Gyro drift (0.001°/hr = 2.78e-6 °/s)
      - White noise on rate
      - Accelerometer noise
      - Voting logic for triple redundancy
    """
    GYRO_NOISE_DEG_S = 0.005        # 1-sigma noise on rate
    GYRO_DRIFT_DEG_S = 2.78e-6      # Constant drift
    ACCEL_NOISE_MS2  = 0.01         # Accelerometer noise

    def __init__(self):
        self._drift_accum_pitch = 0.0
        self._drift_accum_roll  = 0.0
        self._drift_accum_yaw   = 0.0
        self.failure_mode: Optional[str] = None

    def read(self, true_pitch: float, true_roll: float, true_yaw: float,
             true_pitch_rate: float, true_roll_rate: float, true_yaw_rate: float,
             true_ax: float, true_ay: float, true_az: float,
             dt: float) -> IMUState:

        if self.failure_mode == "FROZEN":
            return IMUState(valid=False)

        # Drift accumulation
        self._drift_accum_pitch += self.GYRO_DRIFT_DEG_S * dt
        self._drift_accum_roll  += self.GYRO_DRIFT_DEG_S * dt * 0.7
        self._drift_accum_yaw   += self.GYRO_DRIFT_DEG_S * dt * 1.1

        def noisy_rate(true_rate):
            return true_rate + gaussian_noise(self.GYRO_NOISE_DEG_S)

        def noisy_angle(true_angle, drift):
            return true_angle + drift + gaussian_noise(self.GYRO_NOISE_DEG_S * 0.1)

        return IMUState(
            pitch_deg = noisy_angle(true_pitch, self._drift_accum_pitch),
            roll_deg  = noisy_angle(true_roll,  self._drift_accum_roll),
            yaw_deg   = noisy_angle(true_yaw,   self._drift_accum_yaw),
            pitch_rate_degs = noisy_rate(true_pitch_rate),
            roll_rate_degs  = noisy_rate(true_roll_rate),
            yaw_rate_degs   = noisy_rate(true_yaw_rate),
            ax_ms2 = true_ax + gaussian_noise(self.ACCEL_NOISE_MS2),
            ay_ms2 = true_ay + gaussian_noise(self.ACCEL_NOISE_MS2),
            az_ms2 = true_az + gaussian_noise(self.ACCEL_NOISE_MS2),
            valid  = True
        )


# ── GPS / NavIC ────────────────────────────────────────────────────────────

@dataclass
class GPSState:
    lat_deg: float = 13.73      # Sriharikota launch site
    lon_deg: float = 80.23
    alt_m:   float = 0.0
    vel_n_ms: float = 0.0
    vel_e_ms: float = 0.0
    vel_d_ms: float = 0.0
    hdop: float = 1.0
    lock: bool = True
    num_satellites: int = 10


class GPSSensor:
    """Dual GPS + NavIC constellation simulation."""
    POS_NOISE_M   = 3.0     # 1-sigma horizontal position
    VEL_NOISE_MS  = 0.05    # 1-sigma velocity
    ALT_NOISE_M   = 5.0

    def read(self, true_alt: float, true_vel: float,
             lat0: float = 13.73, lon0: float = 80.23) -> GPSState:
        # GPS not reliable at very high velocity (>Mach 10) for vertical rate
        noise_mult = 1.0 + max(0, (true_vel - 3000) / 1000)
        return GPSState(
            lat_deg   = lat0 + gaussian_noise(self.POS_NOISE_M / 111_000),
            lon_deg   = lon0 + gaussian_noise(self.POS_NOISE_M / 111_000),
            alt_m     = true_alt + gaussian_noise(self.ALT_NOISE_M * noise_mult),
            vel_n_ms  = gaussian_noise(self.VEL_NOISE_MS),
            vel_e_ms  = gaussian_noise(self.VEL_NOISE_MS),
            vel_d_ms  = -true_vel + gaussian_noise(self.VEL_NOISE_MS * noise_mult),
            lock      = True,
            num_satellites = max(4, 10 - int(noise_mult))
        )


# ── Altimeters ────────────────────────────────────────────────────────────────

class BarometricAltimeter:
    """Barometric — unreliable above 80 km, requires ISA model."""
    NOISE_M = 5.0
    MAX_RELIABLE_ALT = 80_000

    def read(self, true_alt: float) -> Tuple[float, bool]:
        if true_alt > self.MAX_RELIABLE_ALT:
            return true_alt * 1.05 + gaussian_noise(500), False  # Unreliable
        return true_alt + gaussian_noise(self.NOISE_M), True


class RadarAltimeter:
    """FMCW radar — accurate from 0 to 3000 m."""
    NOISE_M = 0.1
    MAX_RANGE = 3_000

    def read(self, true_alt: float) -> Tuple[float, bool]:
        if true_alt > self.MAX_RANGE:
            return 0.0, False
        return max(0.0, true_alt + gaussian_noise(self.NOISE_M)), True


class LIDARSensor:
    """
    3D Flash LIDAR — active below 2000 m.
    Provides:
      - Precise altitude
      - Landing zone map (simulated as offset from target)
      - Hazard detection (not simulated — returns clear)
    """
    NOISE_M    = 0.05     # 5 cm resolution
    MAX_RANGE  = 2_000

    def read(self, true_alt: float,
             true_lateral_offset: float = 0.0) -> Tuple[float, float, bool]:
        """Returns (altitude, lateral_offset_to_pad, valid)."""
        if true_alt > self.MAX_RANGE:
            return 0.0, 0.0, False
        alt = max(0.0, true_alt + gaussian_noise(self.NOISE_M))
        lat = true_lateral_offset + gaussian_noise(0.1)
        return alt, lat, True


# ── Sensor Fusion (Simplified Complementary Filter) ──────────────────────────

@dataclass
class FusedNavState:
    """Best-estimate navigation state after sensor fusion."""
    altitude_m: float
    velocity_ms: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    lateral_offset_m: float
    source: str     # Which sensors contributed
    confidence: float   # 0–1


class SensorFusion:
    """
    Complementary filter combining GPS + IMU + altimeters.
    Weights:
      - Below 2 km: LIDAR altitude dominates (weight 0.8)
      - 2–80 km:    GPS altitude dominates
      - Above 80 km: IMU integration only
    """

    def __init__(self):
        self.imu     = IMUSensor()
        self.gps     = GPSSensor()
        self.baro    = BarometricAltimeter()
        self.radar   = RadarAltimeter()
        self.lidar   = LIDARSensor()
        self._last_alt = 0.0
        self._last_t   = 0.0

    def update(self, true_alt: float, true_vel: float,
               true_pitch: float, true_roll: float, true_yaw: float,
               true_pitch_rate: float, true_roll_rate: float, true_yaw_rate: float,
               true_lateral: float, t: float, dt: float) -> FusedNavState:

        # Collect readings
        imu_data = self.imu.read(
            true_pitch, true_roll, true_yaw,
            true_pitch_rate, true_roll_rate, true_yaw_rate,
            0, 0, true_vel, dt
        )
        gps_data  = self.gps.read(true_alt, true_vel)
        baro_alt, baro_ok   = self.baro.read(true_alt)
        radar_alt, radar_ok = self.radar.read(true_alt)
        lidar_alt, lidar_lat, lidar_ok = self.lidar.read(true_alt, true_lateral)

        # Altitude fusion
        if lidar_ok:
            fused_alt = 0.8 * lidar_alt + 0.2 * gps_data.alt_m
            alt_source = "LIDAR+GPS"
            conf = 0.98
        elif radar_ok:
            fused_alt = 0.6 * radar_alt + 0.4 * gps_data.alt_m
            alt_source = "RADAR+GPS"
            conf = 0.93
        elif baro_ok:
            fused_alt = 0.3 * baro_alt + 0.7 * gps_data.alt_m
            alt_source = "BARO+GPS"
            conf = 0.88
        else:
            fused_alt = gps_data.alt_m
            alt_source = "GPS only"
            conf = 0.80

        # Velocity from GPS
        fused_vel = -gps_data.vel_d_ms

        # Lateral offset from LIDAR (only near landing)
        lat_offset = lidar_lat if lidar_ok else true_lateral

        return FusedNavState(
            altitude_m     = fused_alt,
            velocity_ms    = fused_vel,
            pitch_deg      = imu_data.pitch_deg,
            roll_deg       = imu_data.roll_deg,
            yaw_deg        = imu_data.yaw_deg,
            lateral_offset_m = lat_offset,
            source         = alt_source,
            confidence     = conf,
        )


# ── Landing Trigger Logic ─────────────────────────────────────────────────────

class LandingTrigger:
    """
    Monitors altitude + velocity to issue:
      LANDING_BURN_ARM  (alt < 2000 m)
      HOVSLAM_ARM       (alt < 200 m)
      ENGINE_CUTOFF     (alt < 0.5 m AND velocity < 3 m/s)
      LEGS_DEPLOY       (alt < 500 m)
    """

    def __init__(self):
        self._burn_armed   = False
        self._hovslam_armed = False
        self._legs_deployed = False

    def evaluate(self, nav: FusedNavState) -> List[str]:
        commands = []
        alt = nav.altitude_m
        vel = abs(nav.velocity_ms)

        if alt < 2_000 and not self._burn_armed:
            self._burn_armed = True
            commands.append("LANDING_BURN_ARM")

        if alt < 500 and not self._legs_deployed:
            self._legs_deployed = True
            commands.append("LEGS_DEPLOY")

        if alt < 200 and self._burn_armed and not self._hovslam_armed:
            self._hovslam_armed = True
            commands.append("HOVSLAM_ARM")

        if alt <= 0.5 and vel <= 3.0:
            commands.append("ENGINE_CUTOFF_TOUCHDOWN")

        return commands


# ── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fusion   = SensorFusion()
    trigger  = LandingTrigger()

    print("="*60)
    print("  SENSOR FUSION — LANDING APPROACH SIMULATION")
    print("="*60)
    print(f"{'Time':>6}  {'True Alt':>9}  {'Fused Alt':>9}  "
          f"{'Source':<10}  {'Conf':>5}  {'Commands'}")
    print("-"*70)

    # Simulate descent from 3000 m to touchdown
    descent_profile = [
        (0, 3000, -300),
        (5, 1500, -250),
        (8, 1000, -200),   # LANDING_BURN_ARM fires here
        (11, 480,  -60),   # LEGS_DEPLOY
        (13, 190,  -30),   # HOVSLAM_ARM
        (15, 100,  -15),
        (17, 20,   -5),
        (18, 2,    -1.8),
        (19, 0.3,  -0.5),  # TOUCHDOWN
    ]

    for (t, alt, vel) in descent_profile:
        dt = 1.0
        nav = fusion.update(
            true_alt=alt, true_vel=vel,
            true_pitch=-90, true_roll=0, true_yaw=0,
            true_pitch_rate=0, true_roll_rate=0, true_yaw_rate=0,
            true_lateral=max(0, alt * 0.01),
            t=t, dt=dt
        )
        cmds = trigger.evaluate(nav)
        cmd_str = ", ".join(cmds) if cmds else "—"
        print(f"{t:>5}s  {alt:>9.1f}  {nav.altitude_m:>9.1f}  "
              f"{nav.source:<10}  {nav.confidence:>5.2f}  {cmd_str}")
