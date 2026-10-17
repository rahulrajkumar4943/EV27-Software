"""Motor definitions: speed-torque envelope and efficiency map.

Ported from the `emrax208` / `emrax228` / `emrax268` / `dhx_k40` structs shared across
the original MATLAB scripts.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np


@dataclass(eq=False)
class Motor:
    name: str
    max_rpm: float

    # Speed-torque envelope: max continuous torque (Nm) vs. angular speed (rad/s).
    torque_speed_w: np.ndarray
    torque_speed_m: np.ndarray

    # Efficiency map: efficiency (0-1) over a (speed, torque) grid.
    eff_speed: np.ndarray
    eff_torque: np.ndarray
    eff_map: np.ndarray

    # Hard power limit used as a sweep reference point in the design studies (not part
    # of the per-point force balance itself).
    hard_power_limit: float

    def max_torque(self, omega: np.ndarray) -> np.ndarray:
        """Max available torque (Nm) at motor angular speed(s) `omega` (rad/s)."""
        return np.interp(omega, self.torque_speed_w, self.torque_speed_m)

    def max_power(self, omega: np.ndarray) -> np.ndarray:
        """Max available power (W) at motor angular speed(s) `omega` (rad/s)."""
        return self.max_torque(omega) * omega

    def efficiency(self, omega: np.ndarray, torque: np.ndarray) -> np.ndarray:
        """Efficiency (0-1) at motor angular speed(s)/torque(s), clamped to the map.

        Nearest-neighbour lookup: each speed/torque reads the closest grid point. The
        cell boundaries are the midpoints between grid points (exact ties go to the
        lower grid point), the same boundaries `max_torque_for_power` inverts, so the
        two always agree exactly. Values outside the grid read the edge of the map.
        """
        speed_mids, torque_mids = self._eff_midpoints
        ix = np.searchsorted(speed_mids, omega, side="left")
        iy = np.searchsorted(torque_mids, torque, side="left")
        result = self.eff_map[ix, iy]
        return result if np.ndim(omega) or np.ndim(torque) else result.item()

    @cached_property
    def _eff_midpoints(self):
        """Midpoints between neighbouring eff_speed / eff_torque grid points."""
        s, t = self.eff_speed, self.eff_torque
        return 0.5 * (s[:-1] + s[1:]), 0.5 * (t[:-1] + t[1:])

    @cached_property
    def _power_inverse_tables(self):
        """Plain-Python copies of the efficiency map, built once per motor.

        `max_torque_for_power` runs inside the lap sim's innermost loop, and Python
        floats and lists are several times faster than NumPy for arrays this small.

        speed_mids: speeds halfway between eff_speed grid points. The nearest-neighbour
            row for a motor speed is found by binary search (bisect) on these.
        torque_edges: torque cell boundaries. Cell j (nearest grid torque
            eff_torque[j]) covers (torque_edges[j], torque_edges[j+1]]; the outer cells
            extend to +-inf because `efficiency` clamps torque to the map.
        rows: eff_map as a list of lists.
        """
        speed_mids, torque_mids = self._eff_midpoints
        torque_edges = [-math.inf] + torque_mids.tolist() + [math.inf]
        # Edges pulled a hair (relative 1e-9) into each cell. `max_torque_for_power`
        # only returns torques strictly between these, so when the sim recomputes the
        # torque from the speed trace, round-off (~1e-12 relative) can't push it across
        # a boundary into a neighbouring cell with a different efficiency.
        lower_safe = [e + _EDGE_MARGIN * abs(e) if math.isfinite(e) else e for e in torque_edges]
        upper_safe = [e - _EDGE_MARGIN * abs(e) if math.isfinite(e) else e for e in torque_edges]
        return speed_mids.tolist(), lower_safe, upper_safe, self.eff_map.tolist()

    def max_torque_for_power(self, omega: float, p_elec: float) -> float:
        """Largest torque (Nm) this motor can make at speed `omega` (rad/s) while
        drawing at most `p_elec` (W) electrically, i.e. the largest T with
        T * omega / efficiency(omega, T) <= p_elec.

        Exact inverse of the nearest-neighbour lookup in `efficiency`. At a fixed
        speed the lookup always reads one row of the map, and along that row the
        efficiency is constant inside each torque cell. Inside cell j electrical power
        is T * omega / eff_j, which is linear in T, so the torque that uses exactly
        p_elec is p_elec * eff_j / omega. Electrical power is not monotone across cells
        (it drops wherever efficiency steps up), so scan from the highest-torque cell
        downward and return the first torque that actually lies inside its cell.

        Does not apply the torque-speed envelope; take the min with `max_torque`.
        """
        if omega <= 0.0 or math.isinf(p_elec):
            return math.inf  # power doesn't constrain torque at standstill
        if p_elec <= 0.0:
            return 0.0
        speed_mids, lower_safe, upper_safe, rows = self._power_inverse_tables
        # Nearest speed row by binary search; same boundaries as `efficiency`.
        row = rows[bisect_left(speed_mids, omega)]
        x = p_elec / omega
        for j in range(len(row) - 1, -1, -1):
            t = x * row[j]
            if t > lower_safe[j]:
                return t if t < upper_safe[j + 1] else upper_safe[j + 1]
        return 0.0  # unreachable: the lowest cell extends to -inf

    def with_power_limit(self, hard_power_limit: float) -> "Motor":
        return replace(self, hard_power_limit=hard_power_limit)


# Relative distance kept from efficiency-map cell boundaries in max_torque_for_power.
_EDGE_MARGIN = 1e-9

RPM_TO_RAD_S = 0.10472  # 2*pi/60, matches the original scripts' scale factor


def _rpm(*values: float) -> np.ndarray:
    return RPM_TO_RAD_S * np.asarray(values, dtype=float)


EMRAX_208 = Motor(
    name="Emrax 208",
    max_rpm=7000,
    torque_speed_w=_rpm(0, 4000, 4500, 5000, 5500, 6000, 7000),
    torque_speed_m=np.array([140, 135, 132, 128, 122, 115, 0], dtype=float),
    eff_speed=_rpm(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000),
    eff_torque=np.array([20, 40, 60, 80, 100, 120, 140], dtype=float),
    eff_map=np.array(
        [
            [86, 88, 89, 88, 86, 84, 82],
            [88, 94, 94, 94, 92, 89, 85],
            [89, 94.5, 95.5, 96, 94.5, 91, 85],
            [89, 95.5, 96, 96, 95, 92, 85],
            [89, 95.5, 96, 96, 95, 92, 85],
            [89, 95, 96, 95.75, 94.75, 92, 85],
            [88, 94.5, 96, 95.5, 94.5, 92, 85],
            [86, 94, 95, 95.25, 94.5, 92, 85],
            [80, 87, 94, 94.5, 94.5, 91, 85],
            [80, 80, 86, 94, 94, 90, 86],
        ],
        dtype=float,
    )
    / 100.0,
    hard_power_limit=61.5e3,
)

EMRAX_228 = Motor(
    name="Emrax 228",
    max_rpm=5000,
    torque_speed_w=_rpm(0, 2000, 3000, 4000, 5000, 5001),
    torque_speed_m=np.array([240, 240, 234, 228, 216, 0], dtype=float),
    eff_speed=EMRAX_208.eff_speed * 5500 / 6000,
    eff_torque=EMRAX_208.eff_torque * 240 / 140,
    eff_map=EMRAX_208.eff_map.copy(),
    hard_power_limit=80e3,
)

EMRAX_268 = Motor(
    name="Emrax 268",
    max_rpm=4500,
    torque_speed_w=_rpm(0, 2000, 4500, 4501),
    torque_speed_m=np.array([500, 500, 450, 0], dtype=float),
    eff_speed=EMRAX_208.eff_speed * 4500 / 6000,
    eff_torque=EMRAX_208.eff_torque * 500 / 140,
    eff_map=EMRAX_208.eff_map.copy(),
    hard_power_limit=80e3,
)

DHX_K40 = Motor(
    name="DHX K40",
    max_rpm=6000,
    torque_speed_w=_rpm(0, 6000, 6001),
    torque_speed_m=np.array([80, 80, 0], dtype=float),
    eff_speed=np.array([0, 7000], dtype=float),
    eff_torque=np.array([0, 80], dtype=float),
    eff_map=np.full((2, 2), 0.93),
    hard_power_limit=40e3,
)
