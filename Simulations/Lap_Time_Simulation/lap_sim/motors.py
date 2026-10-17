"""Motor definitions: speed-torque envelope and efficiency map.

Ported from the `emrax208` / `emrax228` / `emrax268` / `dhx_k40` structs shared across
the original MATLAB scripts.
"""

from __future__ import annotations

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
        """Efficiency (0-1) at motor angular speed(s)/torque(s), clamped to the map."""
        w = np.clip(omega, self.eff_speed[0], self.eff_speed[-1])
        t = np.clip(torque, self.eff_torque[0], self.eff_torque[-1])
        return _nearest_2d(self.eff_speed, self.eff_torque, self.eff_map, w, t)

    @cached_property
    def _torque_cell_edges(self) -> np.ndarray:
        """Torque boundaries of the nearest-neighbour cells in `efficiency`.

        Cell j (nearest grid torque eff_torque[j]) covers (edges[j], edges[j+1]].
        The outer cells extend to +-inf because `efficiency` clamps torque to the map.
        """
        mids = 0.5 * (self.eff_torque[:-1] + self.eff_torque[1:])
        return np.concatenate(([-np.inf], mids, [np.inf]))

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
        if omega <= 0.0 or not np.isfinite(p_elec):
            return np.inf  # power doesn't constrain torque at standstill
        if p_elec <= 0.0:
            return 0.0
        w = min(max(omega, self.eff_speed[0]), self.eff_speed[-1])
        row = self.eff_map[np.abs(self.eff_speed - w).argmin()]
        edges = self._torque_cell_edges
        for j in range(row.shape[0] - 1, -1, -1):
            t = min(edges[j + 1], p_elec * row[j] / omega)
            if t > edges[j]:
                return t
        return 0.0  # unreachable: the lowest cell extends to -inf

    def with_power_limit(self, hard_power_limit: float) -> "Motor":
        return replace(self, hard_power_limit=hard_power_limit)


def _nearest_2d(xg: np.ndarray, yg: np.ndarray, z: np.ndarray, x, y):
    """Nearest-neighbor lookup on a regular (xg, yg) grid; z has shape (len(xg), len(yg))."""
    ix = np.abs(xg[:, None] - np.atleast_1d(x)[None, :]).argmin(axis=0)
    iy = np.abs(yg[:, None] - np.atleast_1d(y)[None, :]).argmin(axis=0)
    result = z[ix, iy]
    return result if np.ndim(x) or np.ndim(y) else result.item()


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
