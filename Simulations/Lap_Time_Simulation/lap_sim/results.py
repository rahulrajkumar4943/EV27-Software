"""Output of a single lap simulation."""

from __future__ import annotations

from dataclasses import dataclass 

import numpy as np


@dataclass(eq=False)
class LapStats:
    """Per-point (and lap-average) power/force figures for a finished speed trace."""

    dt: np.ndarray
    tt: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    k: np.ndarray
    ptraction: np.ndarray
    pbrakes: np.ndarray
    pdrag: np.ndarray
    pkinetic: np.ndarray
    pmotor: np.ndarray
    pelectric: np.ndarray
    tmotor: np.ndarray
    wmotor: np.ndarray
    numotor: np.ndarray
    avg_traction: float
    avg_brakes: float
    avg_drag: float
    avg_motor: float
    avg_electric: float
    energy_j: float  # electrical energy for this one lap, joules
    batt_energy: float  # pack energy consumed this lap, kWh; None if car.battery is None
    batt_i: np.ndarray  # pack current trace, A; None if car.battery is None
    batt_v: np.ndarray  # pack terminal voltage trace, V; None if car.battery is None
    batt_soc: np.ndarray  # pack SOC trace, 0-1 fraction; None if car.battery is None
    cell_T: np.ndarray
    batt_p_limit: (
        np.ndarray
    )  # cell-model-derived available pack power trace, W; None if car.battery is None

    @classmethod
    def from_dynamics_output(cls, out: dict) -> "LapStats":
        # LapStats(arguments from out dict...)
        # **out is like dt=out["dt"], tt=out["tt"], ax=out["ax"], ...
        # populates a new LapStats object with this data
        return cls(**out)


@dataclass(eq=False)
class LapResult:
    """A completed lap simulation: the feasible speed trace plus derived stats."""

    track: "object"
    car: "object"
    regulations: "object"
    dx: float
    x: np.ndarray  # distance around the lap, meters
    vv: np.ndarray  # speed trace, m/s
    stats: LapStats
    splits: np.ndarray  # cumulative time at the end of each segment, seconds
    segment_corner_limit: (
        np.ndarray
    )  # per-segment corner/battery speed limit from this lap's pre-pass, m/s

    @property
    def lap_time(self) -> float:
        return float(self.stats.tt[-1])

    def split_time(self, from_segment: int, to_segment: int) -> float:
        """Elapsed time between the ends of two (0-indexed) segments."""
        return float(self.splits[to_segment] - self.splits[from_segment])

    def total_energy_wh(self, n_laps: int = 1) -> float:
        """Electrical energy (Wh) to complete `n_laps` at this lap's average power."""
        return self.stats.avg_electric * self.lap_time * n_laps / 3600.0
