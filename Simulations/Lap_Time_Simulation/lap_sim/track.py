"""Track model: a closed loop made of straight/constant-radius segments.

Presets below reproduce the four tracks used across the original MATLAB scripts
(`icahn_loop`, `accel_event`, `skidpad_event`, `endur_event`) — these segment tables were
identical, byte-for-byte, in every source file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(eq=False)
class Track:
    description: str
    location: str
    air_density: float
    lengths: np.ndarray  # segment lengths, meters
    curvature: np.ndarray  # signed 1/radius per segment (0 = straight), 1/meters
    limits: np.ndarray  # externally-imposed speed cap per segment, m/s (inf = none)

    def __post_init__(self):
        self.lengths = np.asarray(self.lengths, dtype=float)
        self.curvature = np.asarray(self.curvature, dtype=float)
        self.limits = np.asarray(self.limits, dtype=float)
        if not (len(self.lengths) == len(self.curvature) == len(self.limits)):
            raise ValueError("lengths, curvature, and limits must be the same length")

    @property
    def num_segments(self) -> int:
        return len(self.lengths)

    @property
    def cumulative_length(self) -> np.ndarray:
        return np.cumsum(self.lengths)

    @property
    def total_length(self) -> float:
        return float(self.cumulative_length[-1])

    def segment_index(self, x) -> np.ndarray:
        """Index of the segment containing position(s) `x` along the lap."""
        cumlen = self.cumulative_length
        idx = np.searchsorted(cumlen, x, side="left")
        return np.clip(idx, 0, self.num_segments - 1)

    def curvature_at(self, x) -> np.ndarray:
        return self.curvature[self.segment_index(x)]

    def limit_at(self, x) -> np.ndarray:
        return self.limits[self.segment_index(x)]


def _pi_over(numerator: float, denominator: float) -> float:
    return numerator * np.pi / denominator


ICAHN_LOOP = Track(
    description="Icahn Stadium Test Loop",
    location="New York, NY",
    air_density=1.293,
    lengths=[
        60,
        11.75 * np.pi,
        20,
        15.25 * np.pi / 2,
        10,
        18.75 * np.pi / 2,
        40,
        11.75 * np.pi / 2,
        19,
        21.75 * np.pi / 2,
        14,
        3.25 * np.pi / 2,
        5,
        6.75 * np.pi / 2,
    ],
    curvature=[
        0,
        -1 / 11.75,
        0,
        1 / 15.25,
        0,
        -1 / 18.75,
        0,
        -1 / 11.75,
        0,
        -1 / 21.75,
        0,
        1 / 3.25,
        0,
        -1 / 6.75,
    ],
    limits=np.full(14, np.inf),
)

ACCEL_EVENT = Track(
    description="FSAE Acceleration Event",
    location="Brooklyn, MI",
    air_density=1.293,
    lengths=[10, 0.3, 75, 75, 10],
    curvature=[0, 0, 0, 0, 0],
    limits=[0.1, np.inf, np.inf, np.inf, 0.1],
)

_R_SKIDPAD = 18.25 / 2  # min 15.25/2 + track/2, max 21.25/2 - track/2
SKIDPAD_EVENT = Track(
    description="FSAE Skidpad Event",
    location="Brooklyn, MI",
    air_density=1.293,
    lengths=[
        _R_SKIDPAD,
        _R_SKIDPAD,
        2 * np.pi * _R_SKIDPAD,
        2 * np.pi * _R_SKIDPAD,
        2 * np.pi * _R_SKIDPAD,
        2 * np.pi * _R_SKIDPAD,
        _R_SKIDPAD,
    ],
    curvature=[0, 0, 1 / _R_SKIDPAD, 1 / _R_SKIDPAD, -1 / _R_SKIDPAD, -1 / _R_SKIDPAD, 0],
    limits=[0.1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
)

_ENDUR_LENGTHS_RAW = np.array(
    [
        7,
        5 * np.pi / 6,
        2 * np.pi / 2.8,
        3,
        2 * np.pi / 3,
        4,
        1.5 * np.pi / 6,
        1.5,
        1 * np.pi / 6,
        1.4 * np.pi / 4,
        1.2 * np.pi / 3,
        0.7 * np.pi / 5,
        1.5,
        1.5 * np.pi / 3,
        1 * np.pi / 2.5,
        5 * np.pi / 10,
        5 * np.pi / 18,
        7,
        1.5 * np.pi / 10,
        1 * np.pi / 2.2,
        1,
        1.5 * np.pi / 2.8,
        3,
        1 * np.pi / 1.2,
        0.9,
        1.5 * np.pi / 2.5,
        0.7,
        0.5 * np.pi / 1.8,
        5 * np.pi / 12,
        2,
        6.5 * np.pi / 3.5,
        2 * np.pi / 3.5,
        1.5,
        1 * np.pi / 3.5,
        3 * np.pi / 6.5,
        2,
        0.7 * np.pi / 5,
        0.5 * np.pi / 4.6,
        5,
        3 * np.pi / 9,
        3 * np.pi / 5,
        1 * np.pi / 3.5,
        1 * np.pi / 2.5,
        1 * np.pi / 2.5,
        1 * np.pi / 2.5,
        1.5 * np.pi / 3.25,
        2.5,
        1 * np.pi / 2.3,
        1.7 * np.pi / 1.3,
        1.3 * np.pi / 2.6,
        3.5,
        0.5 * np.pi / 4,
        1 * np.pi / 2.3,
        1 * np.pi / 2.5,
        1 * np.pi / 2.3,
        1,
        0.5 * np.pi / 2.4,
        1.3,
        0.5 * np.pi / 2.3,
        1.5,
        0.8 * np.pi / 2.1,
        1,
        0.8 * np.pi / 2.1,
        0.8,
        0.9 * np.pi / 1.4,
        2.1,
        1.5 * np.pi / 3,
        1.5,
        1 * np.pi / 2.7,
        1.7,
        1.2 * np.pi / 4.5,
        2,
        1 * np.pi / 3.4,
        1.5454,
        2.4123,
    ]
)

_ENDUR_CURVATURE_RAW = np.array(
    [
        0,
        1 / 5,
        -1 / 2,
        0,
        1 / 2,
        0,
        -1 / 1.5,
        0,
        1 / 1,
        -1 / 1.2,
        1 / 1.2,
        -1 / 0.7,
        0,
        -1 / 1.5,
        1 / 1,
        -1 / 5,
        1 / 5,
        0,
        -1 / 1.5,
        1 / 1,
        0,
        -1 / 1.5,
        0,
        -1 / 1,
        0,
        1 / 1.5,
        0,
        -1 / 0.5,
        1 / 5,
        0,
        -1 / 7,
        1 / 2,
        0,
        -1 / 1,
        1 / 3,
        0,
        -1 / 0.7,
        1 / 0.5,
        0,
        1 / 3,
        -1 / 3,
        1 / 1,
        -1 / 1,
        1 / 1,
        -1 / 1,
        1 / 2,
        0,
        -1 / 1,
        1 / 1.7,
        -1 / 1.3,
        0,
        -1 / 0.5,
        1 / 1,
        -1 / 1,
        1 / 1,
        0,
        -1 / 0.5,
        0,
        1 / 0.5,
        0,
        -1 / 0.8,
        0,
        1 / 0.8,
        0,
        -1 / 0.9,
        0,
        -1 / 1.5,
        0,
        -1 / 1,
        0,
        1.2 / 1,
        0,
        -1 / 1,
        0,
        1 / 3,
    ]
) 

# The original scales the endurance layout so its total lap length is exactly 1000 m,
# scaling curvature by the inverse factor so physical corner radii are preserved.
_ENDUR_SCALE = 1000.0 / _ENDUR_LENGTHS_RAW.sum()

ENDURANCE_EVENT = Track(
    description="FSAE Endurance Event Michigan 2024",
    location="Brooklyn, MI",
    air_density=1.293,
    lengths=_ENDUR_LENGTHS_RAW * _ENDUR_SCALE,
    curvature=_ENDUR_CURVATURE_RAW / _ENDUR_SCALE,
    limits=np.full(len(_ENDUR_LENGTHS_RAW), np.inf),
)
