"""Battery pack model: a series/parallel arrangement of a single `Cell` chemistry.

`Car.battery` feeds `dynamics.py` a per-instant available electrical power (from the
cell equivalent-circuit model in `cell.py`) and tracks pack SOC/energy over the course
of a lap. `Car.battery` stays optional (`None`)
"""

from __future__ import annotations

from dataclasses import dataclass

from .cell import Cell


class Battery(Cell):
    def __init__(self, series: float = 1, parallel: float = 1, cell_type: str = "ampace_jp50"):
        super().__init__(cell_type)
        self._series = series
        self._parallel = parallel
        self._cell_v = self._max_v
        self._soc = 1
        self._dt = 1
        self._i = 1000
        self._pack_v_max = self._max_v * self._series
        self._pack_v_nom = self._nom_v * self._series
        self._pack_energy = self._q_nom / 3600 * self._nom_v * self._series * self._parallel / 1000
        self._cell_T = 25.0
        self._batt_mass = self._m * series * parallel

    def available_power(self) -> float:
        """Max electrical power (W) the pack can deliver at this SOC/temperature."""
        # if the battery is out of charge then return 0
        if self._soc <= 0.0:
            return 0.0
        
        power = (
            self._series # number of cells wired in series (?)
            * self._cell_v # voltage of a single cell under load
            * (self._parallel * self.cell_curr_limits(self._i, self._dt)) 
            # ^ cells in parallel * max current a single cell can discharge
        )
        
        return min(power, 80e3)

    def available_energy(self) -> float:
        return self._parallel * self._series * self._cell_v * self._q_nom / 3600 * self._soc / 1000

    @property
    def voltage(self) -> float:
        """Terminal voltage (V) at this SOC and current draw."""
        return self._series * self._cell_v

    @property
    def pack_energy(self) -> float:
        return self._pack_energy

    @property
    def soc(self) -> float:
        return self._soc

    @property
    def cell_T(self) -> float:
        return self._cell_T

    @property
    def batt_ocv(self) -> float:
        return self.ocv * self._series

    @property
    def batt_r0(self) -> float:
        return self.r_0 / self._parallel**2

    def step(self, i, dt):
        self._i = i
        self._dt = dt

        self._soc, self._cell_v, self._ocv, self._cell_T = self._step(i / self._parallel, dt)


@dataclass(eq=False)
class StaticVoltageBattery:
    """Placeholder battery: fixed voltage window, no SOC/thermal/current effects.

    Mirrors the `hv.vmax` / `hv.vnom` fields already present on `Car` in the original
    MATLAB (tracked as data, never used by the physics). Useful as a starting point to
    subclass once real cell data is needed but the full `Battery`/`Cell` model is more
    than a design study needs.
    """

    vmax: float
    vnom: float
    max_power: float = float("inf")

    def available_power(self) -> float:
        
        return min(self.max_power, 80e3)

    @property
    def voltage(self) -> float:
        return self.vnom
