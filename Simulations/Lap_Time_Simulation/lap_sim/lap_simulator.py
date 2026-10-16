"""Adapter between the OOP data model (Car/Track/Regulations) and the numeric core."""

from __future__ import annotations

import numpy as np

import cProfile # profiler

from . import dynamics
from .regulations import Regulations
from .results import LapResult, LapStats
from .track import Track
from .vehicle import Car

MAX_BATTERY_POWER = 80e3

class LapSimulator:
    """Runs one point-mass lap simulation for a given car/track/ruleset.

    `dx` is the (approximate) simulation step in meters; it is adjusted slightly so the
    track length divides evenly into a whole number of steps.
    """

    def __init__(self, regulations: Regulations, track: Track, car: Car, dx: float = 0.01):
        if car.l_base is None:
            raise ValueError(
                f"{car.name}: wheelbase (car.l_base) is required to run a lap simulation "
                "(used by the weight-transfer traction limit)."
            )
        self.regulations = regulations
        self.track = track
        self.car = car
        self.dx_target = dx

    def run(self, warm_start: np.ndarray = None) -> LapResult:
        """`warm_start`, if given, is a per-segment speed array (same shape as
        `track.curvature`, typically a previous lap's `segment_corner_limit`) used to
        seed each segment's corner-limit solve instead of a cold v0=0.0 start -- see
        `run_multi_lap`.

        Runs in two passes when a battery is present. Pass 1 solves the speed trace
        with `battery=None` -- braking zones (backward marches) come out fully correct
        here since braking never depends on the battery (no regen modeled), and this
        pass is otherwise identical to the no-battery case. Pass 2
        (`dynamics.battery_forward_pass`) overlays the battery's acceleration limit onto
        that trace using forward marches only, which is what lets it step the battery's
        SOC/voltage/temperature sequentially as each point is finalized without a later
        point ever retroactively invalidating an earlier one -- see
        `battery_forward_pass`'s docstring for why the combined forward+backward version
        can't safely do that.
        """
        track, car, regs = self.track, self.car, self.regulations
        drivetrain = car.drivetrain
        motor = drivetrain.motor

        total_length = track.total_length
        n_points = int(np.ceil(total_length / self.dx_target))
        dx = total_length / n_points
        x = np.arange(n_points) * dx

        seg_idx_per_point = track.segment_index(x)
        curvature_by_point = track.curvature[seg_idx_per_point]

        # Apply the regulation's mechanical power limit; the electrical power ceiling
        # is enforced separately inside force_balance().
        effective_power_limit = min(
            regs.power_limit,
            MAX_BATTERY_POWER * drivetrain.efficiency,
        )


        base_args = (
            car.mass,
            car.cg_height_m,
            car.l_base,
            car.aero.cda,
            car.aero.cla,
            track.air_density,
            car.tire,
            # regs.power_limit, 
            effective_power_limit,
            drivetrain.ratio,
            float(drivetrain.count),
            drivetrain.efficiency,
            motor,
            None,
        )

        # if warm_start is None:
        #     segment_corner_limit = np.array(
        #         [dynamics.corner_speed_limit(k, *base_args) for k in track.curvature]
        #     )
        # else:
        #     segment_corner_limit = np.array(
        #         [
        #             dynamics.corner_speed_limit(k, *base_args, v0=v0)
        #             for k, v0 in zip(track.curvature, warm_start)
        #         ]
        #     )

        # replaced above with below


        if warm_start is None:
            segment_corner_limit = np.array([
                dynamics.corner_speed_limit(
                    k,
                    *base_args,
                    motor_efficiency_fn=motor.efficiency,
                    max_electrical_power=MAX_BATTERY_POWER,
                )
                for k in track.curvature
            ])
        else:
            segment_corner_limit = np.array([
                dynamics.corner_speed_limit(
                    k,
                    *base_args,
                    v0=v0,
                    motor_efficiency_fn=motor.efficiency,
                    max_electrical_power=MAX_BATTERY_POWER,
                )
                for k, v0 in zip(track.curvature, warm_start)
            ])



        segment_gross_limit = np.minimum(segment_corner_limit, track.limits)
        point_limit = segment_gross_limit[seg_idx_per_point]

        # vv_base = dynamics.trace_speed_profile(
        #     point_limit, curvature_by_point, seg_idx_per_point, track.limits, dx, *base_args[:-1]
        # )

        # replaced above with below

        vv_base = dynamics.trace_speed_profile(
            point_limit,
            curvature_by_point,
            seg_idx_per_point,
            track.limits,
            dx,
            *base_args[:-1],
            max_electrical_power=MAX_BATTERY_POWER,
        )       

        if car.battery is None:
            vv = vv_base
            batt_i = batt_v = batt_soc = cell_T = batt_p_limit = None
        else:
            vv, batt_i, batt_v, batt_soc, cell_T, batt_p_limit = dynamics.battery_forward_pass(
                vv_base,
                curvature_by_point,
                dx,
                car.mass,
                car.cg_height_m,
                car.l_base,
                car.aero.cda,
                car.aero.cla,
                track.air_density,
                car.tire,
                # regs.power_limit, 
                effective_power_limit,
                drivetrain.ratio,
                float(drivetrain.count),
                drivetrain.efficiency,
                motor,
                car.battery,
                car.ecms,
                max_electrical_power=MAX_BATTERY_POWER,
            )

        stats_dict = dynamics.power_and_energy(
            vv,
            dx,
            curvature_by_point,
            car.mass,
            car.aero.cda,
            track.air_density,
            drivetrain.ratio,
            drivetrain.efficiency,
            car.tire,
            motor,
            car.battery,
            batt_i,
            batt_v,
            batt_soc,
            cell_T,
            batt_p_limit,
        )
        stats = LapStats.from_dynamics_output(stats_dict)

        splits = _segment_finish_times(seg_idx_per_point, stats.tt, track.num_segments)

        return LapResult(
            track=track,
            car=car,
            regulations=regs,
            dx=dx,
            x=x,
            vv=vv,
            stats=stats,
            splits=splits,
            segment_corner_limit=segment_corner_limit,
        )

    def run_multi_lap(self, n_laps: int) -> LapResult:
        """Run `n_laps` consecutive laps back-to-back, carrying `car.battery`'s state
        (SOC, RC-branch voltages, temperature) -- and, if set, `car.ecms`'s state (its
        price `lam` and cumulative distance traveled) -- forward from one lap into the
        next, and concatenate the per-point traces into one combined result.

        Each lap re-solves the corner/battery-limited speed trace against whatever
        state the battery is in by that point, so a degraded pack legitimately shows up
        as a slower trace in later laps -- unlike scaling a single lap's time by
        `n_laps`, which is only exact when the car has no battery (nothing state-
        dependent changes lap to lap, so one solve is already the true answer).

        Each lap's corner-limit pre-pass is warm-started from the previous lap's
        result (consecutive laps' battery-derived limits are usually close together),
        rather than cold-starting every lap from v0=0.0.
        """
        laps = []
        warm_start = None
        for lapnum in range(n_laps):
            # print(f"[Endurance] Starting lap {lapnum + 1}/{n_laps}")

            # PROFILE FIRST LAP
            # if lapnum == 0:  # Profile the first lap
            #     profiler = cProfile.Profile()
            #     lap = profiler.runcall(self.run, warm_start=warm_start)
            #     print("\n[Profiler] First-lap hotspots:", flush=True)
            #     profiler.print_stats(sort="cumtime")
            # else:
            #     lap = self.run(warm_start=warm_start)

            lap = self.run(warm_start=warm_start)
            warm_start = lap.segment_corner_limit
            laps.append(lap)
        return _concat_laps(laps)


def _concat_laps(laps: list) -> LapResult:
    """Stitch consecutive `LapResult`s (same track/car/dx) into one combined result,
    offsetting each lap's distance and elapsed time by the cumulative total so far.
    """
    first = laps[0]
    total_length = first.track.total_length

    x = np.concatenate([lap.x + i * total_length for i, lap in enumerate(laps)])
    vv = np.concatenate([lap.vv for lap in laps])

    time_offsets = np.cumsum([0.0] + [lap.lap_time for lap in laps[:-1]])
    dt = np.concatenate([lap.stats.dt for lap in laps])
    tt = np.concatenate([lap.stats.tt + off for lap, off in zip(laps, time_offsets)])

    per_point_fields = [
        "ax",
        "ay",
        "k",
        "ptraction",
        "pbrakes",
        "pdrag",
        "pkinetic",
        "pmotor",
        "pelectric",
        "tmotor",
        "wmotor",
        "numotor",
    ]
    combined = {
        f: np.concatenate([getattr(lap.stats, f) for lap in laps]) for f in per_point_fields
    }

    has_battery = first.stats.batt_i is not None
    battery_fields = ["batt_i", "batt_v", "batt_soc", "cell_T", "batt_p_limit"]
    for f in battery_fields:
        combined[f] = (
            np.concatenate([getattr(lap.stats, f) for lap in laps]) if has_battery else None
        )

    def time_avg(power):
        return float(np.sum(dt * power) / np.sum(dt))

    energy_j = float(np.sum(combined["pelectric"] * dt))
    batt_energy = (
        float(np.sum(combined["batt_i"] * combined["batt_v"] * dt)) / 3.6e6 if has_battery else None
    )

    stats = LapStats(
        dt=dt,
        tt=tt,
        **combined,
        avg_traction=time_avg(combined["ptraction"]),
        avg_brakes=time_avg(combined["pbrakes"]),
        avg_drag=time_avg(combined["pdrag"]),
        avg_motor=time_avg(combined["pmotor"]),
        avg_electric=time_avg(combined["pelectric"]),
        energy_j=energy_j,
        batt_energy=batt_energy,
    )

    splits = laps[-1].splits + time_offsets[-1]

    return LapResult(
        track=first.track,
        car=first.car,
        regulations=first.regulations,
        dx=first.dx,
        x=x,
        vv=vv,
        stats=stats,
        splits=splits,
        segment_corner_limit=laps[-1].segment_corner_limit,
    )


def _segment_finish_times(
    seg_idx_per_point: np.ndarray, tt: np.ndarray, num_segments: int
) -> np.ndarray:
    """Cumulative lap time at the last simulation point falling in each segment.

    A segment shorter than the simulation step can end up with no points of its own; it
    is left at 0, matching the original script's behavior.
    """
    splits = np.zeros(num_segments)
    segments = np.arange(num_segments)
    boundaries = np.clip(
        np.searchsorted(seg_idx_per_point, segments, side="right") - 1,
        0,
        len(seg_idx_per_point) - 1,
    )
    hit = seg_idx_per_point[boundaries] == segments
    splits[hit] = tt[boundaries[hit]]
    return splits
