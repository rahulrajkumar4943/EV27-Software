"""Reproduces point_mass_sim_final.m: accel, skidpad, Icahn loop, and endurance runs for
a single car, with the original script's plots and FSAE points calculation.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lap_sim import (
    ACCEL_EVENT,
    EMRAX_208,
    EMRAX_268,
    ENDURANCE_EVENT,
    EV27,
    FSAE_EV,
    ICAHN_LOOP,
    SKIDPAD_EVENT,
    Battery,
    CompetitionScorer,
    LapSimulator,
    plotting,
)
from lap_sim.ecms import EcmsController, linear_soc_schedule, linear_temp_schedule
from lap_sim.vehicle import Aero, Car, Drivetrain, HighVoltageSystem

DX = 0.005  # matches point_mass_sim_final.m's lapsim resolution
LAPS = 22
SOC_DERATE = 0.0
TEMP_START = 30.0  # realistic pre-warmed/hot-day ambient, not the sim's cold-start 25 C default
TEMP_END = 60.0
TEMP_DERATE = 0.0
# Tuned via experiments/tune_ecms.py's staged sweep against build_ev27()'s 140s3p pack
# and the current cell data -
# - "nominal" (h=50 W/m2K): the thermal budget is lax enough that ECMS is effectively
#   only pacing the SOC schedule; final SOC lands close to the target with room to spare
#   on temperature.
# - "reduced" (h=20 W/m2K):
REDUCED_COOLING = False  # matches cell.py's current _COOLING_H_W_M2K = 20.0

_ECMS_GAINS_NOMINAL = dict(
    k=200.0,
    k_temp=15600.6,
    s1_kp=10.0,
    s1_ki=2.0,
    s1_max=66.1,
    s2_kp=0.2,
    s2_ki=0.002,
    s2_max=40.0,
    s2_0=1.0,
)
_ECMS_GAINS_REDUCED_COOLING = dict(
    k=300.0,
    k_temp=13157.9,
    s1_kp=10.0,
    s1_ki=0.5,
    s1_max=45.5,
    s2_kp=0.7,
    s2_ki=0.0015,
    s2_max=40.0,
    s2_0=1.0,
)
_ecms_gains = _ECMS_GAINS_REDUCED_COOLING if REDUCED_COOLING else _ECMS_GAINS_NOMINAL
ECMS_K = _ecms_gains["k"]
ECMS_K_TEMP = _ecms_gains["k_temp"]
ECMS_S1_KP = _ecms_gains["s1_kp"]
ECMS_S1_KI = _ecms_gains["s1_ki"]
ECMS_S1_MAX = _ecms_gains["s1_max"]
ECMS_S2_KP = _ecms_gains["s2_kp"]
ECMS_S2_KI = _ecms_gains["s2_ki"]
ECMS_S2_MAX = _ecms_gains["s2_max"]
ECMS_S2_0 = _ecms_gains["s2_0"]
total_distance = LAPS * ENDURANCE_EVENT.total_length
soc_ref_fn = linear_soc_schedule(total_distance, soc_start=1.0, derate=SOC_DERATE)
temp_ref_fn = linear_temp_schedule(
    total_distance, temp_start=TEMP_START, temp_end=TEMP_END, derate=TEMP_DERATE
)


def build_ev27() -> Car:
    """The EV26B spec specific to point_mass_sim_final.m (differs from the other
    scripts' EV26B in mass, aero, gear ratio, HV window, and motor power cap)."""
    motor = EMRAX_268
    return Car(
        name="EV27 Events",
        mass=200 + 60,
        cg=np.array([742.44, 0.0, 248.52]),
        aero=Aero(cda=1.7, cla=3.05),
        tire=EV27.tire,
        drivetrain=Drivetrain(motor=motor, ratio=4.3, efficiency=0.96, count=1),
        hv=HighVoltageSystem(vmax=255, vnom=216),
        l_base=1.530,
        # Fresh Battery per call
        battery=Battery(series=116, parallel=3, cell_type="ampace_jp50"),
    )


def main():
    car = build_ev27()
    if car.battery is not None:
        car.battery._T = (
            TEMP_START  # realistic pre-warmed/hot-day start, not the sim's cold-start 25 C default
        )
    # car = EV25
    scorer = CompetitionScorer()

    accel = LapSimulator(FSAE_EV, ACCEL_EVENT, car, dx=DX).run()
    print(
        f"[Accel]    peak electric power = {accel.stats.pelectric.max() / 1e3:.1f} kW, "
        f"top speed = {accel.vv.max() * 3.6:.1f} km/h"
    )
    plotting.plot_lap_overview([accel])
    accel_time = accel.split_time(1, 2)
    print(f"[Accel]    time = {accel_time:.3f} s")

    skidpad = LapSimulator(FSAE_EV, SKIDPAD_EVENT, car, dx=DX).run()
    plotting.plot_lap_overview([skidpad])
    skidpad_time = skidpad.split_time(2, 3)
    print(f"[Skidpad]  time = {skidpad_time:.3f} s")

    icahn = LapSimulator(FSAE_EV, ICAHN_LOOP, car, dx=DX).run()
    plotting.plot_lap_overview([icahn])
    icahn_time = icahn.lap_time
    print(f"[Icahn]    lap time = {icahn_time:.3f} s")

    endurance_regs = FSAE_EV

    # ECMS is only attached here, not on `car` above
    endur_car = car.replace(
        ecms=EcmsController(
            soc_ref_fn=soc_ref_fn,
            temp_ref_fn=temp_ref_fn,
            k=ECMS_K,
            k_temp=ECMS_K_TEMP,
            s1_kp=ECMS_S1_KP,
            s1_ki=ECMS_S1_KI,
            s1_max=ECMS_S1_MAX,
            s2_kp=ECMS_S2_KP,
            s2_ki=ECMS_S2_KI,
            s2_max=ECMS_S2_MAX,
            s2_0=ECMS_S2_0,
        )
    )
    # print("[Endurance] Initializing simulation...", flush=True)
    endur_sim = LapSimulator(endurance_regs, ENDURANCE_EVENT, endur_car, dx=0.5)
    # print("[Endurance] Initializing done", flush=True)
    # print("[Endurance] Starting simulation", flush=True)
    if car.battery is not None:
        endur = endur_sim.run_multi_lap(22)
        endur_time = endur.lap_time
    else:
        endur = endur_sim.run()
        endur_time = endur.lap_time * 22
    print(
        f"[Endur]    peak electric power = {endur.stats.pelectric.max() / 1e3:.1f} kW, "
        f"top speed = {endur.vv.max() * 3.6:.1f} km/h"
    )
    plotting.plot_lap_overview([endur])
    fig_speed, ax_speed = plt.subplots(figsize=(8, 5))
    plotting.plot_lap_speed_traces(endur, ax=ax_speed)
    ax_speed.set_title("Endurance speed by lap")
    if car.battery is not None:
        plotting.plot_battery_stats([endur])
        final_energy = endur.stats.batt_energy
        print(f"[Endur] battery energy usage = {final_energy:.1f} kWh")

    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(7, 6))
    ax_v.plot(endur.vv)
    ax_v.set_ylabel("Speed (m/s)")
    ax_a.plot(endur.stats.ax)
    ax_a.set_ylabel("ax (m/s^2)")
    fig.suptitle("Endurance speed & longitudinal acceleration trace")

    fig_gg, ax_gg = plt.subplots()
    ax_gg.plot(endur.stats.ax, endur.stats.ay, "*")
    ax_gg.set_xlabel("ax (m/s^2)")
    ax_gg.set_ylabel("ay (m/s^2)")
    ax_gg.set_title("Endurance G-G diagram")

    avg_power = endur.stats.avg_electric
    energy_wh = avg_power * endur_time / 3600.0
    print(
        f"[Endur]    22-lap time = {endur_time:.1f} s, avg electric power = "
        f"{avg_power / 1e3:.2f} kW, energy = {energy_wh:.1f} Wh"
    )

    score = scorer.score(accel_time, skidpad_time, endur_time * 0.8 / 22, endur_time, energy_wh)
    print(
        f"[Score]    total = {score.total:.1f}  "
        f"(accel {score.accel:.1f}, skidpad {score.skidpad:.1f}, "
        f"autocross {score.autocross:.1f}, endurance {score.endurance:.1f}, "
        f"efficiency {score.efficiency:.1f})"
    )

    plt.show()


if __name__ == "__main__":
    main()
