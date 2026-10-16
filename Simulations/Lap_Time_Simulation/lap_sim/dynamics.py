"""The point-mass lap simulation core.

The algorithm, in three stages:

1. **Corner speed limits** — for every distinct track curvature, time-march a point mass
   at that curvature until its longitudinal acceleration settles to ~0; that steady-state
   speed is the fastest the car can sustain through a corner of that radius (limited by
   grip, downforce, and drag).
2. **Speed-trace evolution** — starting from those per-segment limits, walk around the
   closed loop once. Wherever the limit rises from one point to the next, forward-march a
   traction/power/motor-limited acceleration profile until it rejoins the limit curve;
   wherever it falls, back-march a braking profile the same way. Combining every such
   profile with the base limit curve via an elementwise minimum yields a single feasible
   speed trace for the whole lap.
3. **Power/energy accounting** — from the finished speed trace, derive per-point
   traction/brake/drag power, motor torque and efficiency (via the efficiency map), and
   electrical power, plus lap-average figures.

Stage 2 (`trace_speed_profile`/`_march`) is the hot loop.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.806  # m/s^2, matches the original scripts' constant
MAX_WEIGHT_TRANSFER = 0.9  # cap on the linear load-transfer fraction below (see force_balance)



# helper function to calculate electrical power for a give tractive acceleration
# same equations as power_and_energy() function
def electrical_power_for_tractive_accel(
    v,
    tractive_accel,
    mass,
    tire,
    ratio,
    drivetrain_efficiency,
    motor_efficiency_fn,
):
    """Electrical input power required for a given tractive acceleration.

    tractive_accel is wheel tractive force divided by mass.
    It is NOT net vehicle acceleration after aerodynamic drag.
    """

    if v <= 0.0 or tractive_accel <= 0.0:
        return 0.0

    f_traction = mass * tractive_accel

    omega = v / tire.radius * ratio
    torque = (
        f_traction * tire.radius
        / ratio
        / drivetrain_efficiency
    )

    eta_motor = float(motor_efficiency_fn(omega, torque))

    if not np.isfinite(eta_motor) or eta_motor <= 0.0:
        raise ValueError(
            f"Invalid motor efficiency: {eta_motor}"
        )

    p_wheel = f_traction * v

    return p_wheel / drivetrain_efficiency / eta_motor


# we can calculate power from acceleration but not really the other way round
# best sol. is to use binary search to find the best acceleration that fits the 80kw limit

def limit_tractive_accel_by_electrical_power(
    v,
    tractive_accel_max,
    max_electrical_power,
    mass,
    tire,
    ratio,
    drivetrain_efficiency,
    motor_efficiency_fn,
):
    if tractive_accel_max <= 0.0 or v <= 0.0:
        return max(0.0, tractive_accel_max)

    if not np.isfinite(max_electrical_power):
        return tractive_accel_max

    # Check whether the original maximum acceleration already satisfies
    # the electrical power limit. Avoid binary search if it does.
    p_max = electrical_power_for_tractive_accel(
        v,
        tractive_accel_max,
        mass,
        tire,
        ratio,
        drivetrain_efficiency,
        motor_efficiency_fn,
    )

    if p_max <= max_electrical_power:
        return tractive_accel_max

    lo = 0.0
    hi = tractive_accel_max

    # 30 iterations are sufficient for a precise acceleration bound.
    for _ in range(30):
        mid = 0.5 * (lo + hi)

        p_mid = electrical_power_for_tractive_accel(
            v,
            mid,
            mass,
            tire,
            ratio,
            drivetrain_efficiency,
            motor_efficiency_fn,
        )

        if p_mid <= max_electrical_power:
            lo = mid
        else:
            hi = mid

    return lo


def force_balance(
    v,
    k,
    mass,
    cg_height,
    wheelbase,
    cda,
    cla,
    air_density,
    tire,
    power_limit,
    ratio,
    count,
    efficiency,
    motor_w,
    motor_m,
    battery,
    t: float = None,
    battery_power_override: float = None,
    motor_efficiency_fn=None,
    max_electrical_power=None,
):
    """Longitudinal brake/accel limits (m/s^2) at speed `v` and curvature `k`.

    Combines a friction-circle-derated grip limit with drag, then caps the achievable
    acceleration by whichever of (electrical power, battery power, weight-transfer-limited
    traction, motor torque curve) is most restrictive.

    `battery_power_override`, when given, replaces `battery.available_power()` as the
    battery's power ceiling for this call
    """
    downforce = 0.5 * air_density * cla * v * v
    fz = GRAVITY * mass + downforce
    fy = v * v * k * mass

    potential_fx = tire.get_fx(fz, fy)

    drag = 0.5 * air_density * cda * v * v
    brake = (-potential_fx - drag) / mass

    power_limited_accel = power_limit / mass / v if v > 0.0 else np.inf



    # if battery is None:
    #     battery_limited_accel = np.inf
    # elif battery_power_override is not None:
    #     battery_limited_accel = battery_power_override / mass / v if v > 0.0 else np.inf
    # else:
    #     battery_limited_accel = battery.available_power() / mass / v if v > 0.0 else np.inf


    # replaced above with below
    # assuming battery.available_power() and battery_power_override are in watts

    battery_power_limit = np.inf if battery is None else battery.available_power()

    if battery_power_override is not None:
        battery_power_limit = min(
            battery_power_limit,
            max(0.0, battery_power_override),
        )

    electrical_power_limit = battery_power_limit
    if max_electrical_power is not None:
        electrical_power_limit = min(
            electrical_power_limit,
            max_electrical_power,
        )


    # Linear load-transfer model -- only valid up to "all the load has transferred to
    # one axle" (weight_transfer -> 1). At high enough potential_fx (e.g. a lot of
    # downforce) the raw fraction can reach or exceed that, sending the denominator
    # below through zero and traction_limited_accel to +-inf; cap it short of 1 so the
    # corner-speed search never hits that singularity.
    weight_transfer = min(
        (cg_height / wheelbase) * potential_fx / mass / GRAVITY, MAX_WEIGHT_TRANSFER
    )
    traction_limited_accel = 0.5 * potential_fx / mass / (1.0 - weight_transfer)

    omega = v / tire.radius * ratio
    max_torque = np.interp(omega, motor_w, motor_m)
    motor_limited_accel = max_torque * count * ratio * efficiency / tire.radius / mass

    # accel_limit = min(
    #     power_limited_accel, traction_limited_accel, motor_limited_accel, battery_limited_accel
    # )
    # accel = accel_limit - drag / mass
    # return brake, accel

    # replaced above with below

    accel_limit = min(
        power_limited_accel,
        traction_limited_accel,
        motor_limited_accel,
    )

    if motor_efficiency_fn is not None:
        accel_limit = limit_tractive_accel_by_electrical_power(
            v=v,
            tractive_accel_max=accel_limit,
            max_electrical_power=electrical_power_limit,
            mass=mass,
            tire=tire,
            ratio=ratio,
            drivetrain_efficiency=efficiency,
            motor_efficiency_fn=motor_efficiency_fn,
        )

    accel = accel_limit - drag / mass

    return brake, accel




# instead of simulating thousands of tiny time steps to find the steady-state corner speed, 
# solve directly for the speed where acceleration reaches zero using a bracketed bisection search.
def corner_speed_limit(
    k,
    mass,
    cg_height,
    wheelbase,
    cda,
    cla,
    air_density,
    tire,
    power_limit,
    ratio,
    count,
    efficiency,
    motor,
    battery,
    v0=0.0,
    time_step=0.001,
    accel_tolerance=0.001 * GRAVITY,
    max_steps=100_000,
    motor_efficiency_fn=None,
    max_electrical_power=None,
):
    """Find steady-state corner speed using a bracketed speed search.

    Instead of time-marching in 1 ms increments, find the speed at which
    the longitudinal acceleration changes from positive to non-positive.
    """

    def acceleration_at(v):
        _, accel = force_balance(
            v,
            k,
            mass,
            cg_height,
            wheelbase,
            cda,
            cla,
            air_density,
            tire,
            power_limit,
            ratio,
            count,
            efficiency,
            motor.torque_speed_w,
            motor.torque_speed_m,
            battery,
            motor_efficiency_fn=motor_efficiency_fn,
            max_electrical_power=max_electrical_power,
        )
        return accel

    # If the car cannot accelerate from rest at this curvature,
    # there is no positive steady-state corner speed.
    if acceleration_at(0.0) <= 0.0:
        return 0.0

    # Find an upper speed where acceleration is non-positive.
    lo = 0.0
    hi = max(5.0, v0 * 1.2)

    while acceleration_at(hi) > 0.0:
        hi *= 1.5

        if hi > 200.0:
            raise RuntimeError(
                f"Could not bracket corner speed for curvature {k}. "
                f"Acceleration remains positive at {hi / 1.5:.1f} m/s."
            )

    # Bisect the speed interval.
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        accel = acceleration_at(mid)

        if accel > 0.0:
            lo = mid
        else:
            hi = mid

        if hi - lo < 1e-3:
            break

    return 0.5 * (lo + hi)



def _march(
    direction,
    start_idx,
    v0,
    dx,
    t,
    limits_snapshot,
    max_steps,
    curvature_by_point,
    mass,
    cg_height,
    wheelbase,
    cda,
    cla,
    air_density,
    tire,
    power_limit,
    ratio,
    count,
    efficiency,
    motor_w,
    motor_m,
    battery,
    motor_efficiency_fn=None,
    max_electrical_power=None,
):
    """Forward (accel) or backward (brake) march from `start_idx`, until the profile
    rejoins `limits_snapshot`. Returns a full-length array equal to `limits_snapshot`
    except along the visited stretch, plus the acceleration at the starting point.
    """
    n = limits_snapshot.shape[0]
    profile = limits_snapshot.copy()
    idx = start_idx
    v = v0
    step = dx if direction > 0 else -dx
    steps = 0

    while v <= limits_snapshot[idx] and steps < max_steps:
        k = curvature_by_point[idx]
        brake, accel = force_balance(
            v,
            k,
            mass,
            cg_height,
            wheelbase,
            cda,
            cla,
            air_density,
            tire,
            power_limit,
            ratio,
            count,
            efficiency,
            motor_w,
            motor_m,
            battery,
            t,
            motor_efficiency_fn=motor_efficiency_fn,
            max_electrical_power=max_electrical_power,
        )
        a = accel if direction > 0 else brake
        v_sq = v * v + 2.0 * a * step
        v = np.sqrt(v_sq) if v_sq > 0.0 else 0.0
        profile[idx] = v
        idx = (idx + direction) % n
        steps += 1
    return profile


def trace_speed_profile(
    point_limit,
    curvature_by_point,
    seg_idx_per_point,
    track_limits,
    dx,
    mass,
    cg_height,
    wheelbase,
    cda,
    cla,
    air_density,
    tire,
    power_limit,
    ratio,
    count,
    efficiency,
    motor,
    max_electrical_power=None,
):
    """The feasible speed trace around one lap, battery-blind (no `battery` param at
    all

    `point_limit` is the per-point speed ceiling before considering accel/braking
    capability (corner grip limits and any externally-imposed segment limits).
    Wherever it steps up or down between consecutive points, an accel/brake profile is
    marched out from that transition and merged in via an elementwise minimum — the
    classic point-mass lap-sim technique.

    Returns `vv`.
    """
    n = point_limit.shape[0]
    vv = point_limit.copy()
    t = 0.0
    for i in range(n):
        p = (i - 1) % n
        max_steps = n - i
        if point_limit[p] < point_limit[i]:
            profile = _march(
                1,
                i,
                point_limit[p],
                dx,
                t,
                vv,
                max_steps,
                curvature_by_point,
                mass,
                cg_height,
                wheelbase,
                cda,
                cla,
                air_density,
                tire,
                power_limit,
                ratio,
                count,
                efficiency,
                motor.torque_speed_w,
                motor.torque_speed_m,
                None,
                motor_efficiency_fn=motor.efficiency,
                max_electrical_power=max_electrical_power,
            )
            vv = np.minimum(vv, profile)
        elif point_limit[p] > point_limit[i]:
            profile = _march(
                -1,
                p,
                point_limit[i],
                dx,
                t,
                vv,
                max_steps,
                curvature_by_point,
                mass,
                cg_height,
                wheelbase,
                cda,
                cla,
                air_density,
                tire,
                power_limit,
                ratio,
                count,
                efficiency,
                motor.torque_speed_w,
                motor.torque_speed_m,
                None,
                motor_efficiency_fn=motor.efficiency,
                max_electrical_power=max_electrical_power,
            )
            vv = np.minimum(vv, profile)
    return vv


def battery_forward_pass(
    vv_base,
    curvature_by_point,
    dx,
    mass,
    cg_height,
    wheelbase,
    cda,
    cla,
    air_density,
    tire,
    power_limit,
    ratio,
    count,
    efficiency,
    motor,
    battery,
    ecms=None,
    max_electrical_power=None,
):
    """Overlay battery-limited acceleration onto an already-resolved, battery-blind
    speed trace `vv_base` (no regen, braking never depends on the battery, so
    `vv_base`'s braking zones are already final and this pass never needs to touch
    them).

    When `ecms` (an `EcmsController`) is given, the battery isn't simply granted its full
    physical `available_power()` at each point, `vv_base`'s own point-to-point speed
    change already says what current the car would draw here if energy were free
    (`i_request`); ECMS decides how much of that to actually
    grant, gated by the running price `ecms.lam`, and that gated power (not the raw
    physical ceiling) is what `force_balance` sees as the battery's limit for this point.

    A single forward integration, point by point: from each point's actual (possibly
    battery-reduced) speed, take one battery-aware acceleration step and cap the result
    at `vv_base[i]`

    Returns `(vv, batt_i, batt_v, batt_soc, cell_t, batt_p_limit)`.
    """
    n = vv_base.shape[0]
    vv = vv_base.copy()
    batt_i = np.empty(n)
    batt_v = np.empty(n)
    batt_soc = np.empty(n)
    cell_t = np.empty(n)
    batt_p_limit = np.empty(n)

    for i in range(n):
        p = (i - 1) % n

        battery_power_override = None
        if ecms is not None:
            dt_i = dx / vv_base[i] if vv_base[i] > 0.0 else np.inf
            accel_request = (vv_base[i] ** 2 - vv_base[p] ** 2) / (2.0 * dx)
            f_drag_i = 0.5 * air_density * cda * vv_base[i] ** 2
            f_tract_request = accel_request * mass + f_drag_i
            power_request = f_tract_request * vv_base[i] if f_tract_request > 0.0 else 0.0
            i_request = power_request / battery.voltage

            i_ecms = ecms.command_current(
                battery, i_request, battery.batt_ocv, battery.batt_r0, dt_i
            )
            battery_power_override = i_ecms * battery.voltage

        _, accel = force_balance(
            vv[p],
            curvature_by_point[p],
            mass,
            cg_height,
            wheelbase,
            cda,
            cla,
            air_density,
            tire,
            power_limit,
            ratio,
            count,
            efficiency,
            motor.torque_speed_w,
            motor.torque_speed_m,
            battery,
            battery_power_override=battery_power_override,
            motor_efficiency_fn=motor.efficiency,
            max_electrical_power=max_electrical_power,
        )
        v_sq = vv[p] * vv[p] + 2.0 * accel * dx
        v_candidate = np.sqrt(v_sq) if v_sq > 0.0 else 0.0
        vv[i] = min(v_candidate, vv_base[i])

        batt_i[i] = step_battery(
            battery, motor, vv, dx, mass, i, p, cda, air_density, efficiency, tire, ratio
        )
        batt_v[i] = battery.voltage
        batt_soc[i] = battery.soc
        cell_t[i] = battery.cell_T
        batt_p_limit[i] = battery.available_power()

        if ecms is not None:
            dt_actual = dx / vv[i] if vv[i] > 0.0 else np.inf
            ecms.update_s1(battery.soc, dx, dt_actual)
            ecms.update_s2(battery.cell_T, dx, dt_actual)

    return vv, batt_i, batt_v, batt_soc, cell_t, batt_p_limit


def step_battery(battery, motor, vv, dx, mass, i, p, cda, air_density, efficiency, tire, ratio):
    dt = dx / vv[i]
    ax = (vv[i] ** 2 - vv[p] ** 2) / (2.0 * dx)
    f_drag = 0.5 * cda * air_density * vv[i] ** 2
    f_tract = ax * mass + f_drag
    power = f_tract * vv[i] if f_tract > 0.0 else 0.0
    power = power / efficiency
    tmotor = f_tract * tire.radius / ratio / efficiency if f_tract > 0.0 else 0.0
    wmotor = vv[i] / tire.radius * ratio
    numotor = motor.efficiency(wmotor, tmotor)
    power_electric = power / numotor if f_tract > 0.0 else 0.0
    current = power_electric / battery.voltage
    battery.step(current, dt)
    return current


def power_and_energy(
    vv,
    dx,
    curvature_by_point,
    mass,
    cda,
    air_density,
    ratio,
    efficiency,
    tire,
    motor,
    battery,
    batt_i,
    batt_v,
    batt_soc,
    cell_T,
    batt_p_limit,
):
    """Per-point power/energy accounting for a finished speed trace `vv`.

    Returns a dict of per-point arrays plus lap-average figures. Vectorized (not
    JIT-compiled) since it is a single pass with no sequential dependency.
    """

    # by the time this function is called, the lap is done and were just calculating the 
    # energy etc required for that
    dt = dx / vv
    tt = np.cumsum(dt)

    ax = (np.roll(vv, -1) - vv) / dt
    # trace_speed_profile doesn't guarantee vv[0] and vv[-1] agree even though they're
    # the same physical point (the lap's start/finish) -- the two ends can be governed
    # by different, uncoordinated accel/brake marches. Wrapping across that gap here
    # produces an unphysical multi-hundred-kW acceleration spike at the last point every
    # lap. Reusing the last interior (trustworthy) derivative instead of the wraparound
    # one is a reporting-layer patch, not a fix for the underlying discontinuity itself.
    ax[-1] = ax[-2]
    ay = vv**2 * curvature_by_point

    f = mass * ax
    drag = 0.5 * cda * air_density * vv**2
    pkinetic = -f * vv
    pdrag = -drag * vv
    ft = f + drag
    # if total force is positive aka car accelerating, then set traction_mask[i] to True
    traction_mask = ft > 0

    # traction mask is bool array of whether the car is accelerating at each point

    # if accelerating then ptraction is total force * vehicle speed
    ptraction = np.where(traction_mask, ft * vv, 0.0)

    # if accelerating then pmotor is ptraction / efficiency ... mechanical power required from the motor
    pmotor = np.where(traction_mask, ptraction / efficiency, 0.0)

    # rotational torque required from electric motor
    tmotor = np.where(traction_mask, ft * tire.radius / ratio / efficiency, 0.0)

    # rotational speed of the motor in radians/second
    wmotor = vv / tire.radius * ratio

    # efficiency of the motor
    # issue is that this is a lookup map
    # for pelectric to be limited to 80kw, only vv has to go down, 
    # but lookup tables cant exactly be inverted like functions
    numotor = motor.efficiency(wmotor, tmotor)

    
    # if it is accelerating then pmotor / numotor calculates electrical power
    # by dividing motor power by motor efficiency 
    pelectric = np.where(traction_mask, pmotor / numotor, 0.0) # problem line

    # braking power when car is decelerating WHEN NOT ACCELERATING
    pbrakes = np.where(~traction_mask, ft * vv, 0.0)

    batt_energy_usage = (
        None if battery is None else float(np.sum(batt_i * batt_v * dt)) / 3.6e6
    )  # J -> kWh

    def time_avg(power):
        return float(np.sum(dt * power) / np.sum(dt))

    return {
        "dt": dt,
        "tt": tt,
        "ax": ax,
        "ay": ay,
        "k": curvature_by_point,
        "ptraction": ptraction,
        "pbrakes": pbrakes,
        "pdrag": pdrag,
        "pkinetic": pkinetic,
        "pmotor": pmotor,
        "pelectric": pelectric,
        "tmotor": tmotor,
        "wmotor": wmotor,
        "numotor": numotor,
        "avg_traction": time_avg(ptraction),
        "avg_brakes": time_avg(pbrakes),
        "avg_drag": time_avg(pdrag),
        "avg_motor": time_avg(pmotor),
        "avg_electric": time_avg(pelectric),
        "energy_j": float(np.sum(pelectric * dt)),
        "batt_energy": batt_energy_usage,
        "batt_i": batt_i,
        "batt_v": batt_v,
        "batt_soc": batt_soc,
        "cell_T": cell_T,
        "batt_p_limit": batt_p_limit,
    }
