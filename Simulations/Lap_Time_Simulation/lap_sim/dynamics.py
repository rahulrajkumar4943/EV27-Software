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

Power convention: `power_limit`, `battery.available_power()` and any ECMS-granted power
are ELECTRICAL power (accumulator output). `force_balance` turns the tightest of them
into a maximum acceleration with `max_accel_for_power`, which inverts the same chain
`power_and_energy` uses (drivetrain efficiency, then the motor efficiency map), so the
returned acceleration never needs more electrical power than the limit allows.
"""

from __future__ import annotations

import numpy as np

GRAVITY = 9.806  # m/s^2, matches the original scripts' constant
MAX_WEIGHT_TRANSFER = 0.9  # cap on the linear load-transfer fraction below (see force_balance)


def max_accel_for_power(v, p_elec_limit, mass, tire, ratio, count, efficiency, motor):
    """Largest tractive acceleration (m/s^2, before drag) whose electrical draw stays
    within `p_elec_limit` (W) at speed `v`.

    Exact inverse of the chain in `power_and_energy`:
        F -> wheel power -> / efficiency -> motor power -> / motor efficiency -> electrical
    `Motor.max_torque_for_power` inverts the efficiency-map lookup; the power is split
    equally across `count` motors.
    """
    omega = v / tire.radius * ratio
    t_per_motor = motor.max_torque_for_power(omega, p_elec_limit / count)
    if not np.isfinite(t_per_motor):
        return np.inf  # standstill or no power limit: power doesn't constrain accel
    return t_per_motor * count * ratio * efficiency / tire.radius / mass


def wheel_to_electrical_power(f_tract, v, tire, ratio, count, efficiency, motor):
    """Electrical power (W) drawn to deliver tractive force `f_tract` (N) at speed `v`.

    Same chain as `step_battery` / `power_and_energy`. Zero when not driving (no regen).
    """
    if f_tract <= 0.0 or v <= 0.0:
        return 0.0
    omega = v / tire.radius * ratio
    t_per_motor = f_tract * tire.radius / ratio / efficiency / count
    return f_tract * v / efficiency / motor.efficiency(omega, t_per_motor)


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
    motor=None,
):
    """Longitudinal brake/accel limits (m/s^2) at speed `v` and curvature `k`.

    Combines a friction-circle-derated grip limit with drag, then caps the achievable
    acceleration by whichever of (electrical power, battery power, weight-transfer-limited
    traction, motor torque curve) is most restrictive.

    `power_limit` and the battery's power are electrical. The tighter of the two is
    converted to a maximum acceleration through the motor efficiency map, so `motor`
    (the Motor object) is required.

    `battery_power_override`, when given, replaces `battery.available_power()` as the
    battery's power ceiling for this call
    """
    if motor is None:
        raise ValueError("force_balance needs the Motor object (motor=...)")

    downforce = 0.5 * air_density * cla * v * v
    fz = GRAVITY * mass + downforce
    fy = v * v * k * mass

    potential_fx = tire.get_fx(fz, fy)

    drag = 0.5 * air_density * cda * v * v
    brake = (-potential_fx - drag) / mass

    # Electrical power ceiling (accumulator output): the power cap and, if a battery is
    # modelled, its available (or ECMS-granted) power -- whichever is tighter.
    p_elec_limit = power_limit
    if battery is not None:
        if battery_power_override is not None:
            p_elec_limit = min(p_elec_limit, battery_power_override)
        else:
            p_elec_limit = min(p_elec_limit, battery.available_power())

    # Largest acceleration whose electrical draw stays within p_elec_limit, found by
    # inverting the motor efficiency map.
    power_limited_accel = max_accel_for_power(
        v, p_elec_limit, mass, tire, ratio, count, efficiency, motor
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

    accel_limit = min(power_limited_accel, traction_limited_accel, motor_limited_accel)
    accel = accel_limit - drag / mass
    return brake, accel


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
    scan_step=2.0,
    v_tol=1e-4,
    v_cap=150.0,
):
    """Steady-state speed (m/s) for curvature `k`: the lowest speed at which the net
    acceleration a(v) from `force_balance` reaches zero, found by bracketing and
    bisection.

    a(v) is positive at low speed and turns negative once grip (corners) or
    power/motor torque/drag (straights) runs out; that crossing is the fastest speed
    the car can sustain at this curvature. The search:
      1. scans upward from 0 in `scan_step` increments until a(v) <= 0, which brackets
         the first crossing (the same one the old time-march found starting from rest);
      2. bisects that bracket down to `v_tol`.
    About 30 force_balance calls per curvature. Unlike time-marching, bisection doesn't
    need a(v) to be continuous: where a(v) jumps from positive to negative (the motor's
    max-rpm cutoff, steps in the nearest-neighbour efficiency map) it converges onto
    the jump instead of oscillating around it until `max_steps` runs out.

    Returns the top of the bracket's a(v) > 0 side, so the result is always a speed the
    car can sustain. A NaN from the tire model counts as "not sustainable". If a(v)
    never turns negative below `v_cap`, returns `v_cap`.

    `v0`, `time_step`, `accel_tolerance` and `max_steps` belonged to the old
    time-marching search; they are still accepted so existing calls keep working, but
    are ignored.
    """

    def net_accel(v):
        _, a = force_balance(
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
            motor=motor,
        )
        return a

    lo = 0.0
    if not net_accel(lo) > 0.0:
        return 0.0

    # 1. Bracket: step up until a(v) is no longer positive.
    while True:
        hi = min(lo + scan_step, v_cap)
        if not net_accel(hi) > 0.0:
            break
        if hi >= v_cap:
            return v_cap
        lo = hi

    # 2. Bisect: a(lo) > 0 and a(hi) <= 0 throughout.
    while hi - lo > v_tol:
        mid = 0.5 * (lo + hi)
        if net_accel(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


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
    motor,
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
            motor=motor,
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
                motor,
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
                motor,
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
            # Electrical power this point would draw (same chain as step_battery), since
            # force_balance treats the ECMS grant as electrical power.
            power_request = wheel_to_electrical_power(
                f_tract_request, vv_base[i], tire, ratio, count, efficiency, motor
            )
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
            motor=motor,
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
    ax = (vv[i] - vv[p]) / dt
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
    dt = dx / vv
    tt = np.cumsum(dt)

    # Same kinematics the speed trace was built with (v_next^2 = v^2 + 2*a*dx), so this
    # recovers exactly the acceleration force_balance allowed at each point. The plain
    # finite difference (v_next - v) / dt comes out slightly lower, and with the
    # nearest-neighbour efficiency map a slightly lower torque can fall into a less
    # efficient cell and report more than the power limit.
    ax = (np.roll(vv, -1) ** 2 - vv**2) / (2.0 * dx)
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
    traction_mask = ft > 0

    ptraction = np.where(traction_mask, ft * vv, 0.0)
    pmotor = np.where(traction_mask, ptraction / efficiency, 0.0)
    tmotor = np.where(traction_mask, ft * tire.radius / ratio / efficiency, 0.0)
    wmotor = vv / tire.radius * ratio
    numotor = motor.efficiency(wmotor, tmotor)
    pelectric = np.where(traction_mask, pmotor / numotor, 0.0)
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
