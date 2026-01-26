# Analysis: What Information Do We Actually Have?

## The Fundamental Problem

Our measurements are:
- **SCADA**: P_inj, Q_inj, V_mag at all buses (99 values)
- **PMU**: V_mag, Theta at 12 buses (24 values)

Our unknowns in measurement function:
- Line parameters: R, X (2 values)
- Loads: P_load, Q_load at ~33 buses (but these are time-varying!)

## The Issue

In the simulation:
- Loads change randomly each timestep: `fluctuation = np.random.uniform(0.9, 1.1)`
- Power flow is run with these specific loads
- Measurements are extracted

In the filter measurement function:
- We set R, X
- **But we don't know what loads to use!**
- We're using stale load values from the network object
- This causes the ~3% measurement error

## Solution Options

### Option 1: Estimate Loads Too (Joint Estimation)
State vector: [R, X, P_load_1, ..., P_load_n, Q_load_1, ..., Q_load_n]
- Pro: Theoretically correct
- Con: Too many parameters, observability issues

### Option 2: Use Base Loads (No Fluctuations)
Assume loads are constant at base values
- Pro: Simple
- Con: Doesn't match simulation (which has fluctuations)

### Option 3: Infer Loads from Measurements
Use P_inj and Q_inj to back-calculate loads
- P_load = P_gen - P_inj (at load buses, P_gen = 0)
- Pro: Uses available measurements
- Con: Requires knowing network topology

### Option 4: Fix Simulation (Remove Load Fluctuations)
Make loads constant in simulation
- Pro: Matches assumption that loads are known
- Con: Less realistic

## Recommended Solution

**Option 3** is most realistic: Infer loads from injection measurements.

At each load bus:
- P_load_i = -P_inj_i (since P_gen_i = 0 for load buses)
- Q_load_i = -Q_inj_i

This way:
1. Simulation generates random load fluctuations
2. These cause power injections
3. We measure the injections
4. We infer the loads from injections
5. We use inferred loads in measurement function

This matches reality: we measure injections and infer loads!
