Create a Python script that generates a synthetic DRAM Shmoo Plot (2D pass/fail map) with realistic ISI and jitter modeling.

## Context
In DRAM I/O characterization, a Shmoo Plot maps Pass/Fail results across a 2D grid:
- X-axis: Sampling timing offset (in UI units, e.g., -0.5 UI to +0.5 UI)
- Y-axis: Vref voltage level (in mV)
Each (timing, vref) point is tested with multiple data patterns. A point is PASS only if ALL patterns pass across ALL Monte Carlo iterations.

## Signal Model

### 1) Pulse waveform (per bit)
- Use RC exponential for rise/fall edges, NOT sine waves:
  - Rising: V(t) = V_low + (V_high - V_low) × (1 - exp(-t / τ_rise))
  - Falling: V(t) = V_high - (V_high - V_low) × (1 - exp(-t / τ_fall))
- Each bit occupies 1 UI (Unit Interval)
- Parameters: τ_rise = 0.15 UI, τ_fall = 0.15 UI, V_high = 1.0V, V_low = 0.0V
- Use 64 samples per UI for smooth waveform resolution