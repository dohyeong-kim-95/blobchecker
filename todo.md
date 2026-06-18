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

### 2) Channel model (ISI generation) — DISCRETE TAPS ONLY
- Model ISI as a discrete FIR filter (tap-delay line), NOT continuous exponential tails
- Apply convolution: received_bits = np.convolve(transmitted_bits, isi_taps, mode='same')
- ISI tap coefficients:
  - taps = [1.0, -0.15, -0.08, -0.04]  (main cursor + 3 post-cursors)
- This creates pattern-dependent voltage shifts at the sampling point
- DO NOT combine with continuous exponential tails (that would double-count ISI)
- Note: With 16 preamble bits of 0, the filter boundary effects do not affect
  the target bit (position 17). Verify received[17] matches expected ISI calculation.

### 3) Soft voltage saturation (replaces hard clipping)
After ISI convolution, voltage levels may exceed [V_low, V_high] range
(e.g., undershoot to -0.15V after a 1→0 transition with ISI, or
overshoot above V_high). Do NOT use hard clipping (max/min), which
creates non-differentiable kinks and unrealistic high-frequency artifacts.