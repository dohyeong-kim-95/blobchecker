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

Instead, apply a smooth saturation function:

    def soft_saturate(v, v_low=0.0, v_high=1.0, k=10.0):
        mid = (v_low + v_high) / 2
        span = (v_high - v_low) / 2
        return mid + span * np.tanh(k * (v - mid) / span)

Parameters:
- k=10.0: moderate saturation. Increase k for harder limiting (k→∞ = hard clip),
  decrease k for softer (k→0 = linear, no limiting)
- This function is C∞ (infinitely differentiable): no kinks, no artifacts
- Physically motivated: models MOSFET output driver saturation

Apply soft_saturate to the ISI-affected voltage levels BEFORE reconstructing
the continuous waveform with RC exponential transitions.

### 4) Waveform reconstruction from ISI-affected voltage levels
After discrete ISI convolution + soft saturation, each UI has an effective
voltage level (which may be non-binary due to ISI, e.g., 0.85V or 0.05V).
Reconstruct the continuous waveform as follows:

1. Create time array with 64 samples per UI
2. For each UI boundary (bit transition), apply RC exponential:
   - If V_next > V_current (rising):
     V(t) = V_current + (V_next - V_current) × (1 - exp(-t_local / τ_rise))
     where t_local = time since the transition edge
   - If V_next < V_current (falling):
     V(t) = V_current + (V_next - V_current) × (1 - exp(-t / τ_fall))
   - If V_next == V_current: flat (no transition)
3. IMPORTANT: DCD shifts the START of each transition, not the target level.
   - '0→1' edge starts at: ideal_edge_time + dcd_offset (shifted right)
   - '1→0' edge starts at: ideal_edge_time - dcd_offset (shifted left)
   - Between the DCD-shifted edge and the next edge, the RC exponential
     transition occurs from V_current toward V_next
4. Result: continuous voltage waveform with ISI + DCD effects,
   sampleable at any timing offset

### 5) Data patterns
- Test ALL 8 patterns of 3-bit sequences: 000, 001, 010, 011, 100, 101, 110, 111
- The MIDDLE bit (index 1) of each 3-bit sequence is the "target bit" being sampled
- Use 16 preamble bits (all 0) + 3-bit pattern + 16 postamble bits (all 0)
- Total = 35 bits per pattern. This ensures ISI from taps[1:4] fully decays.

### 6) Jitter model
Two components applied at different stages:

**DCD (Deterministic Jitter — Duty Cycle Distortion):**
- Applied to bit boundary POSITIONS (edge timing), NOT to sampled voltage
- For each bit boundary between consecutive bits:
  - If transition is '0'→'1' (rising edge): shift edge RIGHT by +0.01 UI
  - If transition is '1'→'0' (falling edge): shift edge LEFT by -0.01 UI
  - If no transition (0→0 or 1→1): no DCD applied
- Physical meaning: HIGH pulse width deviates from 50% duty cycle
- DCD is applied during waveform reconstruction (step 4 above), not as a separate pass

**RJ (Random Jitter):**
- Gaussian distribution: RJ = σ_RJ × randn() where σ_RJ = 0.015 UI
- Applied to the SAMPLING POINT only (not to edge positions)
- For each (timing, vref) grid point, run N=100 Monte Carlo iterations
- Each iteration independently perturbs the sampling instant:
  t_sample = t_ideal + timing_offset + RJ_perturbation

**DJ_ISI (ISI-induced Deterministic Jitter):**
- Already captured by the discrete ISI tap convolution in step 2 — do NOT add separately.

### 7) Pass/Fail decision
At each (timing_offset, vref_level) point:
- For each of 8 patterns:
  - Generate 35-bit sequence (16 preamble + pattern + 16 postamble)
  - Apply discrete ISI convolution → soft_saturate → reconstruct continuous waveform with DCD
  - For each of 100 Monte Carlo iterations:
    - Sample voltage at: (target_bit_center + timing_offset + RJ_perturbation)
    - If target_bit == 1: PASS if V_sampled > Vref
    - If target_bit == 0: PASS if V_sampled < Vref
  - Pattern passes only if ALL 100 MC iterations pass
- Overall point PASS only if ALL 8 patterns pass