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

## Implementation Structure

```python
def generate_shmoo(
    tau_rise: float = 0.15,       # UI
    tau_fall: float = 0.15,       # UI
    isi_taps: list = [1.0, -0.15, -0.08, -0.04],
    sigma_RJ: float = 0.015,      # UI
    dcd_offset: float = 0.01,     # UI
    sat_k: float = 10.0,          # soft saturation sharpness
    n_mc: int = 100,              # Monte Carlo iterations
    n_timing: int = 101,          # timing grid points
    n_vref: int = 101,            # vref grid points
    samples_per_ui: int = 64,     # waveform resolution
) -> np.ndarray:
    """Returns shmoo[n_vref, n_timing] with 1=PASS, 0=FAIL"""

### Vectorization strategy
- **Vectorize**: timing×vref grid operations (101×101 = 10,201 points at once)
- **Vectorize**: waveform generation (use np.convolve for ISI)
- **Loop acceptable**: over 8 patterns (unavoidable — different bit sequences)
- **Loop acceptable**: over Monte Carlo iterations (100 iterations, each with independent RJ)
- **DO NOT**: loop over individual timing or vref points in Python

## Output Requirements

1. Generate 2D shmoo map as numpy array: shape (n_vref, n_timing), dtype=bool
2. Plot using matplotlib with:
   - X-axis: Timing offset (UI), range [-0.5, 0.5]
   - Y-axis: Vref level (mV), range [200, 800]
   - PASS = green/white, FAIL = red/black
   - Title: "Synthetic DRAM Shmoo Plot (ISI + DCD + RJ)"
3. Generate an eye diagram:
   - For each of the 8 patterns, extract the waveform from -0.5 UI to +0.5 UI
     relative to the target bit center
   - Overlay all 8 patterns in one figure with different colors and a legend
   - Add horizontal dashed line at Vref=500mV and vertical dashed line at timing=0
   - Use soft_saturate to show smooth saturation curves, no hard kinks
4. Print a summary table:
   - For each pattern: effective eye height (mV), effective eye width (UI)
   - Eye height = min(V_high_at_center) - max(V_low_at_center) across all patterns
   - Eye width = timing range where ALL patterns pass at Vref=500mV

## Code quality
- Type hints and docstrings on all functions
- Separate into modules: signal_gen.py, channel_model.py, jitter_model.py, shmoo_eval.py, plotting.py
- Include `if __name__ == "__main__"` block with default parameters
- Use numpy.random.seed(42) for reproducibility
- Print execution time for the full shmoo generation

## Parameter sweep guidance
Allow user to adjust via generate_shmoo() parameters:
- More ISI (larger tap magnitudes) → smaller eye
- More RJ (larger sigma_RJ) → fuzzier shmoo boundaries
- More DCD (larger dcd_offset) → asymmetric eye (left/right width differs)
- Softer saturation (smaller sat_k) → smoother but less realistic clipping
- Harder saturation (larger sat_k) → closer to real MOSFET rail behavior
```