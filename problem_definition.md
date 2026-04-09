# Problem Definition: 2D Binary Map Estimation

## Summary
Estimate a 1000×1000 binary map by querying f(x,y)→{0,1} sequentially,
using a GP surrogate for adaptive sampling.

## Input / Output
- Oracle: `f(x,y) → {0,1}`, deterministic
- Init budget: 50×200; target: 1000×1000
- Output: binary map `M̂[x,y]`

## Map Structure
- One main 1-chunk with smooth boundary
- Possible interior 0-holes
- Rare 1-outliers in background (missable)

## Surrogate
GP Regression (RBF), threshold at 0.5. Sparse GP above ~10k pts.

## Success Criteria
- Pixel accuracy ≥ 95%
- Significant holes detected
- Outliers: may be missed
- Elapsed time ≤ 10 s (50×200 grid)
- Peak memory ≤ 100 MB (50×200 grid)

## Algorithm
1. Space-filling init (Sobol/LHS)
2. Fit GP; predict map
3. Adaptive loop: boundary sampling + interior probing
4. Stop at budget or convergence

## Constraints
- GP is O(n³); sparse approx needed at scale
- Small holes risk being missed without interior probing

## Open Questions
- Min hole size threshold?
- Hard budget vs. convergence stopping?
- Boundary/interior sampling ratio?
- sklearn dense GP failed perf constraints (~160 s, ~700 MB on 50×200); Sparse GP under consideration
