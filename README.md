# HardcoreBosonChainLoss.jl

**Interacting hard-core bosons in a one-dimensional chain with local
one-body loss.** The Julia solver uses ITensorMPS for pure-state quantum-jump
trajectories and an independent dense Lindblad reference for one to four
sites.

## Model and method

Each site has occupation 0 or 1. With `ℏ=1`,

```text
H = -J Σᵢ (b†ᵢ bᵢ₊₁ + b†ᵢ₊₁ bᵢ) + V Σᵢ nᵢ nᵢ₊₁,
Lᵢ = √γᵢ bᵢ.
```

The bath is a prescribed Markovian local particle-loss channel. There is no
injection, thermal bath, or particle-number conservation in the open model.
The MPS uses spin-½ states `Up` (occupied) and `Dn` (empty); hard-core boson
operators map to `S+` and `S-` without fermionic sign strings.

### Numerical quantities and physical meaning

| In the code | Physical meaning |
|---|---|
| `model.sites` | The ordered lattice sites; each local Hilbert space has empty and occupied states. |
| `hopping` (`J`) | Amplitude for a boson to exchange with a neighboring vacancy. |
| `interaction` (`V`) | Energy paid when neighboring sites are both occupied. |
| `loss_rates[i]` (`γᵢ`) | Rate for a bath to remove a boson at site `i`. |
| `occupations(state)[i]` | `⟨nᵢ⟩ = ⟨Sᶻᵢ⟩ + 1/2`, the probability that site `i` is occupied in one trajectory. |
| `jumps` | Detected loss events as `(time, site)` pairs; each event lowers the particle number by one. |
| `mean_occupation`, `se_occupation` | Ensemble estimate of `Tr(nᵢρ)` and its sampling standard error. |
| `dense_evolve(...).rho` | Density matrix evolved by the Lindblad equation without trajectory sampling. |

Here `time`, `dt`, and `γᵢ⁻¹` use the same time unit because `ℏ=1`.
The interaction changes how particles move before they are lost; it does not
itself remove particles. The expected total particle number is
`sum(mean_occupation)` and decreases only through the loss channels.

For each time interval, the code applies coherent half-step bond gates,
non-unitary local no-jump gates, then reversed coherent half-step gates.
The squared norm of the unnormalized propagated MPS gives the no-jump
probability. If a jump occurs, its site is sampled with weights
`γᵢ⟨nᵢ⟩` from the propagated state, and `S-` is
applied. Jumps are placed at step ends, so sampled observables have a
finite-step bias. Reduce `dt` to check it. Tensor truncation adds a separate
error: increase `maxdim` and reduce `cutoff` to check that effect.

## Use

Julia 1.11+ with ITensors.jl and ITensorMPS.jl is required. From this folder:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. test/runtests.jl
julia --project=. example.jl
julia --project=. benchmark.jl
```

```julia
using HardcoreBosonChainLoss, Random
model = Model(4; hopping=1, interaction=0.7,
              loss_rates=[0, 0.5, 0, 0])
initial = [1, 0, 1, 0]
one = trajectory(model, initial, 0.6; dt=0.02,
                 rng=MersenneTwister(1))
many = ensemble(model, initial, 0.6, 400; dt=0.02,
                rng=MersenneTwister(2))
println(one.jumps, " ", many.mean_occupation)
```

`trajectory` returns the final MPS, site occupations, and `(time, site)`
jump record. `ensemble` returns mean site occupations, their standard errors,
and mean jump count. `dense_evolve` returns a density matrix and site means
for at most four sites. Inputs are checked, and the random generator can be
seeded for reproducibility. The [tutorial](chain_tutorial.ipynb)
shows the analytic one-site loss law and dense comparison.

## Checks and limits

Tests compare the single-site survival law `exp(-γt)`, closed interacting
chain evolution against dense unitary/Lindblad propagation, three-site
lossy ensemble means against the dense solution, number conservation when
loss is zero, trace, positivity, and time-step refinement. Monte Carlo
checks use a standard-error tolerance rather than exact agreement.

MPS compression is effective only while trajectory entanglement remains
manageable. Finite-step jump timing and finite sample size affect results.
The dense reference scales exponentially and is intentionally capped at
four sites. An MPS trajectory is a stochastic pure state, so its density
matrix is recovered only after averaging many independent records.

Method context: [Daley, *Advances in Physics* 63, 77 (2014)](https://arxiv.org/abs/1405.6694)
and [Malo et al., *Physical Review A* 97, 053614 (2018)](https://doi.org/10.1103/PhysRevA.97.053614).

## Provenance

Reference repositories: `Open-Systems-and-Current-Fluctuations` and
`Local_Temperature`, consulted for context only. Their vectorized
density-operator implementations are not used in this trajectory solver.
ITensors.jl and ITensorMPS.jl retain their own licenses.
