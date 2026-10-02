module HardcoreBosonChainLoss

using ITensors, ITensorMPS, LinearAlgebra, Random, Statistics

export Model, trajectory, ensemble, dense_evolve, occupations

"""Open hard-core-boson chain with nearest-neighbor hopping/interaction.

Each site is empty or occupied. Local jumps are `sqrt(loss_rates[i]) * b_i`.
The finite-chain Hamiltonian is `-J Σ(b†ᵢbᵢ₊₁ + h.c.) + V Σnᵢnᵢ₊₁`.
"""
struct Model
    sites::Vector{Index}
    hopping::Float64
    interaction::Float64
    loss_rates::Vector{Float64}
end

function valid_real(name, x; positive=false, nonnegative=false)
    x isa Real && isfinite(x) && (!positive || x > 0) &&
        (!nonnegative || x >= 0) || throw(ArgumentError("invalid $name"))
    return Float64(x)
end

function Model(nsites::Integer; hopping=1.0, interaction=0.0,
               loss_rates=zeros(nsites))
    nsites >= 1 || throw(ArgumentError("nsites must be positive"))
    j = valid_real("hopping", hopping)
    v = valid_real("interaction", interaction)
    length(loss_rates) == nsites || throw(ArgumentError("loss_rates length must equal nsites"))
    rates = [valid_real("loss rate", x; nonnegative=true) for x in loss_rates]
    return Model(siteinds("S=1/2", nsites), j, v, rates)
end

function checked_initial(model, initial)
    length(initial) == length(model.sites) ||
        throw(ArgumentError("initial occupation length must equal nsites"))
    all(x -> x isa Integer && (x == 0 || x == 1), initial) ||
        throw(ArgumentError("initial occupations must be 0 or 1"))
    return MPS(model.sites, [x == 1 ? "Up" : "Dn" for x in initial])
end

# On the S=1/2 site, |Up⟩ is occupied and n = Sz + 1/2 has eigenvalues 1, 0.
number_op(site) = op("Sz", site) + 0.5*op("Id", site)

"""Return ⟨nᵢ⟩ = ⟨Szᵢ⟩ + 1/2 for each hard-core-boson site."""
function occupations(state::MPS)
    return real.(expect(state, "Sz")) .+ 0.5
end

function prepared_gates(model::Model, step)
    sites = model.sites
    bond = ITensor[]
    for i in 1:length(sites)-1
        s1, s2 = sites[i], sites[i+1]
        # S+ᵢS-ᵢ₊₁ moves one boson left; the adjoint moves it right.
        h = -model.hopping * (op("S+", s1)*op("S-", s2) +
                              op("S-", s1)*op("S+", s2)) +
            model.interaction * number_op(s1)*number_op(s2)
        push!(bond, exp(-0.5im*step*h))
    end
    # exp(-γᵢ Δt nᵢ/2) is the no-click evolution from H_eff = H - iΣγᵢnᵢ/2.
    loss = [exp(-0.5*step*rate*number_op(site))
            for (site, rate) in zip(sites, model.loss_rates)]
    jumps = [op("S-", site) for site in sites]
    return bond, loss, jumps
end

function step_state(state, bond, loss; cutoff, maxdim)
    # Symmetric splitting approximates exp(-i H_eff Δt); truncation is separate.
    state = apply(bond, state; cutoff, maxdim)
    state = apply(loss, state; cutoff, maxdim)
    return apply(reverse(bond), state; cutoff, maxdim)
end

function checked_controls(time, dt, cutoff, maxdim)
    t = valid_real("time", time; nonnegative=true)
    δ = valid_real("dt", dt; positive=true)
    c = valid_real("cutoff", cutoff; nonnegative=true)
    maxdim isa Integer && maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    return t, δ, c
end

function sample_trajectory(model, initial, time, dt, cutoff, maxdim, rng,
                           bond, loss, jumps)
    state = checked_initial(model, initial)
    nsteps = time == 0 ? 0 : ceil(Int, time/dt)
    step = nsteps == 0 ? 0.0 : time/nsteps
    record = Tuple{Float64, Int}[]
    closed = all(iszero, model.loss_rates)
    for k in 1:nsteps
        next = step_state(state, bond, loss; cutoff, maxdim)
        if closed
            # Compression can change a norm slightly even without losses.
            state = next / norm(next)
            continue
        end
        # The unnormalized state's norm squared is the probability of no loss.
        survival = clamp(real(norm(next)^2), 0.0, 1.0)
        survival > 0 || throw(ArgumentError("no-jump norm underflowed; reduce dt"))
        if rand(rng) < survival
            state = next / norm(next)
        else
            # A detected loss removes one occupied particle. Conditional
            # site weights γᵢ⟨nᵢ⟩ use the propagated state at the step end.
            # This placement introduces finite-step bias.
            next /= norm(next)
            weights = model.loss_rates .* occupations(next)
            total = sum(weights)
            total > 0 || throw(ErrorException("jump selected with zero channel weight; reduce dt"))
            threshold = rand(rng)*total
            site = min(searchsortedfirst(cumsum(weights), threshold), length(weights))
            state = apply(jumps[site], next; cutoff, maxdim)
            state /= norm(state)
            push!(record, (k*step, site))
        end
    end
    return (state=state, local_occupation=occupations(state), jumps=record)
end

"""Sample one finite-step MPS quantum-jump trajectory.

`dt` bounds the time step; the actual step divides `time` evenly. Increase
`maxdim` and reduce `cutoff`, then refine `dt`, to audit convergence.
"""
function trajectory(model::Model, initial, time; dt=0.02,
                    cutoff=1e-12, maxdim=64, rng=Random.default_rng())
    t, δ, c = checked_controls(time, dt, cutoff, maxdim)
    nsteps = t == 0 ? 0 : ceil(Int, t/δ)
    step = nsteps == 0 ? 0.0 : t/nsteps
    bond, loss, jumps = prepared_gates(model, step)
    return sample_trajectory(model, initial, t, δ, c, maxdim, rng,
                             bond, loss, jumps)
end

"""Average ⟨nᵢ⟩ and detected loss counts over independent trajectories.

The standard error measures sampling uncertainty, not gate or MPS truncation error.
"""
function ensemble(model::Model, initial, time, samples::Integer; dt=0.02,
                  cutoff=1e-12, maxdim=64, rng=Random.default_rng())
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    t, δ, c = checked_controls(time, dt, cutoff, maxdim)
    nsteps = t == 0 ? 0 : ceil(Int, t/δ)
    step = nsteps == 0 ? 0.0 : t/nsteps
    bond, loss, jumps = prepared_gates(model, step)
    values = Matrix{Float64}(undef, samples, length(model.sites))
    counts = Vector{Int}(undef, samples)
    for sample in 1:samples
        result = sample_trajectory(model, initial, t, δ, c, maxdim, rng,
                                   bond, loss, jumps)
        values[sample, :] = result.local_occupation
        counts[sample] = length(result.jumps)
    end
    return (mean_occupation=vec(mean(values; dims=1)),
            se_occupation=samples == 1 ? zeros(length(model.sites)) :
                vec(std(values; dims=1, corrected=true))/sqrt(samples),
            mean_jumps=mean(counts), samples=samples)
end

"""Evolve ρ with the full Lindblad generator for at most four sites.

The returned local occupations are Tr(nᵢρ); no trajectory sampling is used.
"""
function dense_evolve(model::Model, initial, time)
    nsites = length(model.sites)
    nsites <= 4 || throw(ArgumentError("dense reference supports at most four sites"))
    checked_initial(model, initial)
    t = valid_real("time", time; nonnegative=true)
    dim = 1 << nsites
    b = [zeros(ComplexF64, dim, dim) for _ in 1:nsites]
    for site in 1:nsites, state in 0:dim-1
        bit = 1 << (site-1)
        if state & bit != 0
            b[site][(state ⊻ bit)+1, state+1] = 1
        end
    end
    n = [op' * op for op in b]
    h = zeros(ComplexF64, dim, dim)
    for i in 1:nsites-1
        h += -model.hopping*(b[i]'*b[i+1] + b[i+1]'*b[i]) +
             model.interaction*n[i]*n[i+1]
    end
    identity = Matrix{ComplexF64}(I, dim, dim)
    # Column-stacked vec(ρ): this term is vec(-i[H,ρ]).
    generator = -im*(kron(identity,h)-kron(transpose(h),identity))
    for i in 1:nsites
        jump = sqrt(model.loss_rates[i])*b[i]
        rate_op = jump'*jump
        # Recycling LρL† and anticommutator loss form D[L]ρ.
        generator += kron(conj(jump),jump) -
            (kron(identity,rate_op)+kron(transpose(rate_op),identity))/2
    end
    state = sum(initial[i] << (i-1) for i in 1:nsites)
    rho = zeros(ComplexF64, dim, dim)
    rho[state+1,state+1] = 1
    evolved = reshape(exp(t*generator)*vec(rho), dim, dim)
    means = [real(tr(n[i]*evolved)) for i in 1:nsites]
    return (rho=evolved, local_occupation=means)
end

end
