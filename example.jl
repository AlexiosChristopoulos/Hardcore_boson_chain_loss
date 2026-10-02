using HardcoreBosonChainLoss, Random, Printf

initial = [1, 0, 1, 0]
# Compare two nearest-neighbor interaction energies at the same local loss rate.
for v in (0.0, 1.0)
    model = Model(4; hopping=1, interaction=v,
                  loss_rates=[0.0, 0.5, 0.0, 0.0])
    sampled = ensemble(model, initial, 0.6, 400; dt=0.02,
                       rng=MersenneTwister(7))
    # Dense Lindblad occupations are Tr(nᵢρ); sampled values average loss records.
    exact = dense_evolve(model, initial, 0.6)
    @printf("V=%.1f sampled=%s dense=%s max_error=%.4f mean_jumps=%.3f\n",
            v, string(round.(sampled.mean_occupation; digits=4)),
            string(round.(exact.local_occupation; digits=4)),
            maximum(abs.(sampled.mean_occupation-exact.local_occupation)),
            sampled.mean_jumps)
end
