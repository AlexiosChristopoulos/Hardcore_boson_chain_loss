using HardcoreBosonChainLoss, Random, Printf

model = Model(6; hopping=1, interaction=0.7,
              loss_rates=[0, 0, 0.5, 0, 0, 0])
initial = [1, 0, 1, 0, 1, 0]
# Warm compilation before timing 100 independent stochastic loss records.
ensemble(model, initial, 0.1, 2; dt=0.05, rng=MersenneTwister(1))
elapsed = @elapsed result = ensemble(model, initial, 0.5, 100;
                                     dt=0.025, cutoff=1e-11, maxdim=32,
                                     rng=MersenneTwister(2))
@printf("100 six-site trajectories, 20 steps: %.3f s; total occupancy %.5f\n",
        elapsed, sum(result.mean_occupation))
