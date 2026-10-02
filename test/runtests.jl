using Test, Random, LinearAlgebra, HardcoreBosonChainLoss

@testset "analytic loss and validation" begin
    gamma, t = 0.8, 0.7
    model = Model(1; hopping=0, interaction=0, loss_rates=[gamma])
    exact = dense_evolve(model, [1], t)
    @test exact.local_occupation[1] ≈ exp(-gamma*t) atol=1e-12
    @test real(tr(exact.rho)) ≈ 1 atol=1e-12
    result = ensemble(model, [1], t, 600; dt=0.01, rng=MersenneTwister(8))
    @test abs(result.mean_occupation[1]-exp(-gamma*t)) <
          4result.se_occupation[1]+0.015
    @test result.mean_jumps ≈ 1-result.mean_occupation[1] atol=1e-12
    @test_throws ArgumentError Model(0)
    @test_throws ArgumentError Model(2; loss_rates=[0.1])
    @test_throws ArgumentError Model(2; loss_rates=[0.1, -0.1])
    @test_throws ArgumentError trajectory(model, [2], 1)
    @test_throws ArgumentError trajectory(model, [1], -1)
    @test_throws ArgumentError trajectory(model, [1], 1; dt=0)
    @test_throws ArgumentError ensemble(model, [1], 1, 0)
end

@testset "MPS coherent evolution and dense Lindblad reference" begin
    initial = [1, 1, 0]
    closed = Model(3; hopping=1, interaction=0.7, loss_rates=zeros(3))
    dense = dense_evolve(closed, initial, 0.6)
    fine = trajectory(closed, initial, 0.6; dt=0.01, cutoff=0,
                      maxdim=16, rng=MersenneTwister(1))
    coarse = trajectory(closed, initial, 0.6; dt=0.05, cutoff=0,
                        maxdim=16, rng=MersenneTwister(1))
    fine_error = maximum(abs.(fine.local_occupation-dense.local_occupation))
    coarse_error = maximum(abs.(coarse.local_occupation-dense.local_occupation))
    @test fine_error < 1e-4
    @test fine_error < coarse_error
    @test isempty(fine.jumps)
    @test sum(fine.local_occupation) ≈ 2 atol=1e-10
    open = Model(3; hopping=1, interaction=0.7,
                 loss_rates=[0.0, 0.4, 0.0])
    exact = dense_evolve(open, initial, 0.3)
    sampled = ensemble(open, initial, 0.3, 400; dt=0.02, cutoff=1e-12,
                       maxdim=16, rng=MersenneTwister(3))
    @test all(abs.(sampled.mean_occupation-exact.local_occupation) .<
              4 .* sampled.se_occupation .+ 0.008)
    @test minimum(eigvals(Hermitian(exact.rho))) > -1e-12
    @test real(tr(exact.rho)) ≈ 1 atol=1e-12
    @test_throws ArgumentError dense_evolve(Model(5), ones(Int,5), 0.1)
end
