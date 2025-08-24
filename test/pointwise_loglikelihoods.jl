using Logging: SimpleLogger, with_logger
using OffsetArrays
using PosteriorStats
using Test

@testset "pointwise_loglikelihoods" begin
    @testset "core functionality" begin
        Random.seed!(1234)

        # Helper to make a well-conditioned SPD matrix
        make_spd(D; jitter=1e-3) = begin
            A = randn(D, D)
            Symmetric(A * A' + I * D * jitter)
        end

        # 1) _pd_diag_inv equals diag(inv(Σ)) for a generic SPD matrix
        @testset "_pd_diag_inv matches diag(inv(Σ))" begin
            D = 6
            Σ = make_spd(D)
            pd = PDMat(Matrix(Σ))
            λ = _pd_diag_inv(pd)  # should be a vector of length D
            @test length(λ) == D
            @test λ ≈ diag(inv(Matrix(Σ))) atol = 1e-10 rtol = 1e-8
            @test all(λ .> 0)
        end
    end
end
