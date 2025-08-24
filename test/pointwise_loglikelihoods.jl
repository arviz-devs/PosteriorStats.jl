using Distributions
using Logging: SimpleLogger, with_logger
using LinearAlgebra
using OffsetArrays
using PDMats
using PosteriorStats
using Random
using Test

@testset "pointwise_loglikelihoods" begin
    Random.seed!(1234)

    # Helper to make a well-conditioned SPD matrix
    make_spd_and_inv(D; jitter=1e-3) = begin
        A = randn(D, D)
        Σ = PDMat(A * A' + I * D * jitter)
        J = inv(Matrix(Σ))
        J = PDMat((J .+ J') ./ 2)
        return Σ, J
    end

    # 1) _pd_diag_inv equals diag(inv(Σ)) for a generic SPD matrix
    @testset "_pd_diag_inv matches diag(inv(Σ))" begin
        D = 10
        Σ, _ = make_spd_and_inv(D)
        λ = PosteriorStats._pd_diag_inv(Σ)  # should be a vector of length D
        @test length(λ) == D
        @test λ ≈ diag(inv(Matrix(Σ)))
        @test all(λ .> 0)
    end

    @testset "Diagonal Σ matches Normal per-component logpdf" begin
        D = 10
        μ = randn(D)
        σ = abs.(randn(D)) .+ 0.5
        Σ = PDMat(Diagonal(σ .^ 2))
        dist = MvNormal(μ, Σ)
        y = randn(D)

        logl = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl, y, dist)

        logl_ref = logpdf.(Normal.(μ, σ), y)
        @test logl ≈ logl_ref
    end

    @testset "Consistency: MvNormal == MvNormalCanon" begin
        D = 6
        μ = randn(D)
        Σ, J = make_spd_and_inv(D)
        h = J * μ

        dist_cov = MvNormal(μ, Σ)
        dist_can = MvNormalCanon(h, J)

        y = randn(D)
        logl_cov = similar(y)
        logl_can = similar(y)

        PosteriorStats.pointwise_loglikelihoods!(logl_cov, y, dist_cov)
        PosteriorStats.pointwise_loglikelihoods!(logl_can, y, dist_can)

        @test logl_cov ≈ logl_can
    end

    @testset "Matches conditional Gaussian per-coordinate formula" begin
        D = 10
        μ = randn(D)
        Σ, J = make_spd_and_inv(D)
        dist = MvNormal(μ, Σ)

        y = randn(D)
        logl = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl, y, dist)

        λ = diag(J)
        r = J * (y .- μ)    # = (cov_inv_y - h)
        logl_ref = @. (log(λ) - r^2 / λ - log(2π)) / 2

        @test logl ≈ logl_ref
    end

    @testset "MvNormalCanon direct formula check" begin
        D = 10
        μ = randn(D)
        Σ, J = make_spd_and_inv(D)
        h = J * μ
        dist = MvNormalCanon(h, J)

        y = randn(D)
        logl = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl, y, dist)

        λ = diag(J)
        cov_inv_y = J * y
        r = cov_inv_y .- h
        logl_ref = @. (log(λ) - r^2 / λ - log(2π)) / 2
        @test logl ≈ logl_ref
    end

    @testset "pointwise_loglikelihoods shape and values" begin
        D = 10
        y = randn(D)
        μ1, μ2 = randn(D), randn(D)
        Σ1, J1 = make_spd_and_inv(D)
        Σ2, J2 = make_spd_and_inv(D)
        dists = [MvNormal(μ1, Σ1), MvNormal(μ2, Σ2)]

        logl = PosteriorStats.pointwise_loglikelihoods(y, dists)
        @test size(logl) == (length(dists), length(y))

        # Rows should match calling the in-place kernel
        logl1 = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl1, y, dists[1])
        logl2 = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl2, y, dists[2])
        @test @view(logl[1, :]) ≈ logl1
        @test @view(logl[2, :]) ≈ logl2
    end

    @testset "pointwise_loglikelihoods with MvNormalCanon" begin
        D = 10
        y = randn(D)

        μ1, μ2 = randn(D), randn(D)
        Σ1, J1 = make_spd_and_inv(D)
        Σ2, J2 = make_spd_and_inv(D)

        h1, h2 = J1 * μ1, J2 * μ2
        dists_can = [MvNormalCanon(h1, J1), MvNormalCanon(h2, J2)]

        logl = PosteriorStats.pointwise_loglikelihoods(y, dists_can)
        @test size(logl) == (length(dists_can), length(y))

        # Compare to direct kernel
        logl1 = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl1, y, dists_can[1])
        logl2 = similar(y)
        PosteriorStats.pointwise_loglikelihoods!(logl2, y, dists_can[2])
        @test @view(logl[1, :]) ≈ logl1
        @test @view(logl[2, :]) ≈ logl2
    end

    @testset "Output eltype" begin
        D = 10
        # obs Float32, distribution Float64 -> promote to Float64
        y32 = rand(Float32, D)
        Σ, _ = make_spd_and_inv(D)
        μ = randn(D)
        dist = MvNormal(μ, Σ)
        logl = PosteriorStats.pointwise_loglikelihoods(y32, [dist, dist])
        @test eltype(logl) == Float64

        # obs Int, distribution Float32
        yI = round.(Int, 10 .* rand(D))  # integers
        μf32 = rand(Float32, D)
        Σf32 = PDMat(Diagonal((abs.(rand(Float32, D)) .+ 0.5f0) .^ 2))
        dist32 = MvNormal(μf32, Σf32)
        logl2 = PosteriorStats.pointwise_loglikelihoods(yI, [dist32, dist32])
        @test eltype(logl2) == Float32
    end
end
