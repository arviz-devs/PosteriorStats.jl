using DimensionalData
using Distributions
using LinearAlgebra
using PDMats
using PosteriorStats
using Random
using Test

# Utility functions. To add a new distribution, overload:
# - rand_dist
# - conditional_distribution
# - factorized_distributions (optional)

function rand_pdmat(T::Type{<:Real}, D::Int)
    A = randn(T, D, D)
    S = PDMat(A * A')
    return S
end
rand_pdmat(D::Int) = rand_pdmat(Float64, D)

"""
    rand_dist(dist_type, T, D; factorized=false) -> dist

Randomly generate a distribution.
"""
function rand_dist(::Type{<:MvNormal}, T::Type{<:Real}, D::Int; factorized::Bool=false)
    μ = randn(T, D)
    Σ = factorized ? Diagonal(rand(T, D)) : rand_pdmat(T, D)
    dist = MvNormal(μ, Σ)
    return dist
end
function rand_dist(::Type{<:MvNormalCanon}, T::Type{<:Real}, D::Int; factorized::Bool=false)
    h = randn(T, D)
    J = factorized ? Diagonal(rand(T, D)) : rand_pdmat(T, D)
    dist = MvNormalCanon(h, J)
    return dist
end

"""
    conditional_distribution(dist, y, i) -> ContinuousUnivariateDistribution

Compute a conditional univariate distribution.

Given an array-variate distribution `dist` and an array `y` in its support,
return the univariate distribution of `y[i]` given the other elements of `y`.
"""
function conditional_distribution(dist::MvNormal, y::AbstractVector, i::Int)
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    μ = mean(dist)
    Σ = cov(dist)
    ic = setdiff(eachindex(y), i)
    Σ_ic_i = @views Σ[ic, i]
    Σ_ic = @views Σ[ic, ic]
    inv_Σ_ic_Σ_ic_i = cholesky(Symmetric(Σ_ic)) \ Σ_ic_i
    Σ_cond = Σ[i, i] - inv_Σ_ic_Σ_ic_i' * Σ_ic_i  # Schur complement
    μ_cond = μ[i] + inv_Σ_ic_Σ_ic_i' * @views(y[ic] - μ[ic])
    return Normal(μ_cond, sqrt(Σ_cond))
end
function conditional_distribution(dist::MvNormalCanon, y::AbstractVector, i::Int)
    return conditional_distribution(MvNormal(mean(dist), cov(dist)), y, i)
end

"""
    factorized_distributions(dist) -> Array{<:ContinuousUnivariateDistribution}

Factorize a factorizable array-variate distribution into univariate distributions.
"""
function factorized_distributions(dist::AbstractMvNormal)
    @assert isdiag(cov(dist))
    return Normal.(mean(dist), std(dist))
end

@testset "pointwise loglikelihoods" begin
    @testset "_pd_diag_inv" begin
        @testset for T in (Float32, Float64), D in (5, 10)
            Σ = rand_pdmat(T, D)
            λ = @inferred PosteriorStats._pd_diag_inv(Σ)
            @test length(λ) == D
            @test eltype(λ) == T
            @test λ ≈ diag(inv(Σ))
            @test all(>(0), λ)
        end
    end

    @testset for dist_type in (MvNormal, MvNormalCanon),
        T in (Float32, Float64),
        D in (5, 10)

        test_factorized = true

        @testset "pointwise_loglikelihoods!" begin
            @testset "consistent with conditional distributions" begin
                dist = rand_dist(dist_type, T, D)
                y = rand(dist)
                @assert eltype(y) == T
                log_like = similar(y)
                PosteriorStats.pointwise_loglikelihoods!(log_like, y, dist)
                conditional_dists = conditional_distribution.(
                    Ref(dist), Ref(y), eachindex(y)
                )
                log_like_ref = loglikelihood.(conditional_dists, y)
                @test log_like ≈ log_like_ref
            end

            test_factorized && @testset "consistent with factorized distributions" begin
                dist = rand_dist(dist_type, T, D; factorized=true)
                y = rand(dist)
                @assert eltype(y) == T
                log_like = similar(y)
                PosteriorStats.pointwise_loglikelihoods!(log_like, y, dist)
                factorized_dists = factorized_distributions(dist)
                log_like_ref = loglikelihood.(factorized_dists, y)
                @test log_like ≈ log_like_ref
            end
        end

        @testset "pointwise_loglikelihoods" begin
            ndraws, nchains = 7, 3
            @testset for dim_type in (UnitRange, DimensionalData.Dim)
                if dim_type <: UnitRange
                    # Need to use Base.OneTo to avoid type-piracy promoting to OffsetArray if in scope
                    draws_dim = Base.OneTo(ndraws)
                    chains_dim = Base.OneTo(nchains)
                    y_dim = Base.OneTo(D)
                elseif dim_type <: Dim
                    draws_dim = Dim{:draws}(0:(ndraws - 1))
                    chains_dim = Dim{:chains}(2:(nchains + 1))
                    y_dim = Dim{:y}(-1:(D - 2))
                else
                    throw(ArgumentError("Unsupported dimension type: $dim_type"))
                end
                # NOTE: for DimensionalData, this forms a DimArray
                dists = [rand_dist(dist_type, T, D) for _ in draws_dim, _ in chains_dim]
                @assert size(dists) == (ndraws, nchains)
                y = zeros(T, y_dim)
                rand!(first(dists), y)
                log_like = @inferred PosteriorStats.pointwise_loglikelihoods(y, dists)
                @test size(log_like) == (ndraws, nchains, D)
                @test eltype(log_like) == T
                @test all(isfinite, log_like)

                if dim_type <: Dim
                    @test log_like isa DimArray
                    @test dims(log_like) == (draws_dim, chains_dim, y_dim)
                end

                log_like_ref = similar(log_like, ndraws, nchains, D)
                for draw in 1:ndraws, chain in 1:nchains
                    conditional_dists = conditional_distribution.(
                        Ref(dists[draw, chain]), Ref(y), eachindex(y)
                    )
                    log_like_ref[draw, chain, :] .= loglikelihood.(conditional_dists, y)
                end
                @test log_like ≈ log_like_ref
            end
        end
    end

    @testset "DimensionalData checks" begin
        y = DimArray(randn(3), Y(1:3))
        μ = zeros(3)
        Σ = Matrix(I, 3, 3)
        d = MvNormal(μ, Σ)
        dists = fill(d, (Dim{:draw}(1:100), Dim{:chain}(1:4)))
        pll = PosteriorStats.pointwise_loglikelihoods(y, dists)

        @test pll isa DimArray
        @test dims(pll) == (dims(dists)..., dims(y)...)
    end
end
