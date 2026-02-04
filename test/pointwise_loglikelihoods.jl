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

function _mvnormal(dist::MatrixNormal)
    (; M, U, V) = dist
    return MvNormal(vec(M), kron(V, U))
end

function rand_pdmat(T::Type{<:Real}, D::Int; jitter::Real=T(1e-3))
    A = randn(T, D, D)
    S = PDMat(A * A' + T(jitter) * I)
    return S
end
rand_pdmat(D::Int; kwargs...) = rand_pdmat(Float64, D; kwargs...)

"""
    rand_dist(dist_type, T, D; factorized=false) -> dist

Randomly generate a distribution.
"""
function rand_dist(::Type{<:MvNormal}, T::Type{<:Real}, (D,); factorized::Bool=false)
    μ = randn(T, D)
    Σ = factorized ? Diagonal(rand(T, D)) : rand_pdmat(T, D)
    dist = MvNormal(μ, Σ)
    return dist
end
function rand_dist(::Type{<:MvNormalCanon}, T::Type{<:Real}, (D,); factorized::Bool=false)
    h = randn(T, D)
    J = factorized ? Diagonal(rand(T, D)) : rand_pdmat(T, D)
    dist = MvNormalCanon(h, J)
    return dist
end
function rand_dist(::Type{<:MatrixNormal}, T::Type{<:Real}, (D, K); factorized::Bool=false)
    M = randn(T, D, K)
    if factorized
        U = Diagonal(rand(T, D))
        V = Diagonal(rand(T, K))
    else
        U = rand_pdmat(T, D; jitter=T(1e-1))
        V = rand_pdmat(T, K; jitter=T(1e-1))
    end
    dist = MatrixNormal(M, U, V)
    return convert(MatrixNormal{T}, dist)
end
function rand_dist(::Type{<:MvLogNormal}, T::Type{<:Real}, (D,); factorized::Bool=false)
    norm = rand_dist(MvNormal, T, D; factorized)
    return MvLogNormal(norm)
end
function rand_dist(
    ::Type{<:Distributions.GenericMvTDist}, T::Type{<:Real}, (D,); factorized::Bool=false
)
    @assert !factorized "factorized=true not supported for GenericMvTDist"
    μ = randn(T, D)
    Σ = rand_pdmat(T, D)
    ν = rand(T) * 8 + 2
    return Distributions.GenericMvTDist(ν, μ, Σ)
end
rand_dist(::Type{Normal}, T::Type{<:Real}, (); kwargs...) = Normal(randn(T), rand(T))
function rand_dist(
    ::Type{<:Distributions.MixtureModel{ArrayLikeVariate{N}}},
    T::Type{<:Real},
    sz;
    factorized::Bool=false,
) where {N}
    num_components = 5
    probs = rand(T, num_components)
    probs ./= sum(probs)
    dist_type = (Normal, MvNormal, MatrixNormal)[N + 1]
    dists = [rand_dist(dist_type, T, sz; factorized) for _ in 1:num_components]
    return MixtureModel(dists, probs)
end
function rand_dist(
    ::Type{<:Distributions.ProductDistribution{N,M}},
    T::Type{<:Real},
    sz;
    factorized::Bool=false,
) where {N,M}
    dist_type = (Normal, MvNormal, MatrixNormal)[M + 1]
    sz_dist = sz[(M + 1):N]
    dists = map(Iterators.product(Base.OneTo.(sz_dist)...)) do _
        rand_dist(dist_type, T, sz_dist; factorized)
    end
    return ProductDistribution(dists)
end
if isdefined(Distributions, :Product)
    function rand_dist(
        ::Type{<:Distributions.Product}, T::Type{<:Real}, (D,); factorized::Bool=false
    )
        dists = [rand_dist(Normal, T, (); factorized) for _ in 1:D]
        return Distributions.Product(dists)
    end
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
function conditional_distribution(dist::MatrixNormal, y::AbstractMatrix, i::CartesianIndex)
    vec_y = vec(y)
    vec_dist = _mvnormal(dist)
    vec_i = LinearIndices(y)[i]
    return conditional_distribution(vec_dist, vec_y, vec_i)
end
function conditional_distribution(dist::MvLogNormal, y::AbstractVector, i::Int)
    (; μ, σ) = conditional_distribution(dist.normal, log.(y), i)
    return LogNormal(μ, σ)
end
function conditional_distribution(
    dist::Distributions.GenericMvTDist, y::AbstractVector, i::Int
)
    # https://en.wikipedia.org/wiki/Multivariate_t-distribution#Conditional_Distribution
    (; μ, Σ) = dist
    ν = dist.df
    ic = setdiff(eachindex(y), i)
    Σ_ic_i = @views Σ[ic, i]
    Σ_ic = @views Σ[ic, ic]
    chol_Σ_ic = cholesky(Symmetric(Σ_ic))
    δ = @views y[ic] - μ[ic]
    d = dot(δ, chol_Σ_ic \ δ)
    inv_Σ_ic_Σ_ic_i = chol_Σ_ic \ Σ_ic_i
    Σ_cond = Σ[i, i] - inv_Σ_ic_Σ_ic_i' * Σ_ic_i  # Schur complement
    μ_cond = μ[i] + inv_Σ_ic_Σ_ic_i' * δ
    ν_cond = ν + length(ic)
    σ_cond = sqrt(Σ_cond * (ν + d) / ν_cond)
    return TDist(ν_cond) * σ_cond + μ_cond
end
function conditional_distribution(dist::Distributions.MixtureModel, y::AbstractArray, i)
    return MixtureModel(
        conditional_distribution.(Distributions.components(dist), Ref(y), Ref(i)),
        Distributions.probs(dist),
    )
end
function conditional_distribution(
    dist::Distributions.ProductDistribution{N,M},
    y::AbstractArray{<:Real,N},
    i::CartesianIndex{N},
) where {N,M}
    inds = Tuple(i)
    ind_in_component = inds[1:M]
    ind_component = inds[(M + 1):N]
    dist_i = dist.dists[inds[(M + 1):N]...]
    M == 0 && return dist_i
    y_i = y[fill(Colon(), M)..., ind_component...]
    return conditional_distribution(dist_i, y_i, ind_in_component)
end
function conditional_distribution(
    dist::Distributions.ProductDistribution{1,0}, ::AbstractVector, i::Int
)
    return dist.dists[i]
end
if isdefined(Distributions, :Product)
    function conditional_distribution(dist::Distributions.Product, ::AbstractVector, i::Int)
        return dist.v[i]
    end
end

"""
    factorized_distributions(dist) -> Array{<:ContinuousUnivariateDistribution}

Factorize a factorizable array-variate distribution into univariate distributions.
"""
function factorized_distributions(dist::AbstractMvNormal)
    @assert isdiag(cov(dist))
    return Normal.(mean(dist), sqrt.(var(dist)))
end
function factorized_distributions(dist::MatrixNormal)
    (; M, U, V) = dist
    @assert isdiag(U) && isdiag(V)
    vec_dist = _mvnormal(dist)
    σ = reshape(std(vec_dist), size(M))
    return Normal.(M, σ)
end
function factorized_distributions(dist::MvLogNormal)
    dnorms = factorized_distributions(dist.normal)
    return map(d -> LogNormal(d.μ, d.σ), dnorms)
end
function factorized_distributions(dist::Distributions.MixtureModel)
    return MixtureModel.(
        factorized_distributions.(Distributions.components(dist)),
        Ref(Distributions.probs(dist)),
    )
end
function factorized_distributions(dist::Distributions.ProductDistribution{N,0}) where {N}
    return dist.dists
end
function factorized_distributions(
    dist::Distributions.ProductDistribution{N,1,<:AbstractArray{D}}
) where {N,D<:Union{AbstractMvNormal,MvLogNormal}}
    return stack(map(factorized_distributions, dist.dists))
end
if isdefined(Distributions, :Product)
    factorized_distributions(dist::Distributions.Product) = dist.v
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

    @testset for dist_type in (
            MvNormal, MvNormalCanon, MatrixNormal, MvLogNormal, Distributions.GenericMvTDist
        ),
        T in (Float32, Float64),
        sz in (dist_type <: MultivariateDistribution ? (5, 10) : ((2, 3),))

        test_factorized = !(dist_type <: Distributions.GenericMvTDist)

        @testset "pointwise_conditional_loglikelihoods!" begin
            @testset "consistent with conditional distributions" begin
                dist = rand_dist(dist_type, T, sz)
                y = convert(Array{T}, rand(dist))
                @assert eltype(y) == T
                log_like = similar(y)
                y_inds = ndims(y) > 1 ? CartesianIndices(y) : eachindex(y)
                PosteriorStats.pointwise_conditional_loglikelihoods!(log_like, y, dist)
                conditional_dists = conditional_distribution.(Ref(dist), Ref(y), y_inds)
                log_like_ref = loglikelihood.(conditional_dists, y)
                @test log_like ≈ log_like_ref
            end

            test_factorized && @testset "consistent with factorized distributions" begin
                dist = rand_dist(dist_type, T, sz; factorized=true)
                y = convert(Array{T}, rand(dist))
                @assert eltype(y) == T
                log_like = similar(y)
                PosteriorStats.pointwise_conditional_loglikelihoods!(log_like, y, dist)
                factorized_dists = factorized_distributions(dist)
                log_like_ref = loglikelihood.(factorized_dists, y)
                @test log_like ≈ log_like_ref
            end
        end

        @testset "pointwise_conditional_loglikelihoods" begin
            ndraws, nchains = 7, 3
            @testset for dim_type in (UnitRange, DimensionalData.Dim)
                if dim_type <: UnitRange
                    # Need to use Base.OneTo to avoid type-piracy promoting to OffsetArray if in scope
                    draws_dim = Base.OneTo(ndraws)
                    chains_dim = Base.OneTo(nchains)
                    dists = [
                        rand_dist(dist_type, T, sz) for _ in draws_dim, _ in chains_dim
                    ]
                    y_dims = map(Base.OneTo, size(first(dists)))
                elseif dim_type <: Dim
                    draws_dim = Dim{:draws}(0:(ndraws - 1))
                    chains_dim = Dim{:chains}(2:(nchains + 1))
                    dists = DimArray(
                        [rand_dist(dist_type, T, sz) for _ in draws_dim, _ in chains_dim],
                        (draws_dim, chains_dim),
                    )
                    y_dims = ntuple(length(sz)) do i
                        return Dim{Symbol(:y, i)}(-1:(sz[i] - 2))
                    end
                else
                    throw(ArgumentError("Unsupported dimension type: $dim_type"))
                end
                @assert size(dists) == (ndraws, nchains)
                y = zeros(T, y_dims...)
                rand!(first(dists), y)
                log_like = @inferred PosteriorStats.pointwise_conditional_loglikelihoods(
                    y, dists
                )
                @test size(log_like) == (ndraws, nchains, sz...)
                @test eltype(log_like) == T
                @test all(isfinite, log_like)

                if dim_type <: Dim
                    @test log_like isa DimArray
                    @test dims(log_like) == (draws_dim, chains_dim, y_dims...)
                end

                log_like_ref = similar(log_like, ndraws, nchains, sz...)
                for draw in 1:ndraws, chain in 1:nchains
                    y_inds = ndims(y) > 1 ? CartesianIndices(y) : eachindex(y)
                    cols = ntuple(_ -> Colon(), ndims(y))
                    conditional_dists = conditional_distribution.(
                        Ref(dists[draw, chain]), Ref(y), y_inds
                    )
                    log_like_ref[draw, chain, cols...] .= loglikelihood.(
                        conditional_dists, y
                    )
                end
                @test log_like ≈ log_like_ref
            end
        end
    end
end
