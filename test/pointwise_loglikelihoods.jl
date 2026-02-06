using DimensionalData
using Distributions
using LinearAlgebra
using PDMats
using PosteriorStats
using Random
using Test

# Utility functions. To add a new distribution, overload:
# - rand_dist
# - marginal_distribution
# - conditional_distribution (optional)
# - factorized_distributions (optional)

map_recursive(f, x...) = map(f, x...)
map_recursive(f, x::NamedTuple...) = map(Base.Fix1(map_recursive, f), x...)

mapreduce_recursive(f, op, x...) = op(map(f, x...))
function mapreduce_recursive(f, op, x::NamedTuple...)
    op(map((x...,) -> mapreduce_recursive(f, op, x...), x...))
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
    ::Type{<:MixtureModel{Multivariate}},
    T::Type{<:Real},
    sz,
    config::Symbol;
    factorized::Bool=false,
)
    num_components = 5
    probs = rand(T, num_components)
    probs ./= sum(probs)
    dist_types =
        config === :uniform ? fill(MvNormal, 2) : [MvNormal, Distributions.GenericMvTDist]
    dists = [rand_dist(dist_types[mod1(i, 2)], T, sz; factorized) for i in 1:num_components]
    return MixtureModel(dists, probs)
end
if isdefined(Distributions, :JointOrderStatistics)
    function rand_dist(
        ::Type{<:Distributions.JointOrderStatistics},
        T::Type{<:Real},
        sz,
        (n, ranks);
        factorized::Bool=false,
    )
        @assert !factorized "factorized=true not supported for JointOrderStatistics"
        @assert only(sz) == length(ranks)
        dist = rand_dist(Normal, T, ())
        return Distributions.JointOrderStatistics(dist, n, ranks)
    end
end
if isdefined(Distributions, :ProductDistribution)
    function rand_dist(
        ::Type{<:Distributions.ProductDistribution{N,M}},
        T::Type{<:Real},
        sz;
        factorized::Bool=false,
    ) where {N,M}
        dist_type = (Normal, MvNormal, MatrixNormal)[M + 1]
        sz_dist = sz[(M + 1):N]
        dists = map(Iterators.product(Base.OneTo.(sz_dist)...)) do _
            rand_dist(dist_type, T, sz[1:M]; factorized)
        end
        dist = Distributions.ProductDistribution(dists)
        @assert size(dist) == sz
        return dist
    end
end
if isdefined(Distributions, :Product)
    function rand_dist(::Type{<:Distributions.Product}, T::Type{<:Real}, (D,); kwargs...)
        dists = [rand_dist(Normal, T, ()) for _ in 1:D]
        return Distributions.Product(dists)
    end
end
if isdefined(Distributions, :ProductNamedTupleDistribution)
    function rand_dist(
        ::Type{<:Distributions.ProductNamedTupleDistribution{K,V}},
        T::Type{<:Real},
        sz;
        kwargs...,
    ) where {K,V}
        sz = NamedTuple{K}(sz)
        dists = map(V.types, sz) do dist_type_k, sz_k
            rand_dist(dist_type_k, T, sz_k; kwargs...)
        end
        return Distributions.product_distribution(NamedTuple{K}(dists))
    end
end

function marginal_loglikelihoods(dist, y::AbstractArray)
    return map(CartesianIndices(y)) do i
        if i isa CartesianIndex{1}
            return marginal_loglikelihood(dist, y, LinearIndices(y)[i])
        else
            return marginal_loglikelihood(dist, y, i)
        end
    end
end
marginal_loglikelihoods(dist::UnivariateDistribution, y) = zero(loglikelihood(dist, y))
if isdefined(Distributions, :ProductNamedTupleDistribution)
    function marginal_loglikelihoods(
        dist::Distributions.ProductNamedTupleDistribution, y::NamedTuple
    )
        log_like_full = loglikelihood(dist, y)
        return map(dist.dists, y) do dist_k, y_k
            log_like_k = loglikelihood(dist_k, y_k)
            return map_recursive(marginal_loglikelihoods(dist_k, y_k)) do ll
                return ll .+ (log_like_full - log_like_k)
            end
        end
    end
end

function marginal_loglikelihood(dist::MultivariateDistribution, y::AbstractVector, i::Int)
    ic = setdiff(eachindex(y), i)
    isempty(ic) && return zero(loglikelihood(dist, y))
    return @views loglikelihood(marginal_distribution(dist, ic), y[ic])
end

function marginal_loglikelihood(dist::MatrixNormal, y::AbstractMatrix, i::CartesianIndex)
    i_vec = LinearIndices(y)[i]
    return marginal_loglikelihood(vec(dist), vec(y), i_vec)
end

if isdefined(Distributions, :ProductDistribution)
    function marginal_loglikelihood(
        dist::Distributions.ProductDistribution{1,0}, y::AbstractVector, i::Int
    )
        return loglikelihood(dist, y) - loglikelihood(dist.dists[i], y[i])
    end
    function marginal_loglikelihood(
        dist::Distributions.ProductDistribution{N,M},
        y::AbstractArray{<:Real,N},
        i::CartesianIndex{N},
    ) where {N,M}
        M == 0 && return loglikelihood(dist, y) - loglikelihood(dist.dists[i], y[i])
        i_tuple = Tuple(i)
        i_param = i_tuple[1:M]
        i_param = length(i_param) == 1 ? i_param[1] : CartesianIndex(i_param)
        i_dist = i_tuple[(M + 1):N]
        y_dist = @views y[fill(Colon(), M)..., i_dist...]
        logp = loglikelihood(dist, y)
        logp -= loglikelihood(dist.dists[i_dist...], y_dist)
        logp += marginal_loglikelihood(dist.dists[i_dist...], y_dist, i_param)
        return logp
    end
end

"""
    marginal_distribution(dist::MultivariateDistribution, i)

Compute the marginal distribution of `dist` at the indices `i`.
"""
function marginal_distribution(dist::MvNormal, i)
    μ = mean(dist)
    Σ = cov(dist)
    Σ_i = @views Symmetric(Σ[i, i])
    return MvNormal(μ[i], Σ_i)
end
function marginal_distribution(dist::MvNormalCanon, i)
    μ = mean(dist)
    Σ = cov(dist)
    J_i = @views Symmetric(inv(Symmetric(Σ[i, i])) + I * eps(eltype(μ)))
    h_i = @views J_i * μ[i]
    return MvNormalCanon(h_i, J_i)
end
function marginal_distribution(dist::MvLogNormal, i)
    return MvLogNormal(marginal_distribution(dist.normal, i))
end
function marginal_distribution(dist::Distributions.GenericMvTDist, i)
    μ_i = dist.μ[i]
    Σ_i = @views Symmetric(dist.Σ[i, i])
    return Distributions.GenericMvTDist(dist.df, μ_i, PDMat(Σ_i))
end
function marginal_distribution(dist::MixtureModel, i)
    return MixtureModel(
        marginal_distribution.(Distributions.components(dist), Ref(i)),
        Distributions.probs(dist),
    )
end
if isdefined(Distributions, :JointOrderStatistics)
    function marginal_distribution(dist::Distributions.JointOrderStatistics, i)
        return Distributions.JointOrderStatistics(dist.dist, dist.n, dist.ranks[i])
    end
end
if isdefined(Distributions, :Product)
    function marginal_distribution(dist::Distributions.Product, i)
        return Distributions.Product(dist.v[i])
    end
end

"""
    conditional_distribution(dist, y, i) -> ContinuousUnivariateDistribution

Compute a conditional univariate distribution.

Given an array-variate distribution `dist` and an array `y` in its support,
return the univariate distribution of `y[i]` given the other elements of `y`.
"""
function conditional_distribution(dist::MvNormal, y::AbstractVector, (i,))
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
function conditional_distribution(dist::MvNormalCanon, y::AbstractVector, (i,))
    return conditional_distribution(MvNormal(mean(dist), cov(dist)), y, i)
end
function conditional_distribution(dist::MatrixNormal, y::AbstractMatrix, i::CartesianIndex)
    vec_y = vec(y)
    vec_dist = vec(dist)
    vec_i = LinearIndices(y)[i]
    return conditional_distribution(vec_dist, vec_y, vec_i)
end
function conditional_distribution(dist::MvLogNormal, y::AbstractVector, (i,))
    (; μ, σ) = conditional_distribution(dist.normal, log.(y), i)
    return LogNormal(μ, σ)
end
function conditional_distribution(
    dist::Distributions.GenericMvTDist, y::AbstractVector, (i,)
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
if isdefined(Distributions, :ProductDistribution)
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
        ind_in_component = if length(ind_in_component) == 1
            ind_in_component[1]
        else
            CartesianIndex(ind_in_component)
        end
        return conditional_distribution(dist_i, y_i, ind_in_component)
    end
    function conditional_distribution(
        dist::Distributions.ProductDistribution{1,0}, ::AbstractVector, (i,)
    )
        return dist.dists[i]
    end
end
if isdefined(Distributions, :Product)
    function conditional_distribution(dist::Distributions.Product, ::AbstractVector, (i,))
        return dist.v[i]
    end
end

"""
    factorized_distributions(dist) -> Array{<:ContinuousUnivariateDistribution}

Factorize a factorizable array-variate distribution into univariate distributions.
"""
function factorized_distributions(dist::AbstractMvNormal)
    Σ = cov(dist)
    @assert isdiag(Σ)
    return Normal.(mean(dist), sqrt.(diag(Σ)))
end
function factorized_distributions(dist::MatrixNormal)
    (; M, U, V) = dist
    @assert isdiag(U) && isdiag(V)
    vec_dist = vec(dist)
    σ = reshape(sqrt.(diag(cov(vec_dist))), size(M))
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
if isdefined(Distributions, :ProductDistribution)
    factorized_distributions(dist::Distributions.ProductDistribution{<:Any,0}) = dist.dists
    function factorized_distributions(dist::Distributions.ProductDistribution)
        return stack(map(factorized_distributions, dist.dists))
    end
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

    dists = Any[
        (MvNormal, 1),
        (MvNormal, 5),
        (MvNormalCanon, 1),
        (MvNormalCanon, 5),
        (MatrixNormal, (2, 3)),
        (MatrixNormal, (3, 2)),
        (MatrixNormal, (1, 2)),
        (MatrixNormal, (2, 1)),
        (MvLogNormal, 1),
        (MvLogNormal, 5),
        (Distributions.GenericMvTDist, 1),
        (Distributions.GenericMvTDist, 5),
        (Distributions.MixtureModel{Multivariate}, (1,), :uniform),
        (Distributions.MixtureModel{Multivariate}, (5,), :uniform),
        (Distributions.MixtureModel{Multivariate}, (5,), :nonuniform),
        (Distributions.JointOrderStatistics, (5,), (5, 1:5)),
        (Distributions.JointOrderStatistics, (3,), (5, [1, 3, 5])),
        (Distributions.JointOrderStatistics, (1,), (5, [1])),
        (Distributions.JointOrderStatistics, (1,), (5, [3])),
        (Distributions.JointOrderStatistics, (1,), (5, [5])),
    ]
    if isdefined(Distributions, :ProductDistribution)
        append!(
            dists,
            [
                (Distributions.ProductDistribution{1,0}, (5,)),
                (Distributions.ProductDistribution{2,0}, (5, 3)),
                (Distributions.ProductDistribution{2,1}, (5, 3)),
                (Distributions.ProductDistribution{3,0}, (5, 3, 4)),
                (Distributions.ProductDistribution{3,1}, (5, 3, 4)),
                (Distributions.ProductDistribution{3,2}, (5, 3, 4)),
            ],
        )
    end
    if isdefined(Distributions, :Product)
        push!(dists, (Distributions.Product, (5,)))
    end

    @testset for (dist_type, sz, config...) in dists, T in (Float64, Float32)
        test_factorized = test_conditional = true
        if dist_type <: Distributions.GenericMvTDist
            # multivariate t-distribution is not factorizable
            test_factorized = false
        elseif dist_type <: Distributions.AbstractMixtureModel
            # conditional distribution for mixture models in't a type in Distributions.jl
            test_conditional = false
        elseif isdefined(Distributions, :JointOrderStatistics) &&
            dist_type <: Distributions.JointOrderStatistics
            # joint order statistics is not factorizable, and conditional distribution is not a type in Distributions.jl
            test_factorized = test_conditional = false
        end

        test_conditional && @testset "pointwise_conditional_loglikelihoods!!" begin
            @testset "consistent with conditional distributions" begin
                dist = rand_dist(dist_type, T, sz, config...)
                y = convert(Array{T}, rand(dist))
                @assert eltype(y) == T
                log_like = similar(y)
                y_inds = ndims(y) > 1 ? CartesianIndices(y) : eachindex(y)
                PosteriorStats.pointwise_conditional_loglikelihoods!!(log_like, y, dist)
                conditional_dists = conditional_distribution.(Ref(dist), Ref(y), y_inds)
                log_like_ref = loglikelihood.(conditional_dists, y)
                @test log_like ≈ log_like_ref
            end

            test_factorized && @testset "consistent with factorized distributions" begin
                dist = rand_dist(dist_type, T, sz, config...; factorized=true)
                y = convert(Array{T}, rand(dist))
                @assert eltype(y) == T
                log_like = similar(y)
                PosteriorStats.pointwise_conditional_loglikelihoods!!(log_like, y, dist)
                factorized_dists = factorized_distributions(dist)
                log_like_ref = loglikelihood.(factorized_dists, y)
                @test log_like ≈ log_like_ref
            end

            @testset "consistent with marginal distributions" begin
                dist = rand_dist(dist_type, T, sz, config...)
                y = convert(Array{T}, rand(dist))
                log_like_cond = similar(y)
                PosteriorStats.pointwise_conditional_loglikelihoods!!(
                    log_like_cond, y, dist
                )
                log_like_marginal = marginal_loglikelihoods(dist, y)
                log_like_full = loglikelihood(dist, y)
                log_like_cond_ref = log_like_full .- log_like_marginal
                rtol = T === Float32 ? 1e-3 : 1e-9
                # check that p(y_i | y_{-i}) == p(y) / p(y_{-i})
                @test log_like_cond ≈ log_like_cond_ref rtol=rtol
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
                        rand_dist(dist_type, T, sz, config...) for
                        _ in draws_dim, _ in chains_dim
                    ]
                    y_dims = map(Base.OneTo, size(first(dists)))
                elseif dim_type <: Dim
                    draws_dim = Dim{:draws}(0:(ndraws - 1))
                    chains_dim = Dim{:chains}(2:(nchains + 1))
                    dists = DimArray(
                        [
                            rand_dist(dist_type, T, sz, config...) for
                            _ in draws_dim, _ in chains_dim
                        ],
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
                log_like =
                    if dist_type <: Distributions.AbstractMixtureModel &&
                        only(config) === :nonuniform
                        PosteriorStats.pointwise_conditional_loglikelihoods(y, dists)
                    else
                        @inferred PosteriorStats.pointwise_conditional_loglikelihoods(
                            y, dists
                        )
                    end
                @test size(log_like) == (ndraws, nchains, sz...)
                @test all(isfinite, log_like)

                if dim_type <: Dim
                    @test log_like isa DimArray
                    @test dims(log_like) == (draws_dim, chains_dim, y_dims...)
                end

                log_like_ref = similar(log_like, ndraws, nchains, sz...)
                for draw in 1:ndraws, chain in 1:nchains
                    y_inds = ndims(y) > 1 ? CartesianIndices(y) : eachindex(y)
                    cols = ntuple(_ -> Colon(), ndims(y))
                    dist_k = dists[draw, chain]
                    log_like_ref[draw, chain, cols...] .=
                        loglikelihood(dist_k, y) .- marginal_loglikelihoods(dist_k, y)
                end
                rtol = T === Float32 ? 1e-3 : 1e-9
                @test log_like ≈ log_like_ref rtol=rtol
            end
        end
    end

    @testset "pointwise_conditional_loglikelihoods!! consistency with ReshapedDistribution" begin
        n = 12
        sz = (3, 4)
        dist = rand_dist(MvNormal, Float64, (n,))
        dist_reshaped = Distributions.ReshapedDistribution(dist, sz)
        y_vec = rand(dist)
        y_reshaped = reshape(y_vec, sz)
        log_like_vec = similar(y_vec)
        log_like_reshaped = similar(y_reshaped)
        PosteriorStats.pointwise_conditional_loglikelihoods!!(log_like_vec, y_vec, dist)
        PosteriorStats.pointwise_conditional_loglikelihoods!!(
            log_like_reshaped, y_reshaped, dist_reshaped
        )
        @test log_like_reshaped ≈ reshape(log_like_vec, sz)
    end

    isdefined(Distributions, :ProductNamedTupleDistribution) &&
        @testset "ProductNamedTupleDistribution" begin
            _similar(x::AbstractArray) = similar(x)
            _similar(x::Number) = oftype(x, NaN)
            @testset "pointwise_conditional_loglikelihoods!!" begin
                @testset for T in (Float64, Float32)
                    dist = rand_dist(
                        Distributions.ProductNamedTupleDistribution{
                            (:x, :y),Tuple{Normal,MvNormal}
                        },
                        T,
                        (x=(), y=(3,)),
                    )
                    @testset for dist in
                                 [dist, product_distribution((; w=dist, z=Normal()))]
                        y = rand(dist)
                        log_like = map_recursive(_similar, y)
                        log_like = PosteriorStats.pointwise_conditional_loglikelihoods!!(
                            log_like, y, dist
                        )
                        log_like_full = loglikelihood(dist, y)
                        log_like_ref = map_recursive(
                            x -> log_like_full .- x, marginal_loglikelihoods(dist, y)
                        )
                        @test mapreduce_recursive(isapprox, all, log_like, log_like_ref)
                    end
                end
            end

            @testset "pointwise_conditional_loglikelihoods" begin
                @testset for T in (Float64, Float32), sz in [(10,), (10, 3)]
                    dists = map(Iterators.product(Base.OneTo.(sz)...)) do _
                        rand_dist(
                            Distributions.ProductNamedTupleDistribution{
                                (:x, :y),Tuple{Normal,MvNormal}
                            },
                            T,
                            (x=(), y=(3,)),
                        )
                    end
                    y = rand(dists[1])
                    log_like = PosteriorStats.pointwise_conditional_loglikelihoods(y, dists)
                    @test size(log_like) == sz
                    @test eltype(log_like) == typeof(y)
                    @test all(
                        map(dists, log_like) do dist, log_like_i
                            log_like_full_i = loglikelihood(dist, y)
                            log_like_marginal_i = marginal_loglikelihoods(dist, y)
                            log_like_ref_i = map_recursive(
                                x -> log_like_full_i .- x, log_like_marginal_i
                            )
                            return mapreduce_recursive(
                                isapprox, all, log_like_i, log_like_ref_i
                            )
                        end,
                    )
                end
            end
        end
end
