using DimensionalData
using Distributions
using LinearAlgebra
using PartitionedDistributions
using PosteriorStats
using Random
using Test

function rand_pdmat(T::Type{<:Real}, D::Int; jitter::Real=T(1e-3))
    A = randn(T, D, D)
    return Matrix(Symmetric(A * A' + T(jitter) * I))
end

"""
    rand_dist(dist_type, T, sz[, config]) -> dist

Randomly generate a distribution.
"""
function rand_dist(::Type{<:MvNormal}, T::Type{<:Real}, (D,))
    return MvNormal(randn(T, D), rand_pdmat(T, D))
end
rand_dist(::Type{Normal}, T::Type{<:Real}, ()) = Normal(randn(T), rand(T))
function rand_dist(::Type{<:MatrixNormal}, T::Type{<:Real}, (D, K))
    M = randn(T, D, K)
    U = rand_pdmat(T, D; jitter=T(1e-1))
    V = rand_pdmat(T, K; jitter=T(1e-1))
    return convert(MatrixNormal{T}, MatrixNormal(M, U, V))
end
function rand_dist(::Type{<:Distributions.GenericMvTDist}, T::Type{<:Real}, (D,))
    ν = rand(T) * 8 + 2
    return MvTDist(ν, randn(T, D), rand_pdmat(T, D))
end
function rand_dist(
    ::Type{<:MixtureModel{Multivariate}}, T::Type{<:Real}, sz, config::Symbol
)
    num_components = 5
    probs = rand(T, num_components)
    probs ./= sum(probs)
    dist_types =
        config === :uniform ? fill(MvNormal, 2) : [MvNormal, Distributions.GenericMvTDist]
    dists = [rand_dist(dist_types[mod1(i, 2)], T, sz) for i in 1:num_components]
    return MixtureModel(dists, probs)
end
function rand_dist(
    ::Type{<:Distributions.ProductDistribution{N,M}}, T::Type{<:Real}, sz
) where {N,M}
    dist_type = (Normal, MvNormal, MatrixNormal)[M + 1]
    dists = map(CartesianIndices(sz[(M + 1):N])) do _
        return rand_dist(dist_type, T, sz[1:M])
    end
    dist = Distributions.ProductDistribution(dists)
    @assert size(dist) == sz
    return dist
end

@testset "pointwise loglikelihoods" begin
    @testset "array-variate" begin
        dist_configs = [
            (MvNormal, (1,)),
            (MvNormal, (5,)),
            (MatrixNormal, (2, 3)),
            (Distributions.GenericMvTDist, (5,)),
            (MixtureModel{Multivariate}, (5,), :uniform),
            (MixtureModel{Multivariate}, (5,), :nonuniform),
            (Distributions.ProductDistribution{3,2}, (2, 3, 4)),
        ]
        ndraws, nchains = 7, 3
        @testset for (dist_type, sz, config...) in dist_configs,
            T in (Float64, Float32),
            dim_type in (UnitRange, Dim)

            if dim_type <: UnitRange
                # Need to use Base.OneTo to avoid type-piracy promoting to OffsetArray if in scope
                draws_dim = Base.OneTo(ndraws)
                chains_dim = Base.OneTo(nchains)
                dists = [
                    rand_dist(dist_type, T, sz, config...) for
                    _ in draws_dim, _ in chains_dim
                ]
                y_dims = map(Base.OneTo, sz)
            else
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
            end
            y = zeros(T, y_dims...)
            rand!(first(dists), y)

            log_like = if dist_type <: MixtureModel && only(config) === :nonuniform
                PosteriorStats._pointwise_conditional_loglikelihoods(y, dists)
            else
                @inferred PosteriorStats._pointwise_conditional_loglikelihoods(y, dists)
            end
            @test size(log_like) == (ndraws, nchains, sz...)
            @test all(isfinite, log_like)
            if dim_type <: Dim
                @test log_like isa DimArray
                @test dims(log_like) == (draws_dim, chains_dim, y_dims...)
            end

            # the migration recipe in the deprecation warning gives the same result
            log_like_ref = stack(
                dist -> pointwise_conditional_logpdfs(dist, y), dists; dims=1
            )
            log_like_ref = reshape(log_like_ref, size(dists)..., size(y)...)
            @test log_like ≈ log_like_ref
        end
    end

    @testset "NamedTuple-variate" begin
        @testset for T in (Float64, Float32), sz in [(10,), (10, 3)]
            dists = map(CartesianIndices(sz)) do _
                return product_distribution((
                    a=rand_dist(Normal, T, ()), b=rand_dist(MvNormal, T, (3,))
                ))
            end
            y = rand(first(dists))
            log_like = @inferred PosteriorStats._pointwise_conditional_loglikelihoods(
                y, dists
            )
            @test size(log_like) == sz
            @test eltype(log_like) == typeof(y)
            @test log_like == map(dist -> pointwise_conditional_logpdfs(dist, y), dists)

            # keys of y are reordered to match the distributions
            y_reordered = (; b=y.b, a=y.a)
            @test PosteriorStats._pointwise_conditional_loglikelihoods(
                y_reordered, dists
            ) == log_like
        end
    end
end
