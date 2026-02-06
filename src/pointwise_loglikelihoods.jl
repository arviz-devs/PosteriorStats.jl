@deprecate pointwise_loglikelihoods(y, dists) pointwise_conditional_loglikelihoods(y, dists)

@doc """
    pointwise_conditional_loglikelihoods(y, dists)

Compute pointwise conditional log-likelihoods of `y` for non-factorized distributions.

A non-factorized observation model ``p(y \\mid \\theta)``, where ``y`` is an observation
in its support and ``\\theta`` are model parameters, can be factorized as
``p(y_i \\mid y_{-i}, \\theta) p(y_{-i} \\mid \\theta)``. However, completely factorizing
into individual likelihood terms can be tedious, expensive, and poorly supported by a given
PPL. This utility function computes ``\\log p(y_i \\mid y_{-i}, \\theta)`` terms for all
``i``; the resulting pointwise conditional log-likelihoods can be used e.g. in
[`loo`](@ref).

# Arguments

  - `y`: observed value in the support of the distributions in `dists`.
    If the distribution is array-variate, `y` is an array with shape `(params...,)`.
  - `dists`: array of shape `(draws[, chains])` containing parametrized
    `Distributions.Distribution`s representing a non-factorized observation
    model, one for each posterior draw. The following distributions are currently supported:
    + [`Distributions.MvNormal`](@extref) [Burkner2021](@citep)
    + [`Distributions.MvNormalCanon`](@extref)
    + [`Distributions.MatrixNormal`](@extref)
    + [`Distributions.MvLogNormal`](@extref)
    + `Distributions.GenericMvTDist` [Burkner2021; but uses a more efficient implementation](@citep)
    + [`Distributions.AbstractMixtureModel`](@extref) for mixtures of any of the above multivariate
        distributions
    + [`Distributions.JointOrderStatistics`](@extref) for joint distributions of order statistics
    + [`Distributions.ProductDistribution`](@extref) for products of univariate distributions and
        any of the above array-variate distributions
    + `Distributions.ReshapedDistribution` for any of the above distributions reshaped
    + [`Distributions.ProductNamedTupleDistribution`](@extref) for `NamedTuple`-variate distributions
        comprised of univariate distributions and any of the above distributions.

# Returns

  - `log_like`: Array with pointwise conditional log-likelihood values. If the distributions are array-variate,
      then the shape is `(draws[, chains], params...)` with real values. Otherwise, the shape is `(draws[, chains])`, 
      with values of a similar eltype to `y`.

# References

- [Burkner2021](@cite) Bürkner et al. Comput. Stat. 36 (2021).
- [LOOFactorized](@cite) Vehtari et al. Leave-one-out cross-validation for non-factorized
    models
"""
function pointwise_conditional_loglikelihoods(
    y::AbstractArray{<:Real,N},
    dists::AbstractArray{
        <:Distributions.Distribution{<:Distributions.ArrayLikeVariate{N}},M
    },
) where {M,N}
    T = _loglikelihood_eltype(first(dists), y)
    sample_dims = ntuple(identity, M)
    log_like = similar(y, T, (axes(dists)..., axes(y)...))
    cache = _build_loglikelihood_cache(dists, log_like)
    for (dist, ll) in zip(dists, eachslice(log_like; dims=sample_dims))
        pointwise_conditional_loglikelihoods!!(ll, y, dist; cache...)
    end
    return log_like
end

_build_loglikelihood_cache(dists, log_like) = ()

function _loglikelihood_eltype(dist::Distributions.Distribution, y)
    return typeof(log(one(promote_type(eltype(y), Distributions.partype(dist)))))
end

# Array-variate normal distribution
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormal,
)
    (; μ, Σ) = dist
    λ = _pd_diag_inv(Σ)
    g = Σ \ (y - μ)
    return @. log_like = (log(λ) - g^2 / λ - log2π) / 2
end
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormalCanon,
)
    (; h, J) = dist
    λ = LinearAlgebra.diag(J)
    cov_inv_y = _pdmul(J, y)
    return @. log_like = (log(λ) - (cov_inv_y - h)^2 / λ - log2π) / 2
end
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractMatrix{<:Real},
    y::AbstractMatrix{<:Real},
    dist::Distributions.MatrixNormal,
)
    (; M, U, V) = dist
    λU = _pd_diag_inv(U)
    λV = _pd_diag_inv(V)
    g = U \ (y - M) / V
    return @. log_like = (log(λU) + log(λV') - g^2 / λU / λV' - log2π) / 2
end

# Multivariate log-normal distribution
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvLogNormal,
)
    logy = log.(y)
    pointwise_conditional_loglikelihoods!!(log_like, logy, dist.normal)
    log_like .-= logy
    return log_like
end

# Array-variate t-distribution
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractVector{T},
    y::AbstractVector{<:Real},
    dist::Distributions.GenericMvTDist,
) where {T<:Real}
    (; μ, Σ) = dist
    ν = dist.df
    νi = ν + length(dist) - 1
    α = (νi + 1) / 2
    logc = SpecialFunctions.loggamma(α) - SpecialFunctions.loggamma(νi / 2) - T(logπ) / 2
    λ = _pd_diag_inv(Σ)
    d = y - μ
    g = Σ \ d
    sqmahal = LinearAlgebra.dot(d, g)
    return map!(log_like, λ, g) do λi, gi
        γ = gi^2 / λi
        β = ν + sqmahal - γ
        return logc - α * log1p(γ / β) + (log(λi) - log(β)) / 2
    end
end

# Mixtures of multivariate distributions
# NOTE: rand and loglikelihood for mixture fails on matrix-variate and higher-dimensional distributions
function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.AbstractMixtureModel{Distributions.Multivariate};
    log_like_k::AbstractVector{<:Real}=similar(log_like),
)
    fill!(log_like, -Inf)
    logp_y = first(log_like)

    K = Distributions.ncomponents(dist)
    for (k, w_k) in zip(1:K, Distributions.probs(dist))
        dist_k = Distributions.component(dist, k)
        logp_y_k = log(w_k) + Distributions.loglikelihood(dist_k, y)
        logp_y = LogExpFunctions.logaddexp(logp_y, logp_y_k)
        pointwise_conditional_loglikelihoods!!(log_like_k, y, dist_k)
        log_like .= LogExpFunctions.logaddexp.(log_like, logp_y_k .- log_like_k)
    end

    log_like .= logp_y .- log_like

    return log_like
end
function _build_loglikelihood_cache(
    ::AbstractArray{<:Distributions.AbstractMixtureModel{Distributions.Multivariate}},
    log_like::AbstractArray{<:Real},
)
    sample_dims = ntuple(identity, ndims(log_like) - 1)
    log_like_draw = first(eachslice(log_like; dims=sample_dims))
    return (; log_like_k=similar(log_like_draw))
end

# work around type instability in partype(::AbstractMixtureModel)
# https://github.com/JuliaStats/Distributions.jl/blob/3d304c26f1cffd6a5bcd24fac2318be92877f4d5/src/mixtures/mixturemodel.jl#L170C41-L170C48
function _loglikelihood_eltype(dist::Distributions.AbstractMixtureModel, y::AbstractArray)
    prob_type = eltype(Distributions.probs(dist))
    components = Distributions.components(dist)
    component_type = if isconcretetype(eltype(components))  # all components are the same type
        _loglikelihood_eltype(first(components), y)
    else
        mapreduce(Base.Fix2(_loglikelihood_eltype, y), promote_type, components)
    end
    return promote_type(component_type, typeof(log(oneunit(prob_type))))
end

if isdefined(Distributions, :JointOrderStatistics)
    function pointwise_conditional_loglikelihoods!!(
        log_like::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        dist::Distributions.JointOrderStatistics,
    )
        (; n, ranks) = dist

        if length(ranks) == 1
            log_like[begin] = Distributions.loglikelihood(dist, y)
            return log_like
        end

        udist = dist.dist
        r_ext = Iterators.flatten((0, ranks, n + 1))
        r_iter = Iterators.zip(r_ext, ranks, Iterators.drop(r_ext, 2))
        y_ext = Iterators.flatten((minimum(udist), y, maximum(udist)))
        y_iter = Iterators.zip(y_ext, y, Iterators.drop(y_ext, 2))

        for (i, (r_minus, r_cur, r_plus), (y_minus, y_cur, y_plus)) in
            zip(eachindex(log_like), r_iter, y_iter)
            udist_trunc = if r_minus == 0
                Distributions.truncated(udist; upper=y_plus)
            elseif r_plus == n + 1
                Distributions.truncated(udist; lower=y_minus)
            else
                Distributions.truncated(udist; lower=y_minus, upper=y_plus)
            end
            n_gap = r_plus - r_minus - 1
            r_in_gap = r_cur - r_minus
            dist_ostat = Distributions.OrderStatistic(udist_trunc, n_gap, r_in_gap)
            log_like[i] = Distributions.loglikelihood(dist_ostat, y_cur)
        end

        return log_like
    end
end

# Product of array-variate distributions
if isdefined(Distributions, :ProductDistribution)
    function pointwise_conditional_loglikelihoods!!(
        log_like::AbstractArray{<:Real,N},
        y::AbstractArray{<:Real,N},
        dist::Distributions.ProductDistribution{N,M},
    ) where {N,M}
        if M == 0
            log_like .= Distributions.loglikelihood.(dist.dists, y)
        else
            dims = ntuple(i -> i + M, Val(N - M))  # product dimensions
            for (y_i, log_like_i, dist_i) in
                zip(eachslice(y; dims), eachslice(log_like; dims), dist.dists)
                pointwise_conditional_loglikelihoods!!(log_like_i, y_i, dist_i)
            end
        end
        return log_like
    end
end
if isdefined(Distributions, :Product)
    function pointwise_conditional_loglikelihoods!!(
        log_like::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        dist::Distributions.Product,
    )
        log_like .= Distributions.loglikelihood.(dist.v, y)
        return log_like
    end
end
if isdefined(Distributions, :ProductNamedTupleDistribution)
    function _similar_loglikelihood(
        dist::Distributions.ProductNamedTupleDistribution, y::NamedTuple
    )
        return map(_similar_loglikelihood, dist.dists, y)
    end
    function pointwise_conditional_loglikelihoods(
        y::NamedTuple,
        dists::AbstractArray{<:Distributions.ProductNamedTupleDistribution{K,V}},
    ) where {K,V}
        _y = NamedTuple{K}(y)
        return map(dists) do dist
            log_like = _similar_loglikelihood(dist, _y)
            return pointwise_conditional_loglikelihoods!!(log_like, _y, dist)
        end
    end

    function pointwise_conditional_loglikelihoods!!(
        log_like::NamedTuple,
        y::NamedTuple,
        dist::Distributions.ProductNamedTupleDistribution,
    )
        dists = dist.dists
        _log_like = NamedTuple{keys(log_like)}(log_like)
        _y = NamedTuple{keys(y)}(y)
        return map(dists, _log_like, _y) do dist_k, log_like_k, y_k
            if dist_k isa Distributions.UnivariateDistribution
                return Distributions.loglikelihood(dist_k, y_k)
            else
                return pointwise_conditional_loglikelihoods!!(log_like_k, y_k, dist_k)
            end
        end
    end
end

function pointwise_conditional_loglikelihoods!!(
    log_like::AbstractArray{<:Real,N},
    y::AbstractArray{<:Real,N},
    dist::Distributions.ReshapedDistribution{N},
) where {N}
    y_reshape = reshape(y, size(dist.dist))
    log_like_reshape = reshape(log_like, size(dist.dist))
    pointwise_conditional_loglikelihoods!!(log_like_reshape, y_reshape, dist.dist)
    return log_like
end

# Helper functions

function _pd_diag_inv(A::PDMats.AbstractPDMat)
    T = typeof(float(oneunit(eltype(A))))
    I = LinearAlgebra.Diagonal(ones(T, axes(A, 1)))
    return PDMats.invquad(A, I)
end

# hack to aboid ambiguity with *(::AbstractPDMat, ::DimArray)
_pdmul(A::PDMats.AbstractPDMat, b::StridedVector) = A * b
function _pdmul(A::PDMats.AbstractPDMat, b::AbstractVector)
    T = Base.promote_eltype(A, b)
    y = similar(b, T)
    mul!(y, A, b)
    return y
end

function _similar_loglikelihood(dist::Distributions.UnivariateDistribution, y)
    zero(_loglikelihood_eltype(dist, y))
end
function _similar_loglikelihood(
    dist::Distributions.Distribution{<:Distributions.ArrayLikeVariate}, y
)
    similar(y, _loglikelihood_eltype(dist, y))
end
