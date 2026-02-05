@deprecate pointwise_loglikelihoods(y, dists) pointwise_conditional_loglikelihoods(y, dists)

@doc """
    pointwise_conditional_loglikelihoods(y, dists)

Compute pointwise conditional log-likelihoods of `y` for non-factorized distributions.

A non-factorized observation model ``p(y \\mid \\theta)``, where ``y`` is an array of
observations and ``\\theta`` are model parameters, can be factorized as
``p(y_i \\mid y_{-i}, \\theta) p(y_{-i} \\mid \\theta)``. However, completely factorizing
into individual likelihood terms can be tedious, expensive, and poorly supported by a given
PPL. This utility function computes ``\\log p(y_i \\mid y_{-i}, \\theta)`` terms for all
``i``; the resulting pointwise conditional log-likelihoods can be used e.g. in
[`loo`](@ref).

# Arguments

  - `y`: array of observations with shape `(params...,)`
  - `dists`: array of shape `(draws[, chains])` containing parametrized
    `Distributions.Distribution`s representing a non-factorized observation
    model, one for each posterior draw. The following distributions are currently supported:
    + [`Distributions.MvNormal`](@extref) [Burkner2021](@citep)
    + [`Distributions.MvNormalCanon`](@extref)
    + [`Distributions.MatrixNormal`](@extref)
    + [`Distributions.MvLogNormal`](@extref)
    + `Distributions.GenericMvTDist` [Burkner2021; but uses a more efficient implementation](@citep)
    + [`Distributions.ProductDistribution`](@extref) for products of univariate distributions and
        any of the above array-variate distributions
    + [`Distributions.AbstractMixtureModel`](@extref) for mixtures of the above distributions

# Returns

  - `log_like`: log-likelihood values with shape `(draws[, chains], params...)`

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
        pointwise_conditional_loglikelihoods!(ll, y, dist; cache...)
    end
    return log_like
end

_build_loglikelihood_cache(dists, log_like) = ()

# compute likelihood once to determine eltype of result
function _loglikelihood_eltype(dist::Distributions.Distribution, y::AbstractArray)
    return typeof(log(one(promote_type(eltype(y), Distributions.partype(dist)))))
end

# Array-variate normal distribution
function pointwise_conditional_loglikelihoods!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormal,
)
    (; μ, Σ) = dist
    λ = _pd_diag_inv(Σ)
    g = Σ \ (y - μ)
    return @. log_like = (log(λ) - g^2 / λ - log2π) / 2
end
function pointwise_conditional_loglikelihoods!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormalCanon,
)
    (; h, J) = dist
    λ = LinearAlgebra.diag(J)
    cov_inv_y = _pdmul(J, y)
    return @. log_like = (log(λ) - (cov_inv_y - h)^2 / λ - log2π) / 2
end
function pointwise_conditional_loglikelihoods!(
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
function pointwise_conditional_loglikelihoods!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvLogNormal,
)
    logy = log.(y)
    pointwise_conditional_loglikelihoods!(log_like, logy, dist.normal)
    log_like .-= logy
    return log_like
end

# Array-variate t-distribution
function pointwise_conditional_loglikelihoods!(
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
function pointwise_conditional_loglikelihoods!(
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
        pointwise_conditional_loglikelihoods!(log_like_k, y, dist_k)
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
    if isconcretetype(eltype(Distributions.components(dist)))  # all components are the same type
        component_type = _loglikelihood_eltype(Distributions.component(dist, 1), y)
    else
        component_type = mapfoldl(
            Base.Fix2(_loglikelihood_eltype, y) ∘ Base.Fix1(Distributions.component, dist),
            promote_type,
            1:Distributions.ncomponents(dist),
        )
    end
    return promote_type(component_type, typeof(log(oneunit(prob_type))))
end

# Product of array-variate distributions
if isdefined(Distributions, :ProductDistribution)
    function pointwise_conditional_loglikelihoods!(
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
                pointwise_conditional_loglikelihoods!(log_like_i, y_i, dist_i)
            end
        end
        return log_like
    end
end
if isdefined(Distributions, :Product)
    function pointwise_conditional_loglikelihoods!(
        log_like::AbstractVector{<:Real},
        y::AbstractVector{<:Real},
        dist::Distributions.Product,
    )
        log_like .= Distributions.loglikelihood.(dist.v, y)
        return log_like
    end
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
