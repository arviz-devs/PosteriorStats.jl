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
    T = typeof(log(one(promote_type(eltype(y), Distributions.partype(first(dists))))))
    sample_dims = ntuple(identity, M)
    log_like = similar(y, T, (axes(dists)..., axes(y)...))
    for (dist, ll) in zip(dists, eachslice(log_like; dims=sample_dims))
        pointwise_conditional_loglikelihoods!(ll, y, dist)
    end
    return log_like
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

# Mixtures of array-variate distributions
function pointwise_conditional_loglikelihoods!(
    log_like::AbstractArray{<:Real,N},
    y::AbstractArray{<:Real,N},
    dist::Distributions.AbstractMixtureModel{Distributions.ArrayLikeVariate{N}},
) where {N}
    log_like_component = similar(log_like)
    probs = Distributions.probs(dist)
    components = Distributions.components(dist)
    pointwise_conditional_loglikelihoods!(log_like, y, first(components))
    log_like .+= log.(first(probs))
    for (component, prob) in Iterators.drop(zip(components, probs), 1)
        pointwise_conditional_loglikelihoods!(log_like_component, y, component)
        log_like .= LogExpFunctions.logaddexp.(log_like, log.(prob) .+ log_like_component)
    end
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
