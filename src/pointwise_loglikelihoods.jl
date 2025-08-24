function _pd_diag_inv(A::PDMats.AbstractPDMat)
    T = typeof(float(oneunit(eltype(A))))
    I = LinearAlgebra.Diagonal(ones(T, axes(A, 1)))
    return PDMats.invquad(A, I)
end

function pointwise_loglikelihoods!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormal,
)
    Σ = Distributions.cov(dist)
    λ = _pd_diag_inv(Σ)
    g = Σ \ (y - Distributions.mean(dist))
    return @. log_like = (log(λ) - g^2 / λ - log2π) / 2
end

function pointwise_loglikelihoods!(
    log_like::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormalCanon,
)
    J = Distributions.invcov(dist)
    λ = LinearAlgebra.diag(J)
    cov_inv_y = J * y
    return @. log_like = (log(λ) - (cov_inv_y - dist.h)^2 / λ - log2π) / 2
end

function pointwise_loglikelihoods(
    y::AbstractArray{<:Real,N},
    dists::AbstractArray{
        <:Distributions.Distribution{<:Distributions.ArrayLikeVariate{N}},M
    },
) where {M,N}
    T = typeof(log(one(promote_type(eltype(y), eltype(eltype(dists))))))
    sample_dims = ntuple(identity, M)
    log_like = similar(y, T, (axes(dists)..., axes(y)...))
    for (dist, ll) in zip(dists, eachslice(log_like; dims=sample_dims))
        pointwise_loglikelihood!(ll, y, dist)
    end
    return log_like
end
