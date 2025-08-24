function _pd_diag_inv(A::PDMats.AbstractPDMat)
    T = typeof(float(oneunit(eltype(A))))
    I = Diagonal(ones(T, axes(A, 1)))
    return PDMats.invquad(A, I)
end

function pointwise_loglikelihood!(
    logl::AbstractVector{<:Real}, y::AbstractVector{<:Real}, dist::Distributions.MvNormal
)
    Σ = cov(dist)
    λ = _pd_diag_inv(Σ)
    g = Σ \ (y - mean(dist))
    return @. logl = (log(λ) - g^2 / λ - log2π) / 2
end

function pointwise_loglikelihood!(
    logl::AbstractVector{<:Real},
    y::AbstractVector{<:Real},
    dist::Distributions.MvNormalCanon,
)
    J = invcov(dist)
    λ = diag(J)
    cov_inv_y = J * y
    return @. logl = (log(λ) - (cov_inv_y - dist.h)^2 / λ - log2π) / 2
end

function pointwise_loglikelihoods(
    obs::AbstractArray{<:Real,N},
    dists::AbstractArray{
        <:Distributions.Distribution{<:Distributions.ArrayLikeVariate{N}},M
    },
) where {M,N}
    T = typeof(log(one(promote_type(eltype(obs), eltype(eltype(dists))))))
    sample_dims = ntuple(identity, M)
    logl = similar(obs, T, (axes(dists)..., axes(obs)...))
    for (dist, logl_i) in zip(dists, eachslice(logl; dims=sample_dims))
        pointwise_loglikelihood!(logl_i, obs, dist)
    end
    return logl
end
