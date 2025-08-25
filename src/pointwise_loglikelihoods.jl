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
        pointwise_loglikelihoods!(ll, y, dist)
    end
    return log_like
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
    cov_inv_y = _pdmul(J, y)
    return @. log_like = (log(λ) - (cov_inv_y - dist.h)^2 / λ - log2π) / 2
end

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
