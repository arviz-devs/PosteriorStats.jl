function diag_cov_from_cov_chol(F::Cholesky)
    if F.uplo == 'U'
        Uinv = inv(UpperTriangular(F.factors))
        return map(Base.Fix1(sum, abs2), eachrow(Uinv))
    else
        Linv = inv(LowerTriangular(F.factors))
        return map(Base.Fix1(sum, abs2), eachcol(Linv))
    end
end

whitened_residuals(F::Cholesky, r::AbstractVector) = F \ r
whitened_residuals(F::AbstractMatrix, r::AbstractVector) = F * r

function pointwise_normal_loglikelihood!(
    out::AbstractArray,
    mean::AbstractVector,
    obs::AbstractVector,
    F::Union{Cholesky,AbstractMatrix},
    c::AbstractVector,
)
    r = obs - mean
    g = whitened_residuals(F, r)
    @. out = (log(c) - g^2 / c - log(2π)) / 2
    return nothing
end

function elements_from_d(d::Distributions.MvNormal)
    μ = d.μ
    F = d.Σ.chol
    c = diag_cov_from_cov_chol(F)
    return μ, F, c
end

function elements_from_d(d::Distributions.MvNormalCanon)
    μ = d.μ
    iΣ = d.J
    c = diag(iΣ)
    return μ, iΣ, c
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, d::Distributions.AbstractMvNormal
)
    μ, F, c = elements_from_d(d)
    logl = similar(obs, promote_type(eltype(obs), eltype(μ), eltype(c)))
    pointwise_normal_loglikelihood!(logl, obs, μ, F, c)
    return logl
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, ds::AbstractArray{<:Distributions.AbstractMvNormal}
)
    D = length(obs)
    N = length(ds)
    logl = similar(obs, N, D)

    for i in 1:length(ds)
        @views logl[i, :] .= pointwise_conditional_loglikelihood(obs, ds[i])
    end

    return logl
end

function loo_nonfactorized(
    obs, ds::AbstractArray{<:Distributions.AbstractMvNormal}; kwargs...
)
    return loo(pointwise_conditional_loglikelihood(obs, ds); kwargs...)
end
