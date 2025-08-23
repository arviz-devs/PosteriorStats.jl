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

function pointwise_normal_loglikelihood(
    obs::AbstractVector, means::AbstractArray, Fs::AbstractArray, cs::AbstractArray
)
    num_sample_dims = ndims(means) - 1
    D = size(means, num_sample_dims + 1)
    @assert length(obs) == D
    @assert axes(means)[1:num_sample_dims] == axes(Fs) == axes(cs)[1:num_sample_dims]

    sample_dims = ntuple(identity, num_sample_dims)
    out_axes = (map(i -> axes(means, i), sample_dims)..., Base.OneTo(D))
    out = similar(obs, Base.promote_eltype(obs, means), out_axes)

    foreach(
        (out_i, μ_i, F_i, c_i) ->
            pointwise_normal_loglikelihood!(out_i, μ_i, obs, F_i, c_i),
        eachslice(out; dims=sample_dims),
        eachslice(means; dims=sample_dims),
        Fs,
        eachslice(cs; dims=sample_dims),
    )
    return out
end

function elements_from_d(d::Distributions.MvNormal)
    μ = mean(d)
    Σ = cov(d)
    F = cholesky(Symmetric(Σ))
    c = diag_cov_from_cov_chol(F)
    return μ, F, c
end

function elements_from_d(d::Distributions.MvNormalCanon)
    μ = mean(d)
    Λ = d.J
    F = Λ
    c = diag(Λ)
    return μ, F, c
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, d::Distributions.AbstractMvNormal
)
    μ, F, c = elements_from_d(d)
    out = similar(obs, promote_type(eltype(obs), eltype(μ), eltype(c)))
    pointwise_normal_loglikelihood!(out, μ, obs, F, c)
    return out
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, ds::AbstractArray{<:Distributions.AbstractMvNormal}
)
    @assert !isempty(ds)
    sample_axes = axes(ds)

    d0 = ds[first(CartesianIndices(ds))]
    μ0, F0, c0 = elements_from_d(d0)
    D = length(μ0)
    @assert length(obs) == D

    means = Array{eltype(μ0)}(undef, (map(length, sample_axes)..., D))
    Fs = Array{Any}(undef, map(length, sample_axes)...)
    cs = Array{eltype(c0)}(undef, (map(length, sample_axes)..., D))

    for I in CartesianIndices(ds)
        μ, F, c = elements_from_d(ds[I])
        means[I, :] = μ
        Fs[I] = F
        cs[I, :] = c
    end

    return pointwise_normal_loglikelihood(obs, means, Fs, cs)
end

function loo_nonfactorized(
    obs, ds::AbstractArray{<:Distributions.AbstractMvNormal}; kwargs...
)
    return loo(pointwise_conditional_loglikelihood(obs, ds); kwargs...)
end
