"""
    diag_cov_from_cov_chol(F::LinearAlgebra.Cholesky)

Diagonal of the covariance matrix `Σ = P⁻¹` given a Cholesky factorisation of the
precision matrix `P`.

`F` is the object returned by `cholesky(P)`.
If `F.uplo == 'U'` we have `P = U'U`; if `F.uplo == 'L'` we have `P = LL'`.
Using \(Σ = P^{-1}\) and the identities

Σ = U⁻¹(U⁻¹)' # upper-triangular case
Σ = (L⁻¹)'L⁻¹ # lower-triangular case

each marginal variance is the squared 2-norm of a row (for `U⁻¹`) or a
column (for `L⁻¹`). The function computes these norms without forming `Σ`.

Returns a vector of type `eltype(F.factors)` containing `diag(inv(parent(F)))`.
"""
function diag_cov_from_cov_chol(F::Cholesky)
    if F.uplo == 'U'
        Uinv = inv(UpperTriangular(F.factors))
        return map(Base.Fix1(sum, abs2), eachrow(Uinv))
    else
        Linv = inv(LowerTriangular(F.factors))
        return map(Base.Fix1(sum, abs2), eachcol(Linv))
    end
end

whitened_residual(F::LinearAlgebra.Cholesky, r::AbstractVector) = F \ r
whitened_residual(F::AbstractMatrix,         r::AbstractVector) = F * r

"""
    pointwise_normal_loglikelihood!(out, mean, obs, F, c)

Overwrite `out` with the **per-element** log-likelihood of a Normal model.

1. Compute the residual
   `r = obs .- mean`.

2. Whiten the residual
   `g = F \\ r` if `F isa Cholesky`, otherwise `g = F * r`.

3. Store for each index `i`

out[i] = (log(c[i]) - g[i]^2 / c[i] - log2π) / 2

`c` is a vector of scaling constants (often precisions); `log2π = log(2π)` is
taken from `Base.MathConstants`.

The function mutates and returns `out`.
"""
function pointwise_normal_loglikelihood!(
    out::AbstractArray,
    mean::AbstractVector,
    obs::AbstractVector,
    F::AbstractArray,
    c::AbstractVector)
    r = obs - mean
    g = whitened_residuals(F, r)
    @. out = (log(c) - g^2 / c - log2π) / 2
    return nothing
end

function whitened_residual

function pointwise_normal_loglikelihood(
    obs::AbstractVector,
    means::AbstractArray,
    Fs::AbstractArray,
    cs::AbstractArray
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
        eachslice(out;  dims=sample_dims),
        eachslice(means; dims=sample_dims),
        Fs,
        eachslice(cs;   dims=sample_dims),
    )
    return out
end

function pointwise_normal_loglikelihood(
    obs::AbstractVector,
    means::AbstractArray,
    Fs::AbstractArray,
    cs::AbstractArray
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
        eachslice(out;  dims=sample_dims),
        eachslice(means; dims=sample_dims),
        Fs,
        eachslice(cs;   dims=sample_dims),
    )
    return out
end

prepare_from_covariance(Σ) = begin
    F = cholesky(Symmetric(Σ))
    c = diag_cov_from_cov_chol(F)
    (F, c)
end

prepare_from_precision(Λ) = (Λ, diag(Λ))

function elements_from_d(d::Distributions.MvNormal)
    μ = mean(d); Σ = cov(d)
    F, c = prepare_from_covariance(Σ)
    return μ, F, c
end

function elements_from_d(d::Distributions.MvNormalCanon)
    μ = mean(d); Λ = d.J
    F, c = prepare_from_precision(Λ)
    return μ, F, c
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, d::Distributions.AbstractMvNormal)
    μ, F, c = elements_from_d(d)
    out = similar(obs, promote_type(eltype(obs), eltype(μ), eltype(c)))
    pointwise_normal_loglikelihood!(out, μ, obs, F, c)
    return out
end

function pointwise_conditional_loglikelihood(
    obs::AbstractVector, ds::AbstractArray{<:Distributions.AbstractMvNormal})
    @assert !isempty(ds)
    sample_axes = axes(ds)

    d0 = ds[first(CartesianIndices(ds))]
    μ0, F0, c0 = elements_from_d(d0)
    D = length(μ0);  @assert length(obs) == D

    means = Array{eltype(μ0)}(undef, (map(length, sample_axes)..., D))
    Fs    = Array{Any}(undef, map(length, sample_axes)...)
    cs    = Array{eltype(c0)}(undef, (map(length, sample_axes)..., D))

    for I in CartesianIndices(ds)
        μ, F, c = elements_from_d(ds[I])
        means[I, :] = μ
        Fs[I]       = F
        cs[I, :]    = c
    end

    return pointwise_normal_loglikelihood(obs, means, Fs, cs)
end

loo_nonfactorized(obs, ds::AbstractArray{<:Distributions.AbstractMvNormal}; kwargs...) =
    loo(pointwise_conditional_loglikelihood(obs, ds); kwargs...)
