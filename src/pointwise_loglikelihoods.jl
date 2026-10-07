@deprecate pointwise_loglikelihoods(y, dists) pointwise_conditional_loglikelihoods(y, dists) false

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

# Examples

```jldoctest
julia> using Distributions

julia> dists = [
           MvNormal([ 0.8, -0.9], [1.3  0.7;  0.7 0.5])
           MvNormal([-0.9,  0.6], [2.7 -1.4; -1.4 1.5])
           MvNormal([-0.6,  0.4], [1.0  0.2;  0.2 0.2])
       ];

julia> y = [2.9, 0.4];

julia> PosteriorStats.pointwise_conditional_loglikelihoods(y, dists)
3×2 Matrix{Float64}:
 -0.471721   0.0121882
 -5.77002   -2.81539
 -8.46362   -1.5339
```

# References

- [Burkner2021](@cite) Bürkner et al. Comput. Stat. 36 (2021).
- [LOOFactorized](@cite) Vehtari et al. Leave-one-out cross-validation for non-factorized
    models
"""
function pointwise_conditional_loglikelihoods(y, dists)
    return _pointwise_conditional_loglikelihoods(y, dists)
end

function _pointwise_conditional_loglikelihoods(
    y::AbstractArray{<:Real,N},
    dists::AbstractArray{
        <:Distributions.Distribution{<:Distributions.ArrayLikeVariate{N}},M
    },
) where {M,N}
    T = _loglikelihood_eltype(first(dists), y)
    sample_dims = ntuple(identity, M)
    log_like = similar(y, T, (axes(dists)..., axes(y)...))
    for (dist, ll) in zip(dists, eachslice(log_like; dims=sample_dims))
        PartitionedDistributions.pointwise_conditional_logpdfs!!(ll, dist, y)
    end
    return log_like
end
function _pointwise_conditional_loglikelihoods(
    y::NamedTuple, dists::AbstractArray{<:Distributions.ProductNamedTupleDistribution{K}}
) where {K}
    _y = NamedTuple{K}(y)
    return map(dists) do dist
        return PartitionedDistributions.pointwise_conditional_logpdfs(dist, _y)
    end
end

function _loglikelihood_eltype(dist::Distributions.Distribution, y)
    return typeof(log(one(promote_type(eltype(y), Distributions.partype(dist)))))
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
