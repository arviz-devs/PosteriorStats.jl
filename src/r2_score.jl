"""
    r2_score(y_true::AbstractVector, y_pred::AbstractArray; kwargs...) -> (; r2, r2_std)

``R²`` for linear Bayesian regression models.[Gelman2019](@citep)

The ``R²``, or coefficient of determination, is defined as the proportion of variance in the
data that is explained by the model. For each draw, it is computed as the variance of the
predicted values divided by the variance of the predicted values plus the variance of the
residuals.

The distribution of the ``R²`` scores can then be summarized using a point estimate and a
credible interval (CI).

# Arguments

  - `y_true`: Observed data of length `noutputs`
  - `y_pred`: Predicted data with size `(ndraws[, nchains], noutputs)`

# Keywords

  - `summary::Bool=true`: Whether to return a summary or an array of ``R²`` scores. The
    summary is a named tuple with the point estimate `:r2` and the credible interval
    `:<ci_fun>`.
  - `point_estimate=$(default_point_estimate())`: The function used to compute the point
    estimate of the ``R²`` scores if `summary` is `true`. Supported options are:
    + [`Statistics.mean`](@extref)
    + [`Statistics.median`](@extref)
    + [`StatsBase.mode`](@extref)
  - `ci_fun=eti`: The function used to compute the credible interval if `summary` is
    `true`. Supported options are [`eti`](@ref) and [`hdi`](@ref).
  - `ci_prob=$(default_ci_prob())`: The probability mass to be contained in the credible
    interval.

# Examples

```jldoctest
julia> using ArviZExampleData

julia> idata = load_example_data("regression1d");

julia> y_true = idata.observed_data.y;

julia> y_pred = PermutedDimsArray(idata.posterior_predictive.y, (:draw, :chain, :y_dim_0));

julia> r2_score(y_true, y_pred)
(r2 = 0.683196996216511, eti = 0.6082075654135802 .. 0.7462891653797559)
```

# References

- [Gelman2019](@cite) Gelman et al, The Am. Stat., 73(3) (2019)
"""
function r2_score(
    y_true,
    y_pred;
    summary=true,
    point_estimate=default_point_estimate(),
    ci_fun=default_ci_fun(),
    ci_prob=default_ci_prob(float(Base.promote_eltype(y_true, y_pred))),
)
    r_squared = _r2_samples(y_true, y_pred)
    summary || return r_squared
    r2 = point_estimate(r_squared)
    ci = ci_fun(r_squared; prob=ci_prob)
    ci_name = Symbol(_fname(ci_fun))
    return (; r2, ci_name => ci)
end

function _r2_samples(y_true::AbstractVector, y_pred::AbstractArray)
    @assert ndims(y_pred) ∈ (2, 3)
    corrected = false
    dims = ndims(y_pred)

    var_y_est = dropdims(Statistics.var(y_pred; corrected, dims); dims)
    y_true_reshape = reshape(y_true, ntuple(one, ndims(y_pred) - 1)..., :)
    var_residual = dropdims(Statistics.var(y_pred .- y_true_reshape; corrected, dims); dims)

    # allocate storage for type-stability
    T = typeof(first(var_y_est) / first(var_residual))
    sample_axes = ntuple(Base.Fix1(axes, y_pred), ndims(y_pred) - 1)
    r_squared = similar(y_pred, T, sample_axes)
    r_squared .= var_y_est ./ (var_y_est .+ var_residual)
    return r_squared
end
