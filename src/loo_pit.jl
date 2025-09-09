"""
    loo_pit(y, y_pred, log_weights) -> Union{Real,AbstractArray}

Compute leave-one-out probability integral transform (LOO-PIT) checks.

# Arguments

  - `y`: array of observations with shape `(params...,)`
  - `y_pred`: array of posterior predictive samples with shape `(draws, chains, params...)`.
  - `log_weights`: array of normalized log LOO importance weights with shape
    `(draws, chains, params...)`.

# Returns

  - `pitvals`: LOO-PIT values with same size as `y`. If `y` is a scalar, then `pitvals` is a
    scalar.

LOO-PIT is a marginal posterior predictive check. If ``y_{-i}`` is the array ``y`` of
observations with the ``i``th observation left out, and ``y_i^*`` is a posterior prediction
of the ``i``th observation, then the LOO-PIT value for the ``i``th observation is defined as

```math
P(y_i^* \\le y_i \\mid y_{-i}) = \\int_{-\\infty}^{y_i} p(y_i^* \\mid y_{-i}) \\mathrm{d} y_i^*
```

The LOO posterior predictions and the corresponding observations should have similar
distributions, so if conditional predictive distributions are well-calibrated, then for
continuous data, all LOO-PIT values should be approximately uniformly distributed on
``[0, 1]``. [Gabry2019](@citep)

!!! warning
    For discrete data, the LOO-PIT values will typically not be uniformly distributed on
    ``[0, 1]``, and this function is not recommended.

# Examples

Calculate LOO-PIT values using as test quantity the observed values themselves.

```jldoctest loo_pit1
julia> using ArviZExampleData

julia> idata = load_example_data("centered_eight");

julia> y = idata.observed_data.obs;

julia> y_pred = PermutedDimsArray(idata.posterior_predictive.obs, (:draw, :chain, :school));

julia> log_like = PermutedDimsArray(idata.log_likelihood.obs, (:draw, :chain, :school));

julia> log_weights = loo(log_like).psis_result.log_weights;

julia> loo_pit(y, y_pred, log_weights)
┌ 8-element DimArray{Float64, 1} ┐
├────────────────────────────────┴─────────────────────────────── dims ┐
  ↓ school Categorical{String} ["Choate", …, "Mt. Hermon"] Unordered
└──────────────────────────────────────────────────────────────────────┘
 "Choate"            0.942759
 "Deerfield"         0.641057
 "Phillips Andover"  0.32729
 "Phillips Exeter"   0.581451
 "Hotchkiss"         0.288523
 "Lawrenceville"     0.393741
 "St. Paul's"        0.886175
 "Mt. Hermon"        0.638821
```

Calculate LOO-PIT values using as test quantity the square of the difference between
each observation and `mu`.

```jldoctest loo_pit1
julia> using Statistics

julia> mu = idata.posterior.mu;

julia> T = y .- median(mu);

julia> T_pred = y_pred .- mu;

julia> loo_pit(T .^ 2, T_pred .^ 2, log_weights)
┌ 8-element DimArray{Float64, 1} ┐
├────────────────────────────────┴─────────────────────────────── dims ┐
  ↓ school Categorical{String} ["Choate", …, "Mt. Hermon"] Unordered
└──────────────────────────────────────────────────────────────────────┘
 "Choate"            0.868148
 "Deerfield"         0.27421
 "Phillips Andover"  0.321719
 "Phillips Exeter"   0.193169
 "Hotchkiss"         0.370422
 "Lawrenceville"     0.195601
 "St. Paul's"        0.817408
 "Mt. Hermon"        0.326795
```

# References

- [Gabry2019](@cite) Gabry et al. J. R. Stat. Soc. Ser. A Stat. Soc. 182 (2019).
"""
function loo_pit(
    y::Union{AbstractArray,Number},
    y_pred::AbstractArray,
    log_weights::AbstractArray;
    kwargs...,
)
    sample_dims = (1, 2)
    size(y) == size(y_pred)[3:end] ||
        throw(ArgumentError("data dimensions of `y` and `y_pred` must have the size"))
    size(log_weights) == size(y_pred) ||
        throw(ArgumentError("`log_weights` and `y_pred` must have same size"))
    if all(isinteger, y) && all(isinteger, y_pred)
        @warn "All data and predictions are integer-valued. `loo_pit` will not be " *
            "uniformly distributed on [0, 1] and is not recommended."
    end
    return _loo_pit(y, y_pred, log_weights)
end

function _loo_pit(y::Number, y_pred, log_weights)
    return @views exp.(LogExpFunctions.logsumexp(log_weights[y_pred .≤ y]))
end
function _loo_pit(y::AbstractArray, y_pred, log_weights)
    sample_dims = (1, 2)
    T = typeof(exp(zero(float(eltype(log_weights)))))
    pitvals = similar(y, T)
    param_dims = _otherdims(log_weights, sample_dims)
    # work around for `eachslices` not supporting multiple dims in older Julia versions
    map!(
        pitvals,
        y,
        CartesianIndices(map(Base.Fix1(axes, y_pred), param_dims)),
        CartesianIndices(map(Base.Fix1(axes, log_weights), param_dims)),
    ) do yi, i1, i2
        yi_pred = @views y_pred[:, :, i1]
        lwi = @views log_weights[:, :, i2]
        init = T(-Inf)
        sel_iter = Iterators.flatten((
            init, (lwi_j for (lwi_j, yi_pred_j) in zip(lwi, yi_pred) if yi_pred_j ≤ yi)
        ))
        return clamp(exp(LogExpFunctions.logsumexp(sel_iter)), 0, 1)
    end
    return pitvals
end
