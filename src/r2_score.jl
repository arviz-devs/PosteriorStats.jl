"""
    r2_score(y_true::AbstractVector, y_pred::AbstractArray) -> (; r2, r2_std)

``R²`` for linear Bayesian regression models.[Gelman2019](@citep)

# Arguments

  - `y_true`: Observed data of length `noutputs`
  - `y_pred`: Predicted data with size `(ndraws[, nchains], noutputs)`

# Examples

```jldoctest
julia> using ArviZExampleData

julia> idata = load_example_data("regression1d");

julia> y_true = idata.observed_data.y;

julia> y_pred = PermutedDimsArray(idata.posterior_predictive.y, (:draw, :chain, :y_dim_0));

julia> r2_score(y_true, y_pred) |> pairs
pairs(::NamedTuple) with 2 entries:
  :r2     => 0.683197
  :r2_std => 0.0368838
```

# References

- [Gelman2019](@cite) Gelman et al, The Am. Stat., 73(3) (2019)
"""
function r2_score(y_true, y_pred)
    r_squared = r2_samples(y_true, y_pred)
    return NamedTuple{(:r2, :r2_std)}(StatsBase.mean_and_std(r_squared; corrected=false))
end

"""
    r2_samples(y_true::AbstractVector, y_pred::AbstractArray) -> AbstractVector

``R²`` samples for Bayesian regression models. Only valid for linear models.

See also [`r2_score`](@ref).

# Arguments

  - `y_true`: Observed data of length `noutputs`
  - `y_pred`: Predicted data with size `(ndraws[, nchains], noutputs)`
"""
function r2_samples(y_true::AbstractVector, y_pred::AbstractArray)
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
