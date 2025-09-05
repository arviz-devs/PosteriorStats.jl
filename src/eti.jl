"""
    eti(samples::AbstractVecOrMat{<:Real}; [prob, kwargs...]) -> IntervalSets.ClosedInterval
    eti(samples::AbstractArray{<:Real}; [prob, kwargs...]) -> Array{<:IntervalSets.ClosedInterval}

Estimate the equal-tailed interval (ETI) of `samples` for the probability `prob`.

The ETI of a given probability is the credible interval wih the property that the
probability of being below the interval is equal to the probability of being above it.
That is, it is defined by the `(1-prob)/2` and `1 - (1-prob)/2`
[quantiles](@extref `Statistics.quantile`) of the samples.

See also: [`eti!`](@ref), [`hdi`](@ref), [`hdi!`](@ref).

# Arguments
- `samples`: an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
    present

# Keywords
- `prob`: the probability mass to be contained in the ETI. Default is `$(DEFAULT_CI_PROB)`.
- `kwargs`: remaining keywords are passed to [`Statistics.quantile`](@extref).

# Returns
- `intervals`: If `samples` is a vector or matrix, then a single
    [`IntervalSets.ClosedInterval`](@extref) is returned. Otherwise, an array with the shape
    `(params...,)`, is returned, containing a marginal ETI for each parameter.

!!! note
    Any default value of `prob` is arbitrary. The default value of
    `prob=$(DEFAULT_CI_PROB)` instead of a more common default like `prob=0.95` is
    chosen to reminder the user of this arbitrariness.

# Examples

Here we calculate the 83% ETI for a normal random variable:

```jldoctest eti; setup = :(using Random; Random.seed!(78))
julia> x = randn(2_000);

julia> eti(x; prob=0.83)
-1.3740585250299766 .. 1.2860771129421198
```

We can also calculate the ETI for a 3-dimensional array of samples:

```jldoctest eti; setup = :(using Random; Random.seed!(67))
julia> x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :);

julia> eti(x)
3-element Vector{IntervalSets.ClosedInterval{Float64}}:
 -1.951006825019686 .. 1.9011666217153793
 3.048993174980314 .. 6.9011666217153795
 8.048993174980314 .. 11.90116662171538
```
"""
function eti(
    x::AbstractArray{<:Real};
    prob::Real=DEFAULT_CI_PROB,
    sorted::Bool=false,
    kwargs...,
)
    return eti!(sorted ? x : _copymutable(x); prob, sorted, kwargs...)
end

"""
    eti!(samples::AbstractArray{<:Real}; [prob, kwargs...])

A version of [`eti`](@ref) that partially sorts `samples` in-place while computing the ETI.

See also: [`eti`](@ref), [`hdi`](@ref), [`hdi!`](@ref).
"""
function eti!(x::AbstractArray{<:Real}; prob::Real=DEFAULT_CI_PROB, kwargs...)
    ndims(x) > 0 ||
        throw(ArgumentError("ETI cannot be computed for a 0-dimensional array."))
    0 < prob < 1 || throw(DomainError(prob, "ETI `prob` must be in the range `(0, 1)`."))
    isempty(x) && throw(ArgumentError("ETI cannot be computed for an empty array."))
    return _eti!(x, prob; kwargs...)
end

function _eti!(x::AbstractVecOrMat{<:Real}, prob::Real; kwargs...)
    if any(isnan, x)
        T = float(promote_type(eltype(x), typeof(prob)))
        lower = upper = T(NaN)
    else
        alpha = (1 - prob) / 2
        lower, upper = Statistics.quantile!(vec(x), (alpha, 1 - alpha); kwargs...)
    end
    return IntervalSets.ClosedInterval(lower, upper)
end
function _eti!(x::AbstractArray, prob::Real; kwargs...)
    axes_out = _param_axes(x)
    T = float(promote_type(eltype(x), typeof(prob)))
    interval = similar(x, IntervalSets.ClosedInterval{T}, axes_out)
    for (i, x_slice) in zip(eachindex(interval), _eachparam(x))
        interval[i] = _eti!(x_slice, prob; kwargs...)
    end
    return interval
end
