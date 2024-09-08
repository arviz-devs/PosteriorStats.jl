"""
    hdi(samples::AbstractVecOrMat{<:Real}; [prob, sorted]) -> IntervalSets.ClosedInterval
    hdi(samples::AbstractArray{<:Real}; [prob, sorted]) -> Array{<:IntervalSets.ClosedInterval}

Estimate the unimodal highest density interval (HDI) of `samples` for the probability `prob`.

The HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest
possible interval containing `(100*prob)`% of the probability mass.[^Hyndman1996]
This implementation uses the algorithm of [^ChenShao1999].

See also: [`hdi!`](@ref), [`eti`](@ref), [`eti!`](@ref).

# Arguments
- `samples`: an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
    present

# Keywords
- `prob`: the probability mass to be contained in the HDI. Default is
    `$(DEFAULT_INTERVAL_PROB)`.
- `sorted=false`: if `true`, the input samples are assumed to be sorted.

# Returns
- `intervals`: If `samples` is a vector or matrix, then a single
    `IntervalSets.ClosedInterval` is returned. Otherwise, an array with the shape
    `(params...,)`, is returned, containing a marginal HDI for each parameter.

!!! note
    Any default value of `prob` is arbitrary. The default value of
    `prob=$(DEFAULT_INTERVAL_PROB)` instead of a more common default like `prob=0.95` is
    chosen to reminder the user of this arbitrariness.

[^Hyndman1996]: Rob J. Hyndman (1996) Computing and Graphing Highest Density Regions,
                Amer. Stat., 50(2): 120-6.
                DOI: [10.1080/00031305.1996.10474359](https://doi.org/10.1080/00031305.1996.10474359)
                [jstor](https://doi.org/10.2307/2684423).
[^ChenShao1999]: Ming-Hui Chen & Qi-Man Shao (1999)
                 Monte Carlo Estimation of Bayesian Credible and HPD Intervals,
                 J Comput. Graph. Stat., 8:1, 69-92.
                 DOI: [10.1080/10618600.1999.10474802](https://doi.org/10.1080/00031305.1996.10474359)
                 [jstor](https://doi.org/10.2307/1390921).

# Examples

Here we calculate the 83% HDI for a normal random variable:

```jldoctest hdi; setup = :(using Random; Random.seed!(78))
julia> x = randn(2_000);

julia> hdi(x; prob=0.83)
-1.3826605224220527 .. 1.259817149822839
```

We can also calculate the HDI for a 3-dimensional array of samples:

```jldoctest hdi; setup = :(using Random; Random.seed!(67))
julia> x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :);

julia> hdi(x)
3-element Vector{IntervalSets.ClosedInterval{Float64}}:
 -1.9673956532343464 .. 1.9002831854921525
 3.0326043467656536 .. 6.900283185492152
 8.032604346765654 .. 11.900283185492153
```
"""
function hdi(x::AbstractArray{<:Real}; sorted::Bool=false, kwargs...)
    return hdi!(sorted ? x : _copymutable(x); sorted, kwargs...)
end

"""
    hdi!(samples::AbstractArray{<:Real}; [prob, sorted])

A version of [`hdi`](@ref) that partially sorts `samples` in-place while computing the HDI.

See also: [`hdi`](@ref), [`eti`](@ref), [`eti!`](@ref).
"""
function hdi!(
    x::AbstractArray{<:Real}; prob::Real=DEFAULT_INTERVAL_PROB, sorted::Bool=false
)
    0 < prob < 1 || throw(DomainError(prob, "HDI `prob` must be in the range `(0, 1)`."))
    ndims(x) > 0 ||
        throw(ArgumentError("HDI cannot be computed for a 0-dimensional array."))
    isempty(x) && throw(ArgumentError("HDI cannot be computed for an empty array."))
    return _hdi!(x, prob, sorted)
end

function _hdi!(x::AbstractVector{<:Real}, prob::Real, sorted::Bool)
    n = length(x)
    interval_length = floor(Int, prob * n) + 1
    if any(isnan, x)
        lower = upper = eltype(x)(NaN)
    elseif interval_length == n && !sorted
        lower, upper = extrema(x)
    else
        npoints_to_check = n - interval_length + 1
        sorted || _hdi_sort!(x, interval_length, npoints_to_check)
        lower_range = @view x[begin:(begin - 1 + npoints_to_check)]
        upper_range = @view x[(begin - 1 + interval_length):end]
        lower, upper = argmax(Base.splat(-), zip(lower_range, upper_range))
    end
    return IntervalSets.ClosedInterval(lower, upper)
end
_hdi!(x::AbstractMatrix{<:Real}, prob::Real, sorted::Bool) = _hdi!(vec(x), prob, sorted)
function _hdi!(x::AbstractArray{<:Real}, prob::Real, sorted::Bool)
    axes_out = _param_axes(x)
    T = eltype(x)
    interval = similar(x, IntervalSets.ClosedInterval{T}, axes_out)
    for (i, x_slice) in zip(eachindex(interval), _eachparam(x))
        interval[i] = _hdi!(x_slice, prob, sorted)
    end
    return interval
end

function _hdi_sort!(x, interval_length, npoints_to_check)
    if npoints_to_check < interval_length - 1
        ifirst = firstindex(x)
        iend = lastindex(x)
        # first sort the lower tail in-place
        sort!(x; alg=Base.Sort.PartialQuickSort(ifirst:(ifirst - 1 + npoints_to_check)))
        # now sort the upper tail, avoiding modifying the lower tail
        x_upper = @view x[(ifirst + npoints_to_check):iend]
        sort!(
            x_upper;
            alg=Base.Sort.PartialQuickSort((
                (interval_length - npoints_to_check):(iend - ifirst + 1 - npoints_to_check)
            )),
        )
    else
        sort!(x)
    end
    return x
end
