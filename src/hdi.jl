"""
    HDIEstimationMethod

Abstract type for highest density interval (HDI) estimation methods.

Each method should implement:
- `_hdi_eltype(::HDIEstimationMethod, x) -> eltype(x)`
    Returns the type of elements returned by `_hdi!`.
- `_hdi!(::HDIEstimationMethod, x, prob, sorted) -> _hdi_eltype(method, x)`
    Computes the HDI(s) of `x` for probability mass `prob`.
"""
abstract type HDIEstimationMethod end

"""
    UnimodalHDI <: HDIEstimationMethod

Unimodal HDI estimation method using the method of [^ChenShao1999].

A single interval is returned whose bounds are selected from the sample. This estimator
assumes the true distribution is unimodal.

[^ChenShao1999]: Ming-Hui Chen & Qi-Man Shao (1999)
                 Monte Carlo Estimation of Bayesian Credible and HPD Intervals,
                 J Comput. Graph. Stat., 8:1, 69-92.
                 DOI: [10.1080/10618600.1999.10474802](https://doi.org/10.1080/00031305.1996.10474359)
                 [jstor](https://doi.org/10.2307/1390921).
"""
struct UnimodalHDI <: HDIEstimationMethod end

"""
    MultimodalHDI{M<:DensityEstimationMethod} <: HDIEstimationMethod

HDI estimation method for multimodal distributions.
"""
struct MultimodalHDI{M<:DensityEstimationMethod} <: HDIEstimationMethod
    """Density estimation method."""
    density_method::M
    """If `true`, the HDI is estimated from the sample points with the highest density."""
    sample_based::Bool
end
function MultimodalHDI(density_method=DefaultDensityEstimation(); sample_based::Bool=false)
    return MultimodalHDI(density_method, sample_based)
end

# public aliases for HDI estimation methods
const HDI_ESTIMATION_METHODS = (
    unimodal=UnimodalHDI(),
    multimodal=MultimodalHDI(DefaultDensityEstimation(); sample_based=false),
    multimodal_sample=MultimodalHDI(DefaultDensityEstimation(); sample_based=true),
)

function _hdi_method(
    method::MultimodalHDI{DefaultDensityEstimation},
    x::AbstractArray{<:Real},
    is_discrete::Union{Bool,Nothing};
    kde_func=kde_reflected,
    kwargs...,
)
    _is_discrete = (is_discrete === nothing && all(isinteger, x)) || is_discrete === true
    if _is_discrete
        return MultimodalHDI(DiscreteDensityEstimation(), false)
    else
        return MultimodalHDI(KDEstimation(kde_func; kwargs...), method.sample_based)
    end
end
function _hdi_method(
    method::HDIEstimationMethod, ::AbstractArray{<:Real}, ::Union{Bool,Nothing}; kwargs...
)
    return method
end
function _hdi_method(
    method::Symbol, x::AbstractArray{<:Real}, is_discrete::Union{Bool,Nothing}; kwargs...
)
    return _hdi_method(HDI_ESTIMATION_METHODS[method], x, is_discrete; kwargs...)
end

_hdi_eltype(::UnimodalHDI, x) = IntervalSets.ClosedInterval{eltype(x)}
_hdi_eltype(::MultimodalHDI, x) = Vector{IntervalSets.ClosedInterval{eltype(x)}}

"""
    hdi(samples::AbstractVecOrMat{<:Real}; [prob, sorted, method]) -> IntervalSets.ClosedInterval
    hdi(samples::AbstractArray{<:Real}; [prob, sorted, method]) -> Array{<:IntervalSets.ClosedInterval}

Estimate the highest density interval (HDI) of `samples` for the probability `prob`.

The HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest
possible interval containing `(100*prob)`% of the probability mass.[^Hyndman1996]
This implementation uses the algorithm of [^ChenShao1999].

See also: [`hdi!`](@ref), [`eti`](@ref), [`eti!`](@ref).

# Arguments
- `samples`: an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
    present, a marginal HDI is computed for each.

# Keywords
- `prob`: the probability mass to be contained in the HDI. Default is
    `$(DEFAULT_INTERVAL_PROB)`.
- `sorted=false`: if `true`, the input samples are assumed to be sorted.
- `method::Symbol`: the method used to estimate the HDI. Available options are:
  - `:unimodal`: Assumes a unimodal distribution (default). Bounds are entries in `samples`.
  - `:multimodal`: Fits a kernel density estimator (KDE) to `samples` and finds the HDI of
    the estimated density.
  - `:multimodal_sample`: Like `:multimodal`, but uses the KDE to find the entries in `samples`
     with the highest density and computes the HDI from those points.
- `is_discrete::Union{Bool,Nothing}=nothing`: Specify if the data is discrete. If `nothing`,
    it's automatically determined.
- `kwargs`: For continuous data and multimodal `method`s, remaining keywords are forwarded
    to [`kde_reflected`](@ref).

# Returns
- `intervals`: If `samples` is a vector or matrix, then a single
    `IntervalSets.ClosedInterval` is returned for `:unimodal` method, or a vector of
    `IntervalSets.ClosedInterval` for multimodal methods. For higher dimensional inputs,
    an array with the shape `(params...,)` is returned, containing marginal HDIs for each parameter.

!!! note
    Any default value of `prob` is arbitrary. The default value of
    `prob=$(DEFAULT_INTERVAL_PROB)` instead of a more common default like `prob=0.95` is
    chosen to remind the user of this arbitrariness.

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

```jldoctest hdi
julia> x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :);

julia> hdi(x)
3-element Vector{IntervalSets.ClosedInterval{Float64}}:
 -1.6402043796029502 .. 2.041852066407182
 3.35979562039705 .. 7.041852066407182
 8.35979562039705 .. 12.041852066407182
```

For multimodal distributions, you can use the `:multimodal` method:

```jldoctest hdi
julia> x = vcat(randn(1000), randn(1000) .+ 5);

julia> hdi(x; method=:multimodal)
2-element Vector{IntervalSets.ClosedInterval{Float64}}:
 -1.8882401079608677 .. 2.0017686164555037
 2.9839268475847436 .. 6.9235952578447275
```
"""
function hdi(x::AbstractArray{<:Real}; kwargs...)
    xcopy = _copymutable(x)
    return hdi!(xcopy; kwargs...)
end

"""
    hdi!(samples::AbstractArray{<:Real}; [prob, sorted])

A version of [`hdi`](@ref) that partially sorts `samples` in-place while computing the HDI.

See also: [`hdi`](@ref), [`eti`](@ref), [`eti!`](@ref).
"""
@constprop :aggressive function hdi!(
    x::AbstractArray{<:Real};
    prob::Real=DEFAULT_INTERVAL_PROB,
    is_discrete::Union{Bool,Nothing}=nothing,
    method::Union{Symbol,HDIEstimationMethod}=UnimodalHDI(),
    sorted::Bool=false,
    kwargs...,
)
    0 < prob < 1 || throw(ArgumentError("HDI `prob` must be in the range `(0, 1)`."))
    ndims(x) > 0 ||
        throw(ArgumentError("HDI cannot be computed for a 0-dimensional array."))
    isempty(x) && throw(ArgumentError("HDI cannot be computed for an empty array."))
    _method = _hdi_method(method, x, is_discrete; kwargs...)
    S = _hdi_eltype(_method, x)
    if ndims(x) < 3
        return S(_hdi!(_method, vec(x), prob, sorted))
    else
        axes_out = _param_axes(x)
        interval = similar(x, S, axes_out)
        for (i, x_slice) in zip(eachindex(interval), _eachparam(x))
            interval[i] = _hdi!(_method, vec(x_slice), prob, sorted)
        end
        return interval
    end
end

# unimodal HDI estimation
function _hdi!(::UnimodalHDI, x::AbstractVector, prob, sorted)
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

# multimodal HDI estimation
function _hdi!(method::MultimodalHDI, x::AbstractVector, prob, sorted)
    if !(eltype(x) <: Integer) && any(isnan, x)
        T = float(eltype(x))
        return [IntervalSets.ClosedInterval(T(NaN), T(NaN))]
    end
    if method.sample_based
        sorted || sort!(x)
        densities = density_at(method.density_method, x)
        return _hdi_from_point_densities(x, densities, prob)
    else
        bins, probs = bins_and_probs(method.density_method, x)
        step = bins isa AbstractRange ? Base.step(bins) : 1
        buffer = step isa Integer ? 0 : 0.01
        return _hdi_from_bin_probs(bins, probs, prob; step, buffer)
    end
end

function _hdi_from_bin_probs(bins, probs, prob_target; step=step(bins), buffer=0.01)
    idx_sorted = sortperm(probs; rev=true)
    prob_sum = @views cumsum(probs[idx_sorted])
    isplit = searchsortedfirst(prob_sum, prob_target)
    interval_size = isplit - firstindex(prob_sum) + 1
    idx_in_intervals = first(idx_sorted, interval_size)
    bins_in_intervals = sort(view(bins, idx_in_intervals))

    # split the bins into the individual HDI intervals
    intervals = _split_into_intervals(bins_in_intervals, step, buffer)

    return intervals
end

function _hdi_from_point_densities(points, densities, prob_target)
    interval_size = ceil(Int, prob_target * length(points))
    idx_sorted = partialsortperm(densities, 1:interval_size; rev=true)
    sort!(idx_sorted)

    # split the bin ids into the individual HDI intervals
    idx_intervals = _split_into_intervals(idx_sorted, 1, 0)
    intervals = _id_intervals_to_intervals(points, idx_intervals)

    return intervals
end

function _split_into_intervals(values, step, buffer)
    intervals = IntervalSets.ClosedInterval{eltype(values)}[]
    step_scaled = step * (1 + buffer)
    start = stop = first(values)
    for i in Iterators.drop(values, 1)
        if i > stop + step_scaled
            push!(intervals, IntervalSets.ClosedInterval(start, stop))
            start = i
        end
        stop = i
    end
    push!(intervals, IntervalSets.ClosedInterval(start, stop))
    return intervals
end

function _id_intervals_to_intervals(x, idx_intervals)
    return map(idx_intervals) do idx_interval
        istart, istop = IntervalSets.endpoints(idx_interval)
        return IntervalSets.ClosedInterval(x[istart], x[istop])
    end
end
