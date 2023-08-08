"""
    hdi(samples::AbstractArray{<:Real}; prob=$(DEFAULT_INTERVAL_PROB)) -> (; lower, upper)

Estimate the unimodal highest density interval (HDI) of `samples` for the probability `prob`.

The HDI is the minimum width Bayesian credible interval (BCI). That is, it is the smallest
possible interval containing `(100*prob)`% of the probability mass.[^Hyndman1996]

`samples` is an array of shape `(draws[, chains[, params...]])`. If multiple parameters are
present, then `lower` and `upper` are arrays with the shape `(params...,)`, computed
separately for each marginal.

This implementation uses the algorithm of [^ChenShao1999].

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
julia> using PosteriorStats

julia> x = randn(2_000);

julia> hdi(x; prob=0.83) |> pairs
pairs(::NamedTuple) with 2 entries:
  :lower => -1.38266
  :upper => 1.25982
```

We can also calculate the HDI for a 3-dimensional array of samples:

```jldoctest hdi; setup = :(using Random; Random.seed!(67))
julia> x = randn(1_000, 1, 1) .+ reshape(0:5:10, 1, 1, :);

julia> hdi(x) |> pairs
pairs(::NamedTuple) with 2 entries:
  :lower => [-1.9674, 3.0326, 8.0326]
  :upper => [1.90028, 6.90028, 11.9003]
```
"""
function hdi(x::AbstractArray{<:Real}; kwargs...)
    xcopy = similar(x)
    copyto!(xcopy, x)
    return hdi!(xcopy; kwargs...)
end

"""
    hdi!(samples::AbstractArray{<:Real}; prob=$(DEFAULT_INTERVAL_PROB)) -> (; lower, upper)

A version of [`hdi`](@ref) that sorts `samples` in-place while computing the HDI.
"""
function hdi!(x::AbstractArray{<:Real}; prob::Real=DEFAULT_INTERVAL_PROB)
    0 < prob < 1 || throw(DomainError(prob, "HDI `prob` must be in the range `(0, 1)`."))
    return _hdi!(x, prob)
end

function _hdi!(x::AbstractVector{<:Real}, prob::Real)
    isempty(x) && throw(ArgumentError("HDI cannot be computed for an empty array."))
    n = length(x)
    interval_length = floor(Int, prob * n) + 1
    if any(isnan, x) || interval_length == n
        lower, upper = extrema(x)
    else
        npoints_to_check = n - interval_length + 1
        sort!(x)
        lower_range = @views x[begin:(begin - 1 + npoints_to_check)]
        upper_range = @views x[(begin - 1 + interval_length):end]
        lower, upper = argmax(Base.splat(-), zip(lower_range, upper_range))
    end
    return (; lower, upper)
end
_hdi!(x::AbstractMatrix{<:Real}, prob::Real) = _hdi!(vec(x), prob)
function _hdi!(x::AbstractArray{<:Real}, prob::Real)
    ndims(x) > 0 ||
        throw(ArgumentError("HDI cannot be computed for a 0-dimensional array."))
    axes_out = _param_axes(x)
    lower = similar(x, axes_out)
    upper = similar(x, axes_out)
    for (i, x_slice) in zip(eachindex(lower), _eachparam(x))
        lower[i], upper[i] = _hdi!(x_slice, prob)
    end
    return (; lower, upper)
end
