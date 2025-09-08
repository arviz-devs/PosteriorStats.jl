"""
    FixKeywords(f; kwargs...)
    FixKeywords(f, kwargs)

A type representing the function `(xs...) -> f(xs...; kwargs...)`.
"""
struct FixKeywords{F,KW}
    f::F
    kwargs::KW
end
FixKeywords(f; kwargs...) = FixKeywords(f, NamedTuple(kwargs))
(f::FixKeywords)(args...) = f.f(args...; f.kwargs...)

function _check_log_likelihood(x)
    if any(!isfinite, x)
        @warn "All log likelihood values must be finite, but some are not."
    end
    return nothing
end

"""
    smooth_data(y; dims=:, interp_method=CubicSpline, offset_frac=0.01)

Smooth `y` along `dims` using `interp_method`.

`interp_method` is a 2-argument callabale that takes the arguments `y` and `x` and returns
a DataInterpolations.jl interpolation method, defaulting to a cubic spline interpolator.

`offset_frac` is the fraction of the length of `y` to use as an offset when interpolating.
"""
function smooth_data(
    y;
    dims::Union{Int,Tuple{Int,Vararg{Int}},Colon}=Colon(),
    interp_method=DataInterpolations.CubicSpline,
    offset_frac=1//100,
)
    T = float(eltype(y))
    y_interp = similar(y, T)
    n = dims isa Colon ? length(y) : prod(Base.Fix1(size, y), dims)
    x = range(0, 1; length=n)
    x_interp = range(0 + offset_frac, 1 - offset_frac; length=n)
    _smooth_data!(y_interp, interp_method, y, x, x_interp, dims)
    return y_interp
end

function _smooth_data!(y_interp, interp_method, y, x, x_interp, ::Colon)
    interp = interp_method(vec(y), x)
    interp(vec(y_interp), x_interp)
    return y_interp
end
function _smooth_data!(y_interp, interp_method, y, x, x_interp, dims)
    for (y_interp_i, y_i) in zip(
        eachslice(y_interp; dims=_otherdims(y_interp, dims)),
        eachslice(y; dims=_otherdims(y, dims)),
    )
        interp = interp_method(vec(y_i), x)
        interp(vec(y_interp_i), x_interp)
    end
    return y_interp
end

Base.@pure _typename(::T) where {T} = T.name.name

_astuple(x) = (x,)
_astuple(x::Tuple) = x

function _assimilar(x::AbstractArray, y)
    z = similar(x, eltype(y))
    copyto!(z, y)
    return z
end
_assimilar(x::AbstractArray, y::NamedTuple) = _assimilar(x, values(y))
function _assimilar(x::Tuple, y)
    z = NTuple{length(x),eltype(y)}(y)
    return z
end
function _assimilar(x::NamedTuple, y)
    z = NamedTuple{fieldnames(typeof(x))}(_assimilar(values(x), y))
    return z
end

# included since Base.copymutable is not public
function _copymutable(x::AbstractArray)
    y = similar(x)
    copyto!(y, x)
    return y
end

function _skipmissing(x::AbstractArray)
    Missing <: eltype(x) && return skipmissing(x)
    return x
end

function _cskipmissing(x::AbstractArray)
    Missing <: eltype(x) && return collect(skipmissing(x))
    return x
end

_sortperm(x; kwargs...) = sortperm(collect(x); kwargs...)

_permute(x::AbstractVector, p::AbstractVector) = x[p]
_permute(x::Tuple, p::AbstractVector) = x[p]
function _permute(x::NamedTuple, p::AbstractVector)
    return NamedTuple{_permute(keys(x), p)}(_permute(values(x), p))
end

# TODO: try to find a way to do this that works for arrays with named indices
_indices(x) = keys(x)

_alldims(x) = ntuple(identity, ndims(x))

_otherdims(x, dims) = filter(∉(dims), _alldims(x))

_param_dims(x::AbstractArray) = ntuple(i -> i + 2, max(0, ndims(x) - 2))

_param_axes(x::AbstractArray) = map(Base.Fix1(axes, x), _param_dims(x))

function _params_array(x::AbstractArray, param_dim::Int=3)
    param_dim > 0 || throw(ArgumentError("param_dim must be positive"))
    sample_sizes = ntuple(Base.Fix1(size, x), param_dim - 1)
    return reshape(x, sample_sizes..., :)
end

function _eachparam(x::AbstractArray, param_dim::Int=3)
    return eachslice(_params_array(x, param_dim); dims=param_dim)
end

_maybe_scalar(x) = x
_maybe_scalar(x::AbstractArray{<:Any,0}) = x[]

_logabssubexp(x, y) = LogExpFunctions.logsubexp(reverse(minmax(x, y))...)

# softmax with support for other mappable iterators
_softmax(x::AbstractArray) = LogExpFunctions.softmax(x)
function _softmax(x)
    nrm = LogExpFunctions.logsumexp(x)
    return map(x) do xi
        return exp(xi - nrm)
    end
end

# compute sum and estimate of standard error of sum
function _sum_and_se(x; dims=:)
    s = sum(x; dims)
    n = dims isa Colon ? length(x) : prod(Base.Fix1(size, x), dims)
    se = Statistics.std(x; dims) * sqrt(oftype(one(eltype(s)), n))
    return s, se
end
_sum_and_se(x::Number; kwargs...) = (x, oftype(float(x), NaN))

function _log_mean(logx, log_weights; dims=:)
    log_expectand = logx .+ log_weights
    return LogExpFunctions.logsumexp(log_expectand; dims)
end

function _se_log_mean(
    logx, log_weights; dims=:, log_mean=_log_mean(logx, log_weights; dims)
)
    # variance of mean estimated using self-normalized importance weighting
    # Art B. Owen. (2013) Monte Carlo theory, methods and examples. eq. 9.9
    log_expectand = @. 2 * (log_weights + _logabssubexp(logx, log_mean))
    log_var_mean = LogExpFunctions.logsumexp(log_expectand; dims)
    # use delta method to asymptotically map variance of mean to variance of logarithm of mean
    se_log_mean = @. exp(log_var_mean / 2 - log_mean)
    return se_log_mean
end

"""
    sigdigits_matching_se(x, se; sigdigits_max=7, scale=2) -> Int

Get number of significant digits of `x` so that the last digit of `x` is the first digit of
`se*scale`.
"""
function sigdigits_matching_se(x::Real, se::Real; sigdigits_max::Int=7, scale::Real=2)
    (iszero(x) || !isfinite(x) || !isfinite(se) || !isfinite(scale)) && return 0
    sigdigits_max ≥ 0 || throw(ArgumentError("`sigdigits_max` must be non-negative"))
    se ≥ 0 || throw(ArgumentError("`se` must be non-negative"))
    iszero(se) && return sigdigits_max
    scale > 0 || throw(ArgumentError("`scale` must be positive"))
    first_digit_x = floor(Int, log10(abs(x)))
    last_digit_x = floor(Int, log10(se * scale))
    sigdigits_x = first_digit_x - last_digit_x + 1
    return clamp(sigdigits_x, 0, sigdigits_max)
end

# format a number with the given number of significant digits
# - chooses scientific or decimal notation by whichever is most appropriate
# - shows trailing zeros if significant
# - removes trailing decimal point if no significant digits after decimal point
function _printf_with_sigdigits(v::Real, sigdigits)
    s = sprint(Printf.format, Printf.Format("%#.$(sigdigits)g"), v)
    return replace(s, r"\.($|e)" => s"\1")
end
