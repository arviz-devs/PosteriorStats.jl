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

# extract keywords from a NamedTuple or a collection of key-value pairs
function _extract_keywords(kwargs::NamedTuple, ks::Tuple{Vararg{Symbol}})
    return NamedTuple{filter(∈(keys(kwargs)), ks)}(kwargs)
end
function _extract_keywords(kwargs, ks::Tuple{Vararg{Symbol}})
    return _extract_keywords(NamedTuple(kwargs), ks)
end

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

#
# PrettyTables formatter utility functions
#

"""
    ft_printf_sigdigits(sigdigits[, columns])

Use Printf to format the elements in the `columns` to the number of `sigdigits`.

If `sigdigits` is a `Real`, and `columns` is not specified (or is empty), then the
formatting will be applied to the entire table.
Otherwise, if `sigdigits` is a `Real` and `columns` is a vector, then the elements in the
columns will be formatted to the number of `sigdigits`.
"""
function ft_printf_sigdigits(sigdigits::Int, columns::AbstractVector{Int}=Int[])
    if isempty(columns)
        return (v, _, _) -> begin
            v isa Real || return v
            return _printf_with_sigdigits(v, sigdigits)
        end
    else
        return (v, _, j) -> begin
            v isa Real || return v
            for col in columns
                col == j && return _printf_with_sigdigits(v, sigdigits)
            end
            return v
        end
    end
end

function ft_printf_sigdigits_interval(sigdigits::Int, columns::AbstractVector{Int}=Int[])
    if isempty(columns)
        return (v, _, _) -> begin
            v isa IntervalSets.AbstractInterval || return v
            tuple_string = map(Base.Fix2(_printf_with_sigdigits, sigdigits), extrema(v))
            return join(tuple_string, _interval_delimiter(v))
        end
    else
        return (v, _, j) -> begin
            v isa Tuple{<:Real,Vararg{Real}} || return v
            for col in columns
                col == j || continue
                tuple_string = map(Base.Fix2(_printf_with_sigdigits, sigdigits), extrema(v))
                return join(tuple_string, _interval_delimiter(v))
            end
            return v
        end
    end
end

function _interval_delimiter(x::IntervalSets.AbstractInterval)
    str = sprint(show, "text/plain", x)
    return occursin(" .. ", str) ? " .. " : ".."
end

"""
    ft_printf_sigdigits_matching_se(se_vals[, columns]; kwargs...)

Use Printf to format the elements in the `columns` to sigdigits based on the standard error
column in `se_vals`.

All values are formatted with Printf to the number of significant digits determined by
[`sigdigits_matching_se`](@ref). `kwargs` are forwarded to that function.

`se_vals` must be the same length as any of the columns in the table.
If `columns` is a non-empty vector, then the formatting is only applied to those columns.
Otherwise, the formatting is applied to the entire table.
"""
function ft_printf_sigdigits_matching_se(
    se_vals::AbstractVector, columns::AbstractVector{Int}=Int[]; kwargs...
)
    if isempty(columns)
        return (v, i, _) -> begin
            (v isa Real && se_vals[i] isa Real) || return v
            sigdigits = sigdigits_matching_se(v, se_vals[i]; kwargs...)
            return _printf_with_sigdigits(v, sigdigits)
        end
    else
        return (v, i, j) -> begin
            (v isa Real && se_vals[i] isa Real) || return v
            for col in columns
                if col == j
                    sigdigits = sigdigits_matching_se(v, se_vals[i]; kwargs...)
                    return _printf_with_sigdigits(v, sigdigits)
                end
            end
            return v
        end
    end
end

function _prettytables_rhat_formatter(data)
    cols = findall(
        x -> (x === :rhat || startswith(string(x), "rhat_")), Tables.columnnames(data)
    )
    isempty(cols) && return nothing
    return PrettyTables.ft_printf("%.2f", cols)
end

function _prettytables_integer_formatter(data)
    sch = Tables.schema(data)
    sch === nothing && return nothing
    cols = findall(t -> t <: Integer, sch.types)
    isempty(cols) && return nothing
    return PrettyTables.ft_printf("%d", cols)
end

# formatting functions for special columns
# see https://ronisbr.github.io/PrettyTables.jl/stable/man/formatters/
function _default_prettytables_formatters(data; sigdigits_se=2, sigdigits_default=3)
    formatters = []
    col_names = Tables.columnnames(data)
    for (i, k) in enumerate(col_names)
        for mcse_key in
            (Symbol("mcse_$k"), Symbol("$(k)_mcse"), Symbol("se_$k"), Symbol("$(k)_se"))
            if haskey(data, mcse_key)
                push!(
                    formatters,
                    ft_printf_sigdigits_matching_se(Tables.getcolumn(data, mcse_key), [i]),
                )
                continue
            end
        end
    end
    mcse_cols = findall(col_names) do k
        s = string(k)
        return startswith(s, "mcse_") ||
               endswith(s, "_mcse") ||
               startswith(s, "se_") ||
               endswith(s, "_se")
    end
    isempty(mcse_cols) || push!(formatters, ft_printf_sigdigits(sigdigits_se, mcse_cols))
    ess_cols = findall(_is_ess_label, col_names)
    isempty(ess_cols) || push!(formatters, PrettyTables.ft_printf("%d", ess_cols))
    ft_integer = _prettytables_integer_formatter(data)
    ft_integer === nothing || push!(formatters, ft_integer)
    push!(formatters, ft_printf_sigdigits(sigdigits_default))
    push!(formatters, ft_printf_sigdigits_interval(sigdigits_default))
    return formatters
end

function _show_prettytable(
    io::IO, data; sigdigits_se=2, sigdigits_default=3, extra_formatters=(), kwargs...
)
    formatters = (
        extra_formatters...,
        _default_prettytables_formatters(data; sigdigits_se, sigdigits_default)...,
    )
    col_names = Tables.columnnames(data)
    alignment = [
        eltype(Tables.getcolumn(data, col_name)) <: Real ? :r : :l for col_name in col_names
    ]
    kwargs_new = merge(
        (
            show_subheader=false,
            vcrop_mode=:middle,
            show_omitted_cell_summary=true,
            row_label_alignment=:l,
            formatters,
            alignment,
        ),
        kwargs,
    )
    PrettyTables.pretty_table(io, data; kwargs_new...)
    return nothing
end

function _show_prettytable(
    io::IO,
    ::MIME"text/plain",
    data;
    title_crayon=PrettyTables.Crayon(),
    hlines=:none,
    vlines=:none,
    newline_at_end=false,
    kwargs...,
)
    alignment_anchor_regex = Dict{Int,Vector{Regex}}()
    for (i, k) in enumerate(Tables.columnnames(data))
        v = Tables.getcolumn(data, k)
        if eltype(v) <: Real && !(eltype(v) <: Integer) && !_is_ess_label(k)
            alignment_anchor_regex[i] = [r"\.", r"e", r"^NaN$", r"Inf$"]
        elseif eltype(v) <: IntervalSets.AbstractInterval
            alignment_anchor_regex[i] = [r"\.\."]
        end
    end
    alignment_anchor_fallback = :r
    alignment_anchor_fallback_override = Dict(
        i => :r for (i, k) in enumerate(Tables.columnnames(data)) if _is_ess_label(k)
    )
    return _show_prettytable(
        io,
        data;
        backend=Val(:text),
        title_crayon,
        hlines,
        vlines,
        newline_at_end,
        alignment_anchor_regex,
        alignment_anchor_fallback,
        alignment_anchor_fallback_override,
        kwargs...,
    )
end
function _show_prettytable(
    io::IO, ::MIME"text/html", data; minify=true, max_num_of_rows=25, kwargs...
)
    return _show_prettytable(
        io, data; backend=Val(:html), minify, max_num_of_rows, kwargs...
    )
end

_is_ess_label(k::Symbol) = ((k === :ess) || startswith(string(k), "ess_"))
