"""
$(SIGNATURES)

A container for a column table of values computed by [`summarize`](@ref).

This object implements the Tables and TableTraits interfaces and has a custom `show` method.

$(FIELDS)
"""
struct SummaryStats{D<:NamedTuple}
    """The summary statistics for each variable, with the first entry containing the
    variable names"""
    data::D
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.propertynames(stats::SummaryStats) = propertynames(parent(stats))
Base.getproperty(stats::SummaryStats, nm::Symbol) = getproperty(parent(stats), nm)
Base.keys(stats::SummaryStats) = keys(parent(stats))
Base.haskey(stats::SummaryStats, nm::Symbol) = haskey(parent(stats), nm)
Base.length(stats::SummaryStats) = length(parent(stats))
Base.getindex(stats::SummaryStats, i::Int) = getindex(parent(stats), i)
Base.getindex(stats::SummaryStats, nm::Symbol) = getindex(parent(stats), nm)
function Base.iterate(stats::SummaryStats, i::Int=firstindex(parent(stats)))
    return iterate(parent(stats), i)
end
function Base.merge(stats::SummaryStats, other_stats::SummaryStats...)
    return SummaryStats(merge(parent(stats), map(parent, other_stats)...))
end

#### custom tabular show methods

function Base.show(io::IO, mime::MIME"text/plain", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end
function Base.show(io::IO, mime::MIME"text/html", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end

function _show(io::IO, mime::MIME, stats::SummaryStats; kwargs...)
    data = NamedTuple{eachindex(stats)[2:end]}(parent(stats))
    rhat_formatter = _prettytables_rhat_formatter(data)
    extra_formatters = rhat_formatter === nothing ? () : (rhat_formatter,)
    return _show_prettytable(
        io,
        mime,
        data;
        title="SummaryStats",
        row_labels=stats.variable,
        extra_formatters,
        kwargs...,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
Tables.columnnames(s::SummaryStats) = Tables.columnnames(parent(s))
Tables.getcolumn(s::SummaryStats, i::Int) = Tables.getcolumn(parent(s), i)
Tables.getcolumn(s::SummaryStats, nm::Symbol) = Tables.getcolumn(parent(s), nm)
Tables.rowaccess(::Type{<:SummaryStats}) = true
Tables.rows(s::SummaryStats) = Tables.rows(parent(s))
Tables.schema(s::SummaryStats) = Tables.schema(parent(s))

IteratorInterfaceExtensions.isiterable(::SummaryStats) = true
function IteratorInterfaceExtensions.getiterator(s::SummaryStats)
    return Tables.datavaluerows(Tables.columntable(s))
end

TableTraits.isiterabletable(::SummaryStats) = true

"""
    summarize(data, stats_funs; [var_names]) -> SummaryStats

Compute the summary statistics in `stats_funs` on each param in `data`.

`stats_funs` is a collection of functions that reduces a matrix with shape `(draws, chains)`
to a scalar or a collection of scalars. Alternatively, an item in `stats_funs` may be a
`Pair` of the form `name => fun` specifying the name to be used for the statistic or of the
form `(name1, ...) => fun` when the function returns a collection. When the function returns
a collection, the names in this latter format must be provided.

The variable names may be provided by specifying `var_names`. Otherwise, defaults are
selected from `data`.

To support computing summary statistics from a custom object, overload this method.

See also [`SummaryStats`](@ref).
"""
function summarize end

"""
    summarize(data::AbstractArray, stats_funs; kwargs...) -> SummaryStats

Compute the summary statistics in `stats_funs` on each param in `data`, with size
`(draws, chains, params)`.

# Examples

Compute `mean`, `std` and the Monte Carlo standard error (MCSE) of the mean estimate:

```jldoctest summarize_array; setup = (using Random; Random.seed!(84))
julia> using PosteriorStats, Statistics, StatsBase

julia> x = randn(1000, 4, 3) .+ reshape(0:10:20, 1, 1, :);

julia> summarize(x, (mean, std, :mcse_mean => sem))
SummaryStats
       mean    std  mcse_mean
 1   0.0003  0.990      0.016
 2  10.02    0.988      0.016
 3  19.98    0.988      0.016
```

Avoid recomputing the mean by using `mean_and_std`, and provide parameter names:
```jldoctest summarize_array
julia> summarize(x, ((:mean, :std) => mean_and_std, mad); var_names=[:a, :b, :c])
SummaryStats
         mean    std    mad
 a   0.000305  0.990  0.978
 b  10.0       0.988  0.995
 c  20.0       0.988  0.979
```

Note that when an estimator and its MCSE are both computed, the MCSE is used to determine
the number of significant digits that will be displayed.
"""
function summarize(
    data::AbstractArray{<:Union{Real,Missing},3},
    stats_funs_and_names;
    var_names=axes(data, 3),
)
    names_and_funs = map(_fun_and_name, stats_funs_and_names)
    fnames = map(first, names_and_funs)
    funs = map(last, names_and_funs)
    nt = merge((; variable=var_names), _summarize(data, funs, fnames)...)
    return SummaryStats(nt)
end

"""
    summarize(data; kwargs...)

Compute summary statistics and diagnostics on the `data`.

This method computes default statistics and diagnostics, which may be customized
(see below).

# Keywords

  - `prob_interval::Real`: The value of the `prob` argument to [`hdi`](@ref) used to compute
    the highest density interval. Defaults to $(DEFAULT_INTERVAL_PROB).
  - `defaults::Union{Bool,Symbol}=true`: Whether to compute just statistics (`:stats`), just
    diagnostics (`:diagnostics`), all (`true`), or none (`false`).
  - `stats_funs=()`: A collection of `stats_funs` forwarded to [`summarize`](@ref).
  - `kwargs`: Remaining keywords are forwarded to [`summarize`](@ref).

# Examples

```jldoctest summarize_defaults; setup = (using Random; Random.seed!(33))
julia> using PosteriorStats, Statistics, StatsBase

julia> x = randn(1000, 4, 3) .+ reshape(0:10:20, 1, 1, :);

julia> summarize(x)
SummaryStats
      mean   std  hdi_3%  hdi_97%  mcse_mean  mcse_std  ess_tail  ess_bulk  rh ⋯
 1   0.003  1.00   -1.83     1.89      0.016     0.011      3800      3952  1. ⋯
 2   9.99   0.98    8.19    11.9       0.015     0.011      3996      4045  1. ⋯
 3  20.02   1.01   18.1     21.9       0.017     0.011      3741      3736  1. ⋯
                                                                1 column omitted
```

Compute just the statistics on all variables, and provide the variable names:

```jldoctest summarize_defaults
julia> summarize(x; var_names=[:a, :b, :c], defaults=:stats)
SummaryStats
        mean    std  hdi_3%  hdi_97%
 a   0.00258  1.00    -1.83     1.89
 b   9.99     0.984    8.19    11.9
 c  20.0      1.01    18.1     21.9
```

Compute the default statistics with a 89% HDI, along with the median and median absolute
deviation:

```jldoctest summarize_defaults
julia> summarize(x; defaults=:stats, prob_interval=0.89, stats_funs=(median, mad))
SummaryStats
        mean    std  hdi_5.5%  hdi_94.5%   median    mad
 1   0.00258  1.00      -1.51       1.68   0.0206  1.02
 2   9.99     0.984      8.53      11.6   10.00    0.982
 3  20.0      1.01      18.4       21.5   20.0     0.992
```
"""
function summarize(
    data;
    prob_interval::Real=DEFAULT_INTERVAL_PROB,
    defaults::Union{Bool,Symbol}=true,
    stats_funs=(),
    kwargs...,
)
    user_stats = summarize(data, stats_funs; kwargs...)
    default_stats_funs = if defaults === :stats
        _default_stats_funs(; prob_interval)
    elseif defaults === :diagnostics
        _default_diagnostic_funs()
    elseif defaults === true
        (_default_stats_funs(; prob_interval)..., _default_diagnostic_funs()...)
    else
        return user_stats
    end
    default_stats = summarize(data, default_stats_funs; kwargs...)
    return merge(default_stats, user_stats)
end

function _default_stats_funs(; prob_interval::Real=DEFAULT_INTERVAL_PROB)
    hdi_names = map(Symbol, _prob_interval_to_strings("hdi", prob_interval))
    return (
        (:mean, :std) => StatsBase.mean_and_std, hdi_names => Base.Fix2(_hdi, prob_interval)
    )
end

function _default_diagnostic_funs()
    return (
        :mcse_mean => MCMCDiagnosticTools.mcse,
        :mcse_std => _mcse_std,
        :ess_tail => _ess_tail,
        (:ess_bulk, :rhat) => MCMCDiagnosticTools.ess_rhat,
    )
end

function _prob_interval_to_strings(interval_type, prob; digits=2)
    α = (1 - prob) / 2
    perc_lower = string(round(100 * α; digits))
    perc_upper = string(round(100 * (1 - α); digits))
    return map((perc_lower, perc_upper)) do s
        s = replace(s, r"\.0+$" => "")
        return "$(interval_type)_$s%"
    end
end

function _summarize(data::AbstractArray{<:Any,3}, funs, fun_names)
    return map(fun_names, funs) do fname, f
        val = map(f, eachslice(data; dims=3))
        if fname isa NTuple{<:Any,Symbol}
            return NamedTuple{fname}(getindex.(val, i) for i in 1:length(fname))
        else
            return (; fname => val)
        end
    end
end

_mcse_std(x) = MCMCDiagnosticTools.mcse(x; kind=Statistics.std)
_ess_tail(x) = MCMCDiagnosticTools.ess(x; kind=:tail)
_hdi(x, prob) = hdi(x; prob)

_map_paramslices(f, x) = map(f, eachslice(x; dims=3))
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess), x) = f(x)
_map_paramslices(f::typeof(_ess_tail), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess_rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.mcse), x) = f(x)
_map_paramslices(f::typeof(_mcse_std), x) = f(x)
_map_paramslices(f::Base.Fix2{typeof(_hdi)}, x) = f(x)

_fnames(f::Tuple) = map(Symbol, f)
_fnames(f) = Symbol(f)

_fun_and_name(p::Pair) = _fnames(p.first) => p.second
_fun_and_name(f) = _fnames(f) => f
