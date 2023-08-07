"""
$(SIGNATURES)

A container for a column table of values computed by [`summarize`](@ref).

This object implements the Tables and TableTraits interfaces and has a custom `show` method.

$(FIELDS)
"""
struct SummaryStats{D<:NamedTuple}
    "The name of the collection of summary statistics, used as the table title in display."
    name::String
    """The summary statistics for each parameter, with an optional first column `parameter`
    containing the parameter names."""
    data::D
end
function SummaryStats(data::NamedTuple; name::String="SummaryStats")
    n = length(first(data))
    return SummaryStats(name, merge((parameter=1:n,), data))
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.keys(stats::SummaryStats) = keys(parent(stats))
Base.haskey(stats::SummaryStats, nm::Symbol) = haskey(parent(stats), nm)
Base.length(stats::SummaryStats) = length(parent(stats))
Base.getindex(stats::SummaryStats, i::Int) = getindex(parent(stats), i)
Base.getindex(stats::SummaryStats, nm::Symbol) = getindex(parent(stats), nm)
function Base.iterate(stats::SummaryStats, i::Int=firstindex(parent(stats)))
    return iterate(parent(stats), i)
end
function Base.merge(stats::SummaryStats, other_stats::SummaryStats...)
    return SummaryStats(stats.name, merge(parent(stats), map(parent, other_stats)...))
end
function Base.isequal(stats::SummaryStats, other_stats::SummaryStats)
    return isequal(parent(stats), parent(other_stats))
end
function Base.:(==)(stats::SummaryStats, other_stats::SummaryStats)
    return (parent(stats) == parent(other_stats))
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
        title=stats.name,
        row_labels=parent(stats).parameter,
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
    summarize(data, stats_funs...; name="SummaryStats", [var_names]) -> SummaryStats

Compute the summary statistics in `stats_funs` on each param in `data`.

`stats_funs` is a collection of functions that reduces a matrix with shape `(draws, chains)`
to a scalar or a collection of scalars. Alternatively, an item in `stats_funs` may be a
`Pair` of the form `name => fun` specifying the name to be used for the statistic or of the
form `(name1, ...) => fun` when the function returns a collection. When the function returns
a collection, the names in this latter format must be provided.

If no stats functions are provided, then those specified in [`default_summary_stats`](@ref)
are computed.

`var_names` specifies the names of the parameters in `data`. If not provided, the names are
inferred from `data`.

To support computing summary statistics from a custom object, overload this method
specifying the type of `data`.

See also [`SummaryStats`](@ref), [`default_summary_stats`](@ref), [`default_stats`](@ref),
[`default_diagnostics`](@ref).

# Examples

Compute `mean`, `std` and the Monte Carlo standard error (MCSE) of the mean estimate:

```jldoctest summarize; setup = (using Random; Random.seed!(84))
julia> using PosteriorStats, Statistics, StatsBase

julia> x = randn(1000, 4, 3) .+ reshape(0:10:20, 1, 1, :);

julia> summarize(x, mean, std, :mcse_mean => sem; name="Mean/Std")
Mean/Std
       mean    std  mcse_mean
 1   0.0003  0.990      0.016
 2  10.02    0.988      0.016
 3  19.98    0.988      0.016
```

Avoid recomputing the mean by using `mean_and_std`, and provide parameter names:
```jldoctest summarize
julia> summarize(x, (:mean, :std) => mean_and_std, mad; var_names=[:a, :b, :c])
SummaryStats
         mean    std    mad
 a   0.000305  0.990  0.978
 b  10.0       0.988  0.995
 c  20.0       0.988  0.979
```

Note that when an estimator and its MCSE are both computed, the MCSE is used to determine
the number of significant digits that will be displayed.

```jldoctest summarize
julia> summarize(x; var_names=[:a, :b, :c])
SummaryStats
       mean   std  hdi_3%  hdi_97%  mcse_mean  mcse_std  ess_tail  ess_bulk  r ⋯
 a   0.0003  0.99   -1.92     1.78      0.016     0.012      3567      3663  1 ⋯
 b  10.02    0.99    8.17    11.9       0.016     0.011      3841      3906  1 ⋯
 c  19.98    0.99   18.1     21.9       0.016     0.012      3892      3749  1 ⋯
                                                                1 column omitted
```

Compute just the statistics with an 89% HDI on all parameters, and provide the parameter
names:

```jldoctest summarize
julia> summarize(x, default_stats(; prob_interval=0.89)...; var_names=[:a, :b, :c])
SummaryStats
         mean    std  hdi_5.5%  hdi_94.5%
 a   0.000305  0.990     -1.63       1.52
 b  10.0       0.988      8.53      11.6
 c  20.0       0.988     18.5       21.6
```

Compute the summary stats with the `stat_focus` set to `Statistics.median`:

```jldoctest summarize
julia> summarize(x, default_summary_stats(median)...; var_names=[:a, :b, :c])
SummaryStats
    median    mad  eti_3%  eti_97%  mcse_median  ess_tail  ess_median  rhat
 a   0.004  0.978   -1.83     1.89        0.020      3567        3336  1.00
 b  10.02   0.995    8.17    11.9         0.023      3841        3787  1.00
 c  19.99   0.979   18.1     21.9         0.020      3892        3829  1.00
```
"""
function summarize end

"""
    summarize(data::AbstractArray, stats_funs...; kwargs...) -> SummaryStats

Compute the summary statistics in `stats_funs` on each param in `data`, with size
`(draws, chains, params)`.
"""
@constprop :aggressive function summarize(
    data::AbstractArray{<:Union{Real,Missing},3},
    stats_funs_and_names...;
    name::String="SummaryStats",
    var_names=axes(data, 3),
)
    if isempty(stats_funs_and_names)
        return summarize(data, default_summary_stats()...; name, var_names)
    end
    names_and_funs = map(_fun_and_name, stats_funs_and_names)
    fnames = map(first, names_and_funs)
    funs = map(last, names_and_funs)
    nt = merge((; parameter=var_names), _summarize(data, funs, fnames)...)
    return SummaryStats(name, nt)
end

"""
    default_summary_stats(focus=Statistics.mean; kwargs...)

Combinatiton of [`default_stats`](@ref) and [`default_diagnostics`](@ref) to be used with
[`summarize`](@ref).
"""
function default_summary_stats(focus=Statistics.mean; kwargs...)
    return (default_stats(focus; kwargs...)..., default_diagnostics(focus; kwargs...)...)
end

"""
    default_stats(focus=Statistics.mean; prob_interval=$(DEFAULT_INTERVAL_PROB), kwargs...)

Default statistics to be computed with [`summarize`](@ref).

The value of `focus` determines the statistics to be returned:
- `Statistics.mean`: `mean`, `std`, `hdi_3%`, `hdi_97%`
- `Statistics.median`: `median`, `mad`, `eti_3%`, `eti_97%`

If `prob_interval` is set to a different value than the default, then different HDI and ETI
statistics are computed accordingly. [`hdi`](@ref) refers to the highest-density interval,
while `eti` refers to the equal-tailed interval (i.e. the credible interval computed from
symmetric quantiles).

See also: [`hdi`](@ref)
"""
function default_stats end
default_stats(; kwargs...) = default_stats(Statistics.mean; kwargs...)
function default_stats(
    ::typeof(Statistics.mean); prob_interval::Real=DEFAULT_INTERVAL_PROB, kwargs...
)
    hdi_names = map(Symbol, _prob_interval_to_strings("hdi", prob_interval))
    return (
        (:mean, :std) => StatsBase.mean_and_std ∘ skipmissing,
        hdi_names => x -> hdi(collect(skipmissing(x)); prob=prob_interval),
    )
end
function default_stats(
    ::typeof(Statistics.median); prob_interval::Real=DEFAULT_INTERVAL_PROB, kwargs...
)
    eti_names = map(Symbol, _prob_interval_to_strings("eti", prob_interval))
    prob_tail = (1 - prob_interval) / 2
    p = (prob_tail, 1 - prob_tail)
    return (
        :median => Statistics.median ∘ skipmissing,
        :mad => StatsBase.mad ∘ skipmissing,
        eti_names => Base.Fix2(Statistics.quantile, p) ∘ skipmissing,
    )
end

"""
    default_diagnostics(focus=Statistics.mean; kwargs...)

Default diagnostics to be computed with [`summarize`](@ref).

The value of `focus` determines the diagnostics to be returned:
- `Statistics.mean`: `mcse_mean`, `mcse_std`, `ess_tail`, `ess_bulk`, `rhat`
- `Statistics.median`: `mcse_median`, `ess_tail`, `ess_bulk`, `rhat`
"""
default_diagnostics(; kwargs...) = default_diagnostics(Statistics.mean; kwargs...)
function default_diagnostics(::typeof(Statistics.mean); kwargs...)
    return (
        :mcse_mean => MCMCDiagnosticTools.mcse,
        :mcse_std => _mcse_std,
        :ess_tail => _ess_tail,
        (:ess_bulk, :rhat) => MCMCDiagnosticTools.ess_rhat,
    )
end
function default_diagnostics(::typeof(Statistics.median); kwargs...)
    return (
        :mcse_median => _mcse_median,
        :ess_tail => _ess_tail,
        :ess_median => _ess_median,
        MCMCDiagnosticTools.rhat,
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
        return _map_over_params(fname, f, data)
    end
end

# aggressive constprop allows summarize to be type-inferrable when called by
# another function
@constprop :aggressive function _map_over_params(fname, f, data)
    vals = _map_paramslices(f, data)
    return _namedtuple_of_vals(f, fname, vals)
end

_namedtuple_of_vals(f, fname::Symbol, val) = (; fname => val)
_namedtuple_of_vals(f, ::Nothing, val) = (; _fname(f) => val)
function _namedtuple_of_vals(f, fname::NTuple{N,Symbol}, val::AbstractVector) where {N}
    return NamedTuple{fname}(ntuple(i -> getindex.(val, i), Val(N)))
end
function _namedtuple_of_vals(f, fname::NTuple{N,Symbol}, val::NamedTuple) where {N}
    return NamedTuple{fname}(values(val))
end
function _namedtuple_of_vals(f, ::Nothing, val::AbstractVector{<:NamedTuple{K}}) where {K}
    return NamedTuple{K}(ntuple(i -> getindex.(val, i), length(K)))
end

_fun_and_name(p::Pair) = p
_fun_and_name(f) = nothing => f

_fname(f) = Symbol(_fname_string(f))
@generated function _fname_string(::F) where {F}
    s = replace(string(F), r"^typeof\((.*)\)$" => s"\1")
    # remove module name
    return replace(s, r"^.*\.(.*)$" => s"\1")
end

# curried functions
_mcse_std(x) = MCMCDiagnosticTools.mcse(x; kind=Statistics.std)
_mcse_median(x) = MCMCDiagnosticTools.mcse(x; kind=Statistics.median)
_ess_median(x) = MCMCDiagnosticTools.ess(x; kind=Statistics.median)
_ess_tail(x) = MCMCDiagnosticTools.ess(x; kind=:tail)

# functions that have a 3D array method
_map_paramslices(f, x) = map(f, eachslice(x; dims=3))
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess), x) = f(x)
_map_paramslices(f::typeof(_ess_median), x) = f(x)
_map_paramslices(f::typeof(_ess_tail), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess_rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.mcse), x) = f(x)
_map_paramslices(f::typeof(_mcse_std), x) = f(x)
_map_paramslices(f::typeof(_mcse_median), x) = f(x)