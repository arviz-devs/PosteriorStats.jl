"""
$(TYPEDEF)

A container for a column table of values computed by [`summarize`](@ref).

This object implements the Tables and TableTraits column table interfaces. It has a custom
`show` method.

`SummaryStats` behaves like an `OrderedDict` of columns, where the columns can be accessed
using either `Symbol`s or a 1-based integer index.

$(TYPEDFIELDS)

    SummaryStats([name::String,] data[, parameter_names])
    SummaryStats(data[, parameter_names]; name::String="SummaryStats")

Construct a `SummaryStats` from tabular `data` with optional stats `name` and `param_names`.

`data` must not contain a column `:parameter`, as this is reserved for the parameter names,
which are always in the first column.
"""
struct SummaryStats{D,V<:AbstractVector}
    "The name of the collection of summary statistics, used as the table title in display."
    name::String
    """The summary statistics for each parameter. It must implement the Tables interface."""
    data::D
    "Names of the parameters"
    parameter_names::V
    function SummaryStats(name::String, data, parameter_names::V) where {V}
        coltable = Tables.columns(data)
        :parameter ∈ Tables.columnnames(coltable) &&
            throw(ArgumentError("Column `:parameter` is reserved for parameter names."))
        length(parameter_names) == Tables.rowcount(data) || throw(
            DimensionMismatch(
                "length $(length(parameter_names)) of `parameter_names` does not match number of rows $(Tables.rowcount(data)) in `data`.",
            ),
        )
        return new{typeof(coltable),V}(name, coltable, parameter_names)
    end
end
function SummaryStats(
    data,
    parameter_names::AbstractVector=Base.OneTo(Tables.rowcount(data));
    name::String="SummaryStats",
)
    return SummaryStats(name, data, parameter_names)
end
function SummaryStats(name::String, data)
    return SummaryStats(name, data, Base.OneTo(Tables.rowcount(data)))
end

function _ordereddict(stats::SummaryStats)
    return OrderedCollections.OrderedDict(
        k => Tables.getcolumn(stats, k) for k in Tables.columnnames(stats)
    )
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.keys(stats::SummaryStats) = map(Symbol, Tables.columnnames(stats))
Base.haskey(stats::SummaryStats, nm::Symbol) = nm ∈ keys(stats)
Base.length(stats::SummaryStats) = length(parent(stats)) + 1
Base.getindex(stats::SummaryStats, i::Union{Int,Symbol}) = Tables.getcolumn(stats, i)
function Base.iterate(stats::SummaryStats)
    ncols = length(stats)
    return stats.parameter_names, (2, ncols)
end
function Base.iterate(stats::SummaryStats, (i, ncols)::NTuple{2,Int})
    i > ncols && return nothing
    return Tables.getcolumn(stats, i), (i + 1, ncols)
end
function Base.merge(
    stats::SummaryStats{<:NamedTuple}, other_stats::SummaryStats{<:NamedTuple}...
)
    isempty(other_stats) && return stats
    stats_all = (stats, other_stats...)
    stats_last = last(stats_all)
    return SummaryStats(
        stats_last.name, merge(map(parent, stats_all)...), stats_last.parameter_names
    )
end
function Base.merge(stats::SummaryStats, other_stats::SummaryStats...)
    isempty(other_stats) && return stats
    stats_all = (stats, other_stats...)
    data_merged = merge(map(_ordereddict, stats_all)...)
    parameter_names = pop!(data_merged, :parameter)
    return SummaryStats(last(stats_all).name, data_merged, parameter_names)
end
for f in (:(==), :isequal)
    @eval begin
        function Base.$(f)(stats::SummaryStats, other_stats::SummaryStats)
            colnames1 = Tables.columnnames(stats)
            colnames2 = Tables.columnnames(other_stats)
            vals1 = (Tables.getcolumn(stats, k) for k in colnames1)
            vals2 = (Tables.getcolumn(other_stats, k) for k in colnames2)
            return all(Base.splat($f), zip(colnames1, colnames2)) &&
                   all(Base.splat($f), zip(vals1, vals2))
        end
    end
end

#### custom tabular show methods

function Base.show(io::IO, mime::MIME"text/plain", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end
function Base.show(io::IO, mime::MIME"text/html", stats::SummaryStats; kwargs...)
    return _show(io, mime, stats; kwargs...)
end

function _show(io::IO, mime::MIME, stats::SummaryStats; kwargs...)
    data = parent(stats)
    rhat_formatter = _prettytables_rhat_formatter(data)
    extra_formatters = rhat_formatter === nothing ? () : (rhat_formatter,)
    return _show_prettytable(
        io,
        mime,
        data;
        title=stats.name,
        row_labels=Tables.getcolumn(stats, :parameter),
        extra_formatters,
        kwargs...,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
function Tables.columnnames(s::SummaryStats)
    data_cols = Tables.columnnames(parent(s))
    data_cols isa Tuple && return (:parameter, data_cols...)
    return collect(Iterators.flatten(((:parameter,), data_cols)))
end
function Tables.getcolumn(stats::SummaryStats, i::Int)
    i == 1 && return stats.parameter_names
    return Tables.getcolumn(parent(stats), i - 1)
end
function Tables.getcolumn(stats::SummaryStats, nm::Symbol)
    nm === :parameter && return stats.parameter_names
    return Tables.getcolumn(parent(stats), nm)
end
function Tables.schema(s::SummaryStats)
    data_schema = Tables.schema(parent(s))
    data_schema === nothing && return nothing
    T = eltype(s.parameter_names)
    if data_schema isa Tables.Schema{Nothing,Nothing}
        return Tables.Schema([:parameter; data_schema.names], [T; data_schema.types])
    else
        return Tables.Schema((:parameter, data_schema.names...), (T, data_schema.types...))
    end
end

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
julia> using Statistics, StatsBase

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

Compute the summary stats focusing on `Statistics.median`:

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
    length(var_names) == size(data, 3) || throw(
        DimensionMismatch(
            "length $(length(var_names)) of `var_names` does not match number of parameters $(size(data, 3)) in `data`.",
        ),
    )
    names_and_funs = map(_fun_and_name, stats_funs_and_names)
    fnames = map(first, names_and_funs)
    _check_function_names(fnames)
    funs = map(last, names_and_funs)
    return SummaryStats(name, _summarize(data, funs, fnames), var_names)
end

function _check_function_names(fnames)
    for name in fnames
        name === nothing && continue
        if name === nothing || name isa Symbol || name isa Tuple{Symbol,Vararg{Symbol}}
            continue
        end
        throw(ArgumentError("Function name must be a symbol or a tuple of symbols."))
    end
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
        (:mean, :std) => StatsBase.mean_and_std ∘ _skipmissing,
        hdi_names => x -> hdi(_cskipmissing(x); prob=prob_interval),
    )
end
function default_stats(
    ::typeof(Statistics.median); prob_interval::Real=DEFAULT_INTERVAL_PROB, kwargs...
)
    eti_names = map(Symbol, _prob_interval_to_strings("eti", prob_interval))
    prob_tail = (1 - prob_interval) / 2
    p = (prob_tail, 1 - prob_tail)
    return (
        :median => Statistics.median ∘ _skipmissing,
        :mad => StatsBase.mad ∘ _skipmissing,
        eti_names => Base.Fix2(Statistics.quantile, p) ∘ _skipmissing ∘ vec,
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

# aggressive constprop allows summarize to be type-inferrable when called by
# another function

@constprop :aggressive function _summarize(data::AbstractArray{<:Any,3}, funs, fun_names)
    return merge(map(fun_names, funs) do fname, f
        return _map_over_params(fname, f, data)
    end...)
end

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
_map_paramslices(f::typeof(_ess_median), x) = f(x)
_map_paramslices(f::typeof(_ess_tail), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess_rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.mcse), x) = f(x)
_map_paramslices(f::typeof(_mcse_std), x) = f(x)
_map_paramslices(f::typeof(_mcse_median), x) = f(x)
