const LABEL_COLUMN_NAME = :label

const _DEFAULT_SUMMARY_STATS_KIND_DOCSTRING = """
- `kind::Symbol`: The named collection of summary statistics to be computed:
    + `:all`: Everything in `:stats` and `:diagnostics`
    + `:stats`: `mean`, `std`, `<ci>`
    + `:diagnostics`: `ess_tail`, `ess_bulk`, `rhat`, `mcse_mean`, `mcse_std`
    + `:all_median`: Everything in `:stats_median` and `:diagnostics_median`
    + `:stats_median`: `median`, `mad`, `<ci>`
    + `:diagnostics_median`: `ess_median`, `ess_tail`, `rhat`, `mcse_median`
"""
const _DEFAULT_SUMMARY_STATS_CI_DOCSTRING = """
- `ci_fun=$(default_ci_fun())`: The function to compute the credible interval `<ci>`, if
    any. Supported options are [`eti`](@ref) and [`hdi`](@ref). CI column name is
    `<ci_fun><100*ci_prob>`.
- `ci_prob=$(default_ci_prob())`: The probability mass to be contained in the credible
    interval `<ci>`.
"""

"""
    struct SummaryStats

A container for a column table of values computed by [`summarize`](@ref).

This object implements the Tables and TableTraits column table interfaces. It has a custom
`show` method.

!!! note
    `SummaryStats` behaves like an `OrderedDict` of columns, where the columns can be
    accessed using either `Symbol`s or a 1-based integer index. However, this interface
    is not part of the public API and may change in the future. We recommend using it
    only interactively.

# Constructor

    SummaryStats(data; name="SummaryStats"[, labels])

Construct a `SummaryStats` from tabular `data`.

`data` must implement the Tables interface. If it contains a column `$(LABEL_COLUMN_NAME)`,
this will be used for the row labels or will be replaced with the `labels` if provided.

# Keywords

- `name::AbstractString`: The name of the collection of summary statistics, used as the
    table title in display.
- `labels::AbstractVector`: The names of the parameters in `data`, used as row labels in
    display. If not provided, then the column `$(LABEL_COLUMN_NAME)` in `data` will be
    used if it exists. Otherwise, the parameter names will be numeric indices.
"""
struct SummaryStats{D,N<:AbstractString}
    data::D
    name::N
    function SummaryStats(data, name::N) where {N<:AbstractString}
        _coltable = Tables.columntable(data)
        # define default parameter names if not present, and set as first column
        if !haskey(_coltable, LABEL_COLUMN_NAME)
            data_cols = _coltable
            labels = Base.OneTo(Tables.rowcount(data))
        else
            data_colnames = filter(k -> k !== LABEL_COLUMN_NAME, keys(_coltable))
            data_cols = NamedTuple{data_colnames}(_coltable)
            labels = _coltable[LABEL_COLUMN_NAME]
        end
        coltable = merge((; LABEL_COLUMN_NAME => labels), data_cols)
        return new{typeof(coltable),N}(coltable, name)
    end
end

function SummaryStats(
    data; labels::Union{AbstractVector,Nothing}=nothing, name::AbstractString="SummaryStats"
)
    if labels !== nothing
        length(labels) == Tables.rowcount(data) || throw(
            DimensionMismatch(
                "length $(length(labels)) of `labels` does not match number of rows $(Tables.rowcount(data)) in `data`.",
            ),
        )
        data_with_varnames = merge(
            Tables.columntable(data), (; LABEL_COLUMN_NAME => labels)
        )
    else
        data_with_varnames = data
    end
    return SummaryStats(data_with_varnames, name)
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.keys(stats::SummaryStats) = map(Symbol, Tables.columnnames(stats))
Base.haskey(stats::SummaryStats, nm::Symbol) = nm ∈ keys(stats)
Base.length(stats::SummaryStats) = length(parent(stats))
Base.getindex(stats::SummaryStats, i::Union{Int,Symbol}) = Tables.getcolumn(stats, i)
Base.iterate(stats::SummaryStats, rest...) = iterate(parent(stats), rest...)
function Base.merge(stats::SummaryStats, other_stats::SummaryStats...)
    isempty(other_stats) && return stats
    stats_all = (stats, other_stats...)
    stats_last = last(stats_all)
    return SummaryStats(merge(map(parent, stats_all)...), stats_last.name)
end
for f in (:(==), :isequal)
    @eval begin
        function Base.$(f)(stats::SummaryStats, other_stats::SummaryStats)
            return $(f)(parent(stats), parent(other_stats))
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
    nt = parent(stats)
    data = nt[keys(nt)[2:end]]
    if isempty(default_precision_settings().show_printf)
        rhat_formatter = _prettytables_rhat_formatter(data)
        extra_formatters = rhat_formatter === nothing ? () : (rhat_formatter,)
    else
        extra_formatters = ()
    end
    return _show_prettytable(
        io,
        mime,
        data;
        title=stats.name,
        row_labels=Tables.getcolumn(stats, LABEL_COLUMN_NAME),
        extra_formatters,
        kwargs...,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
Tables.columnnames(s::SummaryStats) = Tables.columnnames(parent(s))
Tables.getcolumn(stats::SummaryStats, i::Int) = Tables.getcolumn(parent(stats), i)
Tables.getcolumn(stats::SummaryStats, nm::Symbol) = Tables.getcolumn(parent(stats), nm)
Tables.schema(s::SummaryStats) = Tables.schema(parent(s))

IteratorInterfaceExtensions.isiterable(::SummaryStats) = true
function IteratorInterfaceExtensions.getiterator(s::SummaryStats)
    return Tables.datavaluerows(Tables.columntable(s))
end

TableTraits.isiterabletable(::SummaryStats) = true

"""
    summarize(data; kind=:all,kwargs...) -> SummaryStats
    summarize(data, stats_funs...; kwargs...) -> SummaryStats

Compute summary statistics on each param in `data`.

# Arguments

- `data`: a 3D array of real samples with shape `(draws, chains, params)` or another object
    for which a `summarize` method is defined.
- `stats_funs`: a collection of functions that reduces a matrix with shape `(draws, chains)`
    to a scalar or a collection of scalars. Alternatively, an item in `stats_funs` may be a
    `Pair` of the form `name => fun` specifying the name to be used for the statistic or of
    the form `(name1, ...) => fun` when the function returns a collection. When the function
    returns a collection, the names in this latter format must be provided.

# Keywords

- `var_names`: a collection specifying the names of the parameters in `data`. If not
    provided, the names the indices of the parameter dimension in `data`.
- `name::String`: the name of the summary statistics, used as the table title in display.
$(_DEFAULT_SUMMARY_STATS_KIND_DOCSTRING)
- `kwargs`: additional keyword arguments passed to [`default_summary_stats`](@ref),
    including:
    $(replace(_DEFAULT_SUMMARY_STATS_CI_DOCSTRING, r"\n" => "\n    "))

See also [`SummaryStats`](@ref), [`default_summary_stats`](@ref)

# Extended Help

## Examples

Compute all summary statistics (the default):\

!!! details "Display precision"
    When an estimator and its MCSE are both computed, the MCSE is used to determine
    the number of significant digits that will be displayed.

```jldoctest summarize; setup = (using Random; Random.seed!(84))
julia> using Statistics, StatsBase

julia> x = randn(1000, 4, 3) .+ reshape(0:10:20, 1, 1, :);

julia> summarize(x)
SummaryStats
       mean   std  eti89          ess_tail  ess_bulk  rhat  mcse_mean  mcse_std
 1   0.0003  0.99  -1.57 .. 1.59      3567      3663  1.00      0.016     0.012
 2  10.02    0.99   8.47 .. 11.6      3841      3906  1.00      0.016     0.011
 3  19.98    0.99   18.4 .. 21.6      3892      3749  1.00      0.016     0.012
```

Compute just the default statistics with a 94% [HDI](@ref hdi), and provide the parameter
names:
```jldoctest summarize
julia> var_names=[:x, :y, :z];

julia> summarize(x; var_names, kind=:stats, ci_fun=hdi, ci_prob=0.94)
SummaryStats
         mean    std  hdi94
 x   0.000275  0.989  -1.92 .. 1.78
 y  10.0       0.988   8.17 .. 11.9
 z  20.0       0.988   18.1 .. 21.9
```

Compute [`Statistics.mean`](@extref), [`Statistics.std`](@extref) and the Monte Carlo
standard error (MCSE) of the mean estimate:
```jldoctest summarize
julia> summarize(x, mean, std, :mcse_mean => sem; name="Mean/Std")
Mean/Std
       mean    std  mcse_mean
 1   0.0003  0.989      0.016
 2  10.02    0.988      0.016
 3  19.98    0.988      0.016
```

Compute multiple [quantiles](@extref `Statistics.quantile`) simultaneously:

```jldoctest summarize
julia> percs = (5, 25, 50, 75, 95);

julia> summarize(x, Symbol.(:q, percs) => Base.Fix2(quantile, percs ./ 100))
SummaryStats
       q5     q25       q50     q75    q95
 1  -1.61  -0.668   0.00447   0.653   1.64
 2   8.41   9.34   10.0      10.7    11.6
 3  18.4   19.3    20.0      20.6    21.6
```

## Extending `summarize` to custom types

To support computing summary statistics from a custom object `MyType`, overload the
method `summarize(::MyType, stats_funs...; kwargs...)`, which should ultimately call
`summarize(::AbstractArray{<:Union{Real,Missing},3}, stats_funs...; other_kwargs...)`,
where `other_kwargs` are the keyword arguments passed to `summarize`.
"""
function summarize end

Base.@constprop :aggressive function summarize(
    data::AbstractArray{<:Union{Real,Missing},3},
    stats_funs_and_names...;
    kind::Union{Symbol,Val}=:all,
    name::String="SummaryStats",
    var_names=axes(data, 3),
    kwargs...,
)
    length(var_names) == size(data, 3) || throw(
        DimensionMismatch(
            "length $(length(var_names)) of `var_names` does not match number of parameters $(size(data, 3)) in `data`.",
        ),
    )
    if isempty(stats_funs_and_names)
        return _summarize(data, default_summary_stats(kind; kwargs...), name, var_names)
    else
        return _summarize(data, stats_funs_and_names, name, var_names)
    end
end
Base.@constprop :aggressive function _summarize(
    data::AbstractArray, stats_funs_and_names, name::String, var_names
)
    names_and_funs = map(_fun_and_name, stats_funs_and_names)
    fnames = map(first, names_and_funs)
    _check_function_names(fnames)
    funs = map(last, names_and_funs)
    return SummaryStats(_summarize(data, funs, fnames); labels=var_names, name)
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
    default_summary_stats(kind::Symbol=:all; kwargs...)

Return a collection of stats functions based on the named preset `kind`.

These functions are then passed to [`summarize`](@ref).

# Arguments

$(_DEFAULT_SUMMARY_STATS_KIND_DOCSTRING)

# Keywords

$(_DEFAULT_SUMMARY_STATS_CI_DOCSTRING)
"""
function default_summary_stats(kind::Symbol=:all; kwargs...)
    return default_summary_stats(Val(kind); kwargs...)
end
function default_summary_stats(::Val{:all}; kwargs...)
    return (_default_stats(; kwargs...)..., _default_diagnostics(; kwargs...)...)
end
default_summary_stats(::Val{:stats}; kwargs...) = _default_stats(; kwargs...)
default_summary_stats(::Val{:diagnostics}; kwargs...) = _default_diagnostics(; kwargs...)
function default_summary_stats(::Val{:all_median}; kwargs...)
    focus = Statistics.median
    return (_default_stats(focus; kwargs...)..., _default_diagnostics(focus; kwargs...)...)
end
function default_summary_stats(::Val{:stats_median}; kwargs...)
    return _default_stats(Statistics.median; kwargs...)
end
function default_summary_stats(::Val{:diagnostics_median}; kwargs...)
    return _default_diagnostics(Statistics.median; kwargs...)
end
function default_summary_stats(::Val{kind}; kwargs...) where {kind}
    throw(
        ArgumentError(
            "Invalid kind: $kind. Must be one of [:all, :stats, " *
            ":diagnostics, :all_median, :stats_median, :diagnostics_median].",
        ),
    )
end

_default_stats(; kwargs...) = _default_stats(Statistics.mean; kwargs...)
function _default_stats(::typeof(Statistics.mean); kwargs...)
    return (
        (:mean, :std) => StatsBase.mean_and_std ∘ _skipmissing, _interval_stat(; kwargs...)
    )
end
function _default_stats(::typeof(Statistics.median); kwargs...)
    return (
        :median => Statistics.median ∘ _skipmissing,
        :mad => StatsBase.mad ∘ _skipmissing,
        _interval_stat(; kwargs...),
    )
end

function _interval_stat(; ci_fun=default_ci_fun(), ci_prob=default_ci_prob(), kwargs...)
    ci_name = Symbol(_fname(ci_fun), _prob_to_string(ci_prob))
    return ci_name => FixKeywords(ci_fun; prob=ci_prob) ∘ _cskipmissing
end

_default_diagnostics(; kwargs...) = _default_diagnostics(Statistics.mean; kwargs...)
function _default_diagnostics(::typeof(Statistics.mean); kwargs...)
    return (
        :ess_tail => FixKeywords(MCMCDiagnosticTools.ess; kind=:tail),
        (:ess_bulk, :rhat) => MCMCDiagnosticTools.ess_rhat,
        :mcse_mean => MCMCDiagnosticTools.mcse,
        :mcse_std => FixKeywords(MCMCDiagnosticTools.mcse; kind=Statistics.std),
    )
end
function _default_diagnostics(::typeof(Statistics.median); kwargs...)
    return (
        :ess_median => FixKeywords(MCMCDiagnosticTools.ess; kind=Statistics.median),
        :ess_tail => FixKeywords(MCMCDiagnosticTools.ess; kind=:tail),
        MCMCDiagnosticTools.rhat,
        :mcse_median => FixKeywords(MCMCDiagnosticTools.mcse; kind=Statistics.median),
    )
end

_prob_to_string(prob; digits=2) = replace(string(round(100 * prob; digits)), r"\.0+$" => "")

# aggressive constprop allows summarize to be type-inferrable when called by
# another function

Base.@constprop :aggressive function _summarize(
    data::AbstractArray{<:Any,3}, funs, fun_names
)
    return merge(map(fun_names, funs) do fname, f
        return _map_over_params(fname, f, data)
    end...)
end

Base.@constprop :aggressive function _map_over_params(fname, f, data)
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

_fname(f) = nameof(f)

# functions that have a 3D array method
_map_paramslices(f, x) = map(f, eachslice(x; dims=3))
_map_paramslices(f::typeof(MCMCDiagnosticTools.ess_rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.rhat), x) = f(x)
_map_paramslices(f::typeof(MCMCDiagnosticTools.mcse), x) = f(x)
_map_paramslices(f::FixKeywords{typeof(MCMCDiagnosticTools.ess)}, x) = f.f(x; f.kwargs...)
_map_paramslices(f::FixKeywords{typeof(MCMCDiagnosticTools.mcse)}, x) = f.f(x; f.kwargs...)
