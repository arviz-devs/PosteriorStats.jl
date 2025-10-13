const LABEL_COLUMN_NAME = :label

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
    rhat_formatter = _prettytables_rhat_formatter(data)
    extra_formatters = rhat_formatter === nothing ? () : (rhat_formatter,)
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
