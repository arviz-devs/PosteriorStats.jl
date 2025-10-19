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
struct SummaryStats{D,L<:Union{Nothing,AbstractVector},N<:AbstractString}
    data::D
    labels::L
    name::N
end

function SummaryStats(
    data; labels::Union{AbstractVector,Nothing}=nothing, name::AbstractString="SummaryStats"
)
    _coltable = Tables.columntable(data)
    if labels !== nothing
        length(labels) == Tables.rowcount(data) || throw(
            DimensionMismatch(
                "length $(length(labels)) of `labels` does not match number of rows $(Tables.rowcount(data)) in `data`.",
            ),
        )
    end
    if haskey(_coltable, LABEL_COLUMN_NAME)
        labels === nothing || throw(
            ArgumentError(
                "Either `labels` or a column named `$(LABEL_COLUMN_NAME)` may be provided, but not both.",
            ),
        )
        data_colnames = filter(k -> k !== LABEL_COLUMN_NAME, keys(_coltable))
        data_cols = NamedTuple{data_colnames}(_coltable)
        _labels = _coltable[LABEL_COLUMN_NAME]
        return SummaryStats(data_cols, _labels, name)
    end
    return SummaryStats(_coltable, labels, name)
end

# forward key interfaces from its parent
Base.parent(stats::SummaryStats) = getfield(stats, :data)
Base.keys(stats::SummaryStats) = Tables.columnnames(stats)
Base.haskey(stats::SummaryStats, nm::Symbol) = nm ∈ keys(stats)
Base.length(stats::SummaryStats) = length(parent(stats)) + 1
Base.getindex(stats::SummaryStats, i::Union{Int,Symbol}) = Tables.getcolumn(stats, i)
Base.iterate(stats::SummaryStats) = (_labels(stats), 2)
function Base.iterate(stats::SummaryStats, i::Int)
    state = iterate(parent(stats), i - 1)
    state === nothing && return nothing
    return (state[1], state[2] + 1)
end
function Base.merge(stats::SummaryStats, other_stats::SummaryStats...)
    isempty(other_stats) && return stats
    stats_all = (stats, other_stats...)
    stats_last = last(stats_all)
    return SummaryStats(
        merge(map(parent, stats_all)...),
        getfield(stats_last, :labels),
        getfield(stats_last, :name),
    )
end
for f in (:(==), :isequal)
    @eval begin
        function Base.$(f)(stats::SummaryStats, other_stats::SummaryStats)
            return $(f)(_labels(stats), _labels(other_stats)) &&
                   $(f)(parent(stats), parent(other_stats))
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
        row_labels=_labels(stats),
        extra_formatters,
        kwargs...,
    )
end

#### Tables interface as column table

_labels(s::SummaryStats) = getfield(s, :labels)
_labels(s::SummaryStats{<:Any,Nothing}) = eachindex(values(parent(s))...)

Tables.istable(::Type{<:SummaryStats}) = true
Tables.columnaccess(::Type{<:SummaryStats}) = true
Tables.columns(s::SummaryStats) = s
Tables.columnnames(s::SummaryStats) = (LABEL_COLUMN_NAME, Tables.columnnames(parent(s))...)
function Tables.getcolumn(stats::SummaryStats, i::Int)
    i == 1 && return _labels(stats)
    return Tables.getcolumn(parent(stats), i - 1)
end
function Tables.getcolumn(stats::SummaryStats, nm::Symbol)
    nm === LABEL_COLUMN_NAME && return _labels(stats)
    return Tables.getcolumn(parent(stats), nm)
end
function Tables.schema(s::SummaryStats)
    labels = _labels(s)
    sch = Tables.schema(parent(s))
    return Tables.Schema((LABEL_COLUMN_NAME, sch.names...), (eltype(labels), sch.types...))
end

IteratorInterfaceExtensions.isiterable(::SummaryStats) = true
function IteratorInterfaceExtensions.getiterator(s::SummaryStats)
    return Tables.datavaluerows(Tables.columntable(s))
end

TableTraits.isiterabletable(::SummaryStats) = true
