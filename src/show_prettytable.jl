# Utilities for displaying tables using PrettyTables.jl

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
