# Utilities for displaying tables using PrettyTables.jl

# formatting functions for special columns
# see https://ronisbr.github.io/PrettyTables.jl/stable/man/usage/#Formatters

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

function _prettytables_integer_formatter(data, columns::AbstractVector{Int}=Int[])
    if isempty(columns)
        return (v, _, _) -> begin
            (v isa Integer) || return v
            return string(v)
        end
    else
        return (v, _, j) -> begin
            (v isa Integer) || return v
            col = findfirst(==(j), columns)
            col === nothing && return v
            return string(v)
        end
    end
end

function _prettytables_se_formatters(data; sigdigits_se=2)
    col_names = Tables.columnnames(data)
    pattern = r"^(?:mcse_|se_)(.*)|^(.*?)(?:_mcse|_se)$"
    formatters = Function[]
    se_cols_inds = Int[]
    for (idx_se_col, se_col) in enumerate(col_names)
        m = match(pattern, string(se_col))
        m === nothing && continue
        push!(se_cols_inds, idx_se_col)
        col = Symbol(something(m.captures[1], m.captures[2]))
        idx_col = findfirst(==(col), col_names)
        idx_col === nothing && continue
        push!(
            formatters,
            ft_printf_sigdigits_matching_se(Tables.getcolumn(data, se_col), [idx_col]),
        )
    end
    if !isempty(se_cols_inds)
        push!(formatters, ft_printf_sigdigits(sigdigits_se, se_cols_inds))
    end
    return formatters
end

function _prettytables_ess_formatter(data)
    cols = findall(_is_ess_label, Tables.columnnames(data))
    isempty(cols) && return nothing
    return PrettyTables.ft_printf("%d", cols)
end

function _prettytables_rhat_formatter(data)
    col_names = Tables.columnnames(data)
    cols = findall(x -> (x === :rhat || startswith(string(x), "rhat_")), col_names)
    isempty(cols) && return nothing
    return PrettyTables.ft_printf("%.2f", cols)
end

function _default_prettytables_formatters(data; sigdigits_se=2, sigdigits_default=3)
    formatters = Union{Function,Nothing}[]
    push!(formatters, _prettytables_integer_formatter(data))
    append!(formatters, _prettytables_se_formatters(data; sigdigits_se))
    push!(formatters, _prettytables_ess_formatter(data))
    push!(formatters, ft_printf_sigdigits(sigdigits_default))
    push!(formatters, ft_printf_sigdigits_interval(sigdigits_default))
    return filter(!isnothing, formatters)
end

# alignment functions for special columns

function _text_alignment(data)
    col_names = Tables.columnnames(data)
    return map(collect(col_names)) do col_name
        eltype(Tables.getcolumn(data, col_name)) <: Real ? :r : :l
    end
end

function _text_alignment_anchor_regex(data)
    alignment_anchor_regex = Dict{Int,Vector{Regex}}()
    for (i, k) in enumerate(Tables.columnnames(data))
        v = Tables.getcolumn(data, k)
        if eltype(v) <: Real && !(eltype(v) <: Integer) && !_is_ess_label(k)
            alignment_anchor_regex[i] = [r"\.", r"e", r"^NaN$", r"Inf$"]
        elseif eltype(v) <: IntervalSets.AbstractInterval
            alignment_anchor_regex[i] = [r"\.\."]
        end
    end
    return alignment_anchor_regex
end

# show a table with PrettyTables.jl with various mime types

function _show_prettytable(
    io::IO,
    data;
    sigdigits_se=2,
    sigdigits_default=3,
    extra_formatters=(),
    alignment=_text_alignment(data),
    show_subheader=false,
    vcrop_mode=:middle,
    show_omitted_cell_summary=true,
    row_label_alignment=:l,
    kwargs...,
)
    formatters = (
        extra_formatters...,
        _default_prettytables_formatters(data; sigdigits_se, sigdigits_default)...,
    )
    PrettyTables.pretty_table(
        io,
        data;
        show_subheader,
        vcrop_mode,
        show_omitted_cell_summary,
        row_label_alignment,
        formatters,
        alignment,
        kwargs...,
    )
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
    alignment_anchor_regex=_text_alignment_anchor_regex(data),
    alignment_anchor_fallback=:r,
    kwargs...,
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

_is_ess_label(k::Symbol) = (k === :ess) || startswith(string(k), "ess_")
