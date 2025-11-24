# Utilities for displaying tables using PrettyTables.jl

@static if pkgversion(PrettyTables).major == 2
    # temporarily support PrettyTables v2 until v3 is more broadly established in the ecosystem
    const IS_PRETTYTABLES_V2 = true
    const PRETTYTABLES_TEXT_FORMAT = (; hlines=:none, vlines=:none)
    const PRETTYTABLES_TEXT_STYLE = (; title_crayon=PrettyTables.Crayon())
    _prettytables_printf_formatter(fmt::String, cols) = PrettyTables.ft_printf(fmt, cols)
else
    const IS_PRETTYTABLES_V2 = false
    const PRETTYTABLES_TEXT_FORMAT = PrettyTables.TextTableFormat(;
        PrettyTables.@text__no_vertical_lines, PrettyTables.@text__no_horizontal_lines
    )
    const PRETTYTABLES_TEXT_STYLE = PrettyTables.TextTableStyle(;
        title=PrettyTables.Crayon()
    )
    _prettytables_printf_formatter(fmt::String, cols) = PrettyTables.fmt__printf(fmt, cols)
end

# formatting functions for special columns
# see https://ronisbr.github.io/PrettyTables.jl/stable/man/usage/#Formatters

# Use Printf to format real elements to the number of `sigdigits`.
function _prettytables_sigdigits_formatter(sigdigits::Int)
    return (v, _, _) -> begin
        v isa Real || return v
        return _printf_with_sigdigits(v, sigdigits)
    end
end
function _prettytables_sigdigits_formatter(sigdigits::Int, columns::AbstractVector{Int})
    isempty(columns) && return _prettytables_sigdigits_formatter(sigdigits)
    return (v, _, j) -> begin
        (v isa Real && j ∈ columns) || return v
        return _printf_with_sigdigits(v, sigdigits)
    end
end

# Use Printf to format interval elements to the number of `sigdigits`.
function _prettytables_interval_formatter(sigdigits::Int)
    return (v, _, _) -> begin
        v isa IntervalSets.AbstractInterval || return v
        tuple_string = map(Base.Fix2(_printf_with_sigdigits, sigdigits), extrema(v))
        return join(tuple_string, _interval_delimiter(v))
    end
end

function _interval_delimiter(x::IntervalSets.AbstractInterval)
    str = sprint(show, "text/plain", x)
    return occursin(" .. ", str) ? " .. " : ".."
end

function _prettytables_sigdigits_from_se_formatter(data, col::Int, se_col::Int; kwargs...)
    se_vals = Tables.getcolumn(data, se_col)
    return (v, i, j) -> begin
        (v isa Real && col == j && se_vals[i] isa Real) || return v
        sigdigits = sigdigits_matching_se(v, se_vals[i]; kwargs...)
        return _printf_with_sigdigits(v, sigdigits)
    end
end

function _prettytables_integer_formatter(data)
    return (v, _, _) -> begin
        (v isa Integer) || return v
        return string(v)
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
        col ∈ col_names || continue
        idx_col = Tables.columnindex(data, col)
        idx_col == 0 && continue
        push!(
            formatters, _prettytables_sigdigits_from_se_formatter(data, idx_col, idx_se_col)
        )
    end
    if !isempty(se_cols_inds)
        push!(formatters, _prettytables_sigdigits_formatter(sigdigits_se, se_cols_inds))
    end
    return formatters
end

function _prettytables_ess_formatter(data)
    cols = findall(_is_ess_label, Tables.columnnames(data))
    isempty(cols) && return nothing
    return _prettytables_printf_formatter("%d", cols)
end

function _prettytables_rhat_formatter(data)
    col_names = Tables.columnnames(data)
    cols = findall(x -> (x === :rhat || startswith(string(x), "rhat_")), col_names)
    isempty(cols) && return nothing
    return _prettytables_printf_formatter("%.2f", cols)
end

function _prettytables_default_formatters(data; sigdigits_se=2, sigdigits_default=3)
    formatters = Union{Function,Nothing}[]
    push!(formatters, _prettytables_integer_formatter(data))
    append!(formatters, _prettytables_se_formatters(data; sigdigits_se))
    push!(formatters, _prettytables_ess_formatter(data))
    push!(formatters, _prettytables_sigdigits_formatter(sigdigits_default))
    push!(formatters, _prettytables_interval_formatter(sigdigits_default))
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
    alignment_anchor_regex = Pair{Int,Vector{Regex}}[]
    for (i, k) in enumerate(Tables.columnnames(data))
        v = Tables.getcolumn(data, k)
        patterns = if eltype(v) <: Real && !(eltype(v) <: Integer) && !_is_ess_label(k)
            [r"\.", r"e", r"^NaN$", r"Inf$"]
        elseif eltype(v) <: IntervalSets.AbstractInterval
            [r"\.\."]
        else
            continue
        end
        push!(alignment_anchor_regex, i => patterns)
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
    show_first_column_label_only=true,
    vertical_crop_mode=:middle,
    row_label_column_alignment=:l,
    kwargs...,
)
    formatters = [
        extra_formatters...,
        _prettytables_default_formatters(data; sigdigits_se, sigdigits_default)...,
    ]
    IS_PRETTYTABLES_V2 && return PrettyTables.pretty_table(
        io,
        data;
        alignment,
        formatters=Tuple(formatters),
        merge(
            (;
                show_subheader=(!show_first_column_label_only),
                vcrop_mode=vertical_crop_mode,
                row_label_alignment=row_label_column_alignment,
            ),
            kwargs,
        )...,
    )
    PrettyTables.pretty_table(
        io,
        data;
        show_first_column_label_only,
        vertical_crop_mode,
        row_label_column_alignment,
        alignment,
        formatters,
        kwargs...,
    )
    return nothing
end

function _show_prettytable(
    io::IO,
    ::MIME"text/plain",
    data;
    style=PRETTYTABLES_TEXT_STYLE,
    table_format=PRETTYTABLES_TEXT_FORMAT,
    new_line_at_end=false,
    title_alignment=:l,
    alignment_anchor_regex=_text_alignment_anchor_regex(data),
    alignment_anchor_fallback=:r,
    kwargs...,
)
    IS_PRETTYTABLES_V2 && return _show_prettytable(
        io,
        data;
        backend=Val(:text),
        title_alignment,
        alignment_anchor_regex=Dict(alignment_anchor_regex),
        alignment_anchor_fallback,
        merge((; style..., table_format..., newline_at_end=new_line_at_end), kwargs)...,
    )
    return _show_prettytable(
        io,
        data;
        backend=:text,
        style,
        table_format,
        new_line_at_end,
        title_alignment,
        alignment_anchor_regex,
        alignment_anchor_fallback,
        kwargs...,
    )
end

function _show_prettytable(
    io::IO, ::MIME"text/html", data; minify=true, maximum_number_of_rows=25, kwargs...
)
    IS_PRETTYTABLES_V2 && return _show_prettytable(
        io,
        data;
        backend=Val(:html),
        minify,
        merge((; max_num_of_rows=maximum_number_of_rows), kwargs)...,
    )
    return _show_prettytable(
        io, data; backend=:html, minify, maximum_number_of_rows, kwargs...
    )
end

_is_ess_label(k::Symbol) = (k === :ess) || startswith(string(k), "ess_")
