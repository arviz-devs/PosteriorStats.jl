"""
    compare(models; kwargs...) -> ModelComparisonResult

Compare models based on their expected log pointwise predictive density (ELPD).

The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
cross-validation (LOO) or using the widely applicable information criterion (WAIC).
[`loo`](@ref) is the default and recommended method for computing the ELPD. For more theory,
see [Spiegelhalter2002](@citet).

# Arguments

  - `models`: a `Tuple`, `NamedTuple`, or `AbstractVector` whose values are either
    [`AbstractELPDResult`](@ref) entries or any argument to `elpd_method`.

# Keywords

  - `weights_method::AbstractModelWeightsMethod=Stacking()`: the method to be used to weight
    the models. See [`model_weights`](@ref) for details
  - `elpd_method=loo`: a method that computes an `AbstractELPDResult` from an argument in
    `models`.
  - `sort::Bool=true`: Whether to sort models by decreasing ELPD.

# Returns

  - [`ModelComparisonResult`](@ref): A container for the model comparison results. The
    fields contain a similar collection to `models`.

# Examples

Compare the centered and non centered models of the eight school problem using the defaults:
[`loo`](@ref) and [`Stacking`](@ref) weights. A custom `myloo` method formates the inputs
as expected by [`loo`](@ref).

```jldoctest compare; filter = [r"└.*", r"(\\d+\\.\\d{3})\\d*" => s"\\1"]
julia> using ArviZExampleData

julia> models = (
           centered=load_example_data("centered_eight"),
           non_centered=load_example_data("non_centered_eight"),
       );

julia> function myloo(idata)
           log_like = PermutedDimsArray(idata.log_likelihood.obs, (2, 3, 1))
           return loo(log_like)
       end;

julia> mc = compare(models; elpd_method=myloo)
┌ Warning: 1 parameters had Pareto shape values 0.7 < k ≤ 1. Resulting importance sampling estimates are likely to be unstable.
└ @ PSIS ~/.julia/packages/PSIS/...
ModelComparisonResult with Stacking weights
               rank  elpd  se_elpd  elpd_diff  se_elpd_diff  weight    p  se_p ⋯
 non_centered     1   -31      1.5       0            0.0       1.0  0.9  0.32 ⋯
 centered         2   -31      1.4       0.03         0.061     0.0  0.9  0.33 ⋯
julia> mc.weight |> pairs
pairs(::NamedTuple) with 2 entries:
  :non_centered => 1.0
  :centered     => 3.50546e-31
```

Compare the same models from pre-computed PSIS-LOO results and computing
[`BootstrappedPseudoBMA`](@ref) weights:

```jldoctest compare; setup = :(using Random; Random.seed!(23))
julia> elpd_results = mc.elpd_result;

julia> compare(elpd_results; weights_method=BootstrappedPseudoBMA())
ModelComparisonResult with BootstrappedPseudoBMA weights
               rank  elpd  se_elpd  elpd_diff  se_elpd_diff  weight    p  se_p ⋯
 non_centered     1   -31      1.5       0            0.0      0.51  0.9  0.32 ⋯
 centered         2   -31      1.4       0.03         0.061    0.49  0.9  0.33 ⋯
```

# References

- [Spiegelhalter2002](@cite) Spiegelhalter et al. J. R. Stat. Soc. B 64 (2002)
"""
function compare(
    inputs;
    weights_method::AbstractModelWeightsMethod=Stacking(),
    elpd_method=loo,
    model_names=_indices(inputs),
    sort::Bool=true,
)
    length(model_names) === length(inputs) ||
        throw(ArgumentError("Length of `model_names` must match length of `inputs`"))
    elpd_results = map(Base.Fix1(_maybe_elpd_results, elpd_method), inputs)
    weights = model_weights(weights_method, elpd_results)
    perm = _sortperm(elpd_results; by=x -> elpd_estimates(x).elpd, rev=true)
    i_elpd_max = first(perm)
    elpd_max_i = elpd_estimates(elpd_results[i_elpd_max]; pointwise=true).elpd
    se_elpd_diff_and = map(elpd_results) do r
        elpd_diff_j = similar(elpd_max_i)
        # workaround for named dimension packages that check dimension names are exact, for
        # cases where dimension names differ
        map!(-, elpd_diff_j, elpd_max_i, elpd_estimates(r; pointwise=true).elpd)
        return _sum_and_se(elpd_diff_j)
    end
    elpd_diff = map(first, se_elpd_diff_and)
    se_elpd_diff = map(last, se_elpd_diff_and)
    rank = _assimilar(elpd_results, (1:length(elpd_results))[perm])
    result = ModelComparisonResult(
        model_names, rank, elpd_diff, se_elpd_diff, weights, elpd_results, weights_method
    )
    sort || return result
    return _permute(result, perm)
end

_maybe_elpd_results(elpd_method, x::AbstractELPDResult; kwargs...) = x
function _maybe_elpd_results(elpd_method, x; kwargs...)
    elpd_result = elpd_method(x; kwargs...)
    elpd_result isa AbstractELPDResult && return elpd_result
    throw(
        ErrorException(
            "Return value of `elpd_method` must be an `AbstractELPDResult`, not `$(typeof(elpd_result))`.",
        ),
    )
end

"""
    ModelComparisonResult

Result of model comparison using ELPD.

This struct implements the Tables and TableTraits interfaces.

Each field returns a collection of the corresponding entry for each model:
$(FIELDS)
"""
struct ModelComparisonResult{E,N,R,W,ER,M}
    "Names of the models, if provided."
    name::N
    "Ranks of the models (ordered by decreasing ELPD)"
    rank::R
    "ELPD of a model subtracted from the largest ELPD of any model"
    elpd_diff::E
    "Standard error of the ELPD difference"
    se_elpd_diff::E
    "Model weights computed with `weights_method`"
    weight::W
    """`AbstactELPDResult`s for each model, which can be used to access useful stats like
    ELPD estimates, pointwise estimates, and Pareto shape values for PSIS-LOO"""
    elpd_result::ER
    "Method used to compute model weights with [`model_weights`](@ref)"
    weights_method::M
end

#### custom tabular show methods

function Base.show(io::IO, mime::MIME"text/plain", r::ModelComparisonResult; kwargs...)
    return _show(io, mime, r; kwargs...)
end
function Base.show(io::IO, mime::MIME"text/html", r::ModelComparisonResult; kwargs...)
    return _show(io, mime, r; kwargs...)
end

function _show(io::IO, mime::MIME, r::ModelComparisonResult; kwargs...)
    row_labels = collect(r.name)
    cols = Tables.columnnames(r)[2:end]
    table = NamedTuple{cols}(Tables.columntable(r))

    weights_method_name = _typename(r.weights_method)
    weights = table.weight
    digits_weights = ceil(Int, -log10(maximum(weights))) + 1
    weight_formatter = PrettyTables.ft_printf(
        "%.$(digits_weights)f", findfirst(==(:weight), cols)
    )
    return _show_prettytable(
        io,
        mime,
        table;
        title="ModelComparisonResult with $(weights_method_name) weights",
        row_labels,
        extra_formatters=(weight_formatter,),
        kwargs...,
    )
end

function _permute(r::ModelComparisonResult, perm)
    return ModelComparisonResult(
        (_permute(getfield(r, k), perm) for k in fieldnames(typeof(r))[1:(end - 1)])...,
        r.weights_method,
    )
end

#### Tables interface as column table

Tables.istable(::Type{<:ModelComparisonResult}) = true
Tables.columnaccess(::Type{<:ModelComparisonResult}) = true
Tables.columns(r::ModelComparisonResult) = r
function Tables.columnnames(::ModelComparisonResult)
    return (:name, :rank, :elpd, :se_elpd, :elpd_diff, :se_elpd_diff, :weight, :p, :se_p)
end
function Tables.getcolumn(r::ModelComparisonResult, i::Int)
    return Tables.getcolumn(r, Tables.columnnames(r)[i])
end
function Tables.getcolumn(r::ModelComparisonResult, nm::Symbol)
    nm ∈ fieldnames(typeof(r)) && return getfield(r, nm)
    if nm ∈ (:elpd, :se_elpd, :p, :se_p)
        return map(e -> getproperty(elpd_estimates(e), nm), r.elpd_result)
    end
    throw(ArgumentError("Unrecognized column name $nm"))
end
Tables.rowaccess(::Type{<:ModelComparisonResult}) = true
Tables.rows(r::ModelComparisonResult) = Tables.rows(Tables.columntable(r))

IteratorInterfaceExtensions.isiterable(::ModelComparisonResult) = true
function IteratorInterfaceExtensions.getiterator(r::ModelComparisonResult)
    return Tables.datavaluerows(Tables.columntable(r))
end

TableTraits.isiterabletable(::ModelComparisonResult) = true
