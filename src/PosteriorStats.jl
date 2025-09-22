module PosteriorStats

using Distributions: Distributions
using DocStringExtensions: FIELDS, FUNCTIONNAME, TYPEDEF, TYPEDFIELDS, SIGNATURES
using FFTW: FFTW
using IrrationalConstants: sqrthalfπ, sqrtπ, sqrt2, logπ, log2π
using IteratorInterfaceExtensions: IteratorInterfaceExtensions
using KernelDensity: KernelDensity
using LinearAlgebra: LinearAlgebra, mul!, norm, normalize
using LogExpFunctions: LogExpFunctions
using Markdown: @doc_str
using MCMCDiagnosticTools: MCMCDiagnosticTools
using Optim: Optim
using OrderedCollections: OrderedCollections
using PrettyTables: PrettyTables
using Preferences: Preferences
using Printf: Printf
using PDMats: PDMats
using PSIS: PSIS, PSISResult, psis, psis!
using Random: Random
using Roots: Roots
using Setfield: Setfield
using SpecialFunctions: SpecialFunctions
using Statistics: Statistics
using StatsBase: StatsBase
using Tables: Tables
using TableTraits: TableTraits

using IntervalSets

# PSIS
export PSIS, PSISResult, psis, psis!

# LOO-CV
export AbstractELPDResult, PSISLOOResult, WAICResult
export elpd_estimates, information_criterion, loo, waic

# Model weighting and comparison
export AbstractModelWeightsMethod, BootstrappedPseudoBMA, PseudoBMA, Stacking, model_weights
export ModelComparisonResult, compare

# Summary statistics
export SummaryStats, summarize
export default_summary_stats

# Credible intervals
export eti, eti!, hdi, hdi!

# Others
export loo_pit, r2_score

const DEFAULT_CI_PROB = 0.94
const INFORMATION_CRITERION_SCALES = (deviance=-2, log=1, negative_log=-1)

include("utils.jl")
include("preferences.jl")
include("show_prettytable.jl")
include("density_estimation.jl")
include("kde.jl")
include("eti.jl")
include("hdi.jl")
include("elpdresult.jl")
include("pointwise_loglikelihoods.jl")
include("loo.jl")
include("waic.jl")
include("model_weights.jl")
include("compare.jl")
include("loo_pit.jl")
include("r2_score.jl")
include("summarize.jl")

end  # module
