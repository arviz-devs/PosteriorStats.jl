using PosteriorStats
using Random
using Test

Random.seed!(97)

@testset "PosteriorStats" begin
    include("helpers.jl")
    include("utils.jl")
    include("show_prettytable.jl")
    include("kde.jl")
    include("density_estimation.jl")
    include("eti.jl")
    include("hdi.jl")
    include("loo.jl")
    include("pointwise_loglikelihoods.jl")
    include("loo_pit.jl")
    include("model_weights.jl")
    include("compare.jl")
    include("r2_score.jl")
    include("summarize.jl")
end
