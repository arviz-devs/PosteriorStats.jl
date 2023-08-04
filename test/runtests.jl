using PosteriorStats
using Random
using Test

Random.seed!(97)

@testset "PosteriorStats" begin
    include("helpers.jl")
    include("utils.jl")
    include("hdi.jl")
    include("loo.jl")
    include("loo_pit.jl")
    include("waic.jl")
    include("model_weights.jl")
    include("compare.jl")
    include("r2_score.jl")
end
