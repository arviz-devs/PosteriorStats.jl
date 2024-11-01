using PosteriorStats
using Test

function test_density_estimation_interface(method, x; dx=1)
    @testset "$method implementation" begin
        @test method isa PosteriorStats.DensityEstimationMethod
        @testset "bins_and_probs" begin
            bins_probs = PosteriorStats.bins_and_probs(method, x)
            @test bins_probs isa Tuple{<:AbstractVector{<:Real},<:AbstractVector{<:Real}}
            bins, probs = bins_probs
            @testset "regularly spaced ordered bins, gaps allowed" begin
                @test issorted(bins)
                if bins isa AbstractRange
                    @test true
                else
                    @test all(isinteger, diff(bins) / dx)
                end
            end
            @testset "probabilities are valid" begin
                @test all(≥(0), probs)
                @test sum(probs) ≈ 1
            end
        end
        method isa PosteriorStats.DiscreteDensityEstimation || @testset "density_at" begin
            density = PosteriorStats.density_at(method, x)
            @test density isa AbstractVector{<:Real}
            @test length(density) == length(x)
            @test all(≥(0), density)
        end
    end
end

@testset "density estimation" begin
    @testset "KDEstimation" begin
        test_density_estimation_interface(
            PosteriorStats.DiscreteDensityEstimation(), rand(1:10, 100)
        )
        test_density_estimation_interface(PosteriorStats.HistogramEstimation(), randn(100))
        test_density_estimation_interface(PosteriorStats.KDEstimation(), randn(100))
    end
end
