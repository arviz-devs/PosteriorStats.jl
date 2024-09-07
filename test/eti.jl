using IntervalSets
using PosteriorStats
using Statistics
using Test

@testset "PosteriorStats.eti" begin
    @testset "AbstractVecOrMat" begin
        @testset for sz in (100, 1_000, (1_000, 2)),
            prob in (0.7, 0.76, 0.8, 0.88),
            T in (Float32, Float64)

            S = Base.promote_eltype(one(T), prob)
            n = prod(sz)
            x = T <: Integer ? rand(T(1):T(30), sz) : randn(T, sz)
            r = @inferred PosteriorStats.eti(x; prob)
            @test r isa ClosedInterval{S}
            l, u = IntervalSets.endpoints(r)
            frac_in_interval = mean(∈(r), x)
            @test frac_in_interval ≈ prob
            @test count(<(l), x) == count(>(u), x)
        end
    end

    @testset "edge cases and errors" begin
        @testset "NaNs returned if contains NaNs" begin
            x = randn(1000)
            x[3] = NaN
            @test isequal(PosteriorStats.eti(x), NaN .. NaN)
        end

        @testset "errors for empty array" begin
            x = Float64[]
            @test_throws ArgumentError PosteriorStats.eti(x)
        end

        @testset "test errors when prob is not in (0, 1)" begin
            x = randn(1_000)
            @testset for prob in (0, 1, -0.1, 1.1, NaN)
                @test_throws DomainError PosteriorStats.eti(x; prob)
            end
        end
    end
end
