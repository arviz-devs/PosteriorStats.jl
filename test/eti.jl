using IntervalSets
using OffsetArrays
using PosteriorStats
using Statistics
using Test

@testset "eti/eti!" begin
    @testset "AbstractVecOrMat" begin
        @testset for sz in (100, 1_000, (1_000, 2)),
            prob in (0.7, 0.76, 0.8, 0.88),
            T in (Float32, Float64, Int64)

            n = prod(sz)
            S = Base.promote_eltype(one(T), prob)
            x = T <: Integer ? rand(T(1):T(30), n) : randn(T, n)
            r = @inferred eti(x; prob)
            @test r isa ClosedInterval{S}
            if !(T <: Integer)
                l, u = IntervalSets.endpoints(r)
                frac_in_interval = mean(∈(r), x)
                @test frac_in_interval ≈ prob
                @test count(<(l), x) == count(>(u), x)
            end

            @test eti!(copy(x); prob) == r
        end
    end

    @testset "edge cases and errors" begin
        @testset "NaNs returned if contains NaNs" begin
            x = randn(1000)
            x[3] = NaN
            @test isequal(eti(x), NaN .. NaN)
        end

        @testset "errors for empty array" begin
            x = Float64[]
            @test_throws ArgumentError eti(x)
        end

        @testset "errors for 0-dimensional array" begin
            x = fill(1.0)
            @test_throws ArgumentError eti(x)
        end

        @testset "test errors when prob is not in (0, 1)" begin
            x = randn(1_000)
            @testset for prob in (0, 1, -0.1, 1.1, NaN)
                @test_throws DomainError eti(x; prob)
            end
        end
    end

    @testset "AbstractArray consistent with AbstractVector" begin
        @testset for sz in ((100, 2), (100, 2, 3), (100, 2, 3, 4)),
            prob in (0.72, 0.81),
            T in (Float32, Float64, Int64)

            x = T <: Integer ? rand(T(1):T(30), sz) : randn(T, sz)
            r = @inferred eti(x; prob)
            if ndims(x) == 2
                @test r isa ClosedInterval
                @test r == eti(vec(x); prob)
            else
                @test r isa Array{<:ClosedInterval,ndims(x) - 2}
                r_slices = dropdims(
                    mapslices(x -> eti(x; prob), x; dims=(1, 2)); dims=(1, 2)
                )
                @test r == r_slices
            end

            @test eti!(copy(x); prob) == r
        end
    end

    @testset "OffsetArray" begin
        @testset for n in (100, 1_000), prob in (0.732, 0.864), T in (Float32, Float64)
            x = randn(T, (n, 2, 3, 4))
            xoff = OffsetArray(x, (-1, 2, -3, 4))
            r = eti(x; prob)
            roff = @inferred eti(xoff; prob)
            @test roff isa OffsetMatrix{<:ClosedInterval}
            @test axes(roff) == (axes(xoff, 3), axes(xoff, 4))
            @test collect(roff) == r
        end
    end
end
