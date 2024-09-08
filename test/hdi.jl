using IntervalSets
using OffsetArrays
using PosteriorStats
using Statistics
using Test

@testset "hdi/hdi!" begin
    @testset "AbstractVector" begin
        @testset for n in (10, 100, 1_000),
            prob in (1 / n, 0.5, 0.73, 0.96, (n - 1 + 0.1) / n),
            T in (Float32, Float64, Int64),
            sorted in (true, false)

            x = T <: Integer ? rand(T(1):T(30), n) : randn(T, n)
            xsort = sort(x)
            r = @inferred hdi(sorted ? xsort : x; prob, sorted)
            @test r isa ClosedInterval{T}
            l, u = IntervalSets.endpoints(r)
            interval_length = floor(Int, prob * n) + 1
            if T <: Integer
                @test sum(x -> l ≤ x ≤ u, x) ≥ interval_length
            else
                @test sum(x -> l ≤ x ≤ u, x) == interval_length
            end
            lind = 1:(n - interval_length + 1)
            uind = interval_length:n
            @assert all(collect(uind) .- collect(lind) .+ 1 .== interval_length)
            @test minimum(xsort[uind] - xsort[lind]) ≈ u - l

            @test hdi!(sorted ? xsort : x; prob, sorted) == r
        end
    end

    @testset "edge cases and errors" begin
        @testset "NaNs returned if contains NaNs" begin
            x = randn(1000)
            x[3] = NaN
            @test isequal(hdi(x), NaN .. NaN)
        end

        @testset "errors for empty array" begin
            x = Float64[]
            @test_throws ArgumentError hdi(x)
        end

        @testset "errors for 0-dimensional array" begin
            x = fill(1.0)
            @test_throws ArgumentError hdi(x)
        end

        @testset "test errors when prob is not in (0, 1)" begin
            x = randn(1_000)
            @testset for prob in (0, 1, -0.1, 1.1, NaN)
                @test_throws DomainError hdi(x; prob)
            end
        end
    end

    @testset "AbstractArray consistent with AbstractVector" begin
        @testset for sz in ((100, 2), (100, 2, 3), (100, 2, 3, 4)),
            prob in (0.72, 0.81),
            T in (Float32, Float64, Int64)

            x = T <: Integer ? rand(T(1):T(30), sz) : randn(T, sz)
            r = @inferred hdi(x; prob)
            if ndims(x) == 2
                @test r isa ClosedInterval{T}
                @test r == hdi(vec(x); prob)
            else
                @test r isa Array{ClosedInterval{T},ndims(x) - 2}
                r_slices = dropdims(
                    mapslices(x -> hdi(x; prob), x; dims=(1, 2)); dims=(1, 2)
                )
                @test r == r_slices
            end

            @test hdi!(copy(x); prob) == r
        end
    end

    @testset "OffsetArray" begin
        @testset for n in (100, 1_000), prob in (0.732, 0.864), T in (Float32, Float64)
            x = randn(T, (n, 2, 3, 4))
            xoff = OffsetArray(x, (-1, 2, -3, 4))
            r = hdi(x; prob)
            roff = @inferred hdi(xoff; prob)
            @test roff isa OffsetMatrix{ClosedInterval{T}}
            @test axes(roff) == (axes(xoff, 3), axes(xoff, 4))
            @test collect(roff) == r
        end
    end
end
