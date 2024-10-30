using Distributions
using IntervalSets
using OffsetArrays
using PosteriorStats
using Statistics
using Test

function exact_hdi(dist::ContinuousUnivariateDistribution, prob::Real)
    if iszero(skewness(dist))
        return quantile(dist, (1 - prob) / 2) .. quantile(dist, 1 - (1 - prob) / 2)
    else
        xmin = islowerbounded(dist) ? minimum(dist) : quantile(dist, 1e-3)
        xmax = isupperbounded(dist) ? maximum(dist) : quantile(dist, 1 - 1e-3)
        bins = range(xmin, xmax; length=1_000_000)
        density = pdf.(dist, bins)
        bin_probs = density ./ sum(density)
        return PosteriorStats._hdi_from_bin_probs(bins, bin_probs, prob)
    end
end
function exact_hdi(dist::DiscreteUnivariateDistribution, prob::Real)
    bins = support(dist)
    bin_probs = pdf.(dist, bins)
    @assert sum(bin_probs) ≈ 1
    return PosteriorStats._hdi_from_bin_probs(bins, bin_probs, prob)
end

# common tests to all methods:
# - insensitive to sorting of input
# - NaNs returned if contains NaNs
# - errors for empty array
# - errors for 0-dimensional array
# - errors when prob is not in (0, 1)
# - AbstractArray consistent with AbstractVector
# - OffsetArray input produces OffsetArray output
# - works correctly for vectors, matrices, and higher-dimensional arrays

# method-specific tests:
# - MultimodalHDI:
#   - for a distribution with n uniformly-weighted, well-separated modes, the bounds for
#     `prob` should contain `n` modes, and the bounds for each mode should be approximately
#     the same as the `prob / n` bounds for just that mode
#   - symmetric continuous distribution:
#     - bounds approximately agree with those computed exactly using symmetric `Distributions.quantile`
#   - discrete distribution:
#     - bounds approximately agree with those that can be exactly computed using
#       `Distributions.support` and `Distributions.pmf`
#     - For unimodal distribution well-sampled, bounds agree with those computed by
#       `UnimodalHDI`
#   - if `sample=true`, bounds are entries from sample

@testset "hdi/hdi!" begin
    methods = [
        :unimodal => PosteriorStats.UnimodalHDI(),
        :multimodal => PosteriorStats.MultimodalHDI(PosteriorStats.KDEstimation(), false),
        :multimodal_sample =>
            PosteriorStats.MultimodalHDI(PosteriorStats.KDEstimation(), true),
    ]

    @testset "Common properties" begin
        @testset "$name" for (name, method) in methods
            @testset "eltype preserved" begin
                @testset for T in (Float32, Float64)
                    x = randn(T, 100)
                    prob = 0.9
                    interval = hdi(x; method, prob)
                    if name === :unimodal
                        @test eltype(interval) === T
                    else
                        @test eltype(interval) === ClosedInterval{T}
                    end
                end
            end

            @testset "order-insensitivity" begin
                x = randn(1_000)
                prob = 0.9
                @test hdi(x; method, prob) == hdi(sort(x); method, prob)
            end

            @testset "NaN handling" begin
                x = randn(1_000)
                x[3] = NaN
                interval = hdi(x; method)
                if method isa PosteriorStats.MultimodalHDI
                    @test isequal(interval, [NaN .. NaN])
                else
                    @test isequal(interval, NaN .. NaN)
                end
            end

            @testset "errors for empty array" begin
                @test_throws ArgumentError hdi(Float64[]; method)
            end

            @testset "errors for 0-dimensional array" begin
                @test_throws ArgumentError hdi(fill(0.0); method)
            end

            @testset "errors for prob not in (0, 1)" begin
                x = randn(1_000)
                @test_throws ArgumentError hdi(x; method, prob=-0.1)
                @test_throws ArgumentError hdi(x; method, prob=0.0)
                @test_throws ArgumentError hdi(x; method, prob=1.0)
                @test_throws ArgumentError hdi(x; method, prob=1.1)
            end

            @testset "Correctly maps over parameter dimensions" begin
                xarr = randn(100, 4, 3, 2)
                xmat = reshape(xarr, 100, 4, :)
                interval_4d = hdi(xarr; method, prob=0.9)
                @test size(interval_4d) == (3, 2)
                interval_3d = hdi(xmat; method, prob=0.9)
                @test size(interval_3d) == (6,)
                @test interval_4d == reshape(interval_3d, size(interval_4d))

                for (i, j) in Iterators.product(axes(xarr, 3), axes(xarr, 4))
                    @test interval_4d[i, j] == hdi(xarr[:, :, i, j]; method, prob=0.9)
                    @test interval_4d[i, j] == hdi(vec(xarr[:, :, i, j]); method, prob=0.9)
                end
            end

            @testset "Output keeps custom axes" begin
                @testset for n in [100], prob in [0.732, 0.864], T in [Float32, Float64]
                    x = randn(T, (n, 2, 3, 4))
                    xoff = OffsetArray(x, (-1, 2, -3, 4))
                    r = hdi(x; prob, method)
                    roff = @inferred hdi(xoff; prob, method)
                    @test roff isa OffsetMatrix{eltype(r)}
                    @test axes(roff) == (axes(xoff, 3), axes(xoff, 4))
                    @test collect(roff) == r
                end
            end
        end
    end

    @testset "public API" begin
        hdi_unimodal(x, prob) = hdi(x; method=:unimodal, prob)
        hdi_multimodal(x, prob) = hdi(x; method=:multimodal, prob)
        hdi_multimodal_sample(x, prob) = hdi(x; method=:multimodal_sample, prob)

        x = rand(1_000)
        @inferred hdi_unimodal(x, 0.9)
        @inferred hdi_multimodal(x, 0.9)
        @inferred hdi_multimodal_sample(x, 0.9)

        @testset "$name" for (name, method) in methods
            @test hdi(x; method=name, prob=0.9) == hdi(x; method, prob=0.9)
        end

        x = rand(Binomial(50, 0.5), 1_000)
        interval_discrete = hdi(
            x;
            method=PosteriorStats.MultimodalHDI(
                PosteriorStats.DiscreteDensityEstimation(), false
            ),
            prob=0.9,
        )
        @test eltype(interval_discrete) === ClosedInterval{Int}
        @test @inferred(hdi_multimodal(x, 0.9)) == interval_discrete
        @test @inferred(hdi_multimodal_sample(x, 0.9)) == interval_discrete
    end

    @testset "Method-specific behavior" begin
        @testset "UnimodalHDI" begin
            method = PosteriorStats.UnimodalHDI()
        end

        @testset "MultimodalHDI" begin
            @testset for sample in [false, true], discrete in [false, true]
                if sample && discrete
                    # this case is not implemented and not accessible via the public API
                    @test_skip false
                    continue
                end
                density_method = if discrete
                    PosteriorStats.DiscreteDensityEstimation()
                else
                    PosteriorStats.KDEstimation()
                end
                method = PosteriorStats.MultimodalHDI(density_method, sample)

                cdist = discrete ? Binomial(10, 0.5) : Normal()

                @testset "correct identification of separated modes" begin
                    @testset for n_modes in [2, 3], prob in [0.8, 0.9]
                        n = 1_000_000
                        centers = 100 .* (0:(n_modes - 1))
                        dmix = MixtureModel(Ref(cdist) .+ centers)
                        x_mixture = rand(dmix, n)

                        bounds_mixture = hdi(x_mixture; method, prob)
                        @test length(bounds_mixture) == n_modes

                        bounds_component = exact_hdi(cdist, prob)

                        for k in 1:n_modes
                            @test bounds_mixture[1].left + centers[k] ≈
                                bounds_mixture[k].left rtol = 0.05
                            @test bounds_mixture[1].right + centers[k] ≈
                                bounds_mixture[k].right rtol = 0.05
                        end
                    end
                end

                sample && @testset "bounds are drawn from entries" begin
                    dmix = MixtureModel([cdist, cdist + 10])
                    x = rand(dmix, 100)
                    intervals = hdi(x; method, prob)
                    for interval in intervals
                        @test interval.left ∈ x
                        @test interval.right ∈ x
                    end
                end
            end
        end
    end
end
