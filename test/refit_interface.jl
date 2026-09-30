using Distributions
using PosteriorStats
using Random
using StatsAPI: StatsAPI, nobs
using Test

# a wrapper that records what it is asked to fit, to check that the interface is enforced by
# the methods and not by a supertype, and that the statistics pass the indices they promise
struct CountingWrapper{Y}
    y::Y
    nrefits::Base.RefValue{Int}
    train_indices::Vector{Vector{Int}}
    eval_indices::Vector{Vector{Int}}
end
CountingWrapper(y) = CountingWrapper(y, Ref(0), Vector{Int}[], Vector{Int}[])

StatsAPI.nobs(wrapper::CountingWrapper) = length(wrapper.y)
function PosteriorStats.refit(rng, wrapper::CountingWrapper, train_indices, eval_indices)
    wrapper.nrefits[] += 1
    push!(wrapper.train_indices, collect(train_indices))
    push!(wrapper.eval_indices, collect(eval_indices))
    return rand(rng, conjugate_normal_posterior(wrapper.y, train_indices), 400, 4)
end
function PosteriorStats.refit_loglikelihoods(
    rng, wrapper::CountingWrapper, mu, eval_indices
)
    return logpdf.(Normal.(mu), reshape(wrapper.y[eval_indices], 1, 1, :))
end

@testset "Refitting interface" begin
    y = [0.3, -1.2, 0.7, 2.4, -0.1]

    @testset "a custom type needs no supertype" begin
        wrapper = CountingWrapper(y)
        @test nobs(wrapper) == length(y)

        fit = PosteriorStats.refit(Xoshiro(0), wrapper, [1, 2, 3, 4], [5])
        @test wrapper.nrefits[] == 1
        @test wrapper.train_indices == [[1, 2, 3, 4]]
        @test wrapper.eval_indices == [[5]]
        @test fit == PosteriorStats.refit(Xoshiro(0), wrapper, [1, 2, 3, 4], [5])
        @test size(PosteriorStats.refit_loglikelihoods(Xoshiro(0), wrapper, fit, [5])) ==
            (400, 4, 1)
    end

    @testset "returned log-likelihoods are validated" begin
        good_wrapper = conjugate_normal_wrapper(y)
        fit = PosteriorStats.refit(Xoshiro(0), good_wrapper, eachindex(y), eachindex(y))
        loo_result = loo(
            PosteriorStats.refit_loglikelihoods(Xoshiro(0), good_wrapper, fit, eachindex(y))
        )

        @testset "number of dimensions" begin
            wrapper = CallableWrapper(
                length(y);
                refit=(_, _, _) -> nothing,
                loglikelihoods=(_, _, _) -> randn(100, 4),
            )
            @test_throws DimensionMismatch reloo(wrapper, loo_result; k_threshold=(-Inf))
        end

        @testset "number of observations" begin
            wrapper = CallableWrapper(
                length(y);
                refit=(_, _, _) -> nothing,
                loglikelihoods=(_, _, eval_indices) ->
                    randn(100, 4, length(eval_indices) + 1),
            )
            @test_throws DimensionMismatch reloo(wrapper, loo_result; k_threshold=(-Inf))
        end
    end

    @testset "seeded map" begin
        rng = Xoshiro(2)
        xs = collect(1:7)
        # `f` receives a seeded copy of `rng`, so its draws depend on the seed only
        f(rng_x, x) = (x, rand(rng_x))
        serial = PosteriorStats._map_seeded(f, copy(rng), xs, 1)
        @test first.(serial) == xs
        @test PosteriorStats._map_seeded(f, copy(rng), xs, 1) == serial
        # different elements get different seeds, and a different rng different draws
        @test allunique(getindex.(serial, 2))
        @test getindex.(PosteriorStats._map_seeded(f, Xoshiro(3), xs, 1), 2) !=
            getindex.(serial, 2)
        @testset for ntasks in (2, 3, 7, 20)
            parallel = PosteriorStats._map_seeded(f, copy(rng), xs, ntasks)
            @test first.(parallel) == xs
            @test getindex.(parallel, 2) == getindex.(serial, 2)
            @test eltype(parallel) === eltype(serial)
        end
        # the rng given is advanced only by drawing the seeds
        rng_after = copy(rng)
        PosteriorStats._map_seeded(f, rng_after, xs, 3)
        @test rng_after == (rng_copy=copy(rng); rand(rng_copy, UInt, length(xs)); rng_copy)
        # an empty input
        @test PosteriorStats._map_seeded(f, copy(rng), Int[], 3) == []
        # a failure in one task is raised after the running calls complete
        g(rng_x, x) = x == 3 ? error("boom") : x
        @test_throws ErrorException PosteriorStats._map_seeded(g, copy(rng), xs, 1)
        @test_throws TaskFailedException PosteriorStats._map_seeded(g, copy(rng), xs, 2)
    end
end
