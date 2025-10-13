using MCMCDiagnosticTools
using PosteriorStats
using Statistics
using StatsBase
using Test

struct SampleWrapper{T,N}
    draws::T
    var_names::N
end

function PosteriorStats.summarize(
    spl::SampleWrapper, stats_funs...; var_names=spl.var_names, kwargs...
)
    x = spl.draws
    return summarize(x, stats_funs...; var_names, kwargs...)
end

_mean_and_std(x) = (mean=mean(x), std=std(x))

@testset "summarize" begin
    @testset "base cases" begin
        x = randn(1_000, 4, 3)
        stats1 = @inferred summarize(x, mean, std, median)
        @test stats1 isa SummaryStats
        @test getfield(stats1, :name) == "SummaryStats"
        @test stats1 == SummaryStats(
            (
                mean=map(mean, eachslice(x; dims=3)),
                std=map(std, eachslice(x; dims=3)),
                median=map(median, eachslice(x; dims=3)),
            );
            labels=axes(x, 3),
        )

        function _compute_stats(x)
            return summarize(x, (:mean, :std) => mean_and_std, :median => median)
        end
        stats2 = @inferred _compute_stats(x)
        @test stats2 == stats1

        stats3 = summarize(x, mean, std; var_names=["a", "b", "c"], name="Stats")
        @test getfield(stats3, :name) == "Stats"
        @test stats3 == SummaryStats(
            (mean=map(mean, eachslice(x; dims=3)), std=map(std, eachslice(x; dims=3)));
            labels=["a", "b", "c"],
        )

        stats4 = summarize(x; var_names=["a", "b", "c"], name="Stats")
        @test getfield(stats4, :name) == "Stats"
        stats5 = summarize(
            x, default_summary_stats()...; var_names=["a", "b", "c"], name="Stats"
        )
        @test stats4 == stats5

        stats6 = summarize(x, _mean_and_std, mad)
        @test haskey(stats6, :mean)
        @test haskey(stats6, :std)
        @test haskey(stats6, :mad)

        @test_throws DimensionMismatch summarize(x, mean; var_names=["a", "b"])
        @test_throws ArgumentError summarize(x, "std" => std)
        @test_throws ArgumentError summarize(x, ("mean", "std") => mean_and_std)
    end

    @testset "default stats function sets" begin
        @testset "array inputs" begin
            x = randn(1_000, 4, 3)

            @testset "all supported kinds accepted" begin
                @testset for kind in [
                    :all,
                    :stats,
                    :diagnostics,
                    :all_median,
                    :stats_median,
                    :diagnostics_median,
                ]
                    stats = summarize(x; kind)
                    @test stats isa SummaryStats
                    @test stats.name == "SummaryStats"
                    kind ∈ [:all, :stats] && @test haskey(stats, :mean)
                    kind ∈ [:all, :diagnostics] && @test haskey(stats, :ess_bulk)
                    kind ∈ [:all_median, :stats_median] && @test haskey(stats, :median)
                    if kind ∈ [:all_median, :diagnostics_median]
                        @test haskey(stats, :ess_median)
                    end
                end
            end

            # not completely type-inferrable due to CI
            stats1 = summarize(x, default_summary_stats()...)
            @test all(
                map(
                    _isapprox,
                    stats1,
                    summarize(
                        x,
                        mean,
                        std,
                        Symbol("eti89") => eti,
                        :ess_tail => (x -> ess(x; kind=:tail)),
                        :ess_bulk => (x -> ess(x; kind=:bulk)),
                        rhat,
                        :mcse_mean => mcse,
                        :mcse_std => (x -> mcse(x; kind=std)),
                    ),
                ),
            )
            stats2 = summarize(x; kind=:all_median, ci_fun=hdi, ci_prob=0.95)
            @test all(
                map(
                    _isapprox,
                    stats2,
                    summarize(
                        x,
                        median,
                        mad,
                        Symbol("hdi95") => (x -> PosteriorStats.hdi(vec(x); prob=0.95)),
                        :ess_median => (x -> ess(x; kind=median)),
                        :ess_tail => (x -> ess(x; kind=:tail)),
                        rhat,
                        :mcse_median => (x -> mcse(x; kind=median)),
                    ),
                ),
            )
            _compute_diagnostics(x) = summarize(x; kind=:diagnostics)
            stats3 = @inferred _compute_diagnostics(x)
            @test all(
                map(
                    _isapprox,
                    stats3,
                    summarize(
                        x,
                        :ess_tail => (x -> ess(x; kind=:tail)),
                        :ess_bulk => (x -> ess(x; kind=:bulk)),
                        rhat,
                        :mcse_mean => mcse,
                        :mcse_std => (x -> mcse(x; kind=std)),
                    ),
                ),
            )

            @test all(
                map(
                    _isapprox,
                    summarize(x; kind=:stats),
                    summarize(x, mean, std, Symbol("eti89") => eti),
                ),
            )

            x2 = convert(Array{Union{Float64,Missing}}, x)
            x2[1, 1, 1] = missing
            stats4 = summarize(x2)
            @test stats4[:mean] ≈ [mean(skipmissing(x2[:, :, 1])); stats1[:mean][2:end]]
            @test stats4[:std] ≈ [std(skipmissing(x2[:, :, 1])); stats1[:std][2:end]]
            @test stats4[Symbol("eti89")] == [
                eti(collect(skipmissing(x2[:, :, 1])))
                stats1[Symbol("eti89")][2:end]
            ]
            for k in (:ess_tail, :ess_bulk, :rhat, :mcse_mean, :mcse_std)
                @test stats4[k][1] === missing
                @test stats4[k][2:end] ≈ stats1[k][2:end]
            end

            stats5 = summarize(x2; kind=:all_median, ci_fun=hdi, ci_prob=0.95)
            @test stats5[:median] ≈
                [median(skipmissing(x2[:, :, 1])); stats2[:median][2:end]]
            @test stats5[:mad] ≈ [mad(skipmissing(x2[:, :, 1])); stats2[:mad][2:end]]
            @test stats5[Symbol("hdi95")] == [
                PosteriorStats.hdi(collect(skipmissing(x2[:, :, 1])); prob=0.95)
                stats2[Symbol("hdi95")][2:end]
            ]
            for k in (:ess_tail, :ess_median, :rhat, :mcse_median)
                @test stats5[k][1] === missing
                @test stats5[k][2:end] ≈ stats2[k][2:end]
            end
        end

        @testset "custom inputs" begin
            x = randn(1_000, 4, 3)
            sample = SampleWrapper(x, ["a", "b", "c"])
            _compute_diagnostics(x) = summarize(x; kind=:diagnostics, name="foo")
            stats1 = @inferred _compute_diagnostics(sample)
            @test stats1 isa SummaryStats
            @test stats1.name == "foo"
            @test stats1 ==
                summarize(x; kind=:diagnostics, name="foo", var_names=["a", "b", "c"])

            stats2 = summarize(sample)
            @test stats2.name == "SummaryStats"
            @test stats2 == summarize(x; var_names=["a", "b", "c"])
        end
    end
end
