using IteratorInterfaceExtensions
using PosteriorStats
using Tables
using TableTraits
using Test

function _isequal(x::ModelComparisonResult, y::ModelComparisonResult)
    return Tables.columntable(x) == Tables.columntable(y)
end

@testset "compare" begin
    data = log_likelihood_eight_schools()
    eight_schools_loo_results = map(loo, data)
    mc1 = @inferred ModelComparisonResult compare(eight_schools_loo_results)

    @testset "basic checks" begin
        @test mc1.name == (:non_centered, :centered)
        @test mc1.rank == (non_centered=1, centered=2)
        @test _isapprox(
            mc1.elpd_diff,
            (
                non_centered=0.0,
                centered=(
                    eight_schools_loo_results.non_centered.estimates.elpd -
                    eight_schools_loo_results.centered.estimates.elpd
                ),
            ),
        )
        @test mc1.elpd_diff.non_centered == 0.0
        @test mc1.elpd_diff.centered > 0
        @test mc1.weight == NamedTuple{(:non_centered, :centered)}(
            PosteriorStats.model_weights(Stacking(),eight_schools_loo_results)
        )
        @test mc1.elpd_result ==
            NamedTuple{(:non_centered, :centered)}(eight_schools_loo_results)

        mc2 = compare(data; elpd_method=loo)
        @test _isequal(mc2, mc1)

        @test_throws ArgumentError compare(eight_schools_loo_results; model_names=[:foo])
    end

    @testset "keywords are forwarded" begin
        mc2 = compare(eight_schools_loo_results; method=PseudoBMA())
        @test !_isequal(mc2, compare(eight_schools_loo_results))
        @test mc2.method === PseudoBMA()
        mc3 = compare(eight_schools_loo_results; sort=false)
        for k in filter(!=(:method), propertynames(mc1))
            if k === :name
                @test getproperty(mc3, k) == reverse(getproperty(mc1, k))
            else
                @test getproperty(mc3, k) ==
                    NamedTuple{(:centered, :non_centered)}(getproperty(mc1, k))
            end
        end
        mc3 = compare(eight_schools_loo_results; model_names=[:a, :b])
        @test mc3.name == [:b, :a]
        mc4 = compare(eight_schools_loo_results; elpd_method=waic)
        @test !_isequal(mc4, mc2)
    end

    @testset "ModelComparisonResult" begin
        @testset "Tables interface" begin
            @test Tables.istable(typeof(mc1))
            @test Tables.columnaccess(typeof(mc1))
            @test Tables.columns(mc1) == mc1
            @test Tables.columnnames(mc1) == (
                :name, :rank, :elpd, :se_elpd, :elpd_diff, :se_elpd_diff, :weight, :p, :se_p
            )
            table = Tables.columntable(mc1)
            for k in (:name, :rank, :elpd_diff, :se_elpd_diff, :weight)
                @test getproperty(table, k) == collect(getproperty(mc1, k))
            end
            for k in (:elpd, :se_elpd, :p, :se_p)
                @test getproperty(table, k) ==
                    collect(map(x -> getproperty(x.estimates, k), mc1.elpd_result))
            end
            for (i, k) in enumerate(Tables.columnnames(mc1))
                @test Tables.getcolumn(mc1, i) == Tables.getcolumn(mc1, k)
            end
            @test_throws ArgumentError Tables.getcolumn(mc1, :foo)
            @test Tables.rowaccess(typeof(mc1))
            @test map(NamedTuple, Tables.rows(mc1)) ==
                map(NamedTuple, Tables.rows(Tables.columntable(mc1)))
        end

        @testset "TableTraits interface" begin
            @test IteratorInterfaceExtensions.isiterable(mc1)
            @test TableTraits.isiterabletable(mc1)
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(mc1), 1))[1]
            @test isequal(
                nt,
                (; (k => Tables.getcolumn(mc1, k)[1] for k in Tables.columnnames(mc1))...),
            )
            nt = collect(Iterators.take(IteratorInterfaceExtensions.getiterator(mc1), 2))[2]
            @test isequal(
                nt,
                (; (k => Tables.getcolumn(mc1, k)[2] for k in Tables.columnnames(mc1))...),
            )
        end

        @testset "show" begin
            mc5 = compare(eight_schools_loo_results; method=PseudoBMA())
            @test sprint(show, "text/plain", mc1) == """
                ModelComparisonResult with Stacking weights
                               rank  elpd  se_elpd  elpd_diff  se_elpd_diff  weight    p  se_p
                 non_centered     1   -31      1.5       0            0.0       1.0  0.9  0.32
                 centered         2   -31      1.4       0.03         0.061     0.0  0.9  0.33"""

            @test sprint(show, "text/plain", mc5) == """
                ModelComparisonResult with PseudoBMA weights
                               rank  elpd  se_elpd  elpd_diff  se_elpd_diff  weight    p  se_p
                 non_centered     1   -31      1.5       0            0.0      0.51  0.9  0.32
                 centered         2   -31      1.4       0.03         0.061    0.49  0.9  0.33"""

            @test startswith(sprint(show, "text/html", mc1), "<table")
        end
    end
end
