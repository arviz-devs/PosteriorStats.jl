using DimensionalData
using InferenceObjects
using PosteriorStats
using Random
using StatsBase
using Test

@testset "utils" begin
    @testset "log_likelihood" begin
        ndraws = 100
        nchains = 4
        nparams = 3
        x = randn(ndraws, nchains, nparams)
        log_like = convert_to_dataset((; x))
        @test PosteriorStats.log_likelihood(log_like) == x
        @test PosteriorStats.log_likelihood(log_like, :x) == x
        @test_throws Exception PosteriorStats.log_likelihood(log_like, :y)
        idata = InferenceData(; log_likelihood=log_like)
        @test PosteriorStats.log_likelihood(idata) == x
        @test PosteriorStats.log_likelihood(idata, :x) == x
        @test_throws Exception PosteriorStats.log_likelihood(idata, :y)

        y = randn(ndraws, nchains)
        log_like = convert_to_dataset((; x, y))
        @test_throws Exception PosteriorStats.log_likelihood(log_like)
        @test PosteriorStats.log_likelihood(log_like, :x) == x
        @test PosteriorStats.log_likelihood(log_like, :y) == y

        idata = InferenceData(; log_likelihood=log_like)
        @test_throws Exception PosteriorStats.log_likelihood(idata)
        @test PosteriorStats.log_likelihood(idata, :x) == x
        @test PosteriorStats.log_likelihood(idata, :y) == y

        # test old InferenceData versions
        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_likelihood=x))
        idata = InferenceData(; sample_stats)
        @test PosteriorStats.log_likelihood(idata) == x

        sample_stats = convert_to_dataset((; lp=randn(ndraws, nchains), log_like=x))
        idata = InferenceData(; sample_stats)
        @test_throws ArgumentError PosteriorStats.log_likelihood(idata)
        @test PosteriorStats.log_likelihood(idata, :log_like) == x

        idata = InferenceData()
        @test_throws ArgumentError PosteriorStats.log_likelihood(idata)
    end

    @testset "observations_and_predictions" begin
        @testset "unique names" begin
            @testset for pred_group in (:posterior, :posterior_predictive),
                y_name in (:y, :z),
                y_pred_name in (y_name, Symbol("$(y_name)_pred"))

                observed_data = namedtuple_to_dataset(
                    (; (y_name => randn(10),)...); default_dims=()
                )
                pred = namedtuple_to_dataset((; (y_pred_name => randn(100, 4, 10),)...))
                y = observed_data[y_name]
                y_pred = pred[y_pred_name]
                idata = InferenceData(; observed_data, pred_group => pred)

                obs_pred_exp = (y_name => y, y_pred_name => y_pred)
                @testset for y_name_hint in (y_name, nothing),
                    y_pred_name_hint in (y_pred_name, nothing)

                    obs_pred = @inferred PosteriorStats.observations_and_predictions(
                        idata, y_name_hint, y_pred_name_hint
                    )
                    @test obs_pred == obs_pred_exp
                end

                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, :foo, :bar
                )
                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, :foo, nothing
                )
                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, nothing, :bar
                )
            end
        end

        @testset "unique pairs" begin
            @testset for pred_group in (:posterior, :posterior_predictive),
                y_name in (:y, :z),
                y_pred_name in (y_name, Symbol("$(y_name)_pred"))

                observed_data = namedtuple_to_dataset(
                    (; (y_name => randn(10), :w => randn(3))...); default_dims=()
                )
                pred = namedtuple_to_dataset((;
                    (y_pred_name => randn(100, 4, 10), :q => randn(100, 4, 2))...
                ))
                y = observed_data[y_name]
                y_pred = pred[y_pred_name]
                idata = InferenceData(; observed_data, pred_group => pred)

                obs_pred_exp = (y_name => y, y_pred_name => y_pred)
                @testset for y_name_hint in (y_name, nothing),
                    y_pred_name_hint in (y_pred_name, nothing)

                    y_name_hint === nothing && y_pred_name_hint !== nothing && continue
                    obs_pred = PosteriorStats.observations_and_predictions(
                        idata, y_name_hint, y_pred_name_hint
                    )
                    @test obs_pred == obs_pred_exp
                end

                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, :foo, :bar
                )
                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, :foo, nothing
                )
                @test_throws Exception PosteriorStats.observations_and_predictions(
                    idata, nothing, :bar
                )
            end
        end

        @testset "non-unique names" begin
            @testset for pred_group in (:posterior, :posterior_predictive),
                pred_suffix in ("", "_pred")

                y_pred_name = Symbol("y$pred_suffix")
                z_pred_name = Symbol("z$pred_suffix")
                observed_data = namedtuple_to_dataset(
                    (; (:y => randn(10), :z => randn(5))...); default_dims=()
                )
                pred = namedtuple_to_dataset((;
                    (z_pred_name => randn(100, 4, 5), y_pred_name => randn(100, 4, 10))...
                ))
                y = observed_data.y
                z = observed_data.z
                y_pred = pred[y_pred_name]
                z_pred = pred[z_pred_name]
                idata = InferenceData(; observed_data, pred_group => pred)

                @testset for (name, pred_name) in ((:y, y_pred_name), (:z, z_pred_name)),
                    pred_name_hint in (pred_name, nothing)

                    @test PosteriorStats.observations_and_predictions(
                        idata, name, pred_name_hint
                    ) == (name => observed_data[name], pred_name => pred[pred_name])
                end

                @test_throws ArgumentError PosteriorStats.observations_and_predictions(idata)
                @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                    idata, nothing, nothing
                )
                @test_throws ErrorException PosteriorStats.observations_and_predictions(
                    idata, :foo, :bar
                )
                @test_throws ErrorException PosteriorStats.observations_and_predictions(
                    idata, :foo, nothing
                )
            end
        end

        @testset "missing groups" begin
            observed_data = namedtuple_to_dataset((; y=randn(10)); default_dims=())
            idata = InferenceData(; observed_data)
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(idata)
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, :y, :y_pred
            )
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, :y, nothing
            )
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, nothing, :y_pred
            )

            posterior_predictive = namedtuple_to_dataset((; y_pred=randn(100, 4, 10)))
            idata = InferenceData(; posterior_predictive)
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(idata)
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, :y, :y_pred
            )
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, :y, nothing
            )
            @test_throws ArgumentError PosteriorStats.observations_and_predictions(
                idata, nothing, :y_pred
            )
        end
    end

    @testset "_assimilar" begin
        @testset for x in ([8, 2, 5], (8, 2, 5), (; a=8, b=2, c=5))
            @test @inferred(PosteriorStats._assimilar((x=1.0, y=2.0, z=3.0), x)) ==
                (x=8, y=2, z=5)
            @test @inferred(PosteriorStats._assimilar((randn(3)...,), x)) == (8, 2, 5)
            dim = Dim{:foo}(["a", "b", "c"])
            y = DimArray(randn(3), dim)
            @test @inferred(PosteriorStats._assimilar(y, x)) == DimArray([8, 2, 5], dim)
        end
    end

    @testset "_sortperm/_permute" begin
        @testset for (x, y) in (
            [3, 1, 4, 2] => [1, 2, 3, 4],
            (3, 1, 4, 2) => (1, 2, 3, 4),
            (x=3, y=1, z=4, w=2) => (y=1, w=2, x=3, z=4),
        )
            perm = PosteriorStats._sortperm(x)
            @test perm == [2, 4, 1, 3]
            @test PosteriorStats._permute(x, perm) == y
        end
    end

    @testset "_eachslice" begin
        x = randn(2, 3, 4)
        slices = PosteriorStats._eachslice(x; dims=(3, 1))
        @test size(slices) == (size(x, 3), size(x, 1))
        slices = collect(slices)
        for i in axes(x, 3), j in axes(x, 1)
            @test slices[i, j] == x[j, :, i]
        end

        @test PosteriorStats._eachslice(x; dims=2) == PosteriorStats._eachslice(x; dims=(2,))

        if VERSION ≥ v"1.9-"
            for dims in ((3, 1), (2, 3), 3)
                @test PosteriorStats._eachslice(x; dims) === eachslice(x; dims)
            end
        end

        da = DimArray(x, (Dim{:a}(1:2), Dim{:b}(['x', 'y', 'z']), Dim{:c}(0:3)))
        for dims in (2, (1, 3), (3, 1), (2, 3), (:c, :a))
            @test PosteriorStats._eachslice(da; dims) === eachslice(da; dims)
        end
    end

    @testset "_draw_chains_params_array" begin
        chaindim = Dim{:chain}(1:4)
        drawdim = Dim{:draw}(1:2:200)
        paramdim1 = Dim{:param1}(0:1)
        paramdim2 = Dim{:param2}([:a, :b, :c])
        dims = (drawdim, chaindim, paramdim1, paramdim2)
        x = DimArray(randn(size(dims)), dims)
        xperm = permutedims(x, (chaindim, drawdim, paramdim1, paramdim2))
        @test @inferred PosteriorStats._draw_chains_params_array(xperm) ≈ x
        xperm = permutedims(x, (paramdim1, chaindim, drawdim, paramdim2))
        @test @inferred PosteriorStats._draw_chains_params_array(xperm) ≈ x
        xperm = permutedims(x, (paramdim1, drawdim, paramdim2, chaindim))
        @test @inferred PosteriorStats._draw_chains_params_array(xperm) ≈ x
    end

    @testset "_logabssubexp" begin
        x, y = rand(2)
        @test @inferred(PosteriorStats._logabssubexp(log(x), log(y))) ≈ log(abs(x - y))
        @test PosteriorStats._logabssubexp(log(y), log(x)) ≈ log(abs(y - x))
    end

    @testset "_sum_and_se" begin
        @testset for n in (100, 1_000), scale in (1, 5)
            x = randn(n) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x)
            @test s ≈ sum(x)
            @test se ≈ StatsBase.sem(x) * n

            x = randn(n, 10) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x; dims=1)
            @test s ≈ sum(x; dims=1)
            @test se ≈ mapslices(StatsBase.sem, x; dims=1) * n

            x = randn(10, n) * scale
            s, se = @inferred PosteriorStats._sum_and_se(x; dims=2)
            @test s ≈ sum(x; dims=2)
            @test se ≈ mapslices(StatsBase.sem, x; dims=2) * n
        end
        @testset "::Number" begin
            @test isequal(PosteriorStats._sum_and_se(2), (2, NaN))
            @test isequal(PosteriorStats._sum_and_se(3.5f0; dims=()), (3.5f0, NaN32))
        end
    end

    @testset "_log_mean" begin
        x = rand(1000)
        logx = log.(x)
        w = rand(1000)
        w ./= sum(w)
        logw = log.(w)
        @test PosteriorStats._log_mean(logx, logw) ≈ log(mean(x, StatsBase.fweights(w)))
        x = rand(1000, 4)
        logx = log.(x)
        @test PosteriorStats._log_mean(logx, logw; dims=1) ≈
            log.(mean(x, StatsBase.fweights(w); dims=1))
    end

    @testset "_se_log_mean" begin
        ndraws = 1_000
        @testset for n in (1_000, 10_000), scale in (1, 5)
            x = rand(n) * scale
            w = rand(n)
            w = StatsBase.weights(w ./ sum(w))
            logx = log.(x)
            logw = log.(w)
            se = @inferred PosteriorStats._se_log_mean(logx, logw)
            se_exp = std(log(mean(rand(n) * scale, w)) for _ in 1:ndraws)
            @test se ≈ se_exp rtol = 1e-1
        end
    end

    @testset "sigdigits_matching_se" begin
        @test PosteriorStats.sigdigits_matching_se(123.456, 0.01) == 5
        @test PosteriorStats.sigdigits_matching_se(123.456, 1) == 3
        @test PosteriorStats.sigdigits_matching_se(123.456, 0.0001) == 7
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.1) == 7
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.2; scale=5) == 6
        @test PosteriorStats.sigdigits_matching_se(1e4, 0.5) == 5
        @test PosteriorStats.sigdigits_matching_se(1e4, 0.5; scale=1) == 6
        @test PosteriorStats.sigdigits_matching_se(1e5, 0.1; sigdigits_max=2) == 2

        # errors
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(123.456, -1)
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(
            123.456, 1; sigdigits_max=-1
        )
        @test_throws ArgumentError PosteriorStats.sigdigits_matching_se(123.456, 1; scale=-1)

        # edge cases
        @test PosteriorStats.sigdigits_matching_se(0.0, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(NaN, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(Inf, 1) == 0
        @test PosteriorStats.sigdigits_matching_se(100, 1; scale=Inf) == 0
        @test PosteriorStats.sigdigits_matching_se(100, Inf) == 0
        @test PosteriorStats.sigdigits_matching_se(100, 0) == 7
        @test PosteriorStats.sigdigits_matching_se(100, 0; sigdigits_max=2) == 2
    end

    @testset "_printf_with_sigdigits" begin
        @test PosteriorStats._printf_with_sigdigits(123.456, 1) == "1.e+02"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 1) == "-1.e+02"
        @test PosteriorStats._printf_with_sigdigits(123.456, 2) == "1.2e+02"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 2) == "-1.2e+02"
        @test PosteriorStats._printf_with_sigdigits(123.456, 3) == "123"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 3) == "-123"
        @test PosteriorStats._printf_with_sigdigits(123.456, 4) == "123.5"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 4) == "-123.5"
        @test PosteriorStats._printf_with_sigdigits(123.456, 5) == "123.46"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 5) == "-123.46"
        @test PosteriorStats._printf_with_sigdigits(123.456, 6) == "123.456"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 6) == "-123.456"
        @test PosteriorStats._printf_with_sigdigits(123.456, 7) == "123.4560"
        @test PosteriorStats._printf_with_sigdigits(-123.456, 7) == "-123.4560"
        @test PosteriorStats._printf_with_sigdigits(123.456, 8) == "123.45600"
        @test PosteriorStats._printf_with_sigdigits(0.00000123456, 1) == "1.e-06"
        @test PosteriorStats._printf_with_sigdigits(0.00000123456, 2) == "1.2e-06"
    end

    @testset "ft_printf_sigdigits" begin
        @testset "all columns" begin
            @testset for sigdigits in 1:5
                ft1 = PosteriorStats.ft_printf_sigdigits(sigdigits)
                for i in 1:10, j in 1:5
                    v = randn()
                    @test ft1(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    @test ft1("foo", i, j) == "foo"
                end
            end
        end
        @testset "subset of columns" begin
            @testset for sigdigits in 1:5
                ft = PosteriorStats.ft_printf_sigdigits(sigdigits, [2, 3])
                for i in 1:10, j in 1:5
                    v = randn()
                    if j ∈ [2, 3]
                        @test ft(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    else
                        @test ft(v, i, j) === v
                    end
                    @test ft("foo", i, j) == "foo"
                end
            end
        end
    end

    @testset "ft_printf_sigdigits_matching_se" begin
        @testset "all columns" begin
            @testset for scale in 1:3
                se = rand(5)
                ft = PosteriorStats.ft_printf_sigdigits_matching_se(se; scale)
                for i in eachindex(se), j in 1:5
                    v = randn()
                    sigdigits = PosteriorStats.sigdigits_matching_se(v, se[i]; scale)
                    @test ft(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                    @test ft("foo", i, j) == "foo"
                end
            end
        end

        @testset "subset of columns" begin
            @testset for scale in 1:3
                se = rand(5)
                ft = PosteriorStats.ft_printf_sigdigits_matching_se(se, [2, 3]; scale)
                for i in eachindex(se), j in 1:5
                    v = randn()
                    if j ∈ [2, 3]
                        sigdigits = PosteriorStats.sigdigits_matching_se(v, se[i]; scale)
                        @test ft(v, i, j) == PosteriorStats._printf_with_sigdigits(v, sigdigits)
                        @test ft("foo", i, j) == "foo"
                    else
                        @test ft(v, i, j) === v
                    end
                end
            end
        end
    end
end
