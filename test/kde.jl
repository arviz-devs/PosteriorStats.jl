using Distributions
using KernelDensity
using PosteriorStats
using Test

@testset "KDE" begin
    @testset "bandwidth_isj" begin
        @testset "converges to normal reference rule" begin
            n = 1_000_000
            x = randn(n)
            bw_norm = (0.75 * n)^(-1//5)  # Scott's (normal reference) rule
            @test !(PosteriorStats.isj_bandwidth(x) ≈ KernelDensity.default_bandwidth(x))
            @test PosteriorStats.isj_bandwidth(x) ≈ bw_norm rtol = 1e-2
        end

        VERSION > v"1.6" && @testset "bimodal consistent with unimodal" begin
            n = 1_000_000
            @testset for dist in (Normal(), TDist(3))
                dmix = MixtureModel([dist, dist + 300])
                x = rand(dmix, 2n)
                xhalf = rand(dist, n)
                bw_half = PosteriorStats.isj_bandwidth(xhalf)
                @test PosteriorStats.isj_bandwidth(x) ≈ bw_half rtol = 0.2
            end
        end
    end

    @testset "kde_reflected" begin
        @testset for dist in (Exponential(), Uniform(2, 10), Normal()),
            n in 10 .^ (4:6),
            npoints in (2048, 4096)

            x = rand(dist, n)
            bandwidth = KernelDensity.default_bandwidth(x)
            bounds = extrema(dist)
            kde_reflect = PosteriorStats.kde_reflected(x; bandwidth, npoints)
            kde_reflect_exact = PosteriorStats.kde_reflected(x; bandwidth, npoints, bounds)
            kde = KernelDensity.kde(x; bandwidth, npoints)
            @testset "basic properties" begin
                @test length(kde_reflect.x) == npoints
                @test sum(kde_reflect.density) * step(kde_reflect.x) ≈ 1
                if all(!isfinite, bounds)
                    @test kde_reflect_exact.x ≈ kde.x
                    @test kde_reflect_exact.density ≈ kde.density rtol = 1e-6
                end
            end
            @testset "obeys correct bounds" begin
                @test minimum(kde_reflect.x) ≥ bounds[1]
                @test maximum(kde_reflect.x) ≤ bounds[2]
            end

            @testset "similar to if we provide exact bounds" begin
                x_min = max(
                    minimum(kde_reflect_exact.x),
                    minimum(kde_reflect.x),
                    quantile(dist, 1e-3),
                )
                x_max = min(
                    maximum(kde_reflect_exact.x),
                    maximum(kde_reflect.x),
                    quantile(dist, 1 - 1e-3),
                )
                x_grid = range(x_min, x_max; length=500_000)
                pdf_reflect = KernelDensity.pdf(kde_reflect, x_grid)
                pdf_reflect_exact = KernelDensity.pdf(kde_reflect_exact, x_grid)
                @test pdf_reflect ≈ pdf_reflect_exact rtol = 1e-2
            end
            @testset "has lower integrated squared error than usual KDE" begin
                x_min = min(
                    minimum(kde_reflect_exact.x), minimum(kde.x), quantile(dist, 1e-3)
                )
                x_max = max(
                    maximum(kde_reflect_exact.x), maximum(kde.x), quantile(dist, 1 - 1e-3)
                )
                x_grid = range(x_min, x_max; length=500_000)
                pdf_reflect = KernelDensity.pdf(kde_reflect, x_grid)
                pdf_reflect_exact = KernelDensity.pdf(kde_reflect_exact, x_grid)
                pdf_direct = KernelDensity.pdf(kde, x_grid)
                pdf_true = pdf.(dist, x_grid)
                ise_reflect = sum(abs2, pdf_reflect - pdf_true)
                ise_reflect_exact = sum(abs2, pdf_reflect_exact - pdf_true)
                ise_direct = sum(abs2, pdf_direct - pdf_true)
                @test ise_reflect ≤ ise_direct ||
                    isapprox(ise_reflect, ise_direct; rtol=1e-3)
                @test ise_reflect_exact ≤ ise_reflect ||
                    isapprox(ise_reflect_exact, ise_reflect; rtol=0.1)
            end
        end
    end
end
